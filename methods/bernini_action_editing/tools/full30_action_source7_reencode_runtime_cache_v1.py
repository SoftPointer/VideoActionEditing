#!/usr/bin/env python3
"""Fail-closed r4 local MIOpen cache preparation and post-child cleanup.

The preparation and cache-cleanup paths are deliberately Python-stdlib-only.
Only the post-cleanup shared-filesystem audit lazily imports the exact pinned
PyTorch build, on CPU, to reopen the seven published bare tensors with
``weights_only=True``.  The numbered child runs ``prepare`` before starting
the controller, which makes a fresh step-scoped ext4 cache tree and proves
ordinary and SQLite writes.  The same compute-node child may run ``cleanup``
only after the controller process returns success and before the numbered
step exits.  Failed controller runs retain the fresh tree for audit and the
step-scoped name can never be reused.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import socket
import sqlite3
import stat
import sys
from typing import Any, Iterable, Mapping, NoReturn, Optional, Sequence
from urllib.parse import quote


EXPERIMENT_ID = "BOX-EXP-014"
RUN_GENERATION = "r4"
PREPARE_SCHEMA = "bernini-full30-action-source7-reencode-runtime-cache-prepare-v2"
CLEANUP_SCHEMA = "bernini-full30-action-source7-reencode-runtime-cache-cleanup-v2"
CONTROLLER_COMPLETION_SCHEMA = (
    "bernini-full30-action-source7-reencode-controller-completion-v4"
)
CACHE_PARENT = Path("/tmp")
CACHE_PREFIX = "BOX-EXP-014-r4"
EXPECTED_STATFS_TYPE = "ext2/ext3"
EXPECTED_STATFS_MAGIC = 0xEF53
EXPECTED_MOUNT_FSTYPE = "ext4"
EXPECTED_MOUNT_POINT = "/"
EXPECTED_MOUNT_SOURCE = "/dev/mapper/vgroot-lvroot"
EXPECTED_COMPUTE_NODE = "auh7-1b-gpu-299"
EXPECTED_MIOPEN_KERNEL_DB_BASENAME = "gfx90a68.ukdb"
EXPECTED_MIOPEN_KERN_DB_COLUMNS = (
    (0, "id", "INTEGER", 0, None, 1),
    (1, "kernel_name", "TEXT", 1, None, 0),
    (2, "kernel_args", "TEXT", 1, None, 0),
    (3, "kernel_blob", "BLOB", 1, None, 0),
    (4, "kernel_hash", "TEXT", 1, None, 0),
    (5, "uncompressed_size", "INT", 1, None, 0),
)
EXPECTED_MIOPEN_KERN_DB_UNIQUE_INDEX = (
    (0, 1, "kernel_name"),
    (1, 2, "kernel_args"),
)
EXPECTED_TORCH_VERSION = "2.7.1+rocm6.3"
EXPECTED_TORCH_HIP_VERSION = "6.3.42131-fa1d09cbd"
EXPECTED_MIOPEN_BACKEND_VERSION = 3003000
EXPECTED_MIOPEN_LIBRARY_PATH = (
    "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/"
    "site-packages/torch/lib/libMIOpen.so"
)
EXPECTED_MIOPEN_LIBRARY_SIZE = 690355265
EXPECTED_MIOPEN_LIBRARY_SHA256 = (
    "1e6cc33ca21951dce12795e6c5d99578e8f2f1754b84a703508df44426b44b52"
)
EXPECTED_MIOPEN_EMBEDDED_VERSION = "3.3.0.a85ca8a54-dirty"
EXPECTED_MIOPEN_USER_DB_MAIN_BASENAMES = frozenset(
    {
        "gfx90a68.HIP.3_3_0_a85ca8a54-dirty.udb.txt",
        "gfx90a68.HIP.3_3_0_a85ca8a54-dirty.ufdb.txt",
    }
)
EXPECTED_MIOPEN_USER_DB_TIME_BASENAMES = frozenset(
    f"{name}.time" for name in EXPECTED_MIOPEN_USER_DB_MAIN_BASENAMES
)
EXPECTED_MIOPEN_KERNEL_CACHE_BASENAMES = frozenset(
    {
        "gfx90a68.ukdb",
        "gfx90a68.ukdb-shm",
        "gfx90a68.ukdb-wal",
    }
)
EXPECTED_MIOPEN_LOCK_DIRECTORY_BASENAME = "miopen-lockfiles"
GLOBAL_MIOPEN_LOCK_ROOT = Path("/tmp/miopen-lockfiles")
EXPECTED_VAE_CONFIG_SHA256 = (
    "f0c1cc1d7decb5badc384f54691746a27a9aeff49f7ebca974e583389342d527"
)
WAN_RESAMPLE_CANDIDATE_GEOMETRIES = (
    (96, 544, 432, 545, 433, 272, 216),
    (192, 272, 216, 273, 217, 136, 108),
    (384, 136, 108, 137, 109, 68, 54),
)
EXACT7_POSTERIOR_SHAPES = {
    "57cda7597d924dbb": [1, 32, 21, 68, 54],
    "6d4a7f95a52e47e9": [1, 32, 21, 68, 54],
    "a0b66487ab68498a": [1, 32, 21, 82, 46],
    "38b113317af14f01": [1, 32, 21, 74, 50],
    "5ae60e8417244e6e": [1, 32, 21, 68, 54],
    "1149c58e43e54add": [1, 32, 21, 68, 54],
    "a535e13301e448d7": [1, 32, 21, 70, 52],
}
EXACT7_FROZEN_ROWS: tuple[Mapping[str, Any], ...] = (
    {
        "iid": "57cda7597d924dbb",
        "source_video_sha256": "6409f59896c50f0d19dff7ac1e67f37362aa57968bc64a21a9a8271f5a85fec8",
        "expected_posterior_shape": [1, 32, 21, 68, 54],
        "group_id": "fb3bfcf6924f5cf4a9de6f2ad6c48c8b6ac53a27139d0b81d4873fcf1dfd9b11",
        "actor_id": "adult-man-paddleboard-cap",
        "scene_id": "outdoor-lake-paddleboard",
    },
    {
        "iid": "6d4a7f95a52e47e9",
        "source_video_sha256": "d8ea67b2f1ada75894cd3d2b55f877fd7ffd3b0f127119288fbdec37c5925b1a",
        "expected_posterior_shape": [1, 32, 21, 68, 54],
        "group_id": "8ade2c60115591a3061017a1eaf5484c3f12be587f9bb6ba71f10ab368909ff3",
        "actor_id": "adult-woman-black-lace-dancer",
        "scene_id": "outdoor-grass-field",
    },
    {
        "iid": "a0b66487ab68498a",
        "source_video_sha256": "62dbdf50c385233087919b686a0c5d064ce1ae47170ac477f6eeb320a885afa7",
        "expected_posterior_shape": [1, 32, 21, 82, 46],
        "group_id": "e43a520e42e7321953372d28465d5e12aece253746037c9bc1e4c081474dc2ce",
        "actor_id": "adult-woman-blonde-hair",
        "scene_id": "indoor-modern-room",
    },
    {
        "iid": "38b113317af14f01",
        "source_video_sha256": "eece8b2a298a5488fb689c736ceb074842ceb61e07090aacd7ea9a34d48e2fd6",
        "expected_posterior_shape": [1, 32, 21, 74, 50],
        "group_id": "5defa9bd595abe28bad4e3941b5ad9e587c7fa2b49c9288840f8d0a1bb9cd977",
        "actor_id": "adult-woman-sleep-mask",
        "scene_id": "indoor-beige-sofa",
    },
    {
        "iid": "5ae60e8417244e6e",
        "source_video_sha256": "88d7bd4601f6f8c16e8a9d0bbdb5cb75f0c77538061e17e30095ecf1a620ea99",
        "expected_posterior_shape": [1, 32, 21, 68, 54],
        "group_id": "2861dbcfbf32e5153f375f30b9cf7e1d02b3d1dcd222768578e50c6ae53b3e02",
        "actor_id": "adult-woman-pigtails-glasses",
        "scene_id": "indoor-floor-by-sofa",
    },
    {
        "iid": "1149c58e43e54add",
        "source_video_sha256": "d24e32daf499850b33b40f13cd537c11fd8a230eda6fa121a24c33e0a79dfe7a",
        "expected_posterior_shape": [1, 32, 21, 68, 54],
        "group_id": "93d16dbf9aa20a9690c654fe9417f26156b178563ba0d7122f41060786ef0525",
        "actor_id": "adult-woman-white-dress-blue-stage",
        "scene_id": "blue-lit-stage-floor",
    },
    {
        "iid": "a535e13301e448d7",
        "source_video_sha256": "70ccabb237fd8a2a159d0cbf40fb53d21fb070c3d280d974aa28ec58d0d1130c",
        "expected_posterior_shape": [1, 32, 21, 70, 52],
        "group_id": "48b23799bc218750dcd7a9efc3fa920e9a83058bac76b15b7b48457eaadec2cd",
        "actor_id": "adult-woman-white-dress-ponytail",
        "scene_id": "outdoor-stone-fountain",
    },
)
EXPECTED_SOURCE_BASE = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
    "VideoEditing/VideoEdit_experiments/goku_action_wan22_20260730T043022Z/"
    "fullmotion_next1000_v17_20260803T133300Z/wan_next1000_v17/samples"
)
EXPECTED_CHECKPOINT_ROOT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4"
)
EXPECTED_CHECKPOINT_CONTENT_MANIFEST = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_appearance_counterfactual_20260808_74ed30c/"
    "runtime/methods/bernini_action_editing/audits/"
    "bernini_r13_ff4c5d4_checkpoint.sha256"
)
EXPECTED_CHECKPOINT_CONTENT_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)
EXPECTED_EXTERNAL_INDEX0 = {
    "iid": "2d2e28871a5a4856",
    "source_video_sha256": "f12797b095b2108140c32e9ff0cf8ec6ff2af9c5e00dadc086d3f3abe02588d9",
    "group_id": "a5b4f1766ed70b7349c91950143a33366cb2b5e20969163392cf8d1a0920d9cb",
    "source_posterior_index0_path": (
        "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
        "VideoEdit_experiments/bernini_preservation_recovery_20260814/runs/"
        "source-only-v3-64-16-8-e2b65a33690a-r1/physical_source_posterior_index0/"
        "2d2e28871a5a4856.source-posterior-index0.pt"
    ),
    "source_posterior_index0_file_sha256": "b6e54e36ca1be4c58c7925dc1a2c11b1f5a3e65443508affbc0c5d9dd6fa9dee",
    "source_posterior_tensor_raw_sha256": "a65a8a7f73f071d555137e96ddcd53a61aee80704d7acc177daa42a44426f8e6",
    "expected_posterior_shape": [1, 32, 21, 82, 46],
    "external_bound_input_only": True,
    "opened_by_exact7_reencode": False,
    "reencoded": False,
}
EXPECTED_NEGATIVE_ACCESS = {
    "source_only_reencode_from_source_video": True,
    "vae_encode_calls_per_source": 1,
    "paired_dataset_accessed": False,
    "legacy_source_target_container_opened": False,
    "synthetic_target_index1_path_read": False,
    "synthetic_target_index1_bytes_read": False,
    "synthetic_target_index1_decoded": False,
    "synthetic_target_index1_filtered_on": False,
    "synthetic_target_index1_hashed": False,
    "target_video_path_present": False,
    "target_video_accessed": False,
}
EXACT7_OUTPUT_FILENAMES = tuple(
    f"{row['iid']}.source-posterior-index0.pt" for row in EXACT7_FROZEN_ROWS
)
COMPLETION_FIELDS = frozenset(
    {
        "schema_version", "experiment_id", "run_generation", "complete",
        "purpose", "scientific_target", "learning_target", "numeric_target",
        "dataset", "steps", "baseline", "core_validation", "holder",
        "release", "runtime_cache", "cuda_miopen_smoke",
        "runtime_cache_post_materialization", "plan", "materialization",
        "external_existing_index0", "external_existing_index0_reencoded",
        "inventory_snapshot_only", "exact8_authority_go_claimed",
        "teacher_cross_disjointness_pending", "optimizer_created",
        "optimizer_updates", "training_authorized", "completion_digest",
    }
) | frozenset(EXPECTED_NEGATIVE_ACCESS)
HOLDER_FIELDS = frozenset(
    {
        "job_id", "step_id", "node", "parent_retained", "parent_cancelled",
        "parent_released", "parent_requeued",
    }
)
RELEASE_BINDING_FIELDS = frozenset(
    {
        "manifest_path", "manifest_file_sha256", "manifest_digest",
        "content_closure_sha1", "release_generation",
    }
)
PLAN_BINDING_FIELDS = frozenset({"path", "file_sha256", "plan_digest"})
MATERIALIZATION_BINDING_FIELDS = frozenset(
    {
        "receipt_path", "receipt_file_sha256", "receipt_digest",
        "post_publish_rows", "all_seven_physical_files_and_tensors_reopened",
    }
)
POST_PUBLISH_ROW_FIELDS = frozenset(
    {
        "iid", "path", "file_sha256", "tensor_sha256",
        "tensor_raw_sha256", "shape",
        "physical_file_and_tensor_reopened_post_publish",
    }
)
RUNTIME_BINDING_FIELDS = frozenset(
    {
        "prepare_receipt_path", "prepare_receipt_file_sha256", "prepare_digest",
        "cache_root", "hostname", "cache_root_device", "cache_root_inode",
        "filesystem", "directories", "environment", "home_unchanged",
        "created_fresh_create_only", "exclusive_fsync_probe",
        "sqlite_commit_reopen_probe", "post_probe_empty_inventory",
        "global_miopen_lock_root_before_torch", "validated_before_torch_import",
        "cache_reusable", "cleanup_policy",
    }
)
FINAL_CACHE_FIELDS = frozenset(
    {
        "captured_after_exact7_materialization", "cache_root", "inventory",
        "miopen_kernel_db_evidence", "miopen_user_db_evidence",
        "scoped_miopen_temp_lock_evidence",
        "global_miopen_lock_root_after_exact7",
        "global_miopen_lock_root_metadata_unchanged",
        "global_miopen_lock_root_members_scanned",
        "global_miopen_lock_root_mutation_attempted",
    }
)
SUBDIRECTORY_NAMES = ("user-db", "kernel-cache", "xdg-cache", "tmp")
_TOKEN_RE = re.compile(r"(?:0|[1-9][0-9]{0,19})")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MODE_0700 = 0o700
_MODE_0600 = 0o600
_MODE_0777 = 0o777
_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_FILE_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
PREPARE_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "run_generation",
        "slurm_job_id",
        "slurm_step_id",
        "cache_parent",
        "cache_root",
        "cache_name",
        "hostname",
        "filesystem",
        "owner_uid",
        "owner_gid",
        "directory_mode",
        "cache_root_device",
        "cache_root_inode",
        "directories",
        "environment",
        "home_unchanged",
        "created_fresh_create_only",
        "canonical_paths_verified",
        "nofollow_verified",
        "exclusive_fsync_probe",
        "sqlite_commit_reopen_probe",
        "post_probe_empty_inventory",
        "global_miopen_lock_root_before_torch",
        "torch_imported_during_prepare",
        "cache_reusable",
        "cleanup_policy",
        "prepare_digest",
    }
)
CLEANUP_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "run_generation",
        "prepare_receipt_path",
        "prepare_receipt_file_sha256",
        "prepare_digest",
        "controller_completion_path",
        "controller_completion_file_sha256",
        "controller_completion_digest",
        "cache_root",
        "cache_root_device",
        "cache_root_inode",
        "cleanup_node",
        "controller_exit_status",
        "cleanup_after_controller_exit",
        "cleanup_before_numbered_step_exit",
        "controller_complete",
        "cache_root_removed",
        "cache_root_reusable",
        "scoped_miopen_temp_lock_root_removed",
        "global_miopen_lock_root_members_scanned",
        "global_miopen_lock_root_cleanup_attempted",
        "cleanup_digest",
    }
)
RETAINED_FAILURE_SCHEMA = (
    "bernini-full30-action-source7-reencode-runtime-cache-retained-failure-v3"
)
RETAINED_FAILURE_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "run_generation",
        "prepare_receipt_path",
        "prepare_receipt_file_sha256",
        "prepare_digest",
        "hostname",
        "cache_root",
        "cache_root_device",
        "cache_root_inode",
        "filesystem",
        "controller_exit_status",
        "controller_complete",
        "cache_root_present",
        "cache_root_retained",
        "cache_root_reusable",
        "scoped_miopen_temp_lock_root_observation",
        "scoped_miopen_temp_lock_root_present",
        "scoped_miopen_temp_lock_root_retained",
        "global_miopen_lock_root_members_scanned",
        "global_miopen_lock_root_cleanup_attempted",
        "retained_failure_digest",
    }
)
PHASE_FAILURE_SCHEMA = (
    "bernini-full30-action-source7-reencode-runtime-cache-phase-failure-v1"
)
PHASE_FAILURE_CHOICES = frozenset(
    {
        "prepare-or-precontroller",
        "controller-or-retained-terminal",
        "cleanup-or-cleanup-receipt-publication",
        "post-cleanup-audit",
    }
)
PHASE_FAILURE_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "run_generation",
        "phase",
        "failure_exit_status",
        "hostname",
        "slurm_job_id",
        "slurm_step_id",
        "cache_root",
        "cache_root_observation",
        "cache_root_present",
        "cache_root_retained",
        "cache_root_absent_at_terminal",
        "cleanup_may_have_removed_cache_before_terminal",
        "cache_root_reusable",
        "artifacts",
        "global_miopen_lock_root_members_scanned",
        "global_miopen_lock_root_cleanup_attempted",
        "success_claimed",
        "final_marker_authorized",
        "phase_failure_digest",
    }
)
SMOKE_FIELDS = frozenset(
    {
        "schema_version", "experiment_id", "run_generation", "complete",
        "prepare_receipt_path", "prepare_receipt_file_sha256",
        "prepare_digest", "cache_root", "environment",
        "miopen_user_db_path_kind", "miopen_custom_cache_dir_kind",
        "miopen_cache_paths_remained_canonical_0700_directories",
        "pre_conv_cache_inventory", "post_conv_cache_inventory",
        "miopen_kernel_db_activity_required",
        "miopen_kernel_db_activity_observed", "miopen_kernel_db_evidence",
        "kernel_cache_claim", "miopen_user_db_claim",
        "miopen_user_db_evidence",
        "scoped_miopen_temp_lock_activity_required",
        "scoped_miopen_temp_lock_activity_observed",
        "scoped_miopen_temp_lock_evidence",
        "tmpdir_cpp_temp_directory_path_redirect_observed",
        "global_miopen_lock_root_before_torch",
        "global_miopen_lock_root_after_smoke",
        "global_miopen_lock_root_metadata_unchanged",
        "global_miopen_lock_root_members_scanned",
        "global_miopen_lock_root_mutation_attempted",
        "torch_import_after_cache_validation", "pinned_runtime",
        "loaded_miopen_library_paths",
        "loaded_miopen_library_unique_exact_path", "backend",
        "device_index", "device_name", "operation", "r2_failure_step",
        "r2_failure_stack_location", "r2_failure_stack_candidate_closure",
        "vae_config_sha256", "geometry_count", "geometries",
        "module_and_input_declared_dtype", "cuda_autocast_dtype",
        "peak_allocated_bytes", "gpu_cache_cleared",
        "gpu_memory_allocated_after_clear", "source_video_opened",
        "source_video_decoded", "vae_encode_calls", "smoke_digest",
    }
)


class Source7RuntimeCacheError(RuntimeError):
    """Raised before an unsafe or unproved runtime cache can be used."""


def fail(message: str) -> NoReturn:
    raise Source7RuntimeCacheError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _short_hostname() -> str:
    return socket.gethostname().split(".", 1)[0]


def _closed_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _token(value: Any, *, label: str) -> str:
    require(type(value) is str and _TOKEN_RE.fullmatch(value) is not None, f"{label} token differs")
    return value


def expected_cache_root(
    job_id: str, step_id: str, *, cache_parent: Path = CACHE_PARENT
) -> Path:
    job = _token(job_id, label="SLURM_JOB_ID")
    step = _token(step_id, label="SLURM_STEP_ID")
    require(cache_parent.is_absolute() and cache_parent.name != "", "cache parent differs")
    return cache_parent / f"{CACHE_PREFIX}-{job}-{step}"


def _metadata(path: Path, *, directory: bool, mode: int, uid: int) -> os.stat_result:
    try:
        value = path.lstat()
    except OSError as error:
        raise Source7RuntimeCacheError(f"runtime cache path unavailable: {path}") from error
    require(not stat.S_ISLNK(value.st_mode), f"runtime cache path is a symlink: {path}")
    kind_matches = stat.S_ISDIR(value.st_mode) if directory else stat.S_ISREG(value.st_mode)
    require(kind_matches, f"runtime cache path kind differs: {path}")
    require(value.st_uid == uid, f"runtime cache owner differs: {path}")
    require(stat.S_IMODE(value.st_mode) == mode, f"runtime cache mode differs: {path}")
    require(path.resolve(strict=True) == path, f"runtime cache path is not canonical: {path}")
    return value


def _decode_mount_field(value: str) -> str:
    return (
        value.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def _filesystem_identity(path: Path) -> Mapping[str, Any]:
    """Bind both Linux statfs and mountinfo identities for gpu299 local /tmp."""

    class LinuxStatFS(ctypes.Structure):
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

    libc = ctypes.CDLL(None, use_errno=True)
    statfs_value = LinuxStatFS()
    if libc.statfs(os.fsencode(path), ctypes.byref(statfs_value)) != 0:
        error_number = ctypes.get_errno()
        raise Source7RuntimeCacheError("statfs for runtime cache failed") from OSError(
            error_number, os.strerror(error_number)
        )
    statfs_magic = int(statfs_value.f_type) & ((1 << (8 * ctypes.sizeof(ctypes.c_long))) - 1)
    statfs_type = EXPECTED_STATFS_TYPE if statfs_magic == EXPECTED_STATFS_MAGIC else "unknown"

    try:
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise Source7RuntimeCacheError("Linux mountinfo is unavailable") from error
    candidates: list[tuple[int, str, str, str]] = []
    path_text = str(path)
    for line in lines:
        if " - " not in line:
            continue
        left, right = line.split(" - ", 1)
        left_fields = left.split()
        right_fields = right.split()
        if len(left_fields) < 6 or len(right_fields) < 2:
            continue
        mount_point = _decode_mount_field(left_fields[4])
        try:
            inside = os.path.commonpath((path_text, mount_point)) == mount_point
        except ValueError:
            inside = False
        if inside:
            candidates.append(
                (len(mount_point), mount_point, right_fields[0], _decode_mount_field(right_fields[1]))
            )
    require(bool(candidates), "runtime cache mount identity is unavailable")
    _, mount_point, fs_type, source = max(candidates)
    local = (
        statfs_type == EXPECTED_STATFS_TYPE
        and statfs_magic == EXPECTED_STATFS_MAGIC
        and fs_type == EXPECTED_MOUNT_FSTYPE
        and mount_point == EXPECTED_MOUNT_POINT
        and source == EXPECTED_MOUNT_SOURCE
        and source.startswith("/dev/")
    )
    return {
        "statfs_type": statfs_type,
        "statfs_magic_hex": f"0x{statfs_magic:x}",
        "mount_fstype": fs_type,
        "mount_point": mount_point,
        "source": source,
        "source_is_local_block_device": source.startswith("/dev/"),
        "local_filesystem": local,
    }


def _open_directory(path: Path) -> int:
    try:
        return os.open(path, _DIRECTORY_OPEN_FLAGS)
    except OSError as error:
        raise Source7RuntimeCacheError(f"cannot open runtime cache directory without following links: {path}") from error


def _fsync_directory(path: Path) -> None:
    descriptor = _open_directory(path)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_fsync_probe(directory: Path, *, uid: int) -> Mapping[str, Any]:
    name = ".box-exp-014-r4-o-excl-fsync-probe"
    payload = b"BOX-EXP-014-r4-exclusive-fsync-probe\n"
    directory_fd = _open_directory(directory)
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(
            name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | _FILE_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            _MODE_0600,
            dir_fd=directory_fd,
        )
        written = os.write(descriptor, payload)
        require(written == len(payload), "exclusive probe short write")
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        require(
            stat.S_ISREG(metadata.st_mode)
            and metadata.st_uid == uid
            and stat.S_IMODE(metadata.st_mode) == _MODE_0600,
            "exclusive probe metadata differs",
        )
        os.close(descriptor)
        descriptor = None
        reopened = os.open(name, os.O_RDONLY | _FILE_NOFOLLOW, dir_fd=directory_fd)
        try:
            observed = os.read(reopened, len(payload) + 1)
        finally:
            os.close(reopened)
        require(observed == payload, "exclusive probe physical reopen differs")
        os.unlink(name, dir_fd=directory_fd)
        os.fsync(directory_fd)
    except OSError as error:
        raise Source7RuntimeCacheError("exclusive O_EXCL/fsync probe failed") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory_fd)
    return {
        "o_excl": True,
        "o_nofollow": True,
        "file_fsync": True,
        "physical_reopen": True,
        "payload_match": True,
        "probe_removed": True,
    }


def _sqlite_commit_reopen_probe(directory: Path, *, uid: int) -> Mapping[str, Any]:
    path = directory / ".box-exp-014-r4-sqlite-probe.sqlite3"
    require(not path.exists() and not path.is_symlink(), "SQLite probe path is not fresh")
    connection: Optional[sqlite3.Connection] = None
    try:
        connection = sqlite3.connect(path, timeout=5.0)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("CREATE TABLE probe (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("INSERT INTO probe VALUES (?, ?)", ("box-exp-014", "r4"))
        connection.commit()
        connection.close()
        connection = None
        os.chmod(path, _MODE_0600, follow_symlinks=False)
        metadata = _metadata(path, directory=False, mode=_MODE_0600, uid=uid)
        require(metadata.st_size > 0, "SQLite probe database is empty")
        descriptor = os.open(path, os.O_RDONLY | _FILE_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        uri = f"file:{quote(str(path), safe='/')}?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True, timeout=5.0)
        row = connection.execute("SELECT key, value FROM probe").fetchone()
        connection.close()
        connection = None
        require(row == ("box-exp-014", "r4"), "SQLite commit/reopen value differs")
        path.unlink()
        for suffix in ("-journal", "-wal", "-shm"):
            sidecar = Path(f"{path}{suffix}")
            if sidecar.exists() or sidecar.is_symlink():
                _metadata(sidecar, directory=False, mode=_MODE_0600, uid=uid)
                sidecar.unlink()
        _fsync_directory(directory)
    except (OSError, sqlite3.Error) as error:
        raise Source7RuntimeCacheError("SQLite create/commit/reopen probe failed") from error
    finally:
        if connection is not None:
            connection.close()
    return {
        "database_created": True,
        "transaction_committed": True,
        "database_file_fsynced": True,
        "readonly_reopen": True,
        "committed_row_match": True,
        "probe_removed": True,
        "sqlite_version": sqlite3.sqlite_version,
    }


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> str:
    require(path.is_absolute() and path.parent.is_dir() and not path.parent.is_symlink(), "receipt parent differs")
    require(path.resolve(strict=False).parent == path.parent.resolve(strict=True), "receipt path is not canonical")
    require(not path.exists() and not path.is_symlink(), "receipt output must be fresh")
    raw = canonical_json_bytes(value) + b"\n"
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _FILE_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            _MODE_0600,
        )
        written = os.write(descriptor, raw)
        require(written == len(raw), "receipt short write")
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        require(path.read_bytes() == raw, "receipt physical reopen differs")
        _fsync_directory(path.parent)
    except OSError as error:
        raise Source7RuntimeCacheError("cannot publish create-only runtime cache receipt") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return hashlib.sha256(raw).hexdigest()


def _load_json(
    path: Path, expected_sha256: str, *, fields: Optional[frozenset[str]],
    label: str, mode: int = _MODE_0600,
) -> Mapping[str, Any]:
    require(type(expected_sha256) is str and _SHA256_RE.fullmatch(expected_sha256) is not None, f"{label} SHA differs")
    _metadata(path, directory=False, mode=mode, uid=os.getuid())
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_sha256, f"{label} file SHA differs")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_closed_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                Source7RuntimeCacheError(f"non-finite JSON constant: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Source7RuntimeCacheError(f"{label} is not valid JSON") from error
    require(type(value) is dict, f"{label} must be one object")
    if fields is not None:
        require(set(value) == fields, f"{label} field closure differs")
    require(raw == canonical_json_bytes(value) + b"\n", f"{label} is not canonical JSON")
    return value


def _read_shared_plain_file_at(
    path: Path, *, label: str, expected_mode: int, maximum_size: int,
    minimum_size: int = 1,
) -> tuple[bytes, os.stat_result]:
    """Read one canonical shared file through an O_NOFOLLOW openat chain."""

    require(
        path.is_absolute()
        and path.name not in {"", ".", ".."}
        and "/" not in path.name,
        f"{label} path differs",
    )
    parent = path.parent
    try:
        parent_metadata = parent.lstat()
        resolved_parent = parent.resolve(strict=True)
    except OSError as error:
        raise Source7RuntimeCacheError(f"{label} parent is unavailable") from error
    require(
        resolved_parent == parent
        and stat.S_ISDIR(parent_metadata.st_mode)
        and not stat.S_ISLNK(parent_metadata.st_mode),
        f"{label} parent is not one canonical plain directory",
    )
    parent_descriptor = _open_directory(parent)
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(
            path.name,
            os.O_RDONLY | _FILE_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_descriptor,
        )
        before = os.fstat(descriptor)
        require(
            stat.S_ISREG(before.st_mode)
            and not stat.S_ISLNK(before.st_mode)
            and before.st_nlink == 1
            and before.st_uid == os.getuid()
            and stat.S_IMODE(before.st_mode) == expected_mode
            and minimum_size <= before.st_size <= maximum_size,
            f"{label} physical metadata differs",
        )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(4 * 1024 * 1024, remaining))
            require(bool(chunk), f"{label} short physical read")
            chunks.append(chunk)
            remaining -= len(chunk)
        require(os.read(descriptor, 1) == b"", f"{label} grew during physical read")
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev, before.st_ino, before.st_nlink, before.st_uid,
            before.st_gid, before.st_mode, before.st_size, before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev, after.st_ino, after.st_nlink, after.st_uid,
            after.st_gid, after.st_mode, after.st_size, after.st_mtime_ns,
            after.st_ctime_ns,
        )
        raw = b"".join(chunks)
        require(
            before_identity == after_identity and len(raw) == before.st_size,
            f"{label} changed during physical read",
        )
        return raw, after
    except OSError as error:
        raise Source7RuntimeCacheError(
            f"cannot physically open {label} without following links"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)


def _load_shared_json_at(
    path: Path, *, expected_sha256: str, expected_mode: int, label: str,
    fields: Optional[frozenset[str]] = None,
) -> Mapping[str, Any]:
    require(
        type(expected_sha256) is str
        and _SHA256_RE.fullmatch(expected_sha256) is not None,
        f"{label} SHA differs",
    )
    raw, _ = _read_shared_plain_file_at(
        path,
        label=label,
        expected_mode=expected_mode,
        maximum_size=32 * 1024 * 1024,
    )
    require(hashlib.sha256(raw).hexdigest() == expected_sha256, f"{label} file SHA differs")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_closed_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                Source7RuntimeCacheError(f"non-finite JSON constant: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Source7RuntimeCacheError(f"{label} is not valid JSON") from error
    require(type(value) is dict, f"{label} must be one object")
    if fields is not None:
        require(set(value) == fields, f"{label} field closure differs")
    require(raw == canonical_json_bytes(value) + b"\n", f"{label} is not canonical JSON")
    return value


def _load_pinned_cpu_torch() -> Any:
    """Import only the launcher-pinned ROCm torch build for CPU deserialization."""

    expected_site_packages = Path(EXPECTED_MIOPEN_LIBRARY_PATH).parents[2]
    expected_package_root = expected_site_packages / "torch"
    if str(expected_site_packages) not in sys.path:
        sys.path.insert(0, str(expected_site_packages))
    try:
        torch = importlib.import_module("torch")
    except Exception as error:
        raise Source7RuntimeCacheError(
            "pinned CPU torch is unavailable for post-srun tensor audit"
        ) from error
    try:
        observed_package_root = Path(torch.__file__).resolve(strict=True).parent
    except (AttributeError, OSError) as error:
        raise Source7RuntimeCacheError("pinned CPU torch package identity differs") from error
    require(
        observed_package_root == expected_package_root
        and str(torch.__version__) == EXPECTED_TORCH_VERSION
        and getattr(torch.version, "hip", None) == EXPECTED_TORCH_HIP_VERSION,
        "pinned CPU torch identity differs",
    )
    return torch


def _producer_tensor_sha256(tensor: Any, torch: Any) -> str:
    """Frozen ``materialize_vae._tensor_sha256`` algorithm."""

    value = tensor.detach().cpu().contiguous()
    header = canonical_json_bytes(
        {"dtype": str(value.dtype), "shape": list(value.shape)}
    )
    raw = value.view(torch.uint8).reshape(-1).numpy().tobytes(order="C")
    return hashlib.sha256(header + b"\0" + raw).hexdigest()


def _producer_tensor_raw_sha256(tensor: Any, torch: Any) -> str:
    """Frozen source7 materializer raw-FP32 tensor hash algorithm."""

    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    raw = value.view(torch.uint8).reshape(-1).numpy().tobytes(order="C")
    return hashlib.sha256(raw).hexdigest()


def _decode_published_bare_tensor(
    raw: bytes, *, expected_shape: Sequence[int], label: str, torch: Any,
) -> tuple[Any, str, str]:
    try:
        value = torch.load(
            io.BytesIO(raw), map_location="cpu", weights_only=True
        )
    except Exception as error:
        raise Source7RuntimeCacheError(
            f"{label} cannot be safely reopened with weights_only=True"
        ) from error
    require(type(value) is torch.Tensor, f"{label} must be one exact bare tensor")
    require(
        value.layout == torch.strided
        and value.device.type == "cpu"
        and value.dtype == torch.float32
        and value.is_contiguous()
        and [int(item) for item in value.shape] == list(expected_shape)
        and bool(torch.isfinite(value).all().item()),
        f"{label} tensor dtype/shape/layout/finite closure differs",
    )
    return (
        value,
        _producer_tensor_sha256(value, torch),
        _producer_tensor_raw_sha256(value, torch),
    )


def _directory_receipt(path: Path, *, uid: int) -> Mapping[str, Any]:
    metadata = _metadata(path, directory=True, mode=_MODE_0700, uid=uid)
    return {
        "path": str(path),
        "owner_uid": metadata.st_uid,
        "owner_gid": metadata.st_gid,
        "mode": stat.S_IMODE(metadata.st_mode),
        "canonical": True,
        "nofollow": True,
    }


def _sha256_from_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def observe_global_miopen_lock_root() -> Mapping[str, Any]:
    """Read only the shared root metadata; never enumerate or mutate members."""

    path = GLOBAL_MIOPEN_LOCK_ROOT
    common = {
        "path": str(path),
        "authoritative_for_this_run": False,
        "root_metadata_read_only": True,
        "members_scanned": False,
        "mutation_attempted": False,
        "cleanup_scope": False,
    }
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return {**common, "exists": False}
    except OSError as error:
        raise Source7RuntimeCacheError(
            "global MIOpen lock root metadata is unavailable"
        ) from error
    require(
        not stat.S_ISLNK(metadata.st_mode)
        and stat.S_ISDIR(metadata.st_mode)
        and path.resolve(strict=True) == path,
        "global MIOpen lock root must be one canonical non-symlink directory",
    )
    return {
        **common,
        "exists": True,
        "owner_uid": metadata.st_uid,
        "owner_gid": metadata.st_gid,
        "mode": stat.S_IMODE(metadata.st_mode),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
    }


def observe_scoped_miopen_temp_lock_root(
    cache_root: Path, *, uid: Optional[int] = None,
    expected_device: Optional[int] = None,
) -> Mapping[str, Any]:
    """Truthfully observe the scoped lock root without inventing activity."""

    owner_uid = os.getuid() if uid is None else uid
    path = cache_root / "tmp" / EXPECTED_MIOPEN_LOCK_DIRECTORY_BASENAME
    absent = {
        "path": str(path),
        "exists": False,
        "kind": "absent",
        "owner_uid": None,
        "owner_gid": None,
        "mode": None,
        "device": None,
        "inode": None,
        "canonical_nofollow_directory": False,
    }
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return absent
    except OSError as error:
        raise Source7RuntimeCacheError(
            "scoped MIOpen temp lock root metadata is unavailable"
        ) from error
    kind = (
        "symlink"
        if stat.S_ISLNK(metadata.st_mode)
        else "directory"
        if stat.S_ISDIR(metadata.st_mode)
        else "file"
        if stat.S_ISREG(metadata.st_mode)
        else "other"
    )
    require(
        kind == "directory"
        and metadata.st_uid == owner_uid
        and stat.S_IMODE(metadata.st_mode) == _MODE_0777
        and (expected_device is None or metadata.st_dev == expected_device)
        and path.resolve(strict=True) == path,
        "scoped MIOpen temp lock root identity differs",
    )
    descriptor = _open_directory(path)
    try:
        reopened = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    require(
        (reopened.st_dev, reopened.st_ino, reopened.st_uid)
        == (metadata.st_dev, metadata.st_ino, metadata.st_uid)
        and stat.S_ISDIR(reopened.st_mode),
        "scoped MIOpen temp lock root nofollow reopen differs",
    )
    return {
        "path": str(path),
        "exists": True,
        "kind": "directory",
        "owner_uid": metadata.st_uid,
        "owner_gid": metadata.st_gid,
        "mode": stat.S_IMODE(metadata.st_mode),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "canonical_nofollow_directory": True,
    }


def _validate_scoped_miopen_temp_lock_root_observation(
    value: Any, *, cache_root: Path,
) -> None:
    fields = {
        "path", "exists", "kind", "owner_uid", "owner_gid", "mode",
        "device", "inode", "canonical_nofollow_directory",
    }
    require(
        type(value) is dict
        and set(value) == fields
        and value.get("path")
        == str(cache_root / "tmp" / EXPECTED_MIOPEN_LOCK_DIRECTORY_BASENAME)
        and type(value.get("exists")) is bool,
        "scoped MIOpen temp lock root observation closure differs",
    )
    if value["exists"]:
        require(
            value.get("kind") == "directory"
            and value.get("owner_uid") == os.getuid()
            and type(value.get("owner_gid")) is int
            and value.get("mode") == _MODE_0777
            and type(value.get("device")) is int
            and type(value.get("inode")) is int
            and value["inode"] > 0
            and value.get("canonical_nofollow_directory") is True,
            "scoped MIOpen temp lock root present observation differs",
        )
    else:
        require(
            value
            == {
                "path": str(
                    cache_root / "tmp" / EXPECTED_MIOPEN_LOCK_DIRECTORY_BASENAME
                ),
                "exists": False,
                "kind": "absent",
                "owner_uid": None,
                "owner_gid": None,
                "mode": None,
                "device": None,
                "inode": None,
                "canonical_nofollow_directory": False,
            },
            "scoped MIOpen temp lock root absent observation differs",
        )


def _validate_global_miopen_lock_observation(value: Any) -> None:
    common_fields = {
        "path",
        "authoritative_for_this_run",
        "root_metadata_read_only",
        "members_scanned",
        "mutation_attempted",
        "cleanup_scope",
        "exists",
    }
    require(type(value) is dict, "global MIOpen lock root observation differs")
    exists = value.get("exists")
    expected_fields = (
        common_fields
        | {"owner_uid", "owner_gid", "mode", "device", "inode"}
        if exists is True
        else common_fields
    )
    require(
        set(value) == expected_fields
        and value.get("path") == str(GLOBAL_MIOPEN_LOCK_ROOT)
        and value.get("authoritative_for_this_run") is False
        and value.get("root_metadata_read_only") is True
        and value.get("members_scanned") is False
        and value.get("mutation_attempted") is False
        and value.get("cleanup_scope") is False
        and type(exists) is bool,
        "global MIOpen lock root observation closure differs",
    )


def expected_miopen_lock_file_basenames(
    cache_root: Path, *, require_existing_parent: bool = True,
) -> Mapping[str, Any]:
    user_db_parent = cache_root / "user-db"
    require(user_db_parent.is_absolute(), "MIOpen lock hash parent must be absolute")
    if require_existing_parent:
        require(
            user_db_parent.resolve(strict=True) == user_db_parent,
            "MIOpen lock hash parent must be the canonical scoped user DB path",
        )
    else:
        require(
            ".." not in user_db_parent.parts
            and str(user_db_parent) == os.path.normpath(str(user_db_parent)),
            "deleted MIOpen lock hash parent must remain lexically canonical",
        )
    try:
        parent_md5 = hashlib.md5(
            str(user_db_parent).encode("utf-8"), usedforsecurity=False
        ).hexdigest()
    except TypeError:  # pragma: no cover - compatibility with older Python
        parent_md5 = hashlib.md5(str(user_db_parent).encode("utf-8")).hexdigest()
    require(
        re.fullmatch(r"[0-9a-f]{32}", parent_md5) is not None,
        "MIOpen lock parent MD5 differs",
    )
    basenames = sorted(
        f"{parent_md5}_{name}.lock"
        for name in EXPECTED_MIOPEN_USER_DB_MAIN_BASENAMES
    )
    return {
        "user_db_parent": str(user_db_parent),
        "user_db_parent_md5": parent_md5,
        "expected_lock_basenames": basenames,
    }


def _inventory_member_policy(
    *, subdirectory_name: str, subdirectory: Path, candidate: Path,
    metadata: os.stat_result,
) -> None:
    relative_to_subdirectory = candidate.relative_to(subdirectory)
    parts = relative_to_subdirectory.parts
    mode = stat.S_IMODE(metadata.st_mode)
    if subdirectory_name == "user-db":
        require(
            stat.S_ISREG(metadata.st_mode)
            and len(parts) == 1
            and candidate.name
            in (
                EXPECTED_MIOPEN_USER_DB_MAIN_BASENAMES
                | EXPECTED_MIOPEN_USER_DB_TIME_BASENAMES
            ),
            "MIOpen user DB inventory member name/kind differs",
        )
        if candidate.name in EXPECTED_MIOPEN_USER_DB_MAIN_BASENAMES:
            require(
                mode == _MODE_0777,
                "MIOpen 3.3 PlainTextDb main file mode must be 0777",
            )
        return
    if subdirectory_name == "kernel-cache":
        require(
            stat.S_ISREG(metadata.st_mode)
            and len(parts) == 1
            and candidate.name in EXPECTED_MIOPEN_KERNEL_CACHE_BASENAMES
            and mode == _MODE_0600,
            "MIOpen kernel cache member name/kind/mode differs",
        )
        return
    if subdirectory_name == "tmp":
        lock_identity = expected_miopen_lock_file_basenames(subdirectory.parent)
        if stat.S_ISDIR(metadata.st_mode):
            require(
                parts == (EXPECTED_MIOPEN_LOCK_DIRECTORY_BASENAME,)
                and mode == _MODE_0777,
                "scoped MIOpen lock directory identity/mode differs",
            )
            return
        require(
            stat.S_ISREG(metadata.st_mode)
            and len(parts) == 2
            and parts[0] == EXPECTED_MIOPEN_LOCK_DIRECTORY_BASENAME
            and candidate.name in lock_identity["expected_lock_basenames"]
            and mode == _MODE_0777,
            "scoped MIOpen lock file name/kind/mode differs",
        )
        return
    require(subdirectory_name == "xdg-cache", "runtime cache subdirectory differs")
    required_mode = _MODE_0700 if stat.S_ISDIR(metadata.st_mode) else _MODE_0600
    require(
        (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode))
        and mode == required_mode,
        "XDG cache member kind/mode differs",
    )


def inventory_cache_directories(
    cache_root: Path, *, uid: Optional[int] = None
) -> Mapping[str, list[Mapping[str, Any]]]:
    """Return an exact, policy-closed, physically reopened cache inventory."""

    owner_uid = os.getuid() if uid is None else uid
    root_metadata = _metadata(
        cache_root, directory=True, mode=_MODE_0700, uid=owner_uid
    )
    try:
        root_member_names = sorted(os.listdir(cache_root))
    except OSError as error:
        raise Source7RuntimeCacheError(
            "runtime cache root member closure is unavailable"
        ) from error
    require(
        root_member_names == sorted(SUBDIRECTORY_NAMES),
        "runtime cache root exact subdirectory closure differs",
    )
    result: dict[str, list[Mapping[str, Any]]] = {}
    for subdirectory_name in SUBDIRECTORY_NAMES:
        subdirectory = cache_root / subdirectory_name
        subdirectory_metadata = _metadata(
            subdirectory, directory=True, mode=_MODE_0700, uid=owner_uid
        )
        require(
            subdirectory_metadata.st_dev == root_metadata.st_dev,
            "runtime cache subdirectory crossed filesystem boundary",
        )
        rows: list[Mapping[str, Any]] = []
        for current, directory_names, file_names in os.walk(
            subdirectory, topdown=True, followlinks=False
        ):
            directory_names.sort()
            file_names.sort()
            current_path = Path(current)
            for name in [*directory_names, *file_names]:
                candidate = current_path / name
                try:
                    metadata = candidate.lstat()
                except OSError as error:
                    raise Source7RuntimeCacheError(
                        "runtime cache inventory member disappeared"
                    ) from error
                require(
                    not stat.S_ISLNK(metadata.st_mode)
                    and metadata.st_uid == owner_uid
                    and metadata.st_dev == root_metadata.st_dev
                    and candidate.resolve(strict=True) == candidate,
                    "runtime cache inventory member identity differs",
                )
                relative = candidate.relative_to(cache_root).as_posix()
                common = {
                    "path": relative,
                    "owner_uid": metadata.st_uid,
                    "owner_gid": metadata.st_gid,
                    "mode": stat.S_IMODE(metadata.st_mode),
                    "device": metadata.st_dev,
                    "inode": metadata.st_ino,
                }
                _inventory_member_policy(
                    subdirectory_name=subdirectory_name,
                    subdirectory=subdirectory,
                    candidate=candidate,
                    metadata=metadata,
                )
                if stat.S_ISDIR(metadata.st_mode):
                    rows.append({**common, "kind": "directory"})
                    continue
                require(
                    stat.S_ISREG(metadata.st_mode)
                    and metadata.st_nlink == 1,
                    "runtime cache inventory file must be one-link regular file",
                )
                descriptor = os.open(
                    candidate,
                    os.O_RDONLY | _FILE_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                )
                try:
                    reopened = os.fstat(descriptor)
                    require(
                        (reopened.st_dev, reopened.st_ino, reopened.st_size, reopened.st_nlink)
                        == (metadata.st_dev, metadata.st_ino, metadata.st_size, 1)
                        and stat.S_ISREG(reopened.st_mode)
                        and reopened.st_uid == owner_uid,
                        "runtime cache physical file reopen identity differs",
                    )
                    digest = _sha256_from_descriptor(descriptor)
                finally:
                    os.close(descriptor)
                rows.append(
                    {
                        **common,
                        "kind": "file",
                        "size": metadata.st_size,
                        "nlink": metadata.st_nlink,
                        "sha256": digest,
                        "physical_reopen": True,
                        "nofollow": True,
                    }
                )
        result[subdirectory_name] = sorted(rows, key=lambda row: row["path"])
    return result


def miopen_user_db_evidence(
    inventory: Mapping[str, list[Mapping[str, Any]]]
) -> Mapping[str, Any]:
    rows = inventory.get("user-db")
    require(type(rows) is list, "MIOpen user DB inventory closure differs")
    main_rows = [
        row
        for row in rows
        if Path(str(row.get("path"))).name in EXPECTED_MIOPEN_USER_DB_MAIN_BASENAMES
    ]
    time_rows = [
        row
        for row in rows
        if Path(str(row.get("path"))).name in EXPECTED_MIOPEN_USER_DB_TIME_BASENAMES
    ]
    main_names = {Path(str(row["path"])).name for row in main_rows}
    allowed_names = (
        EXPECTED_MIOPEN_USER_DB_MAIN_BASENAMES
        | EXPECTED_MIOPEN_USER_DB_TIME_BASENAMES
    )
    require(
        all(
            Path(str(row.get("path"))).parts
            == ("user-db", Path(str(row.get("path"))).name)
            and Path(str(row.get("path"))).name in allowed_names
            and row.get("kind") == "file"
            and row.get("owner_uid") == os.getuid()
            and row.get("nlink") == 1
            and row.get("physical_reopen") is True
            and row.get("nofollow") is True
            and type(row.get("mode")) is int
            and type(row.get("size")) is int
            and type(row.get("sha256")) is str
            and _SHA256_RE.fullmatch(row["sha256"]) is not None
            for row in rows
        ),
        "MIOpen user DB inventory row closure differs",
    )
    for row in time_rows:
        sidecar_name = Path(str(row["path"])).name
        require(
            sidecar_name[:-5] in main_names,
            "MIOpen user DB time sidecar has no corresponding main file",
        )
    require(
        len(main_rows) + len(time_rows) == len(rows)
        and all(row.get("mode") == _MODE_0777 for row in main_rows),
        "MIOpen user DB physical evidence differs",
    )
    return {
        "write_required": False,
        "plaintext_main_write_observed": bool(main_rows),
        "allowed_main_basenames": sorted(EXPECTED_MIOPEN_USER_DB_MAIN_BASENAMES),
        "allowed_time_sidecar_basenames": sorted(
            EXPECTED_MIOPEN_USER_DB_TIME_BASENAMES
        ),
        "main_file_mode_required": _MODE_0777,
        "time_sidecar_mode_recorded_not_pinned": True,
        "files": list(rows),
    }


def validate_scoped_miopen_temp_lock_activity(
    cache_root: Path,
) -> tuple[Mapping[str, list[Mapping[str, Any]]], Mapping[str, Any]]:
    inventory = inventory_cache_directories(cache_root)
    lock_identity = expected_miopen_lock_file_basenames(cache_root)
    rows = inventory["tmp"]
    lock_directory_relative = f"tmp/{EXPECTED_MIOPEN_LOCK_DIRECTORY_BASENAME}"
    directories = [
        row
        for row in rows
        if row.get("path") == lock_directory_relative
        and row.get("kind") == "directory"
    ]
    lock_files = [row for row in rows if row.get("kind") == "file"]
    require(
        len(directories) == 1
        and directories[0].get("mode") == _MODE_0777
        and bool(lock_files)
        and len(rows) == 1 + len(lock_files),
        "scoped MIOpen temp lock activity was not observed",
    )
    require(
        all(
            row.get("path")
            == f"{lock_directory_relative}/{Path(str(row.get('path'))).name}"
            and Path(str(row.get("path"))).name
            in lock_identity["expected_lock_basenames"]
            and row.get("mode") == _MODE_0777
            and row.get("nlink") == 1
            and row.get("physical_reopen") is True
            and row.get("nofollow") is True
            for row in lock_files
        ),
        "scoped MIOpen temp lock file evidence differs",
    )
    evidence = {
        "tmpdir": str(cache_root / "tmp"),
        "lock_directory": str(cache_root / lock_directory_relative),
        "lock_directory_relative_path": lock_directory_relative,
        "lock_directory_mode": directories[0]["mode"],
        "cpp_temp_directory_path_redirect_observed": True,
        "activity_required": True,
        "activity_observed": True,
        "user_db_parent_for_lock_hash": lock_identity["user_db_parent"],
        "user_db_parent_md5": lock_identity["user_db_parent_md5"],
        "expected_lock_basenames": lock_identity["expected_lock_basenames"],
        "lock_files": lock_files,
        "global_tmp_lock_root_authoritative": False,
    }
    return inventory, evidence


def validate_miopen_kernel_cache_activity(
    cache_root: Path,
) -> tuple[Mapping[str, list[Mapping[str, Any]]], Mapping[str, Any]]:
    """Require the exact fresh gfx90a MIOpen user kernel cache DB and rows."""

    before = inventory_cache_directories(cache_root)
    expected_relative = f"kernel-cache/{EXPECTED_MIOPEN_KERNEL_DB_BASENAME}"
    matching = [
        row
        for row in before["kernel-cache"]
        if row.get("path") == expected_relative and row.get("kind") == "file"
    ]
    require(len(matching) == 1, "fresh MIOpen kernel DB gfx90a68.ukdb was not created")
    inventory_row = matching[0]
    require(
        inventory_row["size"] > 0
        and inventory_row["nlink"] == 1
        and inventory_row["mode"] == _MODE_0600
        and inventory_row["physical_reopen"] is True
        and inventory_row["nofollow"] is True,
        "MIOpen kernel DB physical identity differs",
    )
    wal_relative = f"kernel-cache/{EXPECTED_MIOPEN_KERNEL_DB_BASENAME}-wal"
    shm_relative = f"kernel-cache/{EXPECTED_MIOPEN_KERNEL_DB_BASENAME}-shm"
    wal_rows = [row for row in before["kernel-cache"] if row.get("path") == wal_relative]
    shm_rows = [row for row in before["kernel-cache"] if row.get("path") == shm_relative]
    require(
        len(wal_rows) <= 1
        and (not wal_rows or wal_rows[0].get("size") == 0)
        and len(shm_rows) <= 1,
        "MIOpen kernel DB must be checkpointed with absent-or-empty WAL",
    )
    database_path = cache_root / expected_relative
    descriptor = os.open(
        database_path,
        os.O_RDONLY | _FILE_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        metadata = os.fstat(descriptor)
        header = os.read(descriptor, 16)
    finally:
        os.close(descriptor)
    require(
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_uid == os.getuid()
        and metadata.st_nlink == 1
        and header == b"SQLite format 3\x00",
        "MIOpen kernel DB is not a physically reopened owned SQLite file",
    )
    connection: Optional[sqlite3.Connection] = None
    try:
        uri = f"file:{quote(str(database_path), safe='/')}?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True, timeout=5.0)
        connection.execute("PRAGMA query_only=ON")
        quick_check_rows = [row[0] for row in connection.execute("PRAGMA quick_check")]
        table_info = tuple(connection.execute("PRAGMA table_info(kern_db)"))
        index_list = tuple(connection.execute("PRAGMA index_list(kern_db)"))
        index_info = tuple(connection.execute("PRAGMA index_info(idx_kern_db)"))
        kernel_row_count = connection.execute(
            "SELECT COUNT(*) FROM kern_db"
        ).fetchone()
    except sqlite3.Error as error:
        raise Source7RuntimeCacheError(
            "MIOpen kernel DB readonly validation failed"
        ) from error
    finally:
        if connection is not None:
            connection.close()
    require(quick_check_rows == ["ok"], "MIOpen kernel DB quick_check differs")
    require(
        table_info == EXPECTED_MIOPEN_KERN_DB_COLUMNS,
        "MIOpen kernel DB kern_db column schema differs",
    )
    require(
        any(
            len(row) >= 3 and row[1] == "idx_kern_db" and row[2] == 1
            for row in index_list
        )
        and index_info == EXPECTED_MIOPEN_KERN_DB_UNIQUE_INDEX,
        "MIOpen kernel DB unique index schema differs",
    )
    require(
        type(kernel_row_count) is tuple
        and len(kernel_row_count) == 1
        and type(kernel_row_count[0]) is int
        and kernel_row_count[0] > 0,
        "MIOpen kernel DB contains no compiled kernel rows",
    )
    after = inventory_cache_directories(cache_root)
    require(after == before, "MIOpen cache changed during readonly physical validation")
    evidence = {
        "path": str(database_path),
        "relative_path": expected_relative,
        "basename": EXPECTED_MIOPEN_KERNEL_DB_BASENAME,
        "file_sha256": inventory_row["sha256"],
        "file_size": inventory_row["size"],
        "owner_uid": inventory_row["owner_uid"],
        "file_mode": inventory_row["mode"],
        "nlink": inventory_row["nlink"],
        "ordinary_file": True,
        "nofollow_physical_reopen": True,
        "sqlite_header_verified": True,
        "sqlite_readonly_reopen": True,
        "sqlite_immutable_reopen": True,
        "sqlite_quick_check": "ok",
        "wal_relative_path": wal_relative,
        "wal_absent_or_empty": True,
        "wal_observed": bool(wal_rows),
        "wal_size": wal_rows[0]["size"] if wal_rows else None,
        "shm_relative_path": shm_relative,
        "shm_observed": bool(shm_rows),
        "shm_inventory_row": shm_rows[0] if shm_rows else None,
        "kern_db_columns": [list(row) for row in table_info],
        "kern_db_unique_index_name": "idx_kern_db",
        "kern_db_unique_index_columns": [list(row) for row in index_info],
        "kern_db_row_count": kernel_row_count[0],
        "kern_db_nonempty": True,
        "inventory_stable_during_readonly_validation": True,
    }
    return after, evidence


def prepare_runtime_cache(
    *, receipt_output: Path, environ: Optional[Mapping[str, str]] = None,
    cache_parent: Path = CACHE_PARENT,
) -> Mapping[str, Any]:
    env = os.environ if environ is None else environ
    require("torch" not in sys.modules, "torch was imported before runtime cache preparation")
    hostname = _short_hostname()
    require(hostname == EXPECTED_COMPUTE_NODE, "runtime cache prepare node differs")
    job_id = _token(env.get("SLURM_JOB_ID"), label="SLURM_JOB_ID")
    step_id = _token(env.get("SLURM_STEP_ID"), label="SLURM_STEP_ID")
    home = env.get("HOME")
    require(type(home) is str and home.startswith("/") and home != "/", "HOME differs")
    parent = cache_parent
    parent_metadata = _metadata(parent, directory=True, mode=stat.S_IMODE(parent.lstat().st_mode), uid=parent.lstat().st_uid)
    require(parent.resolve(strict=True) == parent, "cache parent must be canonical")
    filesystem = dict(_filesystem_identity(parent))
    require(
        filesystem
        == {
            "statfs_type": EXPECTED_STATFS_TYPE,
            "statfs_magic_hex": f"0x{EXPECTED_STATFS_MAGIC:x}",
            "mount_fstype": EXPECTED_MOUNT_FSTYPE,
            "mount_point": EXPECTED_MOUNT_POINT,
            "source": EXPECTED_MOUNT_SOURCE,
            "source_is_local_block_device": True,
            "local_filesystem": True,
        },
        "runtime cache filesystem must match gpu299 local ext statfs/mount identity",
    )
    root = expected_cache_root(job_id, step_id, cache_parent=parent)
    require(root.parent == parent and root.name.startswith(f"{CACHE_PREFIX}-"), "runtime cache scope differs")
    expected_environment = {
        "HOME": home,
        "MIOPEN_USER_DB_PATH": str(root / "user-db"),
        "MIOPEN_CUSTOM_CACHE_DIR": str(root / "kernel-cache"),
        "XDG_CACHE_HOME": str(root / "xdg-cache"),
        "TMPDIR": str(root / "tmp"),
    }
    require(
        {name: env.get(name) for name in expected_environment}
        == expected_environment,
        "runtime cache environment must be exported before prepare",
    )
    global_miopen_lock_root_before_torch = observe_global_miopen_lock_root()
    parent_fd = _open_directory(parent)
    try:
        before_identity = (parent_metadata.st_dev, parent_metadata.st_ino)
        try:
            os.mkdir(root.name, _MODE_0700, dir_fd=parent_fd)
        except FileExistsError as error:
            raise Source7RuntimeCacheError("runtime cache root is not fresh") from error
        except OSError as error:
            raise Source7RuntimeCacheError("cannot create runtime cache root") from error
        os.fsync(parent_fd)
        after_parent = os.fstat(parent_fd)
        require(before_identity == (after_parent.st_dev, after_parent.st_ino), "cache parent identity changed")
    finally:
        os.close(parent_fd)
    uid = os.getuid()
    root_metadata = _metadata(root, directory=True, mode=_MODE_0700, uid=uid)
    require(root_metadata.st_dev == parent_metadata.st_dev, "runtime cache crossed filesystem boundary")
    root_fd = _open_directory(root)
    try:
        for name in SUBDIRECTORY_NAMES:
            os.mkdir(name, _MODE_0700, dir_fd=root_fd)
        os.fsync(root_fd)
    except OSError as error:
        raise Source7RuntimeCacheError("cannot create runtime cache subdirectories") from error
    finally:
        os.close(root_fd)
    directories = {name: _directory_receipt(root / name, uid=uid) for name in SUBDIRECTORY_NAMES}
    exclusive_probe = _exclusive_fsync_probe(root / "user-db", uid=uid)
    sqlite_probe = _sqlite_commit_reopen_probe(root / "user-db", uid=uid)
    post_probe_empty_inventory = inventory_cache_directories(root, uid=uid)
    require(
        post_probe_empty_inventory
        == {name: [] for name in SUBDIRECTORY_NAMES},
        "fresh runtime cache is not empty after probes",
    )
    unsigned: dict[str, Any] = {
        "schema_version": PREPARE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_generation": RUN_GENERATION,
        "slurm_job_id": job_id,
        "slurm_step_id": step_id,
        "cache_parent": str(parent),
        "cache_root": str(root),
        "cache_name": root.name,
        "hostname": hostname,
        "filesystem": filesystem,
        "owner_uid": uid,
        "owner_gid": root_metadata.st_gid,
        "directory_mode": stat.S_IMODE(root_metadata.st_mode),
        "cache_root_device": root_metadata.st_dev,
        "cache_root_inode": root_metadata.st_ino,
        "directories": directories,
        "environment": expected_environment,
        "home_unchanged": True,
        "created_fresh_create_only": True,
        "canonical_paths_verified": True,
        "nofollow_verified": True,
        "exclusive_fsync_probe": exclusive_probe,
        "sqlite_commit_reopen_probe": sqlite_probe,
        "post_probe_empty_inventory": post_probe_empty_inventory,
        "global_miopen_lock_root_before_torch": global_miopen_lock_root_before_torch,
        "torch_imported_during_prepare": False,
        "cache_reusable": False,
        "cleanup_policy": "cleanup-scoped-cache-including-tmp-locks-on-compute-node-after-controller-success-before-numbered-step-exit;retain-on-failure;never-touch-global-lock-root;never-reuse",
    }
    receipt = {**unsigned, "prepare_digest": object_sha256(unsigned)}
    require(set(receipt) == PREPARE_FIELDS, "prepare receipt field closure differs")
    _write_json_create_only(receipt_output, receipt)
    return receipt


def validate_prepare_receipt(
    *, receipt_path: Path, expected_sha256: str,
    environ: Optional[Mapping[str, str]] = None,
    require_cache_present: bool = True,
    require_cache_empty: bool = True,
) -> Mapping[str, Any]:
    env = os.environ if environ is None else environ
    receipt = _load_json(receipt_path, expected_sha256, fields=PREPARE_FIELDS, label="prepare receipt")
    unsigned = dict(receipt)
    declared = unsigned.pop("prepare_digest", None)
    require(declared == object_sha256(unsigned), "prepare receipt digest differs")
    job_id = _token(env.get("SLURM_JOB_ID"), label="SLURM_JOB_ID")
    step_id = _token(env.get("SLURM_STEP_ID"), label="SLURM_STEP_ID")
    require(
        receipt["schema_version"] == PREPARE_SCHEMA
        and receipt["experiment_id"] == EXPERIMENT_ID
        and receipt["run_generation"] == RUN_GENERATION
        and receipt["hostname"] == EXPECTED_COMPUTE_NODE
        and _short_hostname() == EXPECTED_COMPUTE_NODE
        and receipt["slurm_job_id"] == job_id
        and receipt["slurm_step_id"] == step_id,
        "prepare receipt identity differs",
    )
    parent = Path(receipt["cache_parent"])
    root = expected_cache_root(job_id, step_id, cache_parent=parent)
    require(parent == CACHE_PARENT and receipt["cache_root"] == str(root) and receipt["cache_name"] == root.name, "cache root identity differs")
    expected_environment = {
        "HOME": env.get("HOME"),
        "MIOPEN_USER_DB_PATH": env.get("MIOPEN_USER_DB_PATH"),
        "MIOPEN_CUSTOM_CACHE_DIR": env.get("MIOPEN_CUSTOM_CACHE_DIR"),
        "XDG_CACHE_HOME": env.get("XDG_CACHE_HOME"),
        "TMPDIR": env.get("TMPDIR"),
    }
    require(receipt["environment"] == expected_environment, "runtime cache environment differs")
    require(
        expected_environment
        == {
            "HOME": receipt["environment"]["HOME"],
            "MIOPEN_USER_DB_PATH": str(root / "user-db"),
            "MIOPEN_CUSTOM_CACHE_DIR": str(root / "kernel-cache"),
            "XDG_CACHE_HOME": str(root / "xdg-cache"),
            "TMPDIR": str(root / "tmp"),
        }
        and receipt["home_unchanged"] is True,
        "runtime cache environment binding differs",
    )
    require(
        receipt["created_fresh_create_only"] is True
        and receipt["canonical_paths_verified"] is True
        and receipt["nofollow_verified"] is True
        and receipt["torch_imported_during_prepare"] is False
        and receipt["cache_reusable"] is False
        and receipt["cleanup_policy"] == "cleanup-scoped-cache-including-tmp-locks-on-compute-node-after-controller-success-before-numbered-step-exit;retain-on-failure;never-touch-global-lock-root;never-reuse",
        "runtime cache safety closure differs",
    )
    _validate_global_miopen_lock_observation(
        receipt["global_miopen_lock_root_before_torch"]
    )
    require(
        receipt["exclusive_fsync_probe"]
        == {
            "o_excl": True,
            "o_nofollow": True,
            "file_fsync": True,
            "physical_reopen": True,
            "payload_match": True,
            "probe_removed": True,
        },
        "exclusive probe receipt differs",
    )
    sqlite_probe = receipt["sqlite_commit_reopen_probe"]
    require(
        type(sqlite_probe) is dict
        and set(sqlite_probe)
        == {
            "database_created",
            "transaction_committed",
            "database_file_fsynced",
            "readonly_reopen",
            "committed_row_match",
            "probe_removed",
            "sqlite_version",
        }
        and all(sqlite_probe[field] is True for field in set(sqlite_probe) - {"sqlite_version"})
        and type(sqlite_probe["sqlite_version"]) is str
        and bool(sqlite_probe["sqlite_version"]),
        "SQLite probe receipt differs",
    )
    if require_cache_present:
        uid = os.getuid()
        root_metadata = _metadata(root, directory=True, mode=_MODE_0700, uid=uid)
        filesystem = dict(_filesystem_identity(root))
        require(filesystem == receipt["filesystem"] and filesystem["local_filesystem"] is True, "runtime cache filesystem identity changed")
        require(
            receipt["owner_uid"] == uid
            and receipt["owner_gid"] == root_metadata.st_gid
            and receipt["directory_mode"] == _MODE_0700,
            "runtime cache root metadata differs",
        )
        require(
            receipt["cache_root_device"] == root_metadata.st_dev
            and receipt["cache_root_inode"] == root_metadata.st_ino,
            "runtime cache root device/inode differs",
        )
        observed_directories = {name: _directory_receipt(root / name, uid=uid) for name in SUBDIRECTORY_NAMES}
        require(observed_directories == receipt["directories"], "runtime cache directory identities changed")
        require(
            observe_global_miopen_lock_root()
            == receipt["global_miopen_lock_root_before_torch"],
            "global MIOpen lock root metadata changed before controller use",
        )
        require(
            receipt["post_probe_empty_inventory"]
            == {name: [] for name in SUBDIRECTORY_NAMES},
            "prepare receipt fresh empty inventory differs",
        )
        if require_cache_empty:
            observed_inventory = inventory_cache_directories(root, uid=uid)
            require(
                observed_inventory == receipt["post_probe_empty_inventory"],
                "runtime cache was populated before controller smoke",
            )
    return receipt


def _validate_cleanup_tree(root: Path, *, uid: int, device: int) -> None:
    metadata = root.lstat()
    require(stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode), "cleanup root differs")
    require(metadata.st_uid == uid and metadata.st_dev == device and stat.S_IMODE(metadata.st_mode) == _MODE_0700, "cleanup root metadata differs")
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        current_metadata = current_path.lstat()
        require(
            stat.S_ISDIR(current_metadata.st_mode)
            and not stat.S_ISLNK(current_metadata.st_mode)
            and current_metadata.st_uid == uid
            and current_metadata.st_dev == device,
            "cleanup directory escaped cache identity",
        )
        for name in [*directories, *files]:
            candidate = current_path / name
            candidate_metadata = candidate.lstat()
            require(
                not stat.S_ISLNK(candidate_metadata.st_mode)
                and candidate_metadata.st_uid == uid
                and candidate_metadata.st_dev == device
                and (stat.S_ISDIR(candidate_metadata.st_mode) or stat.S_ISREG(candidate_metadata.st_mode)),
                "cleanup member differs",
            )


def _validate_smoke_for_cleanup(
    smoke: Mapping[str, Any], *, prepare: Mapping[str, Any],
    prepare_receipt_path: Path, expected_prepare_sha256: str,
    require_cache_present: bool = True,
) -> None:
    require(set(smoke) == SMOKE_FIELDS, "smoke receipt field closure differs")
    unsigned_smoke = dict(smoke)
    declared_smoke_digest = unsigned_smoke.pop("smoke_digest", None)
    require(
        declared_smoke_digest == object_sha256(unsigned_smoke),
        "smoke receipt digest differs",
    )
    require(
        smoke.get("experiment_id") == EXPERIMENT_ID
        and smoke.get("run_generation") == RUN_GENERATION
        and smoke.get("complete") is True
        and smoke.get("prepare_receipt_path") == str(prepare_receipt_path)
        and smoke.get("prepare_receipt_file_sha256") == expected_prepare_sha256
        and smoke.get("prepare_digest") == prepare["prepare_digest"]
        and smoke.get("cache_root") == prepare["cache_root"]
        and smoke.get("environment") == prepare["environment"]
        and smoke.get("miopen_user_db_path_kind") == "directory"
        and smoke.get("miopen_custom_cache_dir_kind") == "directory"
        and smoke.get("miopen_cache_paths_remained_canonical_0700_directories") is True
        and smoke.get("torch_import_after_cache_validation") is True
        and smoke.get("backend") == "ROCm-MIOpen-via-torch.backends.cudnn"
        and type(smoke.get("device_index")) is int
        and type(smoke.get("device_name")) is str
        and bool(smoke.get("device_name"))
        and type(smoke.get("peak_allocated_bytes")) is int
        and smoke["peak_allocated_bytes"] > 0
        and smoke.get("gpu_cache_cleared") is True
        and smoke.get("gpu_memory_allocated_after_clear") == 0,
        "smoke identity/environment/resource closure differs",
    )
    smoke_inventory = smoke.get("post_conv_cache_inventory")
    smoke_kernel_db = smoke.get("miopen_kernel_db_evidence")
    smoke_user_db = smoke.get("miopen_user_db_evidence")
    smoke_temp_lock = smoke.get("scoped_miopen_temp_lock_evidence")
    expected_runtime = {
        "torch_version": EXPECTED_TORCH_VERSION,
        "torch_hip_version": EXPECTED_TORCH_HIP_VERSION,
        "miopen_backend_version": EXPECTED_MIOPEN_BACKEND_VERSION,
        "miopen_library_resolved_path": EXPECTED_MIOPEN_LIBRARY_PATH,
        "miopen_library_size": EXPECTED_MIOPEN_LIBRARY_SIZE,
        "miopen_library_sha256": EXPECTED_MIOPEN_LIBRARY_SHA256,
        "miopen_embedded_version": EXPECTED_MIOPEN_EMBEDDED_VERSION,
    }
    require(
        smoke.get("pre_conv_cache_inventory")
        == prepare["post_probe_empty_inventory"]
        == {name: [] for name in SUBDIRECTORY_NAMES},
        "smoke fresh pre-conv cache inventory differs",
    )
    require(
        type(smoke_inventory) is dict
        and set(smoke_inventory) == set(SUBDIRECTORY_NAMES),
        "smoke post-conv cache inventory closure differs",
    )
    require(
        type(smoke_kernel_db) is dict
        and smoke_kernel_db.get("relative_path")
        == f"kernel-cache/{EXPECTED_MIOPEN_KERNEL_DB_BASENAME}"
        and smoke_kernel_db.get("kern_db_nonempty") is True
        and type(smoke_kernel_db.get("kern_db_row_count")) is int
        and smoke_kernel_db["kern_db_row_count"] > 0
        and smoke_kernel_db.get("nofollow_physical_reopen") is True
        and smoke_kernel_db.get("sqlite_readonly_reopen") is True
        and smoke_kernel_db.get("sqlite_immutable_reopen") is True
        and smoke_kernel_db.get("sqlite_quick_check") == "ok"
        and smoke_kernel_db.get("wal_absent_or_empty") is True
        and smoke.get("miopen_kernel_db_activity_required") is True
        and smoke.get("miopen_kernel_db_activity_observed") is True,
        "smoke MIOpen kernel DB activity closure differs",
    )
    smoke_kernel_rows = [
        row
        for row in smoke_inventory["kernel-cache"]
        if row.get("path")
        == f"kernel-cache/{EXPECTED_MIOPEN_KERNEL_DB_BASENAME}"
        and row.get("kind") == "file"
    ]
    require(
        len(smoke_kernel_rows) == 1
        and smoke_kernel_rows[0].get("mode") == _MODE_0600
        and smoke_kernel_rows[0].get("owner_uid") == os.getuid()
        and smoke_kernel_rows[0].get("nlink") == 1
        and smoke_kernel_rows[0].get("sha256")
        == smoke_kernel_db.get("file_sha256")
        and smoke_kernel_rows[0].get("size")
        == smoke_kernel_db.get("file_size")
        and smoke_kernel_rows[0].get("physical_reopen") is True
        and smoke_kernel_rows[0].get("nofollow") is True,
        "smoke MIOpen kernel DB inventory/evidence binding differs",
    )
    require(
        smoke.get("kernel_cache_claim")
        == "path-bound;fresh-ukdb-write-required-and-observed",
        "smoke MIOpen kernel-cache claim differs",
    )
    require(
        type(smoke_user_db) is dict
        and smoke.get("miopen_user_db_claim")
        == (
            "path-bound;expected-plaintext-write-observed-not-required"
            if bool(smoke_user_db["plaintext_main_write_observed"])
            else "path-bound;no-write-observed-and-no-write-claim"
        )
        and smoke_user_db == miopen_user_db_evidence(smoke_inventory),
        "smoke MIOpen USER_DB_PATH claim differs",
    )
    lock_rows = smoke_inventory["tmp"]
    lock_directory_relative = f"tmp/{EXPECTED_MIOPEN_LOCK_DIRECTORY_BASENAME}"
    lock_directories = [
        row
        for row in lock_rows
        if row.get("path") == lock_directory_relative
        and row.get("kind") == "directory"
    ]
    lock_files = [row for row in lock_rows if row.get("kind") == "file"]
    lock_identity = expected_miopen_lock_file_basenames(
        Path(prepare["cache_root"]),
        require_existing_parent=require_cache_present,
    )
    require(
        smoke.get("scoped_miopen_temp_lock_activity_required") is True
        and smoke.get("scoped_miopen_temp_lock_activity_observed") is True
        and smoke.get("tmpdir_cpp_temp_directory_path_redirect_observed") is True
        and type(smoke_temp_lock) is dict
        and smoke_temp_lock.get("tmpdir") == str(Path(prepare["cache_root"]) / "tmp")
        and smoke_temp_lock.get("lock_directory")
        == str(Path(prepare["cache_root"]) / lock_directory_relative)
        and smoke_temp_lock.get("lock_directory_relative_path")
        == lock_directory_relative
        and smoke_temp_lock.get("lock_directory_mode") == _MODE_0777
        and smoke_temp_lock.get("cpp_temp_directory_path_redirect_observed") is True
        and smoke_temp_lock.get("activity_required") is True
        and smoke_temp_lock.get("activity_observed") is True
        and smoke_temp_lock.get("user_db_parent_for_lock_hash")
        == lock_identity["user_db_parent"]
        and smoke_temp_lock.get("user_db_parent_md5")
        == lock_identity["user_db_parent_md5"]
        and smoke_temp_lock.get("expected_lock_basenames")
        == lock_identity["expected_lock_basenames"]
        and smoke_temp_lock.get("lock_files") == lock_files
        and smoke_temp_lock.get("global_tmp_lock_root_authoritative") is False
        and len(lock_directories) == 1
        and lock_directories[0].get("mode") == _MODE_0777
        and lock_directories[0].get("owner_uid") == os.getuid()
        and bool(lock_files)
        and all(
            row.get("path")
            == f"{lock_directory_relative}/{Path(str(row.get('path'))).name}"
            and Path(str(row.get("path"))).name
            in lock_identity["expected_lock_basenames"]
            and row.get("mode") == _MODE_0777
            and row.get("owner_uid") == os.getuid()
            and row.get("nlink") == 1
            and row.get("physical_reopen") is True
            and row.get("nofollow") is True
            for row in lock_files
        ),
        "smoke scoped MIOpen temp lock closure differs",
    )
    global_before = smoke.get("global_miopen_lock_root_before_torch")
    global_after = smoke.get("global_miopen_lock_root_after_smoke")
    _validate_global_miopen_lock_observation(global_before)
    _validate_global_miopen_lock_observation(global_after)
    require(
        global_before
        == global_after
        == prepare["global_miopen_lock_root_before_torch"]
        and smoke.get("global_miopen_lock_root_metadata_unchanged") is True
        and smoke.get("global_miopen_lock_root_members_scanned") is False
        and smoke.get("global_miopen_lock_root_mutation_attempted") is False,
        "global MIOpen lock root smoke closure differs",
    )
    require(
        smoke.get("pinned_runtime") == expected_runtime,
        "smoke pinned torch/MIOpen runtime differs",
    )
    require(
        smoke.get("schema_version")
        == "bernini-full30-action-source7-reencode-miopen-conv-smoke-v3"
        and smoke.get("operation")
        == "WanResample-downsample-ZeroPad2d-right-bottom-1-then-Conv2d"
        and smoke.get("r2_failure_step") == "136141.115"
        and smoke.get("r2_failure_stack_location")
        == "diffusers/models/autoencoders/autoencoder_kl_wan.py:298"
        and smoke.get("r2_failure_stack_candidate_closure")
        == "all-three-first-temporal-chunk-spatial-downsample-convs"
        and smoke.get("vae_config_sha256") == EXPECTED_VAE_CONFIG_SHA256
        and smoke.get("geometry_count") == 3
        and smoke.get("module_and_input_declared_dtype") == "torch.float32"
        and smoke.get("cuda_autocast_dtype") == "torch.bfloat16"
        and smoke.get("loaded_miopen_library_paths")
        == [EXPECTED_MIOPEN_LIBRARY_PATH]
        and smoke.get("loaded_miopen_library_unique_exact_path") is True,
        "smoke WanResample/runtime closure differs",
    )
    geometries = smoke.get("geometries")
    require(type(geometries) is list and len(geometries) == 3, "smoke geometry list differs")
    for index, expected in enumerate(WAN_RESAMPLE_CANDIDATE_GEOMETRIES):
        channels, height, width, padded_height, padded_width, out_height, out_width = expected
        expected_boundary_samples = {
            96: [864.0, 576.0, 576.0, 384.0],
            192: [1728.0, 1152.0, 1152.0, 768.0],
            384: [3456.0, 2304.0, 2304.0, 1536.0],
        }[channels]
        row = geometries[index]
        require(
            type(row) is dict
            and row.get("candidate_index") == index
            and row.get("pre_pad_input_shape") == [1, channels, height, width]
            and row.get("conv_input_shape")
            == [1, channels, padded_height, padded_width]
            and row.get("weight_shape") == [channels, channels, 3, 3]
            and row.get("bias_shape") == [channels]
            and row.get("stride") == [2, 2]
            and row.get("padding") == [0, 0]
            and row.get("output_shape") == [1, channels, out_height, out_width]
            and row.get("module_and_input_declared_dtype") == "torch.float32"
            and row.get("cuda_autocast_dtype") == "torch.bfloat16"
            and row.get("output_dtype") == "torch.bfloat16"
            and row.get("boundary_samples_fp32") == expected_boundary_samples
            and row.get("finite") is True
            and row.get("cuda_synchronized") is True,
            f"smoke WanResample geometry {index} differs",
        )


def _validate_final_cache_for_cleanup(
    final_cache: Mapping[str, Any], *, smoke: Mapping[str, Any], prepare: Mapping[str, Any]
) -> None:
    root = Path(prepare["cache_root"])
    observed_inventory, observed_kernel_db = validate_miopen_kernel_cache_activity(root)
    observed_lock_inventory, observed_temp_lock = (
        validate_scoped_miopen_temp_lock_activity(root)
    )
    require(
        observed_lock_inventory == observed_inventory,
        "final cache changed between kernel and scoped lock validation",
    )
    observed_user_db = miopen_user_db_evidence(observed_inventory)
    observed_global_lock = observe_global_miopen_lock_root()
    require(
        final_cache
        == {
            "captured_after_exact7_materialization": True,
            "cache_root": str(root),
            "inventory": observed_inventory,
            "miopen_kernel_db_evidence": observed_kernel_db,
            "miopen_user_db_evidence": observed_user_db,
            "scoped_miopen_temp_lock_evidence": observed_temp_lock,
            "global_miopen_lock_root_after_exact7": observed_global_lock,
            "global_miopen_lock_root_metadata_unchanged": True,
            "global_miopen_lock_root_members_scanned": False,
            "global_miopen_lock_root_mutation_attempted": False,
        },
        "post-materialization cache evidence differs physically",
    )
    smoke_kernel_db = smoke["miopen_kernel_db_evidence"]
    require(
        observed_kernel_db["relative_path"] == smoke_kernel_db["relative_path"]
        and observed_kernel_db["kern_db_row_count"]
        >= smoke_kernel_db["kern_db_row_count"]
        and observed_kernel_db["kern_db_nonempty"] is True,
        "post-materialization MIOpen kernel DB regressed from pre-source smoke",
    )
    require(
        observed_global_lock
        == smoke["global_miopen_lock_root_before_torch"]
        == smoke["global_miopen_lock_root_after_smoke"]
        == prepare["global_miopen_lock_root_before_torch"],
        "global MIOpen lock root metadata changed before cleanup",
    )


def cleanup_runtime_cache(
    *, prepare_receipt_path: Path, expected_prepare_sha256: str,
    controller_completion_path: Path, expected_controller_completion_sha256: str,
    cleanup_receipt_output: Path, controller_exit_status: int,
    environ: Optional[Mapping[str, str]] = None,
) -> Mapping[str, Any]:
    require(controller_exit_status == 0, "controller did not exit successfully")
    require(
        _short_hostname() == EXPECTED_COMPUTE_NODE,
        "runtime cache cleanup must run on the preparing compute node",
    )
    if environ is None:
        preliminary = _load_json(
            prepare_receipt_path,
            expected_prepare_sha256,
            fields=PREPARE_FIELDS,
            label="post-child prepare receipt",
        )
        env: Mapping[str, str] = {
            "SLURM_JOB_ID": preliminary["slurm_job_id"],
            "SLURM_STEP_ID": preliminary["slurm_step_id"],
            **preliminary["environment"],
        }
    else:
        env = environ
    prepare = validate_prepare_receipt(
        receipt_path=prepare_receipt_path,
        expected_sha256=expected_prepare_sha256,
        environ=env,
        require_cache_present=True,
        require_cache_empty=False,
    )
    completion = _load_json(
        controller_completion_path,
        expected_controller_completion_sha256,
        fields=None,
        label="controller completion",
        mode=0o400,
    )
    require(type(completion.get("completion_digest")) is str, "controller completion digest is absent")
    unsigned_completion = dict(completion)
    completion_digest = unsigned_completion.pop("completion_digest")
    require(completion_digest == object_sha256(unsigned_completion), "controller completion digest differs")
    runtime_cache = completion.get("runtime_cache")
    smoke = completion.get("cuda_miopen_smoke")
    final_cache = completion.get("runtime_cache_post_materialization")
    require(
        completion.get("complete") is True
        and completion.get("run_generation") == RUN_GENERATION
        and type(runtime_cache) is dict
        and runtime_cache.get("prepare_receipt_path") == str(prepare_receipt_path)
        and runtime_cache.get("prepare_receipt_file_sha256") == expected_prepare_sha256
        and runtime_cache.get("prepare_digest") == prepare["prepare_digest"]
        and runtime_cache.get("cache_root") == prepare["cache_root"]
        and type(smoke) is dict
        and smoke.get("complete") is True
        and smoke.get("prepare_digest") == prepare["prepare_digest"]
        and smoke.get("source_video_opened") is False
        and smoke.get("source_video_decoded") is False
        and smoke.get("vae_encode_calls") == 0,
        "controller/cache/smoke binding differs",
    )
    _validate_smoke_for_cleanup(
        smoke,
        prepare=prepare,
        prepare_receipt_path=prepare_receipt_path,
        expected_prepare_sha256=expected_prepare_sha256,
    )
    require(type(final_cache) is dict, "post-materialization cache evidence is absent")
    _validate_final_cache_for_cleanup(final_cache, smoke=smoke, prepare=prepare)
    root = Path(prepare["cache_root"])
    root_metadata = root.lstat()
    require(
        root_metadata.st_dev == prepare["cache_root_device"]
        and root_metadata.st_ino == prepare["cache_root_inode"],
        "cleanup cache root device/inode differs",
    )
    _validate_cleanup_tree(root, uid=os.getuid(), device=root_metadata.st_dev)
    shutil.rmtree(root)
    require(not root.exists() and not root.is_symlink(), "runtime cache cleanup did not remove root")
    _fsync_directory(root.parent)
    unsigned: dict[str, Any] = {
        "schema_version": CLEANUP_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_generation": RUN_GENERATION,
        "prepare_receipt_path": str(prepare_receipt_path),
        "prepare_receipt_file_sha256": expected_prepare_sha256,
        "prepare_digest": prepare["prepare_digest"],
        "controller_completion_path": str(controller_completion_path),
        "controller_completion_file_sha256": expected_controller_completion_sha256,
        "controller_completion_digest": completion_digest,
        "cache_root": str(root),
        "cache_root_device": root_metadata.st_dev,
        "cache_root_inode": root_metadata.st_ino,
        "cleanup_node": _short_hostname(),
        "controller_exit_status": controller_exit_status,
        "cleanup_after_controller_exit": True,
        "cleanup_before_numbered_step_exit": True,
        "controller_complete": True,
        "cache_root_removed": True,
        "cache_root_reusable": False,
        "scoped_miopen_temp_lock_root_removed": True,
        "global_miopen_lock_root_members_scanned": False,
        "global_miopen_lock_root_cleanup_attempted": False,
    }
    receipt = {**unsigned, "cleanup_digest": object_sha256(unsigned)}
    require(set(receipt) == CLEANUP_FIELDS, "cleanup receipt field closure differs")
    _write_json_create_only(cleanup_receipt_output, receipt)
    return receipt


def _validate_prepare_receipt_after_cleanup(
    receipt: Mapping[str, Any],
) -> None:
    """Validate the signed prepare authority without reopening deleted /tmp."""

    require(set(receipt) == PREPARE_FIELDS, "post-cleanup prepare field closure differs")
    unsigned = dict(receipt)
    declared = unsigned.pop("prepare_digest", None)
    require(
        declared == object_sha256(unsigned),
        "post-cleanup prepare receipt digest differs",
    )
    job_id = _token(receipt.get("slurm_job_id"), label="prepare SLURM_JOB_ID")
    step_id = _token(receipt.get("slurm_step_id"), label="prepare SLURM_STEP_ID")
    cache_parent = Path(str(receipt.get("cache_parent")))
    require(cache_parent == CACHE_PARENT, "post-cleanup cache parent differs")
    root = expected_cache_root(job_id, step_id, cache_parent=cache_parent)
    expected_filesystem = {
        "statfs_type": EXPECTED_STATFS_TYPE,
        "statfs_magic_hex": f"0x{EXPECTED_STATFS_MAGIC:x}",
        "mount_fstype": EXPECTED_MOUNT_FSTYPE,
        "mount_point": EXPECTED_MOUNT_POINT,
        "source": EXPECTED_MOUNT_SOURCE,
        "source_is_local_block_device": True,
        "local_filesystem": True,
    }
    environment = receipt.get("environment")
    require(
        receipt.get("schema_version") == PREPARE_SCHEMA
        and receipt.get("experiment_id") == EXPERIMENT_ID
        and receipt.get("run_generation") == RUN_GENERATION
        and receipt.get("cache_parent") == str(cache_parent)
        and receipt.get("cache_root") == str(root)
        and receipt.get("cache_name") == root.name
        and receipt.get("hostname") == EXPECTED_COMPUTE_NODE
        and receipt.get("filesystem") == expected_filesystem
        and receipt.get("owner_uid") == os.getuid()
        and type(receipt.get("owner_gid")) is int
        and receipt.get("directory_mode") == _MODE_0700
        and type(receipt.get("cache_root_device")) is int
        and type(receipt.get("cache_root_inode")) is int
        and receipt["cache_root_device"] >= 0
        and receipt["cache_root_inode"] > 0
        and type(environment) is dict
        and environment
        == {
            "HOME": environment.get("HOME"),
            "MIOPEN_USER_DB_PATH": str(root / "user-db"),
            "MIOPEN_CUSTOM_CACHE_DIR": str(root / "kernel-cache"),
            "XDG_CACHE_HOME": str(root / "xdg-cache"),
            "TMPDIR": str(root / "tmp"),
        }
        and type(environment.get("HOME")) is str
        and environment["HOME"].startswith("/")
        and environment["HOME"] != "/"
        and receipt.get("home_unchanged") is True
        and receipt.get("created_fresh_create_only") is True
        and receipt.get("canonical_paths_verified") is True
        and receipt.get("nofollow_verified") is True
        and receipt.get("torch_imported_during_prepare") is False
        and receipt.get("cache_reusable") is False
        and receipt.get("cleanup_policy")
        == "cleanup-scoped-cache-including-tmp-locks-on-compute-node-after-controller-success-before-numbered-step-exit;retain-on-failure;never-touch-global-lock-root;never-reuse",
        "post-cleanup prepare authority differs",
    )
    directories = receipt.get("directories")
    require(
        type(directories) is dict and set(directories) == set(SUBDIRECTORY_NAMES),
        "post-cleanup prepare directory closure differs",
    )
    for name in SUBDIRECTORY_NAMES:
        row = directories[name]
        require(
            type(row) is dict
            and set(row)
            == {"path", "owner_uid", "owner_gid", "mode", "canonical", "nofollow"}
            and row.get("path") == str(root / name)
            and row.get("owner_uid") == receipt["owner_uid"]
            and type(row.get("owner_gid")) is int
            and row.get("mode") == _MODE_0700
            and row.get("canonical") is True
            and row.get("nofollow") is True,
            f"post-cleanup prepare directory authority differs: {name}",
        )
    require(
        receipt.get("exclusive_fsync_probe")
        == {
            "o_excl": True,
            "o_nofollow": True,
            "file_fsync": True,
            "physical_reopen": True,
            "payload_match": True,
            "probe_removed": True,
        },
        "post-cleanup prepare exclusive probe differs",
    )
    sqlite_probe = receipt.get("sqlite_commit_reopen_probe")
    require(
        type(sqlite_probe) is dict
        and set(sqlite_probe)
        == {
            "database_created",
            "transaction_committed",
            "database_file_fsynced",
            "readonly_reopen",
            "committed_row_match",
            "probe_removed",
            "sqlite_version",
        }
        and all(
            sqlite_probe[field] is True
            for field in set(sqlite_probe) - {"sqlite_version"}
        )
        and type(sqlite_probe.get("sqlite_version")) is str
        and bool(sqlite_probe["sqlite_version"]),
        "post-cleanup prepare SQLite probe differs",
    )
    require(
        receipt.get("post_probe_empty_inventory")
        == {name: [] for name in SUBDIRECTORY_NAMES},
        "post-cleanup prepare empty inventory differs",
    )
    _validate_global_miopen_lock_observation(
        receipt.get("global_miopen_lock_root_before_torch")
    )


def _load_released_contract_modules() -> tuple[Any, Any, Any]:
    """Load the sealed producer contracts only during the post-srun audit."""

    method_root = Path(__file__).resolve(strict=True).parents[1]
    tools_root = method_root / "tools"
    for root in (method_root, tools_root):
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    names_and_paths = (
        (
            "full30_action_source7_reencode_plan_v1",
            method_root / "full30_action_source7_reencode_plan_v1.py",
        ),
        (
            "materialize_full30_action_source7_reencode_v1",
            tools_root / "materialize_full30_action_source7_reencode_v1.py",
        ),
        (
            "build_full30_action_source7_reencode_release_v1",
            tools_root / "build_full30_action_source7_reencode_release_v1.py",
        ),
    )
    modules: list[Any] = []
    for name, expected_path in names_and_paths:
        try:
            module = importlib.import_module(name)
            observed_path = Path(module.__file__).resolve(strict=True)
        except Exception as error:
            raise Source7RuntimeCacheError(
                f"released post-srun contract module is unavailable: {name}"
            ) from error
        require(
            observed_path == expected_path.resolve(strict=True),
            f"released post-srun contract module path differs: {name}",
        )
        modules.append(module)
    return modules[0], modules[1], modules[2]


def _validate_release_binding_after_cleanup(
    release_binding: Any, *, release_contract: Any,
) -> None:
    require(
        type(release_binding) is dict
        and set(release_binding) == RELEASE_BINDING_FIELDS,
        "post-cleanup release binding field closure differs",
    )
    manifest_sha = release_binding.get("manifest_file_sha256")
    require(
        type(manifest_sha) is str and _SHA256_RE.fullmatch(manifest_sha) is not None,
        "post-cleanup release manifest SHA differs",
    )
    manifest_path = Path(str(release_binding.get("manifest_path")))
    manifest = _load_shared_json_at(
        manifest_path,
        expected_sha256=manifest_sha,
        expected_mode=0o444,
        label="post-cleanup release manifest",
    )
    try:
        release_contract.validate_manifest(manifest)
    except Exception as error:
        raise Source7RuntimeCacheError(
            "post-cleanup release manifest semantic closure differs"
        ) from error
    require(
        release_binding
        == {
            "manifest_path": str(manifest_path),
            "manifest_file_sha256": manifest_sha,
            "manifest_digest": manifest["manifest_digest"],
            "content_closure_sha1": manifest["content_closure_sha1"],
            "release_generation": RUN_GENERATION,
        },
        "post-cleanup release binding differs physically",
    )
    method_root = Path(__file__).resolve(strict=True).parents[1]
    require(
        manifest.get("member_root") == "methods/bernini_action_editing"
        and type(manifest.get("files")) is list,
        "post-cleanup release member root differs",
    )
    for row in manifest["files"]:
        require(
            type(row) is dict
            and set(row) == {"path", "mode", "size", "sha256"}
            and type(row.get("path")) is str
            and type(row.get("mode")) is int
            and type(row.get("size")) is int
            and row["size"] > 0
            and type(row.get("sha256")) is str
            and _SHA256_RE.fullmatch(row["sha256"]) is not None,
            "post-cleanup release member row differs",
        )
        relative = Path(row["path"])
        require(
            not relative.is_absolute()
            and ".." not in relative.parts
            and relative.as_posix() == row["path"],
            "post-cleanup release member path differs",
        )
        raw, metadata = _read_shared_plain_file_at(
            method_root / relative,
            label=f"post-cleanup extracted release member {row['path']}",
            expected_mode=row["mode"],
            minimum_size=row["size"],
            maximum_size=row["size"],
        )
        require(
            metadata.st_size == row["size"]
            and hashlib.sha256(raw).hexdigest() == row["sha256"],
            f"post-cleanup extracted release member differs: {row['path']}",
        )


def _validate_plan_binding_after_cleanup(
    plan_binding: Any, *, controller_completion_path: Path,
    plan_contract: Any,
) -> Mapping[str, Any]:
    require(
        type(plan_binding) is dict and set(plan_binding) == PLAN_BINDING_FIELDS,
        "post-cleanup plan binding field closure differs",
    )
    expected_path = controller_completion_path.parent / "source7-plan.json"
    expected_sha = plan_binding.get("file_sha256")
    require(
        plan_binding.get("path") == str(expected_path)
        and type(expected_sha) is str
        and _SHA256_RE.fullmatch(expected_sha) is not None,
        "post-cleanup plan path/SHA binding differs",
    )
    value = _load_shared_json_at(
        expected_path,
        expected_sha256=expected_sha,
        expected_mode=0o400,
        label="post-cleanup frozen exact7 plan",
    )
    try:
        expected = plan_contract.validate_plan(plan_contract.canonical_plan())
        plan_contract.validate_plan(value)
    except Exception as error:
        raise Source7RuntimeCacheError(
            "post-cleanup frozen exact7 plan semantic closure differs"
        ) from error
    require(
        {
            row["iid"]: row["expected_posterior_shape"]
            for row in EXACT7_FROZEN_ROWS
        }
        == EXACT7_POSTERIOR_SHAPES,
        "post-cleanup embedded exact7 shape registry differs",
    )
    require(
        value == expected
        and plan_binding.get("plan_digest") == value["plan_digest"],
        "post-cleanup frozen exact7 plan differs",
    )
    frozen_rows = []
    for frozen in EXACT7_FROZEN_ROWS:
        iid = frozen["iid"]
        frozen_rows.append(
            {
                **dict(frozen),
                "source_video_path": str(
                    EXPECTED_SOURCE_BASE / iid / "samples" / iid / "source_video.mp4"
                ),
                "output_filename": f"{iid}.source-posterior-index0.pt",
            }
        )
    require(
        [
            {
                key: row[key]
                for key in (
                    "iid", "source_video_sha256", "expected_posterior_shape",
                    "group_id", "actor_id", "scene_id", "source_video_path",
                    "output_filename",
                )
            }
            for row in value["rows"]
        ]
        == frozen_rows,
        "post-cleanup exact7 IID/source/filename closure differs",
    )
    return value


def _validate_materialization_receipt_authority(
    receipt: Mapping[str, Any], *, materializer_contract: Any,
    plan_binding: Mapping[str, Any], plan: Mapping[str, Any], root: Path,
) -> Mapping[str, Mapping[str, Any]]:
    require(
        set(receipt) == materializer_contract.RECEIPT_FIELDS,
        "post-cleanup materialization receipt field closure differs",
    )
    unsigned = dict(receipt)
    declared = unsigned.pop("receipt_digest", None)
    require(
        type(declared) is str
        and _SHA256_RE.fullmatch(declared) is not None
        and declared == object_sha256(unsigned),
        "post-cleanup materialization receipt digest differs",
    )
    require(
        receipt.get("schema_version") == materializer_contract.SCHEMA_VERSION
        and receipt.get("method") == materializer_contract.METHOD_NAME
        and receipt.get("experiment_id") == EXPERIMENT_ID
        and receipt.get("complete") is True
        and receipt.get("plan") == plan_binding
        and receipt.get("output_root") == str(root)
        and receipt.get("row_count") == 7
        and receipt.get("output_filenames") == list(EXACT7_OUTPUT_FILENAMES)
        and receipt.get("output_exact_member_closure") is True
        and receipt.get("distinct_source_mp4_count") == 7
        and receipt.get("total_vae_encode_calls") == 7
        and receipt.get("posterior_sample_materialized") is False
        and receipt.get("external_existing_index0_opened") is False
        and receipt.get("external_existing_index0_reencoded") is False
        and receipt.get("inventory_snapshot_only") is True
        and receipt.get("exact8_authority_go_claimed") is False
        and receipt.get("teacher_cross_disjointness_pending") is True
        and receipt.get("optimizer_created") is False
        and receipt.get("optimizer_updates") == 0
        and receipt.get("training_authorized") is False,
        "post-cleanup materialization receipt authority differs",
    )
    for key, expected in EXPECTED_NEGATIVE_ACCESS.items():
        require(
            type(receipt.get(key)) is type(expected) and receipt.get(key) == expected,
            f"post-cleanup materialization negative-access authority differs: {key}",
        )
    require(
        receipt.get("external_existing_index0")
        == {
            **EXPECTED_EXTERNAL_INDEX0,
            "opened_by_materializer": False,
            "included_in_exact7_output_files": False,
            "reencoded": False,
        },
        "post-cleanup materialization external index0 authority differs",
    )
    vae = receipt.get("vae_identity")
    require(
        type(vae) is dict
        and set(vae)
        == {
            "checkpoint_root", "checkpoint_content_manifest_path",
            "checkpoint_content_manifest_sha256", "vae_config_sha256",
            "vae_files", "every_vae_file_sha256_verified",
            "posterior_representation", "posterior_sample_materialized",
            "vae_identity_digest",
        },
        "post-cleanup materialization VAE identity field closure differs",
    )
    unsigned_vae = dict(vae)
    declared_vae = unsigned_vae.pop("vae_identity_digest", None)
    vae_files = vae.get("vae_files")
    require(
        vae.get("checkpoint_root") == str(EXPECTED_CHECKPOINT_ROOT)
        and vae.get("checkpoint_content_manifest_path")
        == str(EXPECTED_CHECKPOINT_CONTENT_MANIFEST)
        and vae.get("checkpoint_content_manifest_sha256")
        == EXPECTED_CHECKPOINT_CONTENT_MANIFEST_SHA256
        and vae.get("vae_config_sha256") == EXPECTED_VAE_CONFIG_SHA256
        and type(vae_files) is dict
        and len(vae_files) >= 2
        and all(
            type(name) is str
            and Path(name).parts[0] == "vae"
            and ".." not in Path(name).parts
            and type(digest) is str
            and _SHA256_RE.fullmatch(digest) is not None
            for name, digest in vae_files.items()
        )
        and any(name.endswith((".safetensors", ".bin")) for name in vae_files)
        and vae.get("every_vae_file_sha256_verified") is True
        and vae.get("posterior_representation") == "latent_dist.parameters_fp32"
        and vae.get("posterior_sample_materialized") is False
        and declared_vae == object_sha256(unsigned_vae),
        "post-cleanup materialization VAE identity differs",
    )
    rows = receipt.get("rows")
    require(
        type(rows) is list
        and len(rows) == 7
        and [row.get("iid") for row in rows if type(row) is dict]
        == [row["iid"] for row in EXACT7_FROZEN_ROWS],
        "post-cleanup materialization receipt row order differs",
    )
    by_iid: dict[str, Mapping[str, Any]] = {}
    for planned, frozen, row in zip(plan["rows"], EXACT7_FROZEN_ROWS, rows):
        iid = frozen["iid"]
        require(
            type(row) is dict
            and set(row) == materializer_contract.ROW_RECEIPT_FIELDS,
            f"post-cleanup materialization row field closure differs: {iid}",
        )
        for key, expected in EXPECTED_NEGATIVE_ACCESS.items():
            require(
                type(row.get(key)) is type(expected) and row.get(key) == expected,
                f"post-cleanup materialization row negative authority differs: {iid}.{key}",
            )
        expected_output = root / f"{iid}.source-posterior-index0.pt"
        source_identity = row.get("source_video_stat_identity")
        input_hw = row.get("input_hw")
        shape = frozen["expected_posterior_shape"]
        require(
            row.get("schema_version") == materializer_contract.ROW_SCHEMA
            and row.get("iid") == iid
            and row.get("analysis_split") == planned["analysis_split"] == "fit"
            and row.get("event_id") == planned["event_id"]
            and row.get("actor_kind") == planned["actor_kind"] == "adult-human"
            and row.get("q0_id") == planned["q0_id"]
            and row.get("group_id") == frozen["group_id"]
            and row.get("actor_id") == frozen["actor_id"]
            and row.get("scene_id") == frozen["scene_id"]
            and row.get("source_video_path") == planned["source_video_path"]
            and row.get("source_video_sha256") == frozen["source_video_sha256"]
            and row.get("source_video_sha256_before_decode")
            == frozen["source_video_sha256"]
            and row.get("source_video_sha256_after_decode")
            == frozen["source_video_sha256"]
            and row.get("source_video_pre_post_stat_and_hash_stable") is True
            and type(source_identity) is list
            and len(source_identity) == 4
            and all(type(item) is int for item in source_identity)
            and source_identity[2] > 0
            and row.get("frame_count") == 81
            and row.get("expected_fps") == 25.0
            and type(row.get("reported_fps")) in {int, float}
            and abs(float(row["reported_fps"]) - 25.0) <= 1e-3
            and type(input_hw) is list
            and len(input_hw) == 2
            and all(type(item) is int and item > 0 for item in input_hw)
            and row.get("source_aspect_bucket_hw") == [shape[3] * 8, shape[4] * 8]
            and row.get("posterior_parameters_path") == str(expected_output)
            and type(row.get("posterior_parameters_file_sha256")) is str
            and _SHA256_RE.fullmatch(row["posterior_parameters_file_sha256"])
            is not None
            and type(row.get("posterior_parameters_tensor_sha256")) is str
            and _SHA256_RE.fullmatch(row["posterior_parameters_tensor_sha256"])
            is not None
            and type(row.get("posterior_parameters_tensor_raw_sha256")) is str
            and _SHA256_RE.fullmatch(row["posterior_parameters_tensor_raw_sha256"])
            is not None
            and row.get("posterior_parameters_shape") == shape
            and row.get("posterior_parameters_dtype") == "torch.float32"
            and row.get("posterior_parameters_device") == "cpu"
            and row.get("posterior_parameters_layout") == "torch.strided"
            and row.get("posterior_parameters_contiguous") is True
            and row.get("posterior_parameters_finite") is True
            and row.get("posterior_parameters_bare_tensor") is True
            and row.get("posterior_sample_materialized") is False
            and row.get("physical_file_reopened_after_write") is True
            and row.get("physical_tensor_reopened_after_write") is True
            and row.get("physical_tensor_equal_to_encoded_tensor") is True
            and type(row.get("peak_allocated_bytes")) is int
            and row["peak_allocated_bytes"] >= 0,
            f"post-cleanup materialization row input/output authority differs: {iid}",
        )
        by_iid[iid] = row
    return by_iid


def _validate_published_exact7_after_cleanup(
    materialization: Any, *, controller_completion_path: Path,
    plan_binding: Mapping[str, Any], plan: Mapping[str, Any],
    materializer_contract: Any,
) -> None:
    require(
        type(materialization) is dict
        and set(materialization) == MATERIALIZATION_BINDING_FIELDS
        and materialization.get("all_seven_physical_files_and_tensors_reopened")
        is True,
        "post-cleanup exact7 materialization binding field closure differs",
    )
    root = controller_completion_path.parent / "physical_source_posterior_index0_exact7"
    receipt_path = root / "materialization_receipt.json"
    require(
        materialization.get("receipt_path") == str(receipt_path),
        "post-cleanup materialization receipt canonical path differs",
    )
    try:
        root_metadata = root.lstat()
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise Source7RuntimeCacheError(
            "post-cleanup materialization root is unavailable"
        ) from error
    require(
        resolved_root == root
        and stat.S_ISDIR(root_metadata.st_mode)
        and not stat.S_ISLNK(root_metadata.st_mode)
        and root_metadata.st_uid == os.getuid()
        and stat.S_IMODE(root_metadata.st_mode) == _MODE_0700,
        "post-cleanup materialization root metadata differs",
    )
    receipt_sha = materialization.get("receipt_file_sha256")
    require(
        type(receipt_sha) is str and _SHA256_RE.fullmatch(receipt_sha) is not None,
        "post-cleanup materialization receipt SHA differs",
    )
    receipt = _load_shared_json_at(
        receipt_path,
        expected_sha256=receipt_sha,
        expected_mode=0o400,
        label="post-cleanup materialization receipt",
        fields=materializer_contract.RECEIPT_FIELDS,
    )
    require(
        materialization.get("receipt_digest") == receipt.get("receipt_digest"),
        "post-cleanup materialization receipt digest binding differs",
    )
    receipt_rows = _validate_materialization_receipt_authority(
        receipt,
        materializer_contract=materializer_contract,
        plan_binding=plan_binding,
        plan=plan,
        root=root,
    )
    post_rows = materialization.get("post_publish_rows")
    require(
        type(post_rows) is list
        and len(post_rows) == 7
        and [row.get("iid") for row in post_rows if type(row) is dict]
        == [row["iid"] for row in EXACT7_FROZEN_ROWS],
        "post-cleanup exact7 post-publish row order differs",
    )
    root_descriptor = _open_directory(root)
    try:
        require(
            set(os.listdir(root_descriptor))
            == set(EXACT7_OUTPUT_FILENAMES) | {"materialization_receipt.json"},
            "post-cleanup materialization member closure differs physically",
        )
    finally:
        os.close(root_descriptor)
    torch = _load_pinned_cpu_torch()
    for frozen, post_row in zip(EXACT7_FROZEN_ROWS, post_rows):
        iid = frozen["iid"]
        shape = frozen["expected_posterior_shape"]
        receipt_row = receipt_rows[iid]
        path = root / f"{iid}.source-posterior-index0.pt"
        require(
            type(post_row) is dict
            and set(post_row) == POST_PUBLISH_ROW_FIELDS
            and post_row.get("iid") == iid
            and post_row.get("path") == str(path)
            and post_row.get("shape") == shape
            and post_row.get("physical_file_and_tensor_reopened_post_publish")
            is True,
            f"post-cleanup exact7 post-publish row closure differs: {iid}",
        )
        tensor_bytes = 4
        for dimension in shape:
            tensor_bytes *= dimension
        raw, metadata = _read_shared_plain_file_at(
            path,
            label=f"post-cleanup published posterior {iid}",
            expected_mode=0o400,
            minimum_size=tensor_bytes,
            maximum_size=tensor_bytes + 16 * 1024 * 1024,
        )
        observed_file_sha = hashlib.sha256(raw).hexdigest()
        _, observed_tensor_sha, observed_raw_sha = _decode_published_bare_tensor(
            raw,
            expected_shape=shape,
            label=f"post-cleanup published posterior {iid}",
            torch=torch,
        )
        require(
            metadata.st_nlink == 1
            and metadata.st_uid == os.getuid()
            and stat.S_IMODE(metadata.st_mode) == 0o400
            and metadata.st_size == len(raw)
            and observed_file_sha
            == post_row.get("file_sha256")
            == receipt_row.get("posterior_parameters_file_sha256")
            and observed_tensor_sha
            == post_row.get("tensor_sha256")
            == receipt_row.get("posterior_parameters_tensor_sha256")
            and observed_raw_sha
            == post_row.get("tensor_raw_sha256")
            == receipt_row.get("posterior_parameters_tensor_raw_sha256"),
            f"post-cleanup published posterior physical/hash binding differs: {iid}",
        )


def _validate_completion_receipt_after_cleanup(
    completion: Mapping[str, Any], *, prepare: Mapping[str, Any],
    prepare_receipt_path: Path, expected_prepare_sha256: str,
    controller_completion_path: Path,
) -> None:
    """Revalidate signed controller authority after the numbered step exits."""

    unsigned = dict(completion)
    declared = unsigned.pop("completion_digest", None)
    require(
        type(declared) is str
        and _SHA256_RE.fullmatch(declared) is not None
        and declared == object_sha256(unsigned),
        "post-cleanup controller completion digest differs",
    )
    require(
        set(completion) == COMPLETION_FIELDS,
        "post-cleanup controller completion field closure differs",
    )
    holder = completion.get("holder")
    runtime_binding = completion.get("runtime_cache")
    smoke = completion.get("cuda_miopen_smoke")
    final_cache = completion.get("runtime_cache_post_materialization")
    release_binding = completion.get("release")
    plan_binding = completion.get("plan")
    require(
        completion.get("schema_version") == CONTROLLER_COMPLETION_SCHEMA
        and completion.get("experiment_id") == EXPERIMENT_ID
        and completion.get("run_generation") == RUN_GENERATION
        and completion.get("complete") is True
        and type(holder) is dict
        and set(holder) == HOLDER_FIELDS
        and holder.get("job_id") == 136141
        and holder.get("step_id") == int(prepare["slurm_step_id"])
        and holder.get("node") == EXPECTED_COMPUTE_NODE
        and holder.get("parent_retained") is True
        and holder.get("parent_cancelled") is False
        and holder.get("parent_released") is False
        and holder.get("parent_requeued") is False
        and completion.get("external_existing_index0_reencoded") is False
        and completion.get("inventory_snapshot_only") is True
        and completion.get("exact8_authority_go_claimed") is False
        and completion.get("teacher_cross_disjointness_pending") is True
        and completion.get("optimizer_created") is False
        and completion.get("optimizer_updates") == 0
        and completion.get("training_authorized") is False,
        "post-cleanup controller completion authority differs",
    )
    plan_contract, materializer_contract, release_contract = (
        _load_released_contract_modules()
    )
    _validate_release_binding_after_cleanup(
        release_binding, release_contract=release_contract
    )
    plan = _validate_plan_binding_after_cleanup(
        plan_binding,
        controller_completion_path=controller_completion_path,
        plan_contract=plan_contract,
    )
    require(
        completion.get("purpose") == plan["purpose"]
        and completion.get("scientific_target") == plan["scientific_target"]
        and completion.get("learning_target") == plan["learning_target"]
        and completion.get("numeric_target") == plan["numeric_target"]
        and completion.get("dataset") == plan["dataset"]
        and completion.get("steps") == plan["steps"]
        and completion.get("baseline") == plan["baseline"]
        and completion.get("core_validation") == plan["core_validation"],
        "post-cleanup controller frozen-plan authority differs",
    )
    expected_negative: Mapping[str, Any] = {
        "source_only_reencode_from_source_video": True,
        "vae_encode_calls_per_source": 1,
        "paired_dataset_accessed": False,
        "legacy_source_target_container_opened": False,
        "synthetic_target_index1_path_read": False,
        "synthetic_target_index1_bytes_read": False,
        "synthetic_target_index1_decoded": False,
        "synthetic_target_index1_filtered_on": False,
        "synthetic_target_index1_hashed": False,
        "target_video_path_present": False,
        "target_video_accessed": False,
    }
    for key, value in expected_negative.items():
        observed = completion.get(key)
        require(
            observed is value
            if type(value) is bool
            else type(observed) is type(value) and observed == value,
            f"post-cleanup controller negative-access authority differs: {key}",
        )
    external = completion.get("external_existing_index0")
    materialization = completion.get("materialization")
    require(
        type(external) is dict
        and external == EXPECTED_EXTERNAL_INDEX0
        and type(materialization) is dict
        and set(materialization) == MATERIALIZATION_BINDING_FIELDS
        and type(materialization.get("receipt_path")) is str
        and materialization["receipt_path"].startswith("/")
        and _SHA256_RE.fullmatch(
            str(materialization.get("receipt_file_sha256"))
        )
        is not None
        and _SHA256_RE.fullmatch(str(materialization.get("receipt_digest")))
        is not None
        and materialization.get(
            "all_seven_physical_files_and_tensors_reopened"
        )
        is True,
        "post-cleanup exact7 materialization authority differs",
    )
    post_rows = materialization["post_publish_rows"]
    require(
        type(post_rows) is list and len(post_rows) == len(EXACT7_POSTERIOR_SHAPES),
        "post-cleanup exact7 physical row count differs",
    )
    by_iid: dict[str, Mapping[str, Any]] = {}
    for row in post_rows:
        require(
            type(row) is dict
            and set(row) == POST_PUBLISH_ROW_FIELDS
            and type(row.get("iid")) is str
            and row["iid"] not in by_iid
            and type(row.get("path")) is str
            and row["path"].startswith("/")
            and _SHA256_RE.fullmatch(str(row.get("file_sha256"))) is not None
            and _SHA256_RE.fullmatch(str(row.get("tensor_sha256"))) is not None
            and _SHA256_RE.fullmatch(str(row.get("tensor_raw_sha256")))
            is not None
            and row.get("physical_file_and_tensor_reopened_post_publish") is True,
            "post-cleanup exact7 physical row closure differs",
        )
        by_iid[row["iid"]] = row
    require(
        set(by_iid) == set(EXACT7_POSTERIOR_SHAPES)
        and all(
            by_iid[iid].get("shape") == shape
            for iid, shape in EXACT7_POSTERIOR_SHAPES.items()
        ),
        "post-cleanup exact7 IID/shape authority differs",
    )
    _validate_published_exact7_after_cleanup(
        materialization,
        controller_completion_path=controller_completion_path,
        plan_binding=plan_binding,
        plan=plan,
        materializer_contract=materializer_contract,
    )
    require(
        type(runtime_binding) is dict
        and set(runtime_binding) == RUNTIME_BINDING_FIELDS
        and runtime_binding.get("prepare_receipt_path")
        == str(prepare_receipt_path)
        and runtime_binding.get("prepare_receipt_file_sha256")
        == expected_prepare_sha256
        and runtime_binding.get("prepare_digest") == prepare["prepare_digest"]
        and runtime_binding.get("cache_root") == prepare["cache_root"]
        and runtime_binding.get("hostname") == prepare["hostname"]
        and runtime_binding.get("cache_root_device")
        == prepare["cache_root_device"]
        and runtime_binding.get("cache_root_inode") == prepare["cache_root_inode"]
        and runtime_binding.get("filesystem") == prepare["filesystem"]
        and runtime_binding.get("directories") == prepare["directories"]
        and runtime_binding.get("environment") == prepare["environment"]
        and runtime_binding.get("home_unchanged") is True
        and runtime_binding.get("created_fresh_create_only") is True
        and runtime_binding.get("exclusive_fsync_probe")
        == prepare["exclusive_fsync_probe"]
        and runtime_binding.get("sqlite_commit_reopen_probe")
        == prepare["sqlite_commit_reopen_probe"]
        and runtime_binding.get("post_probe_empty_inventory")
        == prepare["post_probe_empty_inventory"]
        and runtime_binding.get("global_miopen_lock_root_before_torch")
        == prepare["global_miopen_lock_root_before_torch"]
        and runtime_binding.get("validated_before_torch_import") is True
        and runtime_binding.get("cache_reusable") is False
        and runtime_binding.get("cleanup_policy") == prepare["cleanup_policy"],
        "post-cleanup controller/prepare binding differs",
    )
    require(
        type(smoke) is dict and set(smoke) == SMOKE_FIELDS,
        "post-cleanup controller smoke field closure differs",
    )
    unsigned_smoke = dict(smoke)
    smoke_digest = unsigned_smoke.pop("smoke_digest", None)
    require(
        smoke_digest == object_sha256(unsigned_smoke)
        and smoke.get("schema_version")
        == "bernini-full30-action-source7-reencode-miopen-conv-smoke-v3"
        and smoke.get("experiment_id") == EXPERIMENT_ID
        and smoke.get("run_generation") == RUN_GENERATION
        and smoke.get("complete") is True
        and smoke.get("prepare_receipt_path") == str(prepare_receipt_path)
        and smoke.get("prepare_receipt_file_sha256") == expected_prepare_sha256
        and smoke.get("prepare_digest") == prepare["prepare_digest"]
        and smoke.get("cache_root") == prepare["cache_root"]
        and smoke.get("miopen_kernel_db_activity_required") is True
        and smoke.get("miopen_kernel_db_activity_observed") is True
        and smoke.get("scoped_miopen_temp_lock_activity_required") is True
        and smoke.get("scoped_miopen_temp_lock_activity_observed") is True
        and smoke.get("global_miopen_lock_root_metadata_unchanged") is True
        and smoke.get("global_miopen_lock_root_members_scanned") is False
        and smoke.get("global_miopen_lock_root_mutation_attempted") is False
        and smoke.get("source_video_opened") is False
        and smoke.get("source_video_decoded") is False
        and smoke.get("vae_encode_calls") == 0,
        "post-cleanup controller smoke authority differs",
    )
    _validate_smoke_for_cleanup(
        smoke,
        prepare=prepare,
        prepare_receipt_path=prepare_receipt_path,
        expected_prepare_sha256=expected_prepare_sha256,
        require_cache_present=False,
    )
    require(
        type(final_cache) is dict
        and set(final_cache) == FINAL_CACHE_FIELDS
        and final_cache.get("captured_after_exact7_materialization") is True
        and final_cache.get("cache_root") == prepare["cache_root"]
        and final_cache.get("global_miopen_lock_root_metadata_unchanged") is True
        and final_cache.get("global_miopen_lock_root_members_scanned") is False
        and final_cache.get("global_miopen_lock_root_mutation_attempted") is False,
        "post-cleanup controller final cache authority differs",
    )
    final_inventory = final_cache.get("inventory")
    final_kernel_db = final_cache.get("miopen_kernel_db_evidence")
    final_user_db = final_cache.get("miopen_user_db_evidence")
    final_temp_lock = final_cache.get("scoped_miopen_temp_lock_evidence")
    final_global_lock = final_cache.get("global_miopen_lock_root_after_exact7")
    require(
        type(final_inventory) is dict
        and set(final_inventory) == set(SUBDIRECTORY_NAMES)
        and final_user_db == miopen_user_db_evidence(final_inventory)
        and type(final_kernel_db) is dict
        and final_kernel_db.get("relative_path")
        == f"kernel-cache/{EXPECTED_MIOPEN_KERNEL_DB_BASENAME}"
        and final_kernel_db.get("file_mode") == _MODE_0600
        and final_kernel_db.get("nlink") == 1
        and final_kernel_db.get("ordinary_file") is True
        and final_kernel_db.get("nofollow_physical_reopen") is True
        and final_kernel_db.get("sqlite_header_verified") is True
        and final_kernel_db.get("sqlite_readonly_reopen") is True
        and final_kernel_db.get("sqlite_immutable_reopen") is True
        and final_kernel_db.get("sqlite_quick_check") == "ok"
        and final_kernel_db.get("wal_absent_or_empty") is True
        and final_kernel_db.get("kern_db_columns")
        == [list(row) for row in EXPECTED_MIOPEN_KERN_DB_COLUMNS]
        and final_kernel_db.get("kern_db_unique_index_columns")
        == [list(row) for row in EXPECTED_MIOPEN_KERN_DB_UNIQUE_INDEX]
        and type(final_kernel_db.get("kern_db_row_count")) is int
        and final_kernel_db["kern_db_row_count"] > 0
        and final_kernel_db.get("kern_db_nonempty") is True
        and final_kernel_db.get("inventory_stable_during_readonly_validation")
        is True,
        "post-cleanup controller final MIOpen DB authority differs",
    )
    lock_identity = expected_miopen_lock_file_basenames(
        Path(prepare["cache_root"]), require_existing_parent=False
    )
    require(
        type(final_temp_lock) is dict
        and final_temp_lock.get("activity_required") is True
        and final_temp_lock.get("activity_observed") is True
        and final_temp_lock.get("cpp_temp_directory_path_redirect_observed")
        is True
        and final_temp_lock.get("user_db_parent_for_lock_hash")
        == lock_identity["user_db_parent"]
        and final_temp_lock.get("user_db_parent_md5")
        == lock_identity["user_db_parent_md5"]
        and final_temp_lock.get("expected_lock_basenames")
        == lock_identity["expected_lock_basenames"]
        and final_temp_lock.get("global_tmp_lock_root_authoritative") is False
        and type(final_temp_lock.get("lock_files")) is list
        and bool(final_temp_lock["lock_files"]),
        "post-cleanup controller final scoped lock authority differs",
    )
    _validate_global_miopen_lock_observation(final_global_lock)
    require(
        final_global_lock
        == smoke["global_miopen_lock_root_before_torch"]
        == smoke["global_miopen_lock_root_after_smoke"]
        == prepare["global_miopen_lock_root_before_torch"],
        "post-cleanup controller final global lock authority differs",
    )


def validate_cleanup_receipt(
    *, cleanup_receipt_path: Path, expected_cleanup_sha256: str,
    prepare_receipt_path: Path, expected_prepare_sha256: str,
    controller_completion_path: Path, expected_controller_completion_sha256: str,
) -> Mapping[str, Any]:
    receipt = _load_json(
        cleanup_receipt_path,
        expected_cleanup_sha256,
        fields=CLEANUP_FIELDS,
        label="cleanup receipt",
    )
    unsigned = dict(receipt)
    declared = unsigned.pop("cleanup_digest", None)
    require(declared == object_sha256(unsigned), "cleanup receipt digest differs")
    prepare = _load_json(
        prepare_receipt_path,
        expected_prepare_sha256,
        fields=PREPARE_FIELDS,
        label="prepare receipt after cleanup",
    )
    _validate_prepare_receipt_after_cleanup(prepare)
    completion = _load_json(
        controller_completion_path,
        expected_controller_completion_sha256,
        fields=None,
        label="controller completion after cleanup",
        mode=0o400,
    )
    _validate_completion_receipt_after_cleanup(
        completion,
        prepare=prepare,
        prepare_receipt_path=prepare_receipt_path,
        expected_prepare_sha256=expected_prepare_sha256,
        controller_completion_path=controller_completion_path,
    )
    require(
        receipt["schema_version"] == CLEANUP_SCHEMA
        and receipt["experiment_id"] == EXPERIMENT_ID
        and receipt["run_generation"] == RUN_GENERATION
        and receipt["prepare_receipt_path"] == str(prepare_receipt_path)
        and receipt["prepare_receipt_file_sha256"] == expected_prepare_sha256
        and receipt["prepare_digest"] == prepare.get("prepare_digest")
        and receipt["controller_completion_path"] == str(controller_completion_path)
        and receipt["controller_completion_file_sha256"] == expected_controller_completion_sha256
        and receipt["controller_completion_digest"] == completion.get("completion_digest")
        and receipt["cache_root"] == prepare.get("cache_root")
        and receipt["cache_root_device"] == prepare.get("cache_root_device")
        and receipt["cache_root_inode"] == prepare.get("cache_root_inode")
        and receipt["cleanup_node"] == prepare.get("hostname") == EXPECTED_COMPUTE_NODE
        and receipt["controller_exit_status"] == 0
        and receipt["cleanup_after_controller_exit"] is True
        and receipt["cleanup_before_numbered_step_exit"] is True
        and receipt["controller_complete"] is True
        and receipt["cache_root_removed"] is True
        and receipt["cache_root_reusable"] is False
        and receipt["scoped_miopen_temp_lock_root_removed"] is True
        and receipt["global_miopen_lock_root_members_scanned"] is False
        and receipt["global_miopen_lock_root_cleanup_attempted"] is False,
        "cleanup receipt binding differs",
    )
    return receipt


def record_retained_failure(
    *, prepare_receipt_path: Path, expected_prepare_sha256: str,
    retained_failure_receipt_output: Path, controller_exit_status: int,
    environ: Optional[Mapping[str, str]] = None,
) -> Mapping[str, Any]:
    require(controller_exit_status != 0, "retained failure requires nonzero controller status")
    require(
        _short_hostname() == EXPECTED_COMPUTE_NODE,
        "retained failure receipt must be written on the preparing compute node",
    )
    prepare = validate_prepare_receipt(
        receipt_path=prepare_receipt_path,
        expected_sha256=expected_prepare_sha256,
        environ=environ,
        require_cache_present=True,
        require_cache_empty=False,
    )
    root = Path(prepare["cache_root"])
    metadata = root.lstat()
    filesystem = dict(_filesystem_identity(root))
    require(
        metadata.st_dev == prepare["cache_root_device"]
        and metadata.st_ino == prepare["cache_root_inode"]
        and filesystem == prepare["filesystem"],
        "retained failure cache identity differs",
    )
    scoped_lock_observation = observe_scoped_miopen_temp_lock_root(
        root, uid=os.getuid(), expected_device=metadata.st_dev
    )
    scoped_lock_present = scoped_lock_observation["exists"]
    unsigned: dict[str, Any] = {
        "schema_version": RETAINED_FAILURE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_generation": RUN_GENERATION,
        "prepare_receipt_path": str(prepare_receipt_path),
        "prepare_receipt_file_sha256": expected_prepare_sha256,
        "prepare_digest": prepare["prepare_digest"],
        "hostname": _short_hostname(),
        "cache_root": str(root),
        "cache_root_device": metadata.st_dev,
        "cache_root_inode": metadata.st_ino,
        "filesystem": filesystem,
        "controller_exit_status": controller_exit_status,
        "controller_complete": False,
        "cache_root_present": True,
        "cache_root_retained": True,
        "cache_root_reusable": False,
        "scoped_miopen_temp_lock_root_observation": scoped_lock_observation,
        "scoped_miopen_temp_lock_root_present": scoped_lock_present,
        "scoped_miopen_temp_lock_root_retained": scoped_lock_present,
        "global_miopen_lock_root_members_scanned": False,
        "global_miopen_lock_root_cleanup_attempted": False,
    }
    receipt = {
        **unsigned,
        "retained_failure_digest": object_sha256(unsigned),
    }
    require(
        set(receipt) == RETAINED_FAILURE_FIELDS,
        "retained failure receipt field closure differs",
    )
    _write_json_create_only(retained_failure_receipt_output, receipt)
    return receipt


def validate_retained_failure_receipt(
    *, retained_failure_receipt_path: Path,
    expected_retained_failure_sha256: str,
    prepare_receipt_path: Path,
    expected_prepare_sha256: str,
) -> Mapping[str, Any]:
    receipt = _load_json(
        retained_failure_receipt_path,
        expected_retained_failure_sha256,
        fields=RETAINED_FAILURE_FIELDS,
        label="retained failure receipt",
    )
    unsigned = dict(receipt)
    declared = unsigned.pop("retained_failure_digest", None)
    require(declared == object_sha256(unsigned), "retained failure digest differs")
    prepare = _load_json(
        prepare_receipt_path,
        expected_prepare_sha256,
        fields=PREPARE_FIELDS,
        label="prepare receipt for retained failure",
    )
    root = Path(prepare["cache_root"])
    root_metadata = _metadata(
        root, directory=True, mode=_MODE_0700, uid=os.getuid()
    )
    require(
        root_metadata.st_dev == prepare["cache_root_device"]
        and root_metadata.st_ino == prepare["cache_root_inode"],
        "retained failure cache root is not physically retained",
    )
    observation = receipt.get("scoped_miopen_temp_lock_root_observation")
    _validate_scoped_miopen_temp_lock_root_observation(
        observation, cache_root=root
    )
    observed_now = observe_scoped_miopen_temp_lock_root(
        root,
        uid=os.getuid(),
        expected_device=prepare["cache_root_device"],
    )
    require(
        receipt["schema_version"] == RETAINED_FAILURE_SCHEMA
        and receipt["experiment_id"] == EXPERIMENT_ID
        and receipt["run_generation"] == RUN_GENERATION
        and receipt["prepare_receipt_path"] == str(prepare_receipt_path)
        and receipt["prepare_receipt_file_sha256"] == expected_prepare_sha256
        and receipt["prepare_digest"] == prepare.get("prepare_digest")
        and receipt["hostname"] == prepare.get("hostname") == EXPECTED_COMPUTE_NODE
        and receipt["cache_root"] == prepare.get("cache_root")
        and receipt["cache_root_device"] == prepare.get("cache_root_device")
        and receipt["cache_root_inode"] == prepare.get("cache_root_inode")
        and receipt["filesystem"] == prepare.get("filesystem")
        and type(receipt["controller_exit_status"]) is int
        and receipt["controller_exit_status"] != 0
        and receipt["controller_complete"] is False
        and receipt["cache_root_present"] is True
        and receipt["cache_root_retained"] is True
        and receipt["cache_root_reusable"] is False
        and receipt["scoped_miopen_temp_lock_root_observation"] == observed_now
        and receipt["scoped_miopen_temp_lock_root_present"]
        is observation["exists"]
        and receipt["scoped_miopen_temp_lock_root_retained"]
        is observation["exists"]
        and receipt["global_miopen_lock_root_members_scanned"] is False
        and receipt["global_miopen_lock_root_cleanup_attempted"] is False,
        "retained failure receipt binding differs",
    )
    return receipt


def _optional_artifact_fact(path: Optional[Path]) -> Mapping[str, Any]:
    if path is None:
        return {"path": None, "present": False, "file_sha256": None}
    absolute = path.absolute()
    try:
        metadata = absolute.lstat()
    except FileNotFoundError:
        return {"path": str(absolute), "present": False, "file_sha256": None}
    except OSError as error:
        raise Source7RuntimeCacheError(
            "phase failure artifact metadata is unavailable"
        ) from error
    require(
        not stat.S_ISLNK(metadata.st_mode)
        and stat.S_ISREG(metadata.st_mode)
        and metadata.st_nlink == 1
        and absolute.resolve(strict=True) == absolute,
        "phase failure artifact must be one canonical plain file",
    )
    return {
        "path": str(absolute),
        "present": True,
        "file_sha256": file_sha256(absolute),
    }


def record_phase_failure(
    *, phase_failure_receipt_output: Path, phase: str,
    failure_exit_status: int, prepare_receipt_path: Optional[Path] = None,
    controller_completion_path: Optional[Path] = None,
    cleanup_receipt_path: Optional[Path] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Mapping[str, Any]:
    """Publish a shared terminal even when a scoped cache phase cannot finish."""

    env = os.environ if environ is None else environ
    require(
        phase in PHASE_FAILURE_CHOICES,
        "phase failure phase differs",
    )
    require(
        type(failure_exit_status) is int and 0 < failure_exit_status <= 255,
        "phase failure status differs",
    )
    require(
        _short_hostname() == EXPECTED_COMPUTE_NODE,
        "phase failure terminal must be written on the compute child",
    )
    job_id = _token(env.get("SLURM_JOB_ID"), label="SLURM_JOB_ID")
    step_id = _token(env.get("SLURM_STEP_ID"), label="SLURM_STEP_ID")
    root = expected_cache_root(job_id, step_id, cache_parent=CACHE_PARENT)
    try:
        metadata = root.lstat()
    except FileNotFoundError:
        root_observation: Mapping[str, Any] = {
            "kind": "absent",
            "owner_uid": None,
            "owner_gid": None,
            "mode": None,
            "device": None,
            "inode": None,
            "canonical_non_symlink_directory": False,
        }
        root_present = False
    except OSError as error:
        raise Source7RuntimeCacheError(
            "phase failure cache root metadata is unavailable"
        ) from error
    else:
        if stat.S_ISLNK(metadata.st_mode):
            kind = "symlink"
        elif stat.S_ISDIR(metadata.st_mode):
            kind = "directory"
        elif stat.S_ISREG(metadata.st_mode):
            kind = "file"
        else:
            kind = "other"
        canonical_directory = False
        if kind == "directory":
            try:
                canonical_directory = root.resolve(strict=True) == root
            except OSError:
                canonical_directory = False
        root_observation = {
            "kind": kind,
            "owner_uid": metadata.st_uid,
            "owner_gid": metadata.st_gid,
            "mode": stat.S_IMODE(metadata.st_mode),
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "canonical_non_symlink_directory": canonical_directory,
        }
        root_present = True
    artifacts = {
        "prepare_receipt": _optional_artifact_fact(prepare_receipt_path),
        "controller_completion": _optional_artifact_fact(
            controller_completion_path
        ),
        "cleanup_receipt": _optional_artifact_fact(cleanup_receipt_path),
    }
    cleanup_phase = phase in {
        "cleanup-or-cleanup-receipt-publication",
        "post-cleanup-audit",
    }
    unsigned: dict[str, Any] = {
        "schema_version": PHASE_FAILURE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_generation": RUN_GENERATION,
        "phase": phase,
        "failure_exit_status": failure_exit_status,
        "hostname": _short_hostname(),
        "slurm_job_id": job_id,
        "slurm_step_id": step_id,
        "cache_root": str(root),
        "cache_root_observation": root_observation,
        "cache_root_present": root_present,
        "cache_root_retained": root_present,
        "cache_root_absent_at_terminal": not root_present,
        "cleanup_may_have_removed_cache_before_terminal": (
            cleanup_phase and not root_present
        ),
        "cache_root_reusable": False,
        "artifacts": artifacts,
        "global_miopen_lock_root_members_scanned": False,
        "global_miopen_lock_root_cleanup_attempted": False,
        "success_claimed": False,
        "final_marker_authorized": False,
    }
    receipt = {
        **unsigned,
        "phase_failure_digest": object_sha256(unsigned),
    }
    require(
        set(receipt) == PHASE_FAILURE_FIELDS,
        "phase failure receipt field closure differs",
    )
    _write_json_create_only(phase_failure_receipt_output, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--receipt-output", required=True)
    cleanup = commands.add_parser("cleanup")
    cleanup.add_argument("--prepare-receipt", required=True)
    cleanup.add_argument("--expected-prepare-receipt-sha256", required=True)
    cleanup.add_argument("--controller-completion", required=True)
    cleanup.add_argument("--expected-controller-completion-sha256", required=True)
    cleanup.add_argument("--cleanup-receipt-output", required=True)
    cleanup.add_argument("--controller-exit-status", required=True, type=int)
    audit = commands.add_parser("audit-cleanup")
    audit.add_argument("--cleanup-receipt", required=True)
    audit.add_argument("--expected-cleanup-receipt-sha256", required=True)
    audit.add_argument("--prepare-receipt", required=True)
    audit.add_argument("--expected-prepare-receipt-sha256", required=True)
    audit.add_argument("--controller-completion", required=True)
    audit.add_argument("--expected-controller-completion-sha256", required=True)
    retained = commands.add_parser("record-retained-failure")
    retained.add_argument("--prepare-receipt", required=True)
    retained.add_argument("--expected-prepare-receipt-sha256", required=True)
    retained.add_argument("--retained-failure-receipt-output", required=True)
    retained.add_argument("--controller-exit-status", required=True, type=int)
    audit_retained = commands.add_parser("audit-retained-failure")
    audit_retained.add_argument("--retained-failure-receipt", required=True)
    audit_retained.add_argument("--expected-retained-failure-sha256", required=True)
    audit_retained.add_argument("--prepare-receipt", required=True)
    audit_retained.add_argument("--expected-prepare-receipt-sha256", required=True)
    phase_failure = commands.add_parser("record-phase-failure")
    phase_failure.add_argument("--phase-failure-receipt-output", required=True)
    phase_failure.add_argument("--phase", required=True, choices=sorted(PHASE_FAILURE_CHOICES))
    phase_failure.add_argument("--failure-exit-status", required=True, type=int)
    phase_failure.add_argument("--prepare-receipt")
    phase_failure.add_argument("--controller-completion")
    phase_failure.add_argument("--cleanup-receipt")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_runtime_cache(receipt_output=Path(args.receipt_output))
    elif args.command == "cleanup":
        result = cleanup_runtime_cache(
            prepare_receipt_path=Path(args.prepare_receipt),
            expected_prepare_sha256=args.expected_prepare_receipt_sha256,
            controller_completion_path=Path(args.controller_completion),
            expected_controller_completion_sha256=args.expected_controller_completion_sha256,
            cleanup_receipt_output=Path(args.cleanup_receipt_output),
            controller_exit_status=args.controller_exit_status,
        )
    elif args.command == "audit-cleanup":
        result = validate_cleanup_receipt(
            cleanup_receipt_path=Path(args.cleanup_receipt),
            expected_cleanup_sha256=args.expected_cleanup_receipt_sha256,
            prepare_receipt_path=Path(args.prepare_receipt),
            expected_prepare_sha256=args.expected_prepare_receipt_sha256,
            controller_completion_path=Path(args.controller_completion),
            expected_controller_completion_sha256=args.expected_controller_completion_sha256,
        )
    elif args.command == "record-retained-failure":
        result = record_retained_failure(
            prepare_receipt_path=Path(args.prepare_receipt),
            expected_prepare_sha256=args.expected_prepare_receipt_sha256,
            retained_failure_receipt_output=Path(
                args.retained_failure_receipt_output
            ),
            controller_exit_status=args.controller_exit_status,
        )
    elif args.command == "audit-retained-failure":
        result = validate_retained_failure_receipt(
            retained_failure_receipt_path=Path(args.retained_failure_receipt),
            expected_retained_failure_sha256=args.expected_retained_failure_sha256,
            prepare_receipt_path=Path(args.prepare_receipt),
            expected_prepare_sha256=args.expected_prepare_receipt_sha256,
        )
    else:
        result = record_phase_failure(
            phase_failure_receipt_output=Path(
                args.phase_failure_receipt_output
            ),
            phase=args.phase,
            failure_exit_status=args.failure_exit_status,
            prepare_receipt_path=(
                Path(args.prepare_receipt) if args.prepare_receipt else None
            ),
            controller_completion_path=(
                Path(args.controller_completion)
                if args.controller_completion
                else None
            ),
            cleanup_receipt_path=(
                Path(args.cleanup_receipt) if args.cleanup_receipt else None
            ),
        )
    print(canonical_json_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
