#!/usr/bin/env python3
"""Independent release/runtime/media verifier and no-replace finalizer for v15c-r6."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import stat
import sys
from typing import Any, Mapping, Sequence


RELEASE_SCHEMA = "bernini-source-object-proposal-role-v15c-r6-release"
BOOTSTRAP_SCHEMA = "bernini-source-object-proposal-role-v15c-r6-bootstrap"
COMPLETE_SCHEMA = "bernini-source-object-proposal-role-v15c-r6-complete"
RELEASE_TAG = "v15c-r6"
RELEASE_RELATIVE_PATH = (
    "methods/bernini_action_editing/assets/"
    "e00_source_sam2_proposal_role_probe_v15c_r6_release.json"
)
EXPECTED_CORE_MEMBER_COUNT = 8
EXPECTED_SNAPSHOT_FILE_COUNT = 9
EXPECTED_JOB = "143808"
EXPECTED_NODE = "auh7-1b-gpu-292"
EXPECTED_GPU_NAME = "AMD Instinct MI210"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_INPUT_HASHES = {
    "source": "888789206a3120c0780be8961dee7fdda520502cb95f573fe211b269aaea53de",
    "sam2_checkpoint": "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318",
    "sam2_config": "1dbd6cb6dfebeaf588c7006ee222c6efbfa9049a7ad472a3cdfb2f5d919e8107",
    "r6_receipt": "8f081c990edd84a64ca35e78ca1de3d4ea6cf4b80bfcdec70bf54c51dc9ed959",
    "r6_tensors": "2535193d41a3405460bd152cd77bc61db7ef8ea6ba7cefd98f514f0787acc553",
}
SNAPSHOT_POLICY = {
    "construction_phase": {
        "directory_mode": "0700",
        "member_mode": "0400",
        "receipt_semantics": "historical_observation_before_sealing",
    },
    "sealed_phase": {
        "directory_mode": "0500",
        "member_mode": "0400",
        "receipt_semantics": "current_state_reverified_at_runtime",
    },
    "core_member_count": EXPECTED_CORE_MEMBER_COUNT,
    "snapshot_file_count": EXPECTED_SNAPSHOT_FILE_COUNT,
    "exact_tree_required": True,
    "reject_extra_symlink_pyc": True,
    "single_link_regular_files_required": True,
}
RELEASE_KEYS = {
    "schema_version",
    "tag",
    "core_member_count",
    "snapshot_file_count",
    "members",
    "manifest_relative_path",
    "snapshot_policy",
    "observer_only",
    "human_audit_action",
    "remote_gpu_status",
    "route_status",
    "route_authorized",
    "decode_authorized",
    "training_authorized",
    "release_sha256",
}
OVERLAY_SCHEMA = "bernini-source-object-proposal-role-overlay-v15c-r6"
MEDIA_SCHEMA = "bernini-source-object-proposal-role-media-validation-v15c-r6"
POSTFLIGHT_SCHEMA = "bernini-source-sam2-proposal-role-postflight-v15c-r3"
RESULT_SCHEMA = "bernini-source-object-proposal-role-probe-v15c-r3"
TRACK_SCHEMA = "bernini-source-sam2-proposal-tracks-v15c-r3"
ROLES = ("old_actor", "new_actor", "recipient")
VIDEO_KEYS = ("source", "all", *ROLES)
DISPLAY_FRAMES = [0, 20, 40, 60, 80]
VIDEO_GATE_KEYS = {"frame_count_81", "fps_25", "width_704", "height_1056"}
VIDEO_ROW_KEYS = {
    "relative_path",
    "sha256",
    "frame_count",
    "fps",
    "width",
    "height",
    "gates",
}
OVERLAY_KEYS = {
    "schema_version",
    "status",
    "inputs",
    "files",
    "media_contract",
    "media_validation_receipt_sha256",
    "display_frames",
    "all_role_contact_sheets_present",
    "all_unassigned_rows_include_full_failure_evidence",
    "synchronized_playback",
    "human_audit_action",
    "approve_action_available",
    "threshold_mutation_available",
    "localization_semantically_certified",
    "route_authorized",
    "decode_authorized",
    "training_authorized",
    "receipt_sha256",
}
# Exact contract emitted by build_source_object_proposal_role_v15c_r6_review.py.
# The key "all" is a semantic row name; its on-disk stem is deliberately
# "all_proposals", not "all".  All finalizer paths flow through this mapping.
BUILDER_MEDIA_CONTRACT = {
    "source": {
        "video": "media/source.mp4",
        "contact_sheet": "media/source_f00_20_40_60_80.jpg",
    },
    "all": {
        "video": "media/all_proposals.mp4",
        "contact_sheet": "media/all_proposals_f00_20_40_60_80.jpg",
    },
    "old_actor": {
        "video": "media/old_actor.mp4",
        "contact_sheet": "media/old_actor_f00_20_40_60_80.jpg",
    },
    "new_actor": {
        "video": "media/new_actor.mp4",
        "contact_sheet": "media/new_actor_f00_20_40_60_80.jpg",
    },
    "recipient": {
        "video": "media/recipient.mp4",
        "contact_sheet": "media/recipient_f00_20_40_60_80.jpg",
    },
}
EXPECTED_REVIEW_FILES = {
    "index.html",
    "media_validation.json",
    *(
        relative
        for row in BUILDER_MEDIA_CONTRACT.values()
        for relative in row.values()
    ),
}


class FinalizeV15CR6Error(RuntimeError):
    """The release, runtime, overlay, media, or publication closure differs."""


# The body was promoted from r5; retain one private compatibility spelling so
# every inherited fail-closed branch continues to raise the fresh r6 type.
FinalizeV15CR5Error = FinalizeV15CR6Error


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        raise FinalizeV15CR5Error(f"{label} is not lowercase SHA256")
    return value


def require_exact_keys(value: Any, keys: Sequence[str] | set[str], label: str) -> None:
    if type(value) is not dict or set(value) != set(keys):
        raise FinalizeV15CR5Error(f"{label} exact keys differ")


def _regular(path: Path, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as error:
        raise FinalizeV15CR5Error(f"{label} is absent") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise FinalizeV15CR5Error(f"{label} is not one regular non-symlink file")
    return info


def file_sha256(path: Path) -> str:
    _regular(path, str(path))
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise FinalizeV15CR5Error("file changed while hashing")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def descriptor_sha256(descriptor: int) -> str:
    try:
        original_offset = os.lseek(descriptor, 0, os.SEEK_CUR)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise FinalizeV15CR5Error("trusted Python descriptor is not regular")
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
        os.lseek(descriptor, original_offset, os.SEEK_SET)
    except OSError as error:
        raise FinalizeV15CR5Error("trusted Python descriptor differs") from error
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise FinalizeV15CR5Error("trusted Python descriptor changed")
    return digest.hexdigest()


def read_json(path: Path) -> Mapping[str, Any]:
    _regular(path, "JSON input")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise FinalizeV15CR5Error("JSON input differs") from error
    if type(value) is not dict:
        raise FinalizeV15CR5Error("JSON input is not one object")
    return value


def verify_self_hash(value: Mapping[str, Any], field: str = "receipt_sha256") -> None:
    payload = dict(value)
    claimed = require_sha(payload.pop(field, None), field)
    if claimed != object_sha256(payload):
        raise FinalizeV15CR5Error(f"{field} self-hash differs")


def _relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise FinalizeV15CR5Error(f"{label} path differs")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or value.endswith(".pyc")
        or "__pycache__" in path.parts
    ):
        raise FinalizeV15CR5Error(f"{label} path differs")
    return value


def parse_builder_media_contract(value: Any, label: str) -> Mapping[str, Mapping[str, str]]:
    """Parse the exact contract emitted by the sealed review builder."""
    require_exact_keys(value, set(VIDEO_KEYS), label)
    normalized: dict[str, Mapping[str, str]] = {}
    for key in VIDEO_KEYS:
        row = value[key]
        require_exact_keys(row, {"video", "contact_sheet"}, f"{label} {key}")
        normalized[key] = {
            "video": _relative(row.get("video"), f"{label} {key} video"),
            "contact_sheet": _relative(
                row.get("contact_sheet"), f"{label} {key} contact sheet"
            ),
        }
    if normalized != BUILDER_MEDIA_CONTRACT:
        raise FinalizeV15CR5Error("sealed review-builder media contract differs")
    all_paths = [relative for row in normalized.values() for relative in row.values()]
    if len(all_paths) != 2 * len(VIDEO_KEYS) or len(set(all_paths)) != len(all_paths):
        raise FinalizeV15CR5Error("review-builder media paths are not unique")
    return normalized


def _verify_manifest(manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    require_exact_keys(manifest, RELEASE_KEYS, "release")
    verify_self_hash(manifest, "release_sha256")
    members = manifest.get("members")
    if (
        manifest.get("schema_version") != RELEASE_SCHEMA
        or manifest.get("tag") != RELEASE_TAG
        or manifest.get("core_member_count") != EXPECTED_CORE_MEMBER_COUNT
        or manifest.get("snapshot_file_count") != EXPECTED_SNAPSHOT_FILE_COUNT
        or type(members) is not list
        or len(members) != EXPECTED_CORE_MEMBER_COUNT
        or manifest.get("manifest_relative_path") != RELEASE_RELATIVE_PATH
        or manifest.get("snapshot_policy") != SNAPSHOT_POLICY
        or manifest.get("observer_only") is not True
        or manifest.get("human_audit_action") != "reject_only"
        or manifest.get("remote_gpu_status") != "REMOTE_GPU_UNAUDITED"
        or manifest.get("route_status") != "ROUTE_NO_GO"
        or manifest.get("route_authorized") is not False
        or manifest.get("decode_authorized") is not False
        or manifest.get("training_authorized") is not False
    ):
        raise FinalizeV15CR5Error("release semantics differ")
    paths: list[str] = []
    normalized: list[Mapping[str, Any]] = []
    for row in members:
        require_exact_keys(row, {"path", "sha256", "size"}, "release member")
        relative = _relative(row.get("path"), "release member")
        digest = require_sha(row.get("sha256"), "release member hash")
        size = row.get("size")
        if type(size) is not int or size <= 0:
            raise FinalizeV15CR5Error("release member size differs")
        normalized.append({"path": relative, "sha256": digest, "size": size})
        paths.append(relative)
    if paths != sorted(paths) or len(set(paths)) != EXPECTED_CORE_MEMBER_COUNT:
        raise FinalizeV15CR5Error("release registry differs")
    return tuple(normalized)


def verify_release(
    root: Path, release_path: Path, expected_release_sha256: str
) -> Mapping[str, Any]:
    expected = require_sha(expected_release_sha256, "external release hash")
    root = root.absolute()
    if root.is_symlink() or not root.is_dir():
        raise FinalizeV15CR5Error("release root differs")
    expected_path = root / RELEASE_RELATIVE_PATH
    if release_path.absolute() != expected_path.absolute():
        raise FinalizeV15CR5Error("release placement differs")
    if file_sha256(expected_path) != expected:
        raise FinalizeV15CR5Error("release file hash differs")
    manifest = read_json(expected_path)
    members = _verify_manifest(manifest)
    for row in members:
        path = root / row["path"]
        info = _regular(path, row["path"])
        if info.st_size != row["size"] or file_sha256(path) != row["sha256"]:
            raise FinalizeV15CR5Error("release member bytes differ")
    return manifest


def _scan_tree(root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories = {"."}
    for current_text, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        current = Path(current_text)
        for name in directory_names:
            path = current / name
            info = path.lstat()
            relative = path.relative_to(root).as_posix()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise FinalizeV15CR5Error("tree contains a symlink/non-directory")
            if name == "__pycache__":
                raise FinalizeV15CR5Error("tree contains __pycache__")
            directories.add(relative)
        for name in file_names:
            path = current / name
            info = path.lstat()
            relative = path.relative_to(root).as_posix()
            if (
                stat.S_ISLNK(info.st_mode)
                or not stat.S_ISREG(info.st_mode)
                or name.endswith(".pyc")
                or info.st_nlink != 1
            ):
                raise FinalizeV15CR5Error(
                    "tree contains symlink/non-file/pyc/hardlink"
                )
            files.add(relative)
    return files, directories


def verify_snapshot(
    snapshot: Path, manifest: Mapping[str, Any], expected_release_sha256: str
) -> Mapping[str, Mapping[str, Any]]:
    snapshot = snapshot.absolute()
    expected_files = {RELEASE_RELATIVE_PATH, *(row["path"] for row in manifest["members"])}
    expected_directories = {"."}
    for relative in expected_files:
        parent = PurePosixPath(relative).parent
        while parent.as_posix() != ".":
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    files, directories = _scan_tree(snapshot)
    if files != expected_files or directories != expected_directories:
        raise FinalizeV15CR5Error("snapshot exact tree differs")
    expected_rows = {
        row["path"]: {"sha256": row["sha256"], "size": row["size"]}
        for row in manifest["members"]
    }
    release_file = snapshot / RELEASE_RELATIVE_PATH
    expected_rows[RELEASE_RELATIVE_PATH] = {
        "sha256": require_sha(expected_release_sha256, "snapshot release hash"),
        "size": release_file.stat().st_size,
    }
    observed: dict[str, Mapping[str, Any]] = {}
    for relative in sorted(expected_files):
        path = snapshot / relative
        info = _regular(path, relative)
        digest = file_sha256(path)
        expected = expected_rows[relative]
        if (
            digest != expected["sha256"]
            or info.st_size != expected["size"]
            or stat.S_IMODE(info.st_mode) != 0o400
            or info.st_nlink != 1
        ):
            raise FinalizeV15CR5Error("snapshot member bytes/mode/link-count differ")
        observed[relative] = {"sha256": digest, "size": info.st_size}
    for relative in expected_directories:
        path = snapshot if relative == "." else snapshot / relative
        if stat.S_IMODE(path.lstat().st_mode) != 0o500:
            raise FinalizeV15CR5Error("snapshot directory mode differs")
    return observed


def sealed_snapshot_observation(
    snapshot: Path,
    manifest: Mapping[str, Any],
    expected_release_sha256: str,
) -> Mapping[str, Any]:
    """Reobserve the *current* sealed tree; never relabel it as construction state."""
    files = verify_snapshot(snapshot, manifest, expected_release_sha256)
    _, directories = _scan_tree(snapshot)
    if len(files) != EXPECTED_SNAPSHOT_FILE_COUNT:
        raise FinalizeV15CR5Error("sealed snapshot file count differs")
    return {
        "observation_scope": "current_state_reverified_at_runtime",
        "directory_mode": "0500",
        "member_mode": "0400",
        "directory_count": len(directories),
        "file_count": len(files),
        "exact_tree_verified": True,
        "extras_symlinks_pyc_absent": True,
        "single_link_regular_files_verified": True,
        "files": files,
    }


def verify_bootstrap_receipt(
    receipt_path: Path,
    *,
    run_root: Path,
    snapshot: Path,
    manifest: Mapping[str, Any],
    expected_release_sha256: str,
    expected_bootstrap_sha256: str,
    expected_python_sha256: str,
) -> Mapping[str, Any]:
    if receipt_path.absolute() != run_root / "external_bootstrap_receipt.json":
        raise FinalizeV15CR5Error("bootstrap receipt placement differs")
    receipt = read_json(receipt_path)
    require_exact_keys(
        receipt,
        {
            "schema_version",
            "status",
            "external_bootstrap",
            "python",
            "release",
            "snapshot",
            "execution",
            "authority",
            "receipt_sha256",
        },
        "bootstrap receipt",
    )
    verify_self_hash(receipt)
    external = receipt["external_bootstrap"]
    python = receipt["python"]
    release = receipt["release"]
    snapshot_row = receipt["snapshot"]
    execution = receipt["execution"]
    authority = receipt["authority"]
    require_exact_keys(external, {"path", "sha256", "size"}, "external bootstrap")
    require_exact_keys(
        python,
        {
            "path",
            "sha256",
            "size",
            "startup_flags",
            "trusted_fd",
            "trusted_fd_sha256",
            "trusted_fd_samefile_as_path",
            "argv0",
            "sys_executable",
            "launch_mode",
        },
        "Python",
    )
    require_exact_keys(
        release,
        {
            "source_root",
            "manifest_path",
            "manifest_file_sha256",
            "manifest_internal_sha256",
            "core_member_count",
            "snapshot_file_count",
        },
        "bootstrap release",
    )
    require_exact_keys(
        snapshot_row,
        {"root", "construction_phase", "sealed_phase"},
        "bootstrap snapshot",
    )
    construction = snapshot_row["construction_phase"]
    sealed = snapshot_row["sealed_phase"]
    require_exact_keys(
        construction,
        {
            "observation_scope",
            "directory_mode",
            "member_mode",
            "directory_count",
            "file_count",
            "exact_tree_verified",
            "extras_symlinks_pyc_absent",
            "single_link_regular_files_verified",
        },
        "bootstrap construction-phase observation",
    )
    require_exact_keys(
        sealed,
        {
            "observation_scope",
            "directory_mode",
            "member_mode",
            "directory_count",
            "file_count",
            "exact_tree_verified",
            "extras_symlinks_pyc_absent",
            "single_link_regular_files_verified",
            "files",
        },
        "bootstrap sealed-phase observation",
    )
    require_exact_keys(execution, {"parent_job_id", "node", "run_root"}, "bootstrap execution")
    require_exact_keys(
        authority,
        {
            "observer_only",
            "human_audit_action",
            "remote_gpu_status_before_execution",
            "route_status",
            "route_authorized",
            "decode_authorized",
            "training_authorized",
        },
        "bootstrap authority",
    )
    bootstrap_path = Path(external["path"])
    python_path = Path(python["path"])
    try:
        source_root = bootstrap_path.resolve(strict=True).parents[3]
    except (IndexError, OSError) as error:
        raise FinalizeV15CR5Error("external bootstrap placement differs") from error
    source_manifest = source_root / RELEASE_RELATIVE_PATH
    current_sealed = sealed_snapshot_observation(
        snapshot, manifest, expected_release_sha256
    )
    if (
        receipt.get("schema_version") != BOOTSTRAP_SCHEMA
        or receipt.get("status") != "EXTERNAL_BOOTSTRAP_VERIFIED_AND_SNAPSHOT_SEALED"
        or external.get("sha256") != require_sha(expected_bootstrap_sha256, "bootstrap hash")
        or external.get("size") != _regular(bootstrap_path, "external bootstrap").st_size
        or file_sha256(bootstrap_path) != external["sha256"]
        or python.get("sha256") != require_sha(expected_python_sha256, "Python hash")
        or python.get("size") != _regular(python_path, "Python").st_size
        or file_sha256(python_path) != python["sha256"]
        or python.get("startup_flags") != ["-I", "-S", "-B"]
        or python.get("trusted_fd") != 8
        or python.get("trusted_fd_sha256") != python["sha256"]
        or python.get("trusted_fd_samefile_as_path") is not True
        or python.get("argv0") != str(python_path)
        or python.get("sys_executable") != str(python_path)
        or python.get("launch_mode")
        != "bash_exec_a_canonical_through_proc_fd"
        or release.get("source_root") != str(source_root)
        or release.get("manifest_path") != str(source_manifest)
        or file_sha256(source_manifest) != expected_release_sha256
        or release.get("manifest_file_sha256") != expected_release_sha256
        or release.get("manifest_internal_sha256") != manifest["release_sha256"]
        or release.get("core_member_count") != EXPECTED_CORE_MEMBER_COUNT
        or release.get("snapshot_file_count") != EXPECTED_SNAPSHOT_FILE_COUNT
        or snapshot_row.get("root") != str(snapshot)
        or construction
        != {
            "observation_scope": "historical_observation_before_sealing",
            "directory_mode": "0700",
            "member_mode": "0400",
            "directory_count": current_sealed["directory_count"],
            "file_count": EXPECTED_SNAPSHOT_FILE_COUNT,
            "exact_tree_verified": True,
            "extras_symlinks_pyc_absent": True,
            "single_link_regular_files_verified": True,
        }
        or sealed != current_sealed
        or execution != {"parent_job_id": 143808, "node": EXPECTED_NODE, "run_root": str(run_root)}
        or authority
        != {
            "observer_only": True,
            "human_audit_action": "reject_only",
            "remote_gpu_status_before_execution": "REMOTE_GPU_UNAUDITED",
            "route_status": "ROUTE_NO_GO",
            "route_authorized": False,
            "decode_authorized": False,
            "training_authorized": False,
        }
    ):
        raise FinalizeV15CR5Error("bootstrap receipt semantics differ")
    return receipt


def verify_live_python_authority(
    python: Mapping[str, Any], expected_python_sha256: str
) -> None:
    require_exact_keys(
        python,
        {
            "path",
            "sha256",
            "size",
            "startup_flags",
            "trusted_fd",
            "trusted_fd_sha256",
            "trusted_fd_samefile_as_path",
            "argv0",
            "sys_executable",
            "launch_mode",
        },
        "live Python receipt",
    )
    expected = require_sha(expected_python_sha256, "live Python hash")
    path = Path(python["path"])
    descriptor = python["trusted_fd"]
    if type(descriptor) is not int or descriptor != 8:
        raise FinalizeV15CR5Error("trusted Python FD differs")
    try:
        descriptor_info = os.fstat(descriptor)
        path_info = _regular(path, "live Python")
        executable_info = _regular(Path(sys.executable), "sys.executable")
    except OSError as error:
        raise FinalizeV15CR5Error("live Python identity differs") from error
    descriptor_identity = (descriptor_info.st_dev, descriptor_info.st_ino)
    if (
        sys.executable != str(path)
        or python["argv0"] != str(path)
        or python["sys_executable"] != str(path)
        or python["sha256"] != expected
        or python["trusted_fd_sha256"] != expected
        or descriptor_sha256(descriptor) != expected
        or file_sha256(path) != expected
        or descriptor_identity != (path_info.st_dev, path_info.st_ino)
        or descriptor_identity
        != (executable_info.st_dev, executable_info.st_ino)
        or python["trusted_fd_samefile_as_path"] is not True
        or python["launch_mode"]
        != "bash_exec_a_canonical_through_proc_fd"
    ):
        raise FinalizeV15CR5Error(
            "live Python FD/argv0/sys.executable authority differs"
        )


def verify_inputs(paths: Mapping[str, Path]) -> Mapping[str, Mapping[str, Any]]:
    if set(paths) != set(EXPECTED_INPUT_HASHES):
        raise FinalizeV15CR5Error("input registry differs")
    result: dict[str, Mapping[str, Any]] = {}
    for label in EXPECTED_INPUT_HASHES:
        path = paths[label].absolute()
        info = _regular(path, label)
        digest = file_sha256(path)
        if digest != EXPECTED_INPUT_HASHES[label]:
            raise FinalizeV15CR5Error(f"{label} pin differs")
        result[label] = {"path": str(path), "sha256": digest, "size": info.st_size}
    return result


def verify_runtime(
    *,
    run_root: Path,
    snapshot: Path,
    release_path: Path,
    release_sha256: str,
    bootstrap_receipt: Path,
    bootstrap_sha256: str,
    python_sha256: str,
    inputs: Mapping[str, Path],
    require_live_authority: bool = True,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Mapping[str, Any]]]:
    run_root = run_root.absolute()
    snapshot = snapshot.absolute()
    if (
        run_root.is_symlink()
        or not run_root.is_dir()
        or stat.S_IMODE(run_root.lstat().st_mode) != 0o700
        or snapshot.parent != run_root
        or snapshot.name != "sealed_code_snapshot"
    ):
        raise FinalizeV15CR5Error("runtime placement/mode differs")
    manifest = verify_release(snapshot, release_path, release_sha256)
    receipt = verify_bootstrap_receipt(
        bootstrap_receipt,
        run_root=run_root,
        snapshot=snapshot,
        manifest=manifest,
        expected_release_sha256=release_sha256,
        expected_bootstrap_sha256=bootstrap_sha256,
        expected_python_sha256=python_sha256,
    )
    if require_live_authority:
        verify_live_python_authority(receipt["python"], python_sha256)
    input_rows = verify_inputs(inputs)
    if require_live_authority and (
        os.environ.get("SLURM_JOB_ID") != EXPECTED_JOB
        or socket.gethostname().split(".", 1)[0] != EXPECTED_NODE
        or os.environ.get("V15C_R6_EXTERNAL_BOOTSTRAP") != "1"
    ):
        raise FinalizeV15CR5Error("live execution authority differs")
    return manifest, receipt, input_rows


def _review_relative_video(key: str) -> str:
    try:
        return BUILDER_MEDIA_CONTRACT[key]["video"]
    except KeyError as error:
        raise FinalizeV15CR5Error("builder video key differs") from error


def _review_relative_sheet(key: str) -> str:
    try:
        return BUILDER_MEDIA_CONTRACT[key]["contact_sheet"]
    except KeyError as error:
        raise FinalizeV15CR5Error("builder contact-sheet key differs") from error


def _probe_video(path: Path) -> Mapping[str, Any]:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - remote runtime dependency
        raise FinalizeV15CR5Error("cv2 is unavailable for independent media replay") from error
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FinalizeV15CR5Error("video reopen failed")
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if getattr(frame, "shape", None) is None or tuple(frame.shape[:2]) != (1056, 704):
                raise FinalizeV15CR5Error("decoded video frame geometry differs")
            frame_count += 1
    finally:
        capture.release()
    if frame_count != 81 or abs(fps - 25.0) > 1.0e-6 or width != 704 or height != 1056:
        raise FinalizeV15CR5Error("decoded video contract differs")
    return {"frame_count": frame_count, "fps": fps, "width": width, "height": height}


def _probe_sheet(path: Path) -> Mapping[str, int]:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover
        raise FinalizeV15CR5Error("cv2 is unavailable for contact-sheet replay") from error
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if getattr(image, "shape", None) is None or tuple(image.shape) != (528, 1760, 3):
        raise FinalizeV15CR5Error("contact-sheet reopen differs")
    return {"height": 528, "width": 1760, "channels": 3}


def verify_review_bundle(run_root: Path, *, probe_media: bool = True) -> Mapping[str, Any]:
    run_root = run_root.absolute()
    review = run_root / "review"
    overlay_path = review / "overlay_receipt.json"
    media_path = review / "media_validation.json"
    overlay = read_json(overlay_path)
    require_exact_keys(overlay, OVERLAY_KEYS, "overlay receipt")
    verify_self_hash(overlay)
    require_exact_keys(
        overlay.get("inputs"),
        {"source_sha256", "track_receipt_sha256", "result_sha256", "postflight_sha256"},
        "overlay inputs",
    )
    builder_contract = parse_builder_media_contract(
        overlay.get("media_contract"), "overlay builder media contract"
    )
    expected_review_files = {
        "index.html",
        "media_validation.json",
        *(
            relative
            for row in builder_contract.values()
            for relative in row.values()
        ),
    }
    files = overlay.get("files")
    if (
        overlay.get("schema_version") != OVERLAY_SCHEMA
        or overlay.get("status") != "SYNCHRONIZED_REJECT_ONLY_OVERLAY_COMPLETE"
        or type(files) is not dict
        or set(files) != expected_review_files
        or overlay.get("display_frames") != DISPLAY_FRAMES
        or overlay.get("all_role_contact_sheets_present") is not True
        or overlay.get("all_unassigned_rows_include_full_failure_evidence") is not True
        or overlay.get("synchronized_playback") is not True
        or overlay.get("human_audit_action") != "reject_only"
        or overlay.get("approve_action_available") is not False
        or overlay.get("threshold_mutation_available") is not False
        or overlay.get("localization_semantically_certified") is not False
        or overlay.get("route_authorized") is not False
        or overlay.get("decode_authorized") is not False
        or overlay.get("training_authorized") is not False
    ):
        raise FinalizeV15CR5Error("overlay semantics differ")
    expected_actual_files = expected_review_files | {"overlay_receipt.json"}
    observed_files, _ = _scan_tree(review)
    if observed_files != expected_actual_files:
        raise FinalizeV15CR5Error("review exact file tree differs")
    for relative in sorted(expected_review_files):
        row = files[relative]
        require_exact_keys(row, {"sha256", "size"}, "overlay file")
        path = review / _relative(relative, "overlay file")
        info = _regular(path, relative)
        if (
            file_sha256(path) != require_sha(row.get("sha256"), "overlay file hash")
            or info.st_size != row.get("size")
        ):
            raise FinalizeV15CR5Error("overlay listed bytes differ")
    source_hash = EXPECTED_INPUT_HASHES["source"]
    if (
        overlay["inputs"]
        != {
            "source_sha256": source_hash,
            "track_receipt_sha256": file_sha256(run_root / "tracks/track_receipt.json"),
            "result_sha256": file_sha256(run_root / "result.json"),
            "postflight_sha256": file_sha256(run_root / "postflight.json"),
        }
        or file_sha256(review / builder_contract["source"]["video"]) != source_hash
    ):
        raise FinalizeV15CR5Error("overlay input binding differs")
    media = read_json(media_path)
    require_exact_keys(
        media,
        {
            "schema_version",
            "media_contract",
            "required_contract",
            "display_frames",
            "videos",
            "all_media_gates_pass",
            "receipt_sha256",
        },
        "media validation",
    )
    verify_self_hash(media)
    media_builder_contract = parse_builder_media_contract(
        media.get("media_contract"), "media-validation builder media contract"
    )
    require_exact_keys(media.get("required_contract"), {"frame_count", "fps", "width", "height"}, "media contract")
    videos = media.get("videos")
    if (
        media.get("schema_version") != MEDIA_SCHEMA
        or media_builder_contract != builder_contract
        or media.get("required_contract") != {"frame_count": 81, "fps": 25.0, "width": 704, "height": 1056}
        or media.get("display_frames") != DISPLAY_FRAMES
        or type(videos) is not dict
        or set(videos) != set(VIDEO_KEYS)
        or media.get("all_media_gates_pass") is not True
        or file_sha256(media_path) != overlay.get("media_validation_receipt_sha256")
        or files["media_validation.json"]["sha256"] != file_sha256(media_path)
    ):
        raise FinalizeV15CR5Error("media validation closure differs")
    replayed_videos: dict[str, Mapping[str, Any]] = {}
    verified_media_artifacts: dict[str, Mapping[str, Any]] = {}
    for key in VIDEO_KEYS:
        row = videos[key]
        require_exact_keys(row, VIDEO_ROW_KEYS, "media video row")
        require_exact_keys(row.get("gates"), VIDEO_GATE_KEYS, "media video gates")
        relative = builder_contract[key]["video"]
        sheet_relative = builder_contract[key]["contact_sheet"]
        video_path = review / relative
        sheet_path = review / sheet_relative
        sheet_info = _regular(sheet_path, sheet_relative)
        sheet_sha = file_sha256(sheet_path)
        if (
            row.get("relative_path") != relative
            or row.get("sha256") != file_sha256(video_path)
            or row.get("sha256") != files[relative]["sha256"]
            or row.get("frame_count") != 81
            or type(row.get("fps")) not in {int, float}
            or abs(float(row["fps"]) - 25.0) > 1.0e-6
            or row.get("width") != 704
            or row.get("height") != 1056
            or row.get("gates")
            != {"frame_count_81": True, "fps_25": True, "width_704": True, "height_1056": True}
            or sheet_sha != files[sheet_relative]["sha256"]
            or sheet_info.st_size != files[sheet_relative]["size"]
        ):
            raise FinalizeV15CR5Error("builder media artifact receipt differs")
        artifact: dict[str, Any] = {
            "video": {
                "relative_path": relative,
                "sha256": row["sha256"],
                "size": files[relative]["size"],
            },
            "contact_sheet": {
                "relative_path": sheet_relative,
                "sha256": sheet_sha,
                "size": sheet_info.st_size,
            },
        }
        if probe_media:
            replay = _probe_video(video_path)
            if replay != {key_name: row[key_name] for key_name in ("frame_count", "fps", "width", "height")}:
                raise FinalizeV15CR5Error("media video independent replay differs")
            sheet_replay = _probe_sheet(sheet_path)
            replayed_videos[key] = replay
            artifact["video_replay"] = replay
            artifact["contact_sheet_replay"] = sheet_replay
        verified_media_artifacts[key] = artifact
    index = review / "index.html"
    try:
        index_text = index.read_text(encoding="utf-8")
    except Exception as error:
        raise FinalizeV15CR5Error("review index reopen differs") from error
    if (
        "Only rejection is allowed." not in index_text
        or "approve_action_available:false" not in index_text
        or "route_authorized:false" not in index_text
    ):
        raise FinalizeV15CR5Error("review index authority differs")
    return {
        "overlay_file_sha256": file_sha256(overlay_path),
        "overlay_internal_sha256": overlay["receipt_sha256"],
        "media_validation_file_sha256": file_sha256(media_path),
        "media_validation_internal_sha256": media["receipt_sha256"],
        "listed_files": files,
        "builder_media_contract": builder_contract,
        "verified_media_artifacts": verified_media_artifacts,
        "independently_replayed_videos": replayed_videos,
        "contact_sheet_geometry": {"height": 528, "width": 1760, "channels": 3},
    }


def verify_observer_outputs(run_root: Path) -> Mapping[str, Any]:
    paths = {
        "track_receipt": run_root / "tracks/track_receipt.json",
        "track_tensors": run_root / "tracks/phase_coverage.safetensors",
        "track_output_manifest": run_root / "tracks/output_manifest.json",
        "result": run_root / "result.json",
        "postflight": run_root / "postflight.json",
    }
    values = {key: read_json(path) for key, path in paths.items() if path.suffix == ".json"}
    for key in ("track_receipt", "result", "postflight"):
        verify_self_hash(values[key])
    if (
        values["track_receipt"].get("schema_version") != TRACK_SCHEMA
        or values["result"].get("schema_version") != RESULT_SCHEMA
        or values["postflight"].get("schema_version") != POSTFLIGHT_SCHEMA
        or values["result"].get("route_authorized") is not False
        or values["result"].get("decode_authorized") is not False
        or values["result"].get("training_authorized") is not False
        or values["postflight"].get("human_audit_action") != "reject_only"
        or values["postflight"].get("human_audit_may_authorize_route") is not False
        or values["postflight"].get("route_authorized") is not False
        or values["postflight"].get("decode_authorized") is not False
        or values["postflight"].get("training_authorized") is not False
    ):
        raise FinalizeV15CR5Error("observer output authority differs")
    for path in paths.values():
        _regular(path, str(path))
    return {
        key: {"sha256": file_sha256(path), "size": path.stat().st_size}
        for key, path in paths.items()
    }


def _output_manifest(run_root: Path) -> Mapping[str, Mapping[str, Any]]:
    files, _ = _scan_tree(run_root)
    ignored = {"COMPLETE.manifest.json"}
    observed: dict[str, Mapping[str, Any]] = {}
    for relative in sorted(files):
        if relative in ignored:
            continue
        path = run_root / relative
        info = _regular(path, relative)
        observed[relative] = {"sha256": file_sha256(path), "size": info.st_size}
    return observed


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_noreplace(destination: Path, payload: bytes) -> None:
    temporary = destination.parent / f".{destination.name}.tmp.{os.getpid()}"
    if destination.exists() or destination.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise FinalizeV15CR5Error("COMPLETE destination is not fresh")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise FinalizeV15CR5Error("COMPLETE short write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, destination, follow_symlinks=False)
        _fsync_directory(destination.parent)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    temporary.unlink()
    _fsync_directory(destination.parent)
    if destination.read_bytes() != payload:
        raise FinalizeV15CR5Error("COMPLETE reopen differs")


def write_complete(args: argparse.Namespace) -> None:
    input_paths = _input_paths_from_args(args)
    manifest, bootstrap, inputs = verify_runtime(
        run_root=args.run_root,
        snapshot=args.snapshot,
        release_path=args.release_manifest,
        release_sha256=args.release_sha256,
        bootstrap_receipt=args.bootstrap_receipt,
        bootstrap_sha256=args.bootstrap_sha256,
        python_sha256=args.python_sha256,
        inputs=input_paths,
    )
    if (
        args.job_id != EXPECTED_JOB
        or args.node != EXPECTED_NODE
        or args.gpu_index != "0"
        or args.gpu_name != EXPECTED_GPU_NAME
    ):
        raise FinalizeV15CR5Error("final execution authority differs")
    run_root = args.run_root.absolute()
    observer_outputs = verify_observer_outputs(run_root)
    review = verify_review_bundle(run_root, probe_media=True)
    sealed_observation = sealed_snapshot_observation(
        args.snapshot, manifest, args.release_sha256
    )
    outputs_before = _output_manifest(run_root)
    complete: dict[str, Any] = {
        "schema_version": COMPLETE_SCHEMA,
        "status": "COMPLETE_OBSERVER_ONLY_REJECT_ONLY",
        "execution": {
            "parent_job_id": 143808,
            "node": EXPECTED_NODE,
            "visible_gpu_index": 0,
            "visible_gpu_count": 1,
            "visible_gpu_name": EXPECTED_GPU_NAME,
            "fresh_run_root": str(run_root),
        },
        "code": {
            "snapshot_root": str(args.snapshot.absolute()),
            "core_member_count": EXPECTED_CORE_MEMBER_COUNT,
            "snapshot_file_count": EXPECTED_SNAPSHOT_FILE_COUNT,
            "sealed_snapshot_observation": sealed_observation,
            "release_manifest_file_sha256": args.release_sha256,
            "release_manifest_internal_sha256": manifest["release_sha256"],
            "core_members": manifest["members"],
        },
        "bootstrap": {
            "receipt_file_sha256": file_sha256(args.bootstrap_receipt),
            "receipt_internal_sha256": bootstrap["receipt_sha256"],
            "external_bootstrap_sha256": args.bootstrap_sha256,
            "python_sha256": args.python_sha256,
        },
        "inputs": inputs,
        "observer_outputs": observer_outputs,
        "review": review,
        "outputs": outputs_before,
        "human_audit_action": "reject_only",
        "remote_gpu_package_status_before_this_run": "REMOTE_GPU_UNAUDITED",
        "route_status": "ROUTE_NO_GO",
        "localization_semantically_certified": False,
        "action_success_certified": False,
        "route_authorized": False,
        "decode_authorized": False,
        "training_authorized": False,
    }
    complete["complete_sha256"] = object_sha256(complete)
    payload = canonical_bytes(complete)
    if _output_manifest(run_root) != outputs_before:
        raise FinalizeV15CR5Error("outputs changed before COMPLETE publication")
    destination = run_root / "COMPLETE.manifest.json"
    publish_noreplace(destination, payload)
    if _output_manifest(run_root) != outputs_before:
        raise FinalizeV15CR5Error("outputs changed during COMPLETE publication")
    reopened = read_json(destination)
    verify_self_hash(reopened, "complete_sha256")
    if reopened != complete:
        raise FinalizeV15CR5Error("COMPLETE reopen differs")


def verify_complete(run_root: Path) -> None:
    run_root = run_root.absolute()
    complete = read_json(run_root / "COMPLETE.manifest.json")
    require_exact_keys(
        complete,
        {
            "schema_version",
            "status",
            "execution",
            "code",
            "bootstrap",
            "inputs",
            "observer_outputs",
            "review",
            "outputs",
            "human_audit_action",
            "remote_gpu_package_status_before_this_run",
            "route_status",
            "localization_semantically_certified",
            "action_success_certified",
            "route_authorized",
            "decode_authorized",
            "training_authorized",
            "complete_sha256",
        },
        "COMPLETE",
    )
    verify_self_hash(complete, "complete_sha256")
    execution = complete.get("execution")
    code = complete.get("code")
    bootstrap = complete.get("bootstrap")
    if (
        complete.get("schema_version") != COMPLETE_SCHEMA
        or complete.get("status") != "COMPLETE_OBSERVER_ONLY_REJECT_ONLY"
        or complete.get("human_audit_action") != "reject_only"
        or complete.get("remote_gpu_package_status_before_this_run") != "REMOTE_GPU_UNAUDITED"
        or complete.get("route_status") != "ROUTE_NO_GO"
        or complete.get("localization_semantically_certified") is not False
        or complete.get("action_success_certified") is not False
        or complete.get("route_authorized") is not False
        or complete.get("decode_authorized") is not False
        or complete.get("training_authorized") is not False
    ):
        raise FinalizeV15CR5Error("COMPLETE authority differs")
    require_exact_keys(
        execution,
        {"parent_job_id", "node", "visible_gpu_index", "visible_gpu_count", "visible_gpu_name", "fresh_run_root"},
        "COMPLETE execution",
    )
    require_exact_keys(
        code,
        {
            "snapshot_root",
            "core_member_count",
            "snapshot_file_count",
            "sealed_snapshot_observation",
            "release_manifest_file_sha256",
            "release_manifest_internal_sha256",
            "core_members",
        },
        "COMPLETE code",
    )
    require_exact_keys(
        bootstrap,
        {"receipt_file_sha256", "receipt_internal_sha256", "external_bootstrap_sha256", "python_sha256"},
        "COMPLETE bootstrap",
    )
    if execution != {
        "parent_job_id": 143808,
        "node": EXPECTED_NODE,
        "visible_gpu_index": 0,
        "visible_gpu_count": 1,
        "visible_gpu_name": EXPECTED_GPU_NAME,
        "fresh_run_root": str(run_root),
    }:
        raise FinalizeV15CR5Error("COMPLETE execution differs")
    snapshot = Path(code["snapshot_root"])
    manifest = verify_release(
        snapshot,
        snapshot / RELEASE_RELATIVE_PATH,
        code["release_manifest_file_sha256"],
    )
    current_sealed = sealed_snapshot_observation(
        snapshot, manifest, code["release_manifest_file_sha256"]
    )
    if (
        code["snapshot_root"] != str(run_root / "sealed_code_snapshot")
        or code["core_member_count"] != EXPECTED_CORE_MEMBER_COUNT
        or code["snapshot_file_count"] != EXPECTED_SNAPSHOT_FILE_COUNT
        or code["sealed_snapshot_observation"] != current_sealed
        or code["release_manifest_internal_sha256"] != manifest["release_sha256"]
        or code["core_members"] != manifest["members"]
        or complete.get("outputs") != _output_manifest(run_root)
    ):
        raise FinalizeV15CR5Error("COMPLETE byte replay differs")
    receipt_path = run_root / "external_bootstrap_receipt.json"
    receipt = verify_bootstrap_receipt(
        receipt_path,
        run_root=run_root,
        snapshot=snapshot,
        manifest=manifest,
        expected_release_sha256=code["release_manifest_file_sha256"],
        expected_bootstrap_sha256=bootstrap["external_bootstrap_sha256"],
        expected_python_sha256=bootstrap["python_sha256"],
    )
    verify_live_python_authority(receipt["python"], bootstrap["python_sha256"])
    if (
        os.environ.get("SLURM_JOB_ID") != EXPECTED_JOB
        or socket.gethostname().split(".", 1)[0] != EXPECTED_NODE
        or os.environ.get("V15C_R6_EXTERNAL_BOOTSTRAP") != "1"
    ):
        raise FinalizeV15CR5Error("COMPLETE live execution authority differs")
    if (
        bootstrap["receipt_file_sha256"] != file_sha256(receipt_path)
        or bootstrap["receipt_internal_sha256"] != receipt["receipt_sha256"]
    ):
        raise FinalizeV15CR5Error("COMPLETE bootstrap binding differs")
    input_paths = {key: Path(row["path"]) for key, row in complete["inputs"].items()}
    if complete["inputs"] != verify_inputs(input_paths):
        raise FinalizeV15CR5Error("COMPLETE input replay differs")
    if complete["observer_outputs"] != verify_observer_outputs(run_root):
        raise FinalizeV15CR5Error("COMPLETE observer output replay differs")
    if complete["review"] != verify_review_bundle(run_root, probe_media=True):
        raise FinalizeV15CR5Error("COMPLETE review replay differs")


def _input_paths_from_args(args: argparse.Namespace) -> Mapping[str, Path]:
    return {
        "source": args.source,
        "sam2_checkpoint": args.checkpoint,
        "sam2_config": args.config,
        "r6_receipt": args.r6_receipt,
        "r6_tensors": args.r6_tensors,
    }


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--release-manifest", required=True, type=Path)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--bootstrap-receipt", required=True, type=Path)
    parser.add_argument("--bootstrap-sha256", required=True)
    parser.add_argument("--python-sha256", required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--r6-receipt", required=True, type=Path)
    parser.add_argument("--r6-tensors", required=True, type=Path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    release = subparsers.add_parser("verify-release")
    release.add_argument("--root", required=True, type=Path)
    release.add_argument("--release-manifest", required=True, type=Path)
    release.add_argument("--expected-sha256", required=True)
    runtime = subparsers.add_parser("verify-runtime")
    _add_runtime_arguments(runtime)
    complete = subparsers.add_parser("complete")
    _add_runtime_arguments(complete)
    complete.add_argument("--job-id", required=True)
    complete.add_argument("--node", required=True)
    complete.add_argument("--gpu-index", required=True)
    complete.add_argument("--gpu-name", required=True)
    verify_done = subparsers.add_parser("verify-complete")
    verify_done.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "verify-release":
        verify_release(args.root, args.release_manifest, args.expected_sha256)
    elif args.command == "verify-runtime":
        verify_runtime(
            run_root=args.run_root,
            snapshot=args.snapshot,
            release_path=args.release_manifest,
            release_sha256=args.release_sha256,
            bootstrap_receipt=args.bootstrap_receipt,
            bootstrap_sha256=args.bootstrap_sha256,
            python_sha256=args.python_sha256,
            inputs=_input_paths_from_args(args),
        )
    elif args.command == "complete":
        write_complete(args)
    elif args.command == "verify-complete":
        verify_complete(args.run_root)
    else:  # pragma: no cover
        raise FinalizeV15CR5Error("unknown command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
