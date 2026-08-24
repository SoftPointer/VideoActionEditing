#!/usr/bin/env python3
"""External CPU completion controller for actual-target foundation V3.

The controller never loads a checkpoint or uses a GPU.  It rebuilds every
CaseEvidenceV3 and every gate from persisted non-raw mechanics.  An
authority-valid invocation create-only seals exactly one outcome: ``PASS`` or
``REJECTED``.  Engineering/authority failures create only a non-completion
attempt ledger.  REJECTED is completion evidence, never scientific success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import stat
import sys
from typing import Any, Mapping, NoReturn, Optional, Sequence

import actual_target_foundation_canary_v3 as authority
import actual_target_foundation_runtime_v3 as runtime
import actual_target_foundation_snapshot_v3 as snapshot_v3


SCHEMA = "actual-target-foundation-external-controller-v3"
SEAL_SCHEMA = "actual-target-foundation-completion-seal-v3"
CONTROLLER_STEP_META_SCHEMA = "actual-target-foundation-controller-step-meta-v3"
AUH_PYTHON_BIN = "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
CONTROLLER_DEVICE_ENV = (
    "CUDA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
)
EXTERNAL_FOUNDATION_IMPORT_PREFIXES = (
    "torch",
    "sam2",
    "cotracker",
    "transformers",
)


class ControllerV3Error(RuntimeError):
    pass


def fail(message: str) -> NoReturn:
    raise ControllerV3Error(message)


def _identity(row: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        row.st_dev,
        row.st_ino,
        row.st_mode,
        row.st_size,
        row.st_mtime_ns,
        row.st_ctime_ns,
    )


def _forbidden_foundation_imports() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in EXTERNAL_FOUNDATION_IMPORT_PREFIXES
        )
    )


def _plain_directory_record(path: Path) -> Mapping[str, Any]:
    try:
        snapshot_v3._plain_directory(path)
    except Exception as error:
        raise ControllerV3Error(
            f"directory is not lexical absolute plain: {path}"
        ) from error
    row = path.stat()
    return {
        "path": str(path),
        "device": row.st_dev,
        "inode": row.st_ino,
        "mode": stat.S_IMODE(row.st_mode),
        "no_symlink_components": True,
    }


def stable_file_record(path: Path, *, payload: bool = False) -> Mapping[str, Any]:
    try:
        snapshot_v3._plain_regular_file(path)
    except Exception as error:
        raise ControllerV3Error(f"file is not lexical absolute plain regular: {path}") from error
    before = path.stat()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        inside_before = os.fstat(descriptor)
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        inside_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.stat()
    if not (_identity(before) == _identity(inside_before) == _identity(inside_after) == _identity(after)):
        fail(f"file changed during stable controller read: {path}")
    data = b"".join(chunks)
    value: dict[str, Any] = {
        "path": str(path),
        "device": before.st_dev,
        "inode": before.st_ino,
        "mode": stat.S_IMODE(before.st_mode),
        "byte_count": len(data),
        "mtime_ns": before.st_mtime_ns,
        "ctime_ns": before.st_ctime_ns,
        "sha256": hashlib.sha256(data).hexdigest(),
        "stable_stat_verified": True,
        "no_symlink_components": True,
    }
    if payload:
        value["payload"] = data
    return value


def strict_json_file(path: Path) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    record = dict(stable_file_record(path, payload=True))
    data = record.pop("payload")
    value = authority.strict_json_bytes(data)
    if not isinstance(value, Mapping):
        fail(f"strict JSON file is not one object: {path}")
    return value, record


def _verify_digest(value: Any, label: str) -> None:
    if not isinstance(value, Mapping):
        fail(f"{label} is not an object")
    body = dict(value)
    claim = body.pop("digest", None)
    if claim != authority.object_sha256(body):
        fail(f"{label} digest differs")


def _nul_argv_record(path: Path) -> Mapping[str, Any]:
    record = dict(stable_file_record(path, payload=True))
    data = record.pop("payload")
    if not data or not data.endswith(b"\0"):
        fail(f"NUL argv does not end in one NUL: {path}")
    parts = data[:-1].split(b"\0")
    if not parts or any(not part for part in parts):
        fail(f"NUL argv contains an empty argument: {path}")
    try:
        decoded = [part.decode("utf-8", errors="strict") for part in parts]
    except UnicodeError as error:
        raise ControllerV3Error("NUL argv is not strict UTF-8") from error
    record.update({"argc": len(decoded), "argv": decoded, "nul_terminated": True})
    return record


def write_nul_argv(path: Path, argv: Sequence[str]) -> Mapping[str, Any]:
    if not argv or any("\0" in value for value in argv):
        fail("NUL argv requires nonempty NUL-free arguments")
    payload = b"".join(value.encode("utf-8", errors="strict") + b"\0" for value in argv)
    runtime.create_only_bytes(path, payload, 0o444)
    return _nul_argv_record(path)


def write_step_meta(
    path: Path,
    candidate: Path,
    cache_dir: Path,
    rank_argv_path: Path,
    snapshot_root: Path,
    miopen_user_dir: Path,
    miopen_custom_cache_dir: Path,
) -> Mapping[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    step_id = os.environ.get("SLURM_STEP_ID")
    local_rank = os.environ.get("LOCAL_RANK")
    world_size = os.environ.get("WORLD_SIZE")
    rocr = os.environ.get("ROCR_VISIBLE_DEVICES")
    if not all(isinstance(value, str) and value for value in (job_id, step_id, local_rank, world_size, rocr)):
        fail("step metadata requires real Slurm/rank/device environment")
    if local_rank != "0" or world_size != "1" or "," in rocr:
        fail("step metadata is not exact one-rank/one-device")
    rank_argv = _nul_argv_record(rank_argv_path)
    fixed = authority.load_authority()["fixed_paths"]
    if str(snapshot_root) != fixed["planned_preflip_snapshot_root"] or snapshot_root != Path(os.path.abspath(__file__)).parent:
        fail("rank step is not executing from the fixed immutable snapshot")
    run_root = Path(fixed["fresh_formal_run_root"])
    expected_scratch = {
        "MIOPEN_USER_DB_PATH": run_root / fixed["miopen_user_dirname"],
        "MIOPEN_CUSTOM_CACHE_DIR": run_root
        / fixed["miopen_custom_cache_dirname"],
    }
    observed_scratch = {
        "MIOPEN_USER_DB_PATH": miopen_user_dir,
        "MIOPEN_CUSTOM_CACHE_DIR": miopen_custom_cache_dir,
    }
    if observed_scratch != expected_scratch:
        fail("rank-step MIOpen scratch paths differ from fresh-run authority")
    if any(os.environ.get(name) != str(path) for name, path in expected_scratch.items()):
        fail("rank-step MIOpen environment differs from exact scratch paths")
    if "MIOPEN_DISABLE_CACHE" in os.environ:
        fail("MIOPEN_DISABLE_CACHE must be absent")
    scratch_records = {
        name: _plain_directory_record(path)
        for name, path in expected_scratch.items()
    }
    if any(
        record["mode"] != 0o700 or any(path.iterdir())
        for path, record in zip(expected_scratch.values(), scratch_records.values())
    ):
        fail("rank-step MIOpen scratch is not fresh empty mode 0700")
    snapshot_receipt = snapshot_v3.verify_snapshot(snapshot_root, verify_original=False)
    value = {
        "schema_version": "actual-target-foundation-step-meta-v3",
        "slurm_job_id": job_id,
        "slurm_step_id": step_id,
        "local_rank": 0,
        "world_size": 1,
        "rocr_visible_devices": rocr,
        "hostname": socket.gethostname(),
        "candidate_path": str(candidate),
        "cache_dir": str(cache_dir),
        "rank_argv_path": str(rank_argv_path),
        "rank_argv_sha256": rank_argv["sha256"],
        "rank_argv_argc": rank_argv["argc"],
        "snapshot_root": str(snapshot_root),
        "snapshot_receipt_digest": snapshot_receipt["digest"],
        "snapshot_manifest_file_sha256": snapshot_receipt["manifest_file_sha256"],
        "miopen_environment": {
            name: str(path) for name, path in expected_scratch.items()
        },
        "miopen_disable_cache_present": False,
        "miopen_directories_initially_empty": True,
        "miopen_directory_records": scratch_records,
    }
    receipt = {**value, "digest": authority.object_sha256(value)}
    runtime.create_only_json(path, receipt)
    return receipt


def _torch_is_imported() -> bool:
    return any(name == "torch" or name.startswith("torch.") for name in sys.modules)


def write_controller_step_meta(
    path: Path,
    controller_argv_path: Path,
    snapshot_root: Path,
) -> Mapping[str, Any]:
    """Create the CPU controller-step identity before any live closure imports."""

    job_id = os.environ.get("SLURM_JOB_ID")
    step_id = os.environ.get("SLURM_STEP_ID")
    if not all(isinstance(value, str) and value for value in (job_id, step_id)):
        fail("controller metadata requires one real Slurm step")
    visible = {name: os.environ.get(name) for name in CONTROLLER_DEVICE_ENV}
    if visible != {name: "" for name in CONTROLLER_DEVICE_ENV}:
        fail("controller metadata requires all CUDA/ROCR/HIP visibility masks empty")
    forbidden_imports = _forbidden_foundation_imports()
    if _torch_is_imported() or forbidden_imports:
        fail("controller metadata must be created before any torch/foundation import")
    controller_argv = _nul_argv_record(controller_argv_path)
    if controller_argv["mode"] != 0o444:
        fail("controller NUL argv must be frozen before the controller step")
    fixed = authority.load_authority()["fixed_paths"]
    if (
        str(snapshot_root) != fixed["planned_preflip_snapshot_root"]
        or snapshot_root != Path(os.path.abspath(__file__)).parent
    ):
        fail("controller step is not executing from the fixed immutable snapshot")
    snapshot_receipt = snapshot_v3.verify_snapshot(snapshot_root, verify_original=False)
    controller_wrapper = (
        snapshot_root
        / "scripts"
        / "auh_actual_target_foundation_canary_controller_wrapper_v3.sh"
    )
    controller_wrapper_record = stable_file_record(controller_wrapper)
    value = {
        "schema_version": CONTROLLER_STEP_META_SCHEMA,
        "slurm_job_id": job_id,
        "slurm_step_id": step_id,
        "hostname": socket.gethostname(),
        "cuda_visible_devices": "",
        "rocr_visible_devices": "",
        "hip_visible_devices": "",
        "torch_imported_at_metadata": False,
        "foundation_imports_at_metadata": [],
        "controller_wrapper_path": str(controller_wrapper),
        "controller_wrapper_sha256": controller_wrapper_record["sha256"],
        "controller_argv_path": str(controller_argv_path),
        "controller_argv_sha256": controller_argv["sha256"],
        "controller_argv_argc": controller_argv["argc"],
        "snapshot_root": str(snapshot_root),
        "snapshot_receipt_digest": snapshot_receipt["digest"],
        "snapshot_manifest_file_sha256": snapshot_receipt[
            "manifest_file_sha256"
        ],
    }
    receipt = {**value, "digest": authority.object_sha256(value)}
    runtime.create_only_json(path, receipt)
    return receipt


def freeze_existing_file(path: Path) -> Mapping[str, Any]:
    before = stable_file_record(path)
    os.chmod(path, 0o444)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    runtime._fsync_directory(path.parent)
    after = stable_file_record(path)
    if before["sha256"] != after["sha256"] or before["byte_count"] != after["byte_count"] or after["mode"] != 0o444:
        fail("formal log changed while freezing")
    return after


def freeze_candidate_cache(candidate_path: Path, cache_dir: Path) -> Mapping[str, Any]:
    candidate = freeze_existing_file(candidate_path)
    expected_names = {
        f'{row["pair_id"]}.json' for row in authority.load_preregistration()["pairs"]
    }
    try:
        snapshot_v3._plain_directory(cache_dir)
    except Exception as error:
        raise ControllerV3Error("cache is not a lexical plain directory") from error
    children = list(cache_dir.iterdir())
    observed = {path.name for path in children}
    if (
        observed != expected_names
        or any(path.is_symlink() or not path.is_file() for path in children)
    ):
        fail("cache cannot be frozen because its exact four-file closure differs")
    files = [freeze_existing_file(cache_dir / name) for name in sorted(expected_names)]
    os.chmod(cache_dir, 0o555)
    runtime._fsync_directory(cache_dir)
    runtime._fsync_directory(cache_dir.parent)
    if stat.S_IMODE(cache_dir.stat().st_mode) != 0o555:
        fail("cache directory did not freeze to mode 0555")
    value = {
        "candidate": candidate,
        "cache_directory": str(cache_dir),
        "cache_directory_mode": 0o555,
        "cache_files": files,
    }
    return {**value, "digest": authority.object_sha256(value)}


def _expected_miopen_scratch_paths() -> tuple[Path, Path, Path]:
    fixed = authority.load_authority()["fixed_paths"]
    run_root = Path(fixed["fresh_formal_run_root"])
    return (
        run_root / fixed["miopen_user_dirname"],
        run_root / fixed["miopen_custom_cache_dirname"],
        run_root / fixed["miopen_scratch_closure_filename"],
    )


def _scan_scratch_tree(root: Path, *, require_frozen: bool) -> Mapping[str, Any]:
    root_record = _plain_directory_record(root)
    directories = []
    files = []
    for current_text, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        directory_names.sort()
        file_names.sort()
        current = Path(current_text)
        current_row = current.lstat()
        if stat.S_ISLNK(current_row.st_mode) or not stat.S_ISDIR(current_row.st_mode):
            fail(f"MIOpen scratch directory differs: {current}")
        current_mode = stat.S_IMODE(current_row.st_mode)
        if require_frozen and current_mode != 0o555:
            fail(f"MIOpen scratch directory is not frozen 0555: {current}")
        directories.append(
            {
                "relative_path": "."
                if current == root
                else str(current.relative_to(root)),
                "device": current_row.st_dev,
                "inode": current_row.st_ino,
                "mode": current_mode,
            }
        )
        for name in directory_names:
            child = current / name
            child_row = child.lstat()
            if stat.S_ISLNK(child_row.st_mode) or not stat.S_ISDIR(child_row.st_mode):
                fail(f"MIOpen scratch contains symlink/non-directory: {child}")
        for name in file_names:
            child = current / name
            child_row = child.lstat()
            if stat.S_ISLNK(child_row.st_mode) or not stat.S_ISREG(child_row.st_mode):
                fail(f"MIOpen scratch contains symlink/non-file: {child}")
            record = stable_file_record(child)
            if require_frozen and record["mode"] != 0o444:
                fail(f"MIOpen scratch file is not frozen 0444: {child}")
            files.append(
                {
                    "relative_path": str(child.relative_to(root)),
                    "device": record["device"],
                    "inode": record["inode"],
                    "mode": record["mode"],
                    "byte_count": record["byte_count"],
                    "sha256": record["sha256"],
                }
            )
    value = {
        "root": str(root),
        "root_device": root_record["device"],
        "root_inode": root_record["inode"],
        "root_mode": root_record["mode"],
        "directories": sorted(directories, key=lambda row: row["relative_path"]),
        "files": sorted(files, key=lambda row: row["relative_path"]),
        "no_symlinks": True,
        "frozen": require_frozen,
    }
    return {**value, "digest": authority.object_sha256(value)}


def freeze_miopen_scratch(
    miopen_user_dir: Path,
    miopen_custom_cache_dir: Path,
    closure_path: Path,
) -> Mapping[str, Any]:
    expected_user, expected_custom, expected_closure = _expected_miopen_scratch_paths()
    if (
        miopen_user_dir != expected_user
        or miopen_custom_cache_dir != expected_custom
        or closure_path != expected_closure
        or closure_path.exists()
        or closure_path.is_symlink()
    ):
        fail("MIOpen scratch freeze paths differ from fresh create-only authority")
    roots = (miopen_user_dir, miopen_custom_cache_dir)
    for root in roots:
        if _plain_directory_record(root)["mode"] != 0o700:
            fail("MIOpen scratch root must be mode 0700 before freeze")
        _scan_scratch_tree(root, require_frozen=False)
    for root in roots:
        for current_text, directory_names, file_names in os.walk(
            root, topdown=False, followlinks=False
        ):
            current = Path(current_text)
            for name in sorted(file_names):
                child = current / name
                row = child.lstat()
                if stat.S_ISLNK(row.st_mode) or not stat.S_ISREG(row.st_mode):
                    fail(f"MIOpen scratch changed before freeze: {child}")
                os.chmod(child, 0o444, follow_symlinks=False)
                descriptor = os.open(
                    child, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                )
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            for name in sorted(directory_names):
                child = current / name
                row = child.lstat()
                if stat.S_ISLNK(row.st_mode) or not stat.S_ISDIR(row.st_mode):
                    fail(f"MIOpen scratch changed before directory freeze: {child}")
                os.chmod(child, 0o555, follow_symlinks=False)
                runtime._fsync_directory(child)
        os.chmod(root, 0o555, follow_symlinks=False)
        runtime._fsync_directory(root)
    trees = [_scan_scratch_tree(root, require_frozen=True) for root in roots]
    if trees != [_scan_scratch_tree(root, require_frozen=True) for root in roots]:
        fail("MIOpen scratch changed during post-freeze double scan")
    value = {
        "schema_version": "actual-target-foundation-miopen-scratch-closure-v3r3",
        "trees": trees,
        "tree_count": 2,
        "all_plain_no_symlinks": True,
        "all_files_mode_0444": True,
        "all_directories_mode_0555": True,
    }
    receipt = {**value, "digest": authority.object_sha256(value)}
    runtime.create_only_json(closure_path, receipt)
    return receipt


def _verify_miopen_scratch_closure(path: Path) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    expected_user, expected_custom, expected_closure = _expected_miopen_scratch_paths()
    if path != expected_closure:
        fail("MIOpen scratch closure path differs")
    value, record = strict_json_file(path)
    _verify_digest(value, "MIOpen scratch closure")
    if record["mode"] != 0o444:
        fail("MIOpen scratch closure is not frozen mode 0444")
    live_trees = [
        _scan_scratch_tree(root, require_frozen=True)
        for root in (expected_user, expected_custom)
    ]
    expected_value = {
        "schema_version": "actual-target-foundation-miopen-scratch-closure-v3r3",
        "trees": live_trees,
        "tree_count": 2,
        "all_plain_no_symlinks": True,
        "all_files_mode_0444": True,
        "all_directories_mode_0555": True,
    }
    if value != {**expected_value, "digest": authority.object_sha256(expected_value)}:
        fail("MIOpen scratch closure differs from independent stable tree scan")
    return value, record


def _scan_legacy_tree(root: Path) -> list[Mapping[str, Any]]:
    """Stable lexical tree scan used for frozen legacy snapshot/run roots."""

    _plain_directory_record(root)
    rows = []
    for current_text, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        directory_names.sort()
        file_names.sort()
        current = Path(current_text)
        current_row = current.lstat()
        if stat.S_ISLNK(current_row.st_mode) or not stat.S_ISDIR(
            current_row.st_mode
        ):
            fail(f"legacy tree directory differs: {current}")
        rows.append(
            {
                "relative_path": "."
                if current == root
                else str(current.relative_to(root)),
                "kind": "directory",
                "device": current_row.st_dev,
                "inode": current_row.st_ino,
                "mode": stat.S_IMODE(current_row.st_mode),
            }
        )
        for name in directory_names:
            child = current / name
            child_row = child.lstat()
            if stat.S_ISLNK(child_row.st_mode) or not stat.S_ISDIR(
                child_row.st_mode
            ):
                fail(f"legacy tree contains symlink/non-directory: {child}")
        for name in file_names:
            child = current / name
            child_row = child.lstat()
            if stat.S_ISLNK(child_row.st_mode) or not stat.S_ISREG(
                child_row.st_mode
            ):
                fail(f"legacy tree contains symlink/non-file: {child}")
            record = stable_file_record(child)
            rows.append(
                {
                    "relative_path": str(child.relative_to(root)),
                    "kind": "file",
                    "device": record["device"],
                    "inode": record["inode"],
                    "mode": record["mode"],
                    "byte_count": record["byte_count"],
                    "sha256": record["sha256"],
                }
            )
    return sorted(rows, key=lambda row: (row["relative_path"], row["kind"]))


def _verify_exact_legacy_tree(spec: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(spec, Mapping):
        fail(f"{label} authority is absent")
    root = Path(str(spec.get("root", "")))
    if not root.is_absolute():
        fail(f"{label} root is not absolute")
    first = _scan_legacy_tree(root)
    second = _scan_legacy_tree(root)
    if first != second:
        fail(f"{label} changed during stable double scan")
    value = {"root": str(root), "rows": first}
    digest = authority.object_sha256(value)
    if first != spec.get("rows") or digest != spec.get("canonical_tree_digest"):
        fail(f"{label} exact member/device/inode/mode/hash closure differs")
    receipt = {
        "verified": True,
        "root": str(root),
        "rows": first,
        "canonical_tree_digest": digest,
        "double_scan_stable": True,
        "no_symlinks": True,
        "exact_no_extra_members": True,
    }
    return {**receipt, "digest": authority.object_sha256(receipt)}


def _verify_generic_legacy_snapshot(spec: Any) -> Mapping[str, Any]:
    tree = _verify_exact_legacy_tree(spec, "legacy snapshot")
    root = Path(spec["root"])
    manifest_path = root / str(spec.get("manifest_relative_path", ""))
    manifest, manifest_record = strict_json_file(manifest_path)
    manifest_body = dict(manifest)
    manifest_self = manifest_body.pop("manifest_self_sha256", None)
    expected_manifest_fields = {
        "schema_version",
        "source_closure",
        "snapshot_root",
        "snapshot_file_count",
        "snapshot_files",
        "snapshot_directory_mode",
        "snapshot_file_modes",
        "no_symlinks",
        "immutable_permissions_applied",
        "manifest_self_sha256",
    }
    if (
        set(manifest) != expected_manifest_fields
        or manifest_record["sha256"] != spec.get("manifest_file_sha256")
        or manifest_self != spec.get("manifest_self_sha256")
        or manifest_self != authority.object_sha256(manifest_body)
        or manifest.get("schema_version") != spec.get("manifest_schema_version")
        or manifest.get("snapshot_root") != str(root)
        or manifest.get("snapshot_file_count") != spec.get("snapshot_file_count")
        or manifest.get("snapshot_directory_mode") != 0o555
        or manifest.get("snapshot_file_modes")
        != {"data": 0o444, "executable": 0o555}
        or manifest.get("no_symlinks") is not True
        or manifest.get("immutable_permissions_applied") is not True
    ):
        fail("legacy snapshot generic manifest header/self closure differs")
    source = manifest.get("source_closure")
    if not isinstance(source, Mapping):
        fail("legacy snapshot source closure is absent")
    source_body = dict(source)
    source_digest = source_body.pop("digest", None)
    source_rows = source.get("files")
    if (
        set(source)
        != {
            "source_root",
            "file_count",
            "files",
            "no_symlink_laundering",
            "digest",
        }
        or source_digest != spec.get("source_closure_digest")
        or source_digest != authority.object_sha256(source_body)
        or source.get("file_count") != spec.get("snapshot_file_count")
        or source.get("no_symlink_laundering") is not True
        or not isinstance(source.get("source_root"), str)
        or not Path(source.get("source_root", "")).is_absolute()
        or not isinstance(source_rows, list)
        or len(source_rows) != spec.get("snapshot_file_count")
    ):
        fail("legacy snapshot generic source closure differs")
    source_by_relative = {}
    for row in source_rows:
        if not isinstance(row, Mapping) or set(row) != {
            "relative_path",
            "original_path",
            "original_mode",
            "byte_count",
            "sha256",
            "original_path_components",
        }:
            fail("legacy snapshot source row schema differs")
        relative = row.get("relative_path")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in source_by_relative
            or row.get("original_path")
            != str(Path(source["source_root"]) / relative)
            or not isinstance(row.get("original_mode"), int)
            or isinstance(row.get("original_mode"), bool)
            or not isinstance(row.get("byte_count"), int)
            or isinstance(row.get("byte_count"), bool)
            or row.get("byte_count", -1) < 0
            or not isinstance(row.get("sha256"), str)
            or len(row.get("sha256", "")) != 64
            or not isinstance(row.get("original_path_components"), list)
        ):
            fail("legacy snapshot source row identity/metadata differs")
        source_by_relative[relative] = row
    manifest_rows = manifest.get("snapshot_files")
    if (
        not isinstance(manifest_rows, list)
        or len(manifest_rows) != spec.get("snapshot_file_count")
    ):
        fail("legacy snapshot manifest row count differs")
    observed_files = {
        row["relative_path"]: row
        for row in tree["rows"]
        if row["kind"] == "file"
        and row["relative_path"] != spec["manifest_relative_path"]
    }
    seen = set()
    for row in manifest_rows:
        if not isinstance(row, Mapping) or set(row) != {
            "relative_path",
            "sha256",
            "byte_count",
            "snapshot_mode",
        }:
            fail("legacy snapshot manifest row schema differs")
        relative = row.get("relative_path")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in seen
            or relative not in observed_files
            or relative not in source_by_relative
        ):
            fail("legacy snapshot manifest row identity differs")
        observed = observed_files[relative]
        source_row = source_by_relative[relative]
        if (
            row["sha256"] != observed["sha256"]
            or row["byte_count"] != observed["byte_count"]
            or row["snapshot_mode"] != observed["mode"]
            or row["sha256"] != source_row["sha256"]
            or row["byte_count"] != source_row["byte_count"]
            or row["snapshot_mode"]
            != (0o555 if source_row["original_mode"] & 0o111 else 0o444)
        ):
            fail("legacy snapshot manifest/member closure differs")
        seen.add(relative)
    if seen != set(observed_files) or seen != set(source_by_relative):
        fail("legacy snapshot generic manifest has missing/extra members")
    value = {
        "verified": True,
        "tree": tree,
        "manifest_file": manifest_record,
        "manifest_self_sha256": manifest_self,
        "source_closure_digest": source_digest,
        "snapshot_file_count": len(manifest_rows),
        "generic_legacy_schema_used": True,
        "current_payload_authority_not_used": True,
    }
    return {**value, "digest": authority.object_sha256(value)}


def _verify_v3r1_failed_attempt() -> Mapping[str, Any]:
    live = authority.load_authority()
    spec = live.get("prior_failed_engineering_attempt")
    fixed = live.get("fixed_paths")
    if not isinstance(spec, Mapping) or not isinstance(fixed, Mapping):
        fail("prior failed engineering-attempt authority is absent")
    old_root = Path(str(spec.get("run_root", "")))
    if (
        not old_root.is_absolute()
        or str(old_root) == fixed.get("fresh_formal_run_root")
        or spec.get("immutable_preservation_required") is not True
        or spec.get("relaunch_or_reuse_forbidden") is not True
    ):
        fail("prior failed run-root preservation authority differs")
    legacy_snapshot = _verify_generic_legacy_snapshot(
        spec.get("legacy_snapshot")
    )
    legacy_run = _verify_exact_legacy_tree(
        spec.get("legacy_run_tree"), "legacy run"
    )
    if legacy_run["root"] != str(old_root):
        fail("legacy run tree root differs from failed-attempt root")
    ledger, ledger_record = strict_json_file(old_root / "attempt_ledger.json")
    if (
        ledger.get("digest") != spec["attempt_ledger"].get("digest")
        or ledger.get("failure_reasons")
        != spec["attempt_ledger"].get("failure_reasons")
        or ledger_record["sha256"] != spec["attempt_ledger"].get("sha256")
    ):
        fail("prior failed attempt ledger digest/reasons differ")
    for absent_name in ("candidate.json", "completion_seal.json"):
        absent = old_root / absent_name
        if absent.exists() or absent.is_symlink():
            fail(f"prior failed attempt unexpectedly gained {absent_name}")
    value = {
        "verified": True,
        "legacy_snapshot": legacy_snapshot,
        "legacy_run": legacy_run,
        "attempt_ledger_digest": ledger["digest"],
        "candidate_absent": True,
        "completion_seal_absent": True,
        "immutable_preservation_required": True,
    }
    return {**value, "digest": authority.object_sha256(value)}


def _verify_v3r2_failed_attempt() -> Mapping[str, Any]:
    """Bind the preserved V3R2 failure receipt and independently rescan it."""

    live = authority.load_authority()
    spec = live.get("v3r2_failed_engineering_attempt")
    fixed = live.get("fixed_paths")
    if not isinstance(spec, Mapping) or not isinstance(fixed, Mapping):
        fail("V3R2 failed engineering-attempt authority is absent")
    run_root = Path(str(spec.get("run_root", "")))
    if (
        not run_root.is_absolute()
        or str(run_root) == fixed.get("fresh_formal_run_root")
        or spec.get("immutable_preservation_required") is not True
        or spec.get("relaunch_or_reuse_forbidden") is not True
    ):
        fail("V3R2 failed run-root preservation authority differs")

    receipt_spec = spec.get("failure_closure_receipt")
    if not isinstance(receipt_spec, Mapping):
        fail("V3R2 failure-closure receipt authority is absent")
    receipt_path = Path(str(receipt_spec.get("path", "")))
    receipt, receipt_record = strict_json_file(receipt_path)
    if (
        receipt_record.get("path") != str(receipt_path)
        or any(
            receipt_record.get(name) != receipt_spec.get(name)
            for name in ("sha256", "byte_count", "device", "inode", "mode")
        )
        or receipt_record.get("no_symlink_components") is not True
    ):
        fail("V3R2 failure-closure receipt file record differs")
    receipt_body = dict(receipt)
    receipt_self = receipt_body.pop("receipt_self_sha256", None)
    if (
        receipt_self != receipt_spec.get("self_sha256")
        or receipt_self != authority.object_sha256(receipt_body)
        or receipt.get("schema_version")
        != "actual-target-foundation-failed-run-closure-v3"
        or receipt.get("classification") != "ENGINEERING_FAILURE"
        or receipt.get("scientific_outcome") is not None
    ):
        fail("V3R2 failure-closure receipt self/identity differs")

    receipt_run = receipt.get("run")
    if (
        not isinstance(receipt_run, Mapping)
        or receipt_run.get("root") != str(run_root)
        or receipt_run.get("canonical_tree_digest")
        != spec.get("run_canonical_tree_digest")
        or len(receipt_run.get("rows", ())) != spec.get("run_tree_row_count")
        or receipt_run.get("all_directories_mode_0555") is not True
        or receipt_run.get("all_files_mode_0444") is not True
        or receipt_run.get("cache_empty") is not True
        or receipt_run.get("candidate_absent") is not True
        or receipt_run.get("completion_seal_absent") is not True
    ):
        fail("V3R2 failure-closure run header differs")
    run_tree = _verify_exact_legacy_tree(
        {
            "root": str(run_root),
            "rows": receipt_run["rows"],
            "canonical_tree_digest": spec["run_canonical_tree_digest"],
        },
        "V3R2 failed run",
    )
    if any((run_root / name).exists() or (run_root / name).is_symlink() for name in ("candidate.json", "completion_seal.json")):
        fail("V3R2 failed attempt unexpectedly gained candidate or seal")
    if any((run_root / "cache").iterdir()):
        fail("V3R2 failed attempt cache is no longer empty")

    snapshot_spec = spec.get("legacy_snapshot")
    if not isinstance(snapshot_spec, Mapping):
        fail("V3R2 legacy snapshot authority is absent")
    snapshot_root = Path(str(snapshot_spec.get("root", "")))
    first_snapshot_rows = _scan_legacy_tree(snapshot_root)
    second_snapshot_rows = _scan_legacy_tree(snapshot_root)
    snapshot_digest = authority.object_sha256(
        {"root": str(snapshot_root), "rows": first_snapshot_rows}
    )
    if (
        first_snapshot_rows != second_snapshot_rows
        or len(first_snapshot_rows) != snapshot_spec.get("tree_row_count")
        or snapshot_digest != snapshot_spec.get("canonical_tree_digest")
    ):
        fail("V3R2 legacy snapshot double-scan closure differs")
    snapshot = _verify_generic_legacy_snapshot(
        {**snapshot_spec, "rows": first_snapshot_rows}
    )
    receipt_snapshot = receipt.get("snapshot")
    if (
        not isinstance(receipt_snapshot, Mapping)
        or receipt_snapshot.get("root") != str(snapshot_root)
        or receipt_snapshot.get("canonical_tree_digest") != snapshot_digest
        or receipt_snapshot.get("native_verify_digest")
        != snapshot_spec.get("native_verify_digest")
    ):
        fail("V3R2 failure receipt/snapshot binding differs")

    ledger, ledger_record = strict_json_file(run_root / "attempt_ledger.json")
    log_record = stable_file_record(run_root / "formal.log")
    ledger_spec = spec.get("attempt_ledger")
    log_spec = spec.get("formal_log")
    prior_receipt = (
        ledger.get("launch_evidence", {})
        .get("prior_failed_engineering_attempt", {})
        .get("digest")
    )
    if (
        not isinstance(ledger_spec, Mapping)
        or not isinstance(log_spec, Mapping)
        or ledger_record.get("sha256") != ledger_spec.get("sha256")
        or ledger_record.get("mode") != ledger_spec.get("mode")
        or ledger.get("digest") != ledger_spec.get("digest")
        or ledger.get("failure_reasons") != ledger_spec.get("failure_reasons")
        or log_record.get("sha256") != log_spec.get("sha256")
        or log_record.get("mode") != log_spec.get("mode")
        or prior_receipt != spec.get("transitive_legacy_failure_receipt_digest")
    ):
        fail("V3R2 failed log/ledger/transitive closure differs")
    claim = receipt.get("claim_boundary")
    if (
        not isinstance(claim, Mapping)
        or claim.get("valid_completion_seal") is not False
        or claim.get("representation_admitted") is not False
        or claim.get("scientific_evidence_claimed") is not False
        or claim.get("training_performed") is not False
        or claim.get("generator_loaded") is not False
    ):
        fail("V3R2 failure receipt claim boundary differs")
    value = {
        "verified": True,
        "receipt_file": receipt_record,
        "receipt_self_sha256": receipt_self,
        "run": run_tree,
        "snapshot": snapshot,
        "attempt_ledger_digest": ledger["digest"],
        "transitive_legacy_failure_receipt_digest": prior_receipt,
        "candidate_absent": True,
        "completion_seal_absent": True,
        "immutable_preservation_required": True,
    }
    return {**value, "digest": authority.object_sha256(value)}


def _verify_v3r3_failed_attempt() -> Mapping[str, Any]:
    """Bind the preserved V3R3 NumPy failure receipt and rescan it."""

    live = authority.load_authority()
    spec = live.get("v3r3_failed_engineering_attempt")
    fixed = live.get("fixed_paths")
    if not isinstance(spec, Mapping) or not isinstance(fixed, Mapping):
        fail("V3R3 failed engineering-attempt authority is absent")
    run_root = Path(str(spec.get("run_root", "")))
    if (
        not run_root.is_absolute()
        or str(run_root) == fixed.get("fresh_formal_run_root")
        or spec.get("immutable_preservation_required") is not True
        or spec.get("relaunch_or_reuse_forbidden") is not True
    ):
        fail("V3R3 failed run-root preservation authority differs")

    receipt_spec = spec.get("failure_closure_receipt")
    if not isinstance(receipt_spec, Mapping):
        fail("V3R3 failure-closure receipt authority is absent")
    receipt_path = Path(str(receipt_spec.get("path", "")))
    receipt, receipt_record = strict_json_file(receipt_path)
    if (
        receipt_record.get("path") != str(receipt_path)
        or any(
            receipt_record.get(name) != receipt_spec.get(name)
            for name in ("sha256", "byte_count", "device", "inode", "mode")
        )
        or receipt_record.get("no_symlink_components") is not True
    ):
        fail("V3R3 failure-closure receipt file record differs")
    receipt_body = dict(receipt)
    receipt_self = receipt_body.pop("receipt_self_sha256", None)
    if (
        receipt_self != receipt_spec.get("self_sha256")
        or receipt_self != authority.object_sha256(receipt_body)
        or receipt.get("schema_version")
        != "actual-target-foundation-failed-run-closure-v3"
        or receipt.get("classification") != "ENGINEERING_FAILURE"
        or receipt.get("scientific_outcome") is not None
        or "NumPy ndarray" not in str(receipt.get("failure_boundary", ""))
        or "_cosine" not in str(receipt.get("failure_boundary", ""))
    ):
        fail("V3R3 failure-closure receipt self/identity differs")

    receipt_run = receipt.get("run")
    preservation = receipt.get("preservation_transition")
    if (
        not isinstance(receipt_run, Mapping)
        or receipt_run.get("root") != str(run_root)
        or receipt_run.get("canonical_tree_digest")
        != spec.get("run_canonical_tree_digest")
        or len(receipt_run.get("rows", ())) != spec.get("run_tree_row_count")
        or receipt_run.get("all_directories_mode_0555") is not True
        or receipt_run.get("all_files_mode_0444") is not True
        or receipt_run.get("cache_empty") is not True
        or receipt_run.get("candidate_absent") is not True
        or receipt_run.get("completion_seal_absent") is not True
        or not isinstance(preservation, Mapping)
        or preservation.get("only_changed_paths") != [".", "cache"]
        or preservation.get("before_mode") != 0o700
        or preservation.get("after_mode") != 0o555
        or preservation.get("all_members_inodes_sizes_and_file_hashes_unchanged")
        is not True
        or preservation.get("post_freeze_canonical_tree_digest")
        != spec.get("run_canonical_tree_digest")
    ):
        fail("V3R3 failure-closure run/preservation header differs")
    run_tree = _verify_exact_legacy_tree(
        {
            "root": str(run_root),
            "rows": receipt_run["rows"],
            "canonical_tree_digest": spec["run_canonical_tree_digest"],
        },
        "V3R3 failed run",
    )
    if any(
        (run_root / name).exists() or (run_root / name).is_symlink()
        for name in ("candidate.json", "completion_seal.json")
    ):
        fail("V3R3 failed attempt unexpectedly gained candidate or seal")
    if any((run_root / "cache").iterdir()):
        fail("V3R3 failed attempt cache is no longer empty")

    snapshot_spec = spec.get("legacy_snapshot")
    if not isinstance(snapshot_spec, Mapping):
        fail("V3R3 legacy snapshot authority is absent")
    snapshot_root = Path(str(snapshot_spec.get("root", "")))
    first_snapshot_rows = _scan_legacy_tree(snapshot_root)
    second_snapshot_rows = _scan_legacy_tree(snapshot_root)
    snapshot_digest = authority.object_sha256(
        {"root": str(snapshot_root), "rows": first_snapshot_rows}
    )
    if (
        first_snapshot_rows != second_snapshot_rows
        or len(first_snapshot_rows) != snapshot_spec.get("tree_row_count")
        or snapshot_digest != snapshot_spec.get("canonical_tree_digest")
    ):
        fail("V3R3 legacy snapshot double-scan closure differs")
    snapshot = _verify_generic_legacy_snapshot(
        {**snapshot_spec, "rows": first_snapshot_rows}
    )
    receipt_snapshot = receipt.get("snapshot")
    if (
        not isinstance(receipt_snapshot, Mapping)
        or receipt_snapshot.get("root") != str(snapshot_root)
        or receipt_snapshot.get("canonical_tree_digest") != snapshot_digest
        or receipt_snapshot.get("native_verify_digest")
        != snapshot_spec.get("native_verify_digest")
        or receipt_snapshot.get("manifest_file_sha256")
        != snapshot_spec.get("manifest_file_sha256")
    ):
        fail("V3R3 failure receipt/snapshot binding differs")

    ledger, ledger_record = strict_json_file(run_root / "attempt_ledger.json")
    log_record = stable_file_record(run_root / "formal.log")
    ledger_spec = spec.get("attempt_ledger")
    log_spec = spec.get("formal_log")
    prior_receipt = (
        ledger.get("launch_evidence", {})
        .get("prior_failed_engineering_attempt", {})
        .get("digest")
    )
    if (
        not isinstance(ledger_spec, Mapping)
        or not isinstance(log_spec, Mapping)
        or ledger_record.get("sha256") != ledger_spec.get("sha256")
        or ledger_record.get("mode") != ledger_spec.get("mode")
        or ledger.get("digest") != ledger_spec.get("digest")
        or ledger.get("failure_reasons") != ledger_spec.get("failure_reasons")
        or ledger.get("engineering_failure") is not True
        or ledger.get("valid_completion_seal") is not False
        or log_record.get("sha256") != log_spec.get("sha256")
        or log_record.get("mode") != log_spec.get("mode")
        or prior_receipt
        != spec.get("transitive_prior_failed_attempt_receipt_digest")
    ):
        fail("V3R3 failed log/ledger/transitive closure differs")
    claim = receipt.get("claim_boundary")
    if (
        not isinstance(claim, Mapping)
        or claim.get("valid_completion_seal") is not False
        or claim.get("representation_admitted") is not False
        or claim.get("scientific_evidence_claimed") is not False
        or claim.get("locked_validation_claimed") is not False
        or claim.get("case_aggregate_persisted") is not False
        or claim.get("training_performed") is not False
        or claim.get("optimizer_created") is not False
        or claim.get("parameter_updates") != 0
        or claim.get("generator_loaded") is not False
        or claim.get("generator_forward_calls") != 0
    ):
        fail("V3R3 failure receipt claim boundary differs")
    value = {
        "verified": True,
        "receipt_file": receipt_record,
        "receipt_self_sha256": receipt_self,
        "run": run_tree,
        "snapshot": snapshot,
        "attempt_ledger_digest": ledger["digest"],
        "transitive_prior_failed_attempt_receipt_digest": prior_receipt,
        "candidate_absent": True,
        "completion_seal_absent": True,
        "immutable_preservation_required": True,
    }
    return {**value, "digest": authority.object_sha256(value)}


def _verify_v3r3_sam_layout_source_evidence() -> Mapping[str, Any]:
    """Bind the repaired mask ABI to exact pinned SAM2 source and line bytes."""

    repair = authority.load_authority()["v3r3_engineering_repair_contract"]
    spec = repair.get("sam_pinned_binary_mask_source_evidence")
    if (
        not isinstance(spec, Mapping)
        or set(spec) != {"claim_boundary", "sources"}
        or spec.get("claim_boundary")
        != (
            "the full-storage transpose layout is derived from pinned SAM2 "
            "source bytes, not inferred from the V3R2 compound failure log"
        )
        or not isinstance(spec.get("sources"), list)
        or len(spec["sources"]) != 2
    ):
        fail("V3R3 SAM layout source evidence authority differs")
    expected_roles = {
        "uncompressed_rle_to_mask",
        "automatic_binary_mask_return",
    }
    rows = []
    seen_roles = set()
    for source in spec["sources"]:
        if (
            not isinstance(source, Mapping)
            or set(source)
            != {
                "role",
                "path",
                "sha256",
                "line_start",
                "line_end",
                "line_span_sha256",
            }
            or source.get("role") not in expected_roles
            or source["role"] in seen_roles
            or not isinstance(source.get("line_start"), int)
            or isinstance(source.get("line_start"), bool)
            or not isinstance(source.get("line_end"), int)
            or isinstance(source.get("line_end"), bool)
            or source["line_start"] <= 0
            or source["line_end"] < source["line_start"]
        ):
            fail("V3R3 SAM layout source row differs")
        record = dict(stable_file_record(Path(source["path"]), payload=True))
        payload = record.pop("payload")
        lines = payload.splitlines(keepends=True)
        if (
            record["sha256"] != source["sha256"]
            or len(lines) < source["line_end"]
        ):
            fail("V3R3 pinned SAM source file differs")
        span = b"".join(
            lines[source["line_start"] - 1 : source["line_end"]]
        )
        span_sha256 = hashlib.sha256(span).hexdigest()
        if span_sha256 != source["line_span_sha256"]:
            fail("V3R3 pinned SAM semantic line span differs")
        seen_roles.add(source["role"])
        rows.append(
            {
                "role": source["role"],
                "file": record,
                "line_start": source["line_start"],
                "line_end": source["line_end"],
                "line_span_sha256": span_sha256,
            }
        )
    if seen_roles != expected_roles:
        fail("V3R3 pinned SAM source roles differ")
    value = {"verified": True, "sources": rows}
    return {**value, "digest": authority.object_sha256(value)}


def _verify_prior_failed_attempt() -> Mapping[str, Any]:
    # Keep the historical closure receipt/digest schema stable while making
    # every preflight and zero-GPU sealing path also validate the new repair's
    # exact upstream source semantics.
    _verify_v3r3_sam_layout_source_evidence()
    v3r1 = _verify_v3r1_failed_attempt()
    v3r2 = _verify_v3r2_failed_attempt()
    v3r3 = _verify_v3r3_failed_attempt()
    value = {
        "verified": True,
        "v3r1": v3r1,
        "v3r2": v3r2,
        "v3r3": v3r3,
    }
    return {**value, "digest": authority.object_sha256(value)}


def _static_hydra_config_closure() -> Mapping[str, Any]:
    """Validate frozen Hydra authority without importing SAM/Hydra/torch."""

    spec = authority.load_authority()["sam_hydra_authority"]
    config_path = Path(spec["runtime_config_path"])
    config_record = stable_file_record(config_path)
    if config_record["sha256"] != spec["runtime_config_sha256"]:
        fail("static SAM Hydra YAML closure differs")
    value = {
        "verified": True,
        "config_name": spec["config_name"],
        "runtime_config_path": spec["runtime_config_path"],
        "runtime_config_sha256": spec["runtime_config_sha256"],
        "exact_overrides": list(spec["exact_overrides"]),
        "apply_postprocessing": False,
        "resolved_canonical_json_bytes": spec["resolved_canonical_json_bytes"],
        "resolved_canonical_sha256": spec["resolved_canonical_sha256"],
        "resolved_model_target": spec["resolved_model_target"],
    }
    return {**value, "digest": authority.object_sha256(value)}


def _controller_source_tree_rows(root: Path, suffix: str) -> list[Mapping[str, Any]]:
    try:
        snapshot_v3._plain_directory(root)
    except Exception as error:
        raise ControllerV3Error(
            f"foundation source root is not lexical plain: {root}"
        ) from error
    rows = []
    for current_text, directories, files in os.walk(root, followlinks=False):
        current = Path(current_text)
        current_row = current.lstat()
        if stat.S_ISLNK(current_row.st_mode) or not stat.S_ISDIR(current_row.st_mode):
            fail(f"foundation source directory differs: {current}")
        for name in directories:
            child = current / name
            child_row = child.lstat()
            if stat.S_ISLNK(child_row.st_mode) or not stat.S_ISDIR(child_row.st_mode):
                fail(f"foundation source tree contains symlink/non-directory: {child}")
        for name in files:
            child = current / name
            child_row = child.lstat()
            if stat.S_ISLNK(child_row.st_mode) or not stat.S_ISREG(child_row.st_mode):
                fail(f"foundation source tree contains symlink/non-file: {child}")
            if child.suffix != suffix:
                continue
            record = stable_file_record(child)
            rows.append(
                {
                    "relative_path": str(child.relative_to(root)),
                    "byte_count": record["byte_count"],
                    "sha256": record["sha256"],
                    "mode": record["mode"],
                }
            )
    return sorted(rows, key=lambda row: row["relative_path"])


def _recompute_foundation_source_trees() -> Mapping[str, Any]:
    specs = authority.load_authority()["foundation_source_tree_authority"]
    if not isinstance(specs, list) or len(specs) != 3:
        fail("foundation source tree authority differs")
    trees = []
    seen_roles = set()
    for spec in specs:
        if not isinstance(spec, Mapping) or set(spec) != {
            "role",
            "root",
            "suffix",
            "file_count",
            "manifest_sha256",
        }:
            fail("foundation source tree authority row differs")
        if spec["role"] in seen_roles or spec["suffix"] != ".py":
            fail("foundation source tree role/suffix differs")
        seen_roles.add(spec["role"])
        root = Path(spec["root"])
        rows = _controller_source_tree_rows(root, spec["suffix"])
        if rows != _controller_source_tree_rows(root, spec["suffix"]):
            fail(f"foundation source tree changed during controller scan: {spec['role']}")
        digest = authority.object_sha256(rows)
        if len(rows) != spec["file_count"] or digest != spec["manifest_sha256"]:
            fail(f"foundation source tree manifest differs: {spec['role']}")
        trees.append(
            {
                "role": spec["role"],
                "root": str(root),
                "suffix": spec["suffix"],
                "file_count": len(rows),
                "manifest_sha256": digest,
                "files": rows,
                "double_scan_stable": True,
                "no_symlinks": True,
            }
        )
    value = {"verified": True, "trees": trees}
    return {**value, "digest": authority.object_sha256(value)}


def _static_module_source(module_name: str) -> tuple[Path, str]:
    suffix = "_python_tree"
    roots = {}
    for row in authority.load_authority()["foundation_source_tree_authority"]:
        role = row["role"]
        roots[role[: -len(suffix)] if role.endswith(suffix) else role] = Path(
            row["root"]
        )
    prefix, *parts = module_name.split(".")
    if prefix not in roots or not parts:
        fail(f"foundation module is outside static source authority: {module_name}")
    root = roots[prefix]
    relative_parts = parts if root.name == prefix else [prefix, *parts]
    source = root.joinpath(*relative_parts).with_suffix(".py")
    record = stable_file_record(source)
    return source, record["sha256"]


def _validate_model_binding(value: Any, hydra_config_digest: str) -> None:
    _verify_digest(value, "model binding")
    expected_fields = {
        "verified",
        "classes",
        "preprocessor_and_build_sources",
        "non_tensor_configs",
        "cotracker_runtime_config",
        "hydra_config_digest",
        "digest",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != expected_fields
        or value.get("verified") is not True
    ):
        fail("model binding exact schema/verified flag differs")
    availability = authority.load_availability()
    expected_classes = {
        (row["module"], row["class"]): row["source_sha256"]
        for row in availability["runtime_class_authority"]
    }
    observed_classes: dict[tuple[str, str], str] = {}
    class_rows = value.get("classes")
    if not isinstance(class_rows, list) or len(class_rows) != len(expected_classes):
        fail("foundation class binding cardinality differs")
    for row in class_rows:
        if not isinstance(row, Mapping) or set(row) != {
            "module",
            "class",
            "source_path",
            "source_sha256",
        }:
            fail("foundation class binding row schema differs")
        key = (row["module"], row["class"])
        if key in observed_classes or key not in expected_classes:
            fail("foundation class binding identity differs/duplicates")
        live_path, live_sha = _static_module_source(key[0])
        if (
            row["source_path"] != str(live_path)
            or row["source_sha256"] != live_sha
            or live_sha != expected_classes[key]
        ):
            fail("foundation class source closure differs")
        observed_classes[key] = live_sha
    if set(observed_classes) != set(expected_classes):
        fail("foundation class binding matrix differs")

    non_tensor = authority.load_authority()[
        "preprocessor_and_nontensor_config_authority"
    ]
    expected_source_roles = {
        "sam_build_function": (
            non_tensor["sam_build_function"]["module"],
            non_tensor["sam_build_function"]["name"],
        ),
        "dinov2_processor": (
            non_tensor["dinov2_processor"]["module"],
            non_tensor["dinov2_processor"]["class"],
        ),
        "vjepa2_processor": (
            non_tensor["vjepa2_processor"]["module"],
            non_tensor["vjepa2_processor"]["class"],
        ),
    }
    source_rows = value.get("preprocessor_and_build_sources")
    if not isinstance(source_rows, list) or len(source_rows) != len(
        expected_source_roles
    ):
        fail("preprocessor/build source binding cardinality differs")
    seen_roles = set()
    for row in source_rows:
        if not isinstance(row, Mapping) or set(row) != {
            "role",
            "module",
            "name",
            "source_path",
            "source_sha256",
        }:
            fail("preprocessor/build source binding row schema differs")
        role = row["role"]
        if role in seen_roles or role not in expected_source_roles:
            fail("preprocessor/build source role differs/duplicates")
        module_name, object_name = expected_source_roles[role]
        spec = non_tensor[role]
        live_path = Path(spec["source_path"])
        live_sha = stable_file_record(live_path)["sha256"]
        if (
            row["module"] != module_name
            or row["name"] != object_name
            or row["source_path"] != str(live_path)
            or row["source_sha256"] != live_sha
            or live_sha != spec["source_sha256"]
            or str(live_path) != spec["source_path"]
        ):
            fail("preprocessor/build live source closure differs")
        seen_roles.add(role)
    if seen_roles != set(expected_source_roles):
        fail("preprocessor/build source matrix differs")

    expected_config_roles = {
        "dinov2_processor",
        "vjepa2_processor",
        "dinov2_model_config",
        "vjepa2_model_config",
    }
    config_rows = value.get("non_tensor_configs")
    if not isinstance(config_rows, list) or len(config_rows) != len(
        expected_config_roles
    ):
        fail("non-tensor config binding cardinality differs")
    seen_config_roles = set()
    for row in config_rows:
        if not isinstance(row, Mapping) or set(row) != {
            "role",
            "canonical_config_bytes",
            "canonical_config_sha256",
        }:
            fail("non-tensor config binding row schema differs")
        role = row["role"]
        if role in seen_config_roles or role not in expected_config_roles:
            fail("non-tensor config role differs/duplicates")
        spec = non_tensor[role]
        expected_row = {
            "role": role,
            "canonical_config_bytes": spec["canonical_config_bytes"],
            "canonical_config_sha256": spec["canonical_config_sha256"],
        }
        if dict(row) != expected_row:
            fail("non-tensor config authority closure differs")
        seen_config_roles.add(role)
    if seen_config_roles != expected_config_roles:
        fail("non-tensor config matrix differs")
    if value.get("cotracker_runtime_config") != non_tensor["cotracker_runtime_config"]:
        fail("CoTracker runtime config binding differs")
    if value.get("hydra_config_digest") != hydra_config_digest:
        fail("model binding Hydra digest differs")


def _validate_model_closure(value: Any, hydra_config_digest: str) -> None:
    _verify_digest(value, "model closure")
    if value.get("mode") != "real_frozen_full_tensor_closure" or value.get("verified") is not True:
        fail("model closure is not real/verified")
    before, after = value.get("before"), value.get("after")
    _verify_digest(before, "model before closure")
    _verify_digest(after, "model after closure")
    if before != after or value.get("exact_before_after_equality") is not True:
        fail("model before/after closure differs")
    rows = before.get("tensors")
    fields = {"model", "kind", "name", "shape", "dtype", "device", "data_ptr", "value_sha256", "requires_grad"}
    if not isinstance(rows, list) or before.get("tensor_count") != len(rows) or not rows:
        fail("model tensor closure is empty/incomplete")
    identities = set()
    observed_models = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != fields:
            fail("model tensor closure row fields differ")
        identity = (row["model"], row["kind"], row["name"])
        if identity in identities:
            fail("model tensor closure identity is duplicated")
        identities.add(identity)
        if row["model"] not in {"sam2", "cotracker", "dinov2", "vjepa2"} or row["kind"] not in {"parameter", "buffer"}:
            fail("model tensor identity differs")
        observed_models.add(row["model"])
        shape = row["shape"]
        if not isinstance(shape, list) or any(
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension < 0
            for dimension in shape
        ):
            fail("model tensor shape closure differs")
        numel = math.prod(shape)
        pointer = row["data_ptr"]
        if (
            row["device"] != "cuda:0"
            or not isinstance(pointer, int)
            or isinstance(pointer, bool)
            or pointer < 0
            or (numel > 0 and pointer == 0)
        ):
            fail("model tensor device/data_ptr differs")
        if row["requires_grad"] is not False:
            fail("model tensor frozen/shape closure differs")
        digest = row["value_sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            fail("model tensor value SHA differs")
    if observed_models != {"sam2", "cotracker", "dinov2", "vjepa2"}:
        fail("full tensor closure does not contain every foundation model")
    _validate_model_binding(value.get("binding"), hydra_config_digest)


def _validate_raw_inventory(
    value: Any,
    forward_closure: Mapping[str, Any],
    model_closure: Mapping[str, Any],
) -> None:
    _verify_digest(value, "raw inventory")
    exact_fields = {
        "schema_version",
        "required_categories",
        "opportunity_by_category",
        "produced_by_category",
        "registered_by_category",
        "zeroized_by_category",
        "failure_attempts_by_category",
        "observed_counts",
        "opportunity_total",
        "produced_total",
        "registered_total",
        "zeroized_total",
        "outstanding_count",
        "missing_required_categories",
        "zero_produced_categories",
        "zero_produced_categories_are_valid_abstention",
        "observed_count_keys",
        "model_output_unique_storage_multipliers",
        "model_output_unique_storage_evidence_digest",
        "production_binding_rule",
        "in_scope_storage_boundary",
        "excluded_ephemeral_workspace_boundary",
        "recursive_best_effort_scrub",
        "verified",
        "digest",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != exact_fields
        or value.get("schema_version") != "actual-target-raw-inventory-v3"
        or value.get("recursive_best_effort_scrub") is not True
    ):
        fail("raw inventory exact schema/version differs")
    authority_value = authority.load_authority()
    required = authority_value["raw_inventory_required_categories"]
    scope = authority_value["raw_ownership_contract"]
    output_evidence = scope.get("model_output_unique_storage_evidence")
    availability_rows = {
        (row["module"], row["class"]): row["source_sha256"]
        for row in authority.load_availability()["runtime_class_authority"]
    }
    if (
        scope.get("model_output_unique_storage_multipliers")
        != runtime.MODEL_OUTPUT_UNIQUE_STORAGE_MULTIPLIERS
        or not isinstance(output_evidence, Mapping)
        or set(output_evidence) != {"dinov2", "vjepa2"}
        or any(
            row.get("unique_storage_multiplier")
            != runtime.MODEL_OUTPUT_UNIQUE_STORAGE_MULTIPLIERS[name]
            or availability_rows.get((row.get("module"), row.get("class")))
            != row.get("source_sha256")
            for name, row in output_evidence.items()
            if isinstance(row, Mapping)
        )
        or any(not isinstance(row, Mapping) for row in output_evidence.values())
    ):
        fail("model-output unique-storage source authority differs")
    if value.get("required_categories") != required or value.get("verified") is not True:
        fail("raw inventory required category/verified closure differs")
    registered = value.get("registered_by_category")
    opportunities = value.get("opportunity_by_category")
    produced = value.get("produced_by_category")
    zeroized = value.get("zeroized_by_category")
    failures = value.get("failure_attempts_by_category")
    if not all(
        isinstance(row, Mapping) and set(row) == set(required)
        for row in (opportunities, produced, registered, zeroized, failures)
    ):
        fail("raw inventory per-category matrix differs")
    if any(
        not isinstance(row[name], int)
        or isinstance(row[name], bool)
        or row[name] < 0
        for row in (opportunities, produced, registered, zeroized, failures)
        for name in required
    ):
        fail("raw inventory per-category counters are not nonnegative integers")
    observed = value.get("observed_counts")
    if (
        not isinstance(observed, Mapping)
        or set(observed) != set(runtime.RAW_OBSERVED_COUNT_KEYS)
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in observed.values()
        )
    ):
        fail("raw inventory upstream observed-count matrix differs")
    logical = forward_closure.get("logical_counts")
    before = model_closure.get("before")
    tensor_count = before.get("tensor_count") if isinstance(before, Mapping) else None
    if (
        logical != runtime.EXPECTED_LOGICAL_COUNTS
        or not isinstance(tensor_count, int)
        or isinstance(tensor_count, bool)
        or tensor_count < 0
    ):
        fail("raw production binding inputs differ")
    case_count = len(authority.load_preregistration()["pairs"])
    expected_produced = {
        "compressed_video_hash_buffer": logical["media_decode"],
        "decoded_bgr_frame": observed["decoded_bgr_frames"],
        "decoded_rgb_frame": observed["decoded_rgb_frames"],
        "sam_ann_mask_pre_filter": observed["sam_ann_records_before_filter"],
        "sam_mask_c_contiguous_copy": observed[
            "sam_ann_records_before_filter"
        ],
        "sam_mask_coordinate_indices": observed["sam_mask_coordinate_calls"],
        "dino_processor_input": 2
        * observed["dino_processor_tensor_items"],
        "dino_tokens": observed["dino_model_output_unique_storages"]
        + logical["dinov2"],
        "dino_mask_input": observed["dino_filtered_ann_records"],
        "dino_mask_resized": observed["dino_filtered_ann_records"],
        "dino_mask_cropped": observed["dino_filtered_ann_records"],
        "dino_patch_weights": observed["dino_filtered_ann_records"],
        "dino_patch_support": observed["dino_filtered_ann_records"],
        "dino_pooled_descriptor": 3
        * observed["dino_positive_support_records"],
        "dino_pooled_descriptor_cpu": 2
        * observed["dino_positive_support_records"],
        "node_signature": logical["sam2"] // runtime.PHASES
        + 3 * case_count,
        "cotracker_video": 3 * logical["cotracker"],
        "cotracker_tracks": logical["cotracker"],
        "cotracker_visibility": logical["cotracker"],
        "cotracker_coordinates_cpu": 2 * logical["cotracker"],
        "cotracker_visibility_cpu": 2 * logical["cotracker"],
        "cotracker_group_coordinates": 2
        * observed["cotracker_membership_rows"],
        "cotracker_group_visibility": 2
        * observed["cotracker_membership_rows"],
        "track_signature": logical["cotracker"] + case_count,
        "edge_signature": logical["cotracker"],
        "drop_edge_signature": logical["cotracker"],
        "vjepa_processor_input": 2
        * observed["vjepa_processor_tensor_items"],
        "vjepa_hidden": observed["vjepa_model_output_unique_storages"]
        + 2 * logical["vjepa2"],
        "vjepa_phase_signature": 2 * logical["vjepa2"],
        "model_hash_copy": observed["model_tensor_hash_requests"],
    }
    if (
        opportunities != produced
        or produced != expected_produced
        or produced != registered
        or registered != zeroized
        or any(failures[name] != 0 for name in required)
    ):
        fail("raw inventory opportunity/produced/zeroized closure differs")
    if (
        observed["compressed_video_hash_requests"] != logical["media_decode"]
        or produced["sam_ann_mask_pre_filter"]
        != observed["sam_ann_records_before_filter"]
        or produced["sam_mask_c_contiguous_copy"]
        != observed["sam_ann_records_before_filter"]
        or observed["dino_processor_tensor_items"] != logical["dinov2"]
        or observed["dino_model_output_unique_storages"]
        != logical["dinov2"]
        or observed["dino_filtered_ann_records"]
        > observed["sam_ann_records_before_filter"]
        or observed["dino_filtered_ann_records"]
        > 12 * logical["dinov2"]
        or observed["dino_positive_support_records"]
        > observed["dino_filtered_ann_records"]
        or observed["sam_mask_coordinate_calls"]
        != observed["dino_positive_support_records"]
        or observed["vjepa_processor_tensor_items"] != logical["vjepa2"]
        or observed["vjepa_model_output_unique_storages"]
        != 4 * logical["vjepa2"]
        or observed["cotracker_membership_rows"]
        > 96 * logical["cotracker"]
        or observed["model_tensor_hash_requests"] != 2 * tensor_count
    ):
        fail("raw inventory upstream/fixed-multiplier production binding differs")
    if (
        value.get("opportunity_total") != sum(opportunities.values())
        or value.get("produced_total") != sum(produced.values())
        or value.get("registered_total") != sum(registered.values())
        or value.get("zeroized_total") != sum(zeroized.values())
    ):
        fail("raw inventory aggregate counters differ")
    if value.get("outstanding_count") != 0 or value.get("missing_required_categories") != []:
        fail("raw inventory retains outstanding/missing payload")
    if (
        value.get("zero_produced_categories")
        != sorted(name for name in required if produced[name] == 0)
        or value.get("zero_produced_categories_are_valid_abstention") is not True
        or value.get("observed_count_keys") != list(runtime.RAW_OBSERVED_COUNT_KEYS)
        or value.get("model_output_unique_storage_multipliers")
        != runtime.MODEL_OUTPUT_UNIQUE_STORAGE_MULTIPLIERS
        or value.get("model_output_unique_storage_evidence_digest")
        != authority.object_sha256(output_evidence)
        or scope.get("observed_count_keys")
        != list(runtime.RAW_OBSERVED_COUNT_KEYS)
        or value.get("production_binding_rule")
        != scope.get("production_binding_rule")
        or value.get("in_scope_storage_boundary")
        != scope["included_storage_scope"]
        or value.get("excluded_ephemeral_workspace_boundary")
        != scope["excluded_ephemeral_workspace_scope"]
    ):
        fail("raw inventory opportunity/scope boundary differs")
    if observed["sam_ann_records_before_filter"] != produced["sam_ann_mask_pre_filter"]:
        fail("SAM pre-filter annotation ownership closure differs")
    expected_decoded_frames = sum(
        int(row["frame_count"]) for row in authority.load_decode_receipt()["rows"]
    )
    if (
        registered["compressed_video_hash_buffer"]
        != runtime.EXPECTED_LOGICAL_COUNTS["media_decode"]
        or registered["decoded_bgr_frame"] != expected_decoded_frames
        or registered["decoded_rgb_frame"] != expected_decoded_frames
        or observed.get("decoded_bgr_frames") != expected_decoded_frames
        or observed.get("decoded_rgb_frames") != expected_decoded_frames
    ):
        fail("compressed/BGR/RGB media raw ownership closure differs")


def _validate_media(value: Any) -> None:
    _verify_digest(value, "decoded media closure")
    expected = {
        (row["r1b_ordinal"], row["role"]): (
            row["compressed_sha256"],
            row["frame_count"],
            [720, 1280, 3],
            "uint8",
            row["decoded_rgb_sha256"],
        )
        for row in authority.load_decode_receipt()["rows"]
    }
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != 8:
        fail("decoded media closure is not exactly eight rows")
    keys = [
        (row.get("r1b_ordinal"), row.get("role"))
        for row in rows
        if isinstance(row, Mapping)
    ]
    if len(keys) != 8 or len(set(keys)) != 8:
        fail("decoded media closure keys are missing/duplicated")
    observed = {
        (row.get("r1b_ordinal"), row.get("role")): (
            row.get("compressed_sha256"),
            row.get("frame_count"),
            row.get("shape_hwc"),
            row.get("dtype"),
            row.get("decoded_rgb_sha256"),
        )
        for row in rows
        if isinstance(row, Mapping)
    }
    if value.get("verified") is not True or observed != expected:
        fail("decoded media rows differ")
    if value.get("decode_receipt_file_sha256") != authority.file_sha256(authority.base.DECODE_RECEIPT_PATH):
        fail("decoded media receipt file SHA differs")
    if value.get("decode_receipt_self_sha256") != authority.load_decode_receipt()["decode_receipt_self_sha256"]:
        fail("decoded media receipt self SHA differs")


def _validate_candidate(
    candidate_path: Path,
    cache_dir: Path,
    expected_contract_digest: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Sequence[Mapping[str, Any]]]:
    candidate, candidate_file = strict_json_file(candidate_path)
    if candidate_file["mode"] != 0o444:
        fail("candidate is not frozen mode 0444")
    expected_fields = {
        "schema_version", "experiment_id", "scope", "mechanical_case_evidence", "cases", "aggregate",
        "forward_closure", "raw_ownership", "model_closure", "device_closure", "hydra_config_closure",
        "asset_closure", "decoded_media_closure", "runtime_source_closure", "training_performed",
        "optimizer_created", "parameter_updates", "generator_loaded", "generator_forward_calls",
        "raw_teacher_payload_persisted", "representation_admission_hard_false", "scientific_evidence_claimed",
        "completion_authority", "launch_contract_digest", "digest",
    }
    if set(candidate) != expected_fields:
        fail("candidate exact top-level schema differs")
    _verify_digest(candidate, "candidate")
    contract = runtime.launch_contract()
    if expected_contract_digest != contract["digest"] or candidate["launch_contract_digest"] != contract["digest"]:
        fail("reviewed/live/candidate launch contract differs")
    if candidate["runtime_source_closure"] != contract["source_closure"]:
        fail("candidate runtime source closure differs")
    if any(
        (
            candidate.get("training_performed") is not False,
            candidate.get("optimizer_created") is not False,
            candidate.get("parameter_updates") != 0,
            candidate.get("generator_loaded") is not False,
            candidate.get("generator_forward_calls") != 0,
            candidate.get("raw_teacher_payload_persisted") is not False,
            candidate.get("representation_admission_hard_false") is not True,
            candidate.get("scientific_evidence_claimed") is not False,
        )
    ):
        fail("candidate hard claim boundary differs")

    mechanical = candidate.get("mechanical_case_evidence")
    rows = candidate.get("cases")
    if not isinstance(mechanical, list) or not isinstance(rows, list) or len(mechanical) != 4 or len(rows) != 4:
        fail("candidate four-case evidence matrix differs")
    evidences = []
    for item in mechanical:
        _verify_digest(item, "mechanical case evidence")
        body = dict(item)
        body.pop("digest")
        evidences.append(authority.CaseEvidenceV3.from_mapping(body))
    prereg = authority.load_preregistration()
    rebuilt_rows = [authority.evaluate_case(evidence, prereg) for evidence in evidences]
    if rebuilt_rows != rows:
        fail("candidate evaluated cases are not exact mechanical recomputations")
    rebuilt_aggregate = authority.aggregate_canary(rebuilt_rows, evidences, prereg)
    if rebuilt_aggregate != candidate.get("aggregate"):
        fail("candidate aggregate is not exact four-case recomputation")

    if not cache_dir.is_absolute() or cache_dir.is_symlink() or not cache_dir.is_dir() or stat.S_IMODE(cache_dir.stat().st_mode) != 0o555:
        fail("candidate cache path differs")
    expected_cache = {f"{evidence.pair_id}.json" for evidence in evidences}
    cache_children = list(cache_dir.iterdir())
    observed_cache = {path.name for path in cache_children}
    if (
        observed_cache != expected_cache
        or any(path.is_symlink() or not path.is_file() for path in cache_children)
    ):
        fail("complete case cache closure differs")
    cache_records = []
    for evidence, rebuilt in zip(evidences, rebuilt_rows):
        cache, cache_record = strict_json_file(cache_dir / f"{evidence.pair_id}.json")
        if cache_record["mode"] != 0o444:
            fail("case cache is not frozen mode 0444")
        cache_records.append(cache_record)
        _verify_digest(cache, "case cache")
        _verify_digest(cache.get("case_evidence"), "cached mechanical evidence")
        if cache.get("case_evidence") != next(row for row in mechanical if row["pair_id"] == evidence.pair_id) or cache.get("evaluated_case") != rebuilt:
            fail("case cache does not exactly bind candidate mechanics/evaluation")

    _verify_digest(candidate.get("forward_closure"), "forward closure")
    forward = candidate["forward_closure"]
    if (
        forward.get("verified") is not True
        or forward.get("logical_counts") != runtime.EXPECTED_LOGICAL_COUNTS
        or forward.get("actual_forward_hook_counts") != runtime.EXPECTED_HOOK_COUNTS
        or forward.get("expected_logical_counts") != runtime.EXPECTED_LOGICAL_COUNTS
        or forward.get("expected_actual_forward_hook_counts") != runtime.EXPECTED_HOOK_COUNTS
    ):
        fail("logical/actual foundation forward closure differs")
    hydra_candidate = candidate.get("hydra_config_closure")
    _verify_digest(hydra_candidate, "candidate Hydra config closure")
    _validate_model_closure(candidate.get("model_closure"), hydra_candidate["digest"])
    _validate_raw_inventory(
        candidate.get("raw_ownership"),
        forward,
        candidate.get("model_closure"),
    )

    _verify_digest(candidate.get("device_closure"), "device closure")
    device = candidate["device_closure"]
    if any((device.get("mode") != "real_one_device", device.get("verified") is not True, device.get("type") != "cuda", device.get("index") != 0, device.get("visible_device_count") != 1, not isinstance(device.get("rocr_visible_devices"), str), not device.get("rocr_visible_devices"), "," in device.get("rocr_visible_devices", ""))):
        fail("device closure differs")
    live_hydra = _static_hydra_config_closure()
    if live_hydra.get("verified") is not True or candidate.get("hydra_config_closure") != live_hydra:
        fail("static/runtime Hydra config closure differs")
    live_assets = authority.verify_remote_assets()
    if candidate.get("asset_closure") != live_assets:
        fail("exact live asset closure differs")
    independent_source_trees = _recompute_foundation_source_trees()
    if candidate["asset_closure"].get("foundation_source_trees") != independent_source_trees:
        fail("independent foundation source tree closure differs")
    _validate_media(candidate.get("decoded_media_closure"))
    completion = candidate.get("completion_authority")
    if completion != {
        "candidate_file_presence_is_completion_authority": False,
        "external_controller_required": True,
        "external_controller_valid_outcomes": ["PASS", "REJECTED"],
        "external_completion_seal_written_by_probe": False,
    }:
        fail("candidate completion authority differs")
    return candidate, candidate_file, cache_records


def _validate_run_root_preseal(
    run_root: Path, exact_paths: Mapping[str, Path]
) -> None:
    try:
        snapshot_v3._plain_directory(run_root)
    except Exception as error:
        raise ControllerV3Error("formal run root is not a lexical plain directory") from error
    if exact_paths["seal"].exists() or exact_paths["seal"].is_symlink():
        fail("completion seal must be absent before controller sealing")
    if exact_paths["attempt_ledger"].exists() or exact_paths["attempt_ledger"].is_symlink():
        fail("attempt ledger must be absent before controller sealing")
    allowed = {path.name for path in exact_paths.values()}
    children = list(run_root.iterdir())
    for child in children:
        if child.name not in allowed or child.is_symlink():
            fail(f"formal run root contains an unauthorized sidecar: {child.name}")
        if child in {
            exact_paths["cache"],
            exact_paths["miopen_user"],
            exact_paths["miopen_custom"],
        }:
            if not child.is_dir():
                fail("formal cache/MIOpen path is not a directory")
        elif not child.is_file():
            fail(f"formal run artifact is not a regular file: {child.name}")


def _expected_controller_command(
    *,
    exact_paths: Mapping[str, Path],
    expected_contract_digest: str,
    srun_exit_code: int,
    tee_exit_code: int,
    expected_job_id: str,
    snapshot_root: Path,
) -> list[str]:
    controller_wrapper = str(
        snapshot_root
        / "scripts"
        / "auh_actual_target_foundation_canary_controller_wrapper_v3.sh"
    )
    return [
        controller_wrapper,
        str(snapshot_root),
        "seal",
        "--candidate",
        str(exact_paths["candidate"]),
        "--cache-dir",
        str(exact_paths["cache"]),
        "--seal",
        str(exact_paths["seal"]),
        "--attempt-ledger",
        str(exact_paths["attempt_ledger"]),
        "--expected-contract-digest",
        expected_contract_digest,
        "--srun-exit-code",
        str(srun_exit_code),
        "--tee-exit-code",
        str(tee_exit_code),
        "--expected-job-id",
        expected_job_id,
        "--step-meta",
        str(exact_paths["step_meta"]),
        "--formal-log",
        str(exact_paths["log"]),
        "--srun-argv",
        str(exact_paths["srun_argv"]),
        "--rank-argv",
        str(exact_paths["rank_argv"]),
        "--controller-argv",
        str(exact_paths["controller_argv"]),
        "--controller-step-meta",
        str(exact_paths["controller_step_meta"]),
        "--miopen-scratch-closure",
        str(exact_paths["miopen_scratch_closure"]),
        "--snapshot-root",
        str(snapshot_root),
    ]


def _expected_controller_srun_argv(
    *,
    command: Sequence[str],
    expected_job_id: str,
) -> tuple[list[str], list[str]]:
    mask_export = "--export=ALL"
    formal = [
        "srun",
        "--nodes=1",
        "--ntasks=1",
        "--gpus=0",
        "--cpus-per-task=1",
        "--mem=1G",
        mask_export,
        *command,
    ]
    existing = [
        "srun",
        "--jobid",
        expected_job_id,
        "--exclusive",
        "--exact",
        "--immediate=60",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=1",
        "--gpus=0",
        "--mem=1G",
        mask_export,
        *command,
    ]
    return formal, existing


def seal_outcome(
    *,
    candidate_path: Path,
    cache_dir: Path,
    seal_path: Path,
    expected_contract_digest: str,
    srun_exit_code: int,
    tee_exit_code: int,
    expected_job_id: str,
    step_meta_path: Path,
    formal_log_path: Path,
    srun_argv_path: Path,
    rank_argv_path: Path,
    controller_argv_path: Path,
    controller_step_meta_path: Path,
    miopen_scratch_closure_path: Path,
    snapshot_root: Path,
    attempt_ledger_path: Path,
) -> Mapping[str, Any]:
    fixed = authority.load_authority()["fixed_paths"]
    run_root = Path(fixed["fresh_formal_run_root"])
    exact_paths = {
        "candidate": run_root / fixed["candidate_filename"],
        "cache": run_root / fixed["cache_dirname"],
        "seal": run_root / fixed["seal_filename"],
        "attempt_ledger": run_root / fixed["attempt_ledger_filename"],
        "log": run_root / fixed["formal_log_filename"],
        "srun_argv": run_root / fixed["srun_argv_filename"],
        "rank_argv": run_root / fixed["rank_argv_filename"],
        "step_meta": run_root / fixed["step_meta_filename"],
        "controller_argv": run_root / fixed["controller_argv_filename"],
        "controller_step_meta": run_root
        / fixed["controller_step_meta_filename"],
        "miopen_user": run_root / fixed["miopen_user_dirname"],
        "miopen_custom": run_root / fixed["miopen_custom_cache_dirname"],
        "miopen_scratch_closure": run_root
        / fixed["miopen_scratch_closure_filename"],
    }
    observed_paths = {
        "candidate": candidate_path,
        "cache": cache_dir,
        "seal": seal_path,
        "attempt_ledger": attempt_ledger_path,
        "log": formal_log_path,
        "srun_argv": srun_argv_path,
        "rank_argv": rank_argv_path,
        "step_meta": step_meta_path,
        "controller_argv": controller_argv_path,
        "controller_step_meta": controller_step_meta_path,
        "miopen_user": run_root / fixed["miopen_user_dirname"],
        "miopen_custom": run_root / fixed["miopen_custom_cache_dirname"],
        "miopen_scratch_closure": miopen_scratch_closure_path,
    }
    candidate = None
    candidate_record = {"path": str(candidate_path), "present": candidate_path.exists() and not candidate_path.is_symlink()}
    launch_evidence: dict[str, Any] = {}
    live_contract_digest: Optional[str] = None
    try:
        if observed_paths != exact_paths:
            fail("controller paths differ from fixed fresh run authority")
        launch_evidence["prior_failed_engineering_attempt"] = (
            _verify_prior_failed_attempt()
        )
        controller_step_meta = write_controller_step_meta(
            controller_step_meta_path,
            controller_argv_path,
            snapshot_root,
        )
        _validate_run_root_preseal(run_root, exact_paths)
        if str(snapshot_root) != fixed["planned_preflip_snapshot_root"]:
            fail("snapshot root differs from fixed authority")
        if snapshot_root != Path(os.path.abspath(__file__)).parent:
            fail("controller is not executing from the captured immutable snapshot root")
        snapshot_receipt = snapshot_v3.verify_snapshot(snapshot_root, verify_original=False)
        launch_evidence["snapshot"] = snapshot_receipt
        log_record = stable_file_record(formal_log_path)
        if log_record["mode"] != 0o444:
            fail("formal log is not frozen mode 0444")
        launch_evidence["formal_log"] = log_record
        srun_argv = _nul_argv_record(srun_argv_path)
        rank_argv = _nul_argv_record(rank_argv_path)
        controller_argv = _nul_argv_record(controller_argv_path)
        if any(
            row["mode"] != 0o444
            for row in (srun_argv, rank_argv, controller_argv)
        ):
            fail("captured NUL argv files are not frozen mode 0444")
        wrapper = str(snapshot_root / "scripts" / "auh_actual_target_foundation_canary_rank_wrapper_v3.sh")
        expected_rank_argv = [
            wrapper,
            str(candidate_path),
            str(cache_dir),
            str(step_meta_path),
            str(rank_argv_path),
            str(snapshot_root),
            str(exact_paths["miopen_user"]),
            str(exact_paths["miopen_custom"]),
        ]
        if rank_argv["argv"] != expected_rank_argv:
            fail("rank wrapper NUL argv differs from exact formal invocation")
        launch_authority = authority.load_authority()
        if launch_authority.get("authorized_launch_mode") != "existing_allocation_only":
            fail("authorized launch mode differs from existing-allocation-only authority")
        existing_srun = [
            "srun",
            "--jobid",
            expected_job_id,
            "--exclusive",
            "--exact",
            "--immediate=60",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=16",
            "--gres=gpu:mi210:1",
            "--mem=56G",
            "--export=ALL,LOCAL_RANK=0,WORLD_SIZE=1",
            *expected_rank_argv,
        ]
        if srun_argv["argv"] != existing_srun:
            fail("compute srun is not the exact authorized existing-allocation invocation")
        if int(expected_job_id) not in launch_authority["existing_allocation_contract"]["approved_job_ids"]:
            fail("existing-allocation srun job is not approved")
        controller_command = _expected_controller_command(
            exact_paths=exact_paths,
            expected_contract_digest=expected_contract_digest,
            srun_exit_code=srun_exit_code,
            tee_exit_code=tee_exit_code,
            expected_job_id=expected_job_id,
            snapshot_root=snapshot_root,
        )
        _, existing_controller_srun = (
            _expected_controller_srun_argv(
                command=controller_command,
                expected_job_id=expected_job_id,
            )
        )
        if controller_argv["argv"] != existing_controller_srun:
            fail(
                "controller srun is not the exact authorized existing-allocation invocation"
            )
        launch_evidence["srun_argv"] = srun_argv
        launch_evidence["rank_argv"] = rank_argv
        launch_evidence["controller_argv"] = controller_argv
        step_meta, step_record = strict_json_file(step_meta_path)
        _verify_digest(step_meta, "step metadata")
        expected_step_fields = {
            "schema_version",
            "slurm_job_id",
            "slurm_step_id",
            "local_rank",
            "world_size",
            "rocr_visible_devices",
            "hostname",
            "candidate_path",
            "cache_dir",
            "rank_argv_path",
            "rank_argv_sha256",
            "rank_argv_argc",
            "snapshot_root",
            "snapshot_receipt_digest",
            "snapshot_manifest_file_sha256",
            "miopen_environment",
            "miopen_disable_cache_present",
            "miopen_directories_initially_empty",
            "miopen_directory_records",
            "digest",
        }
        if step_record["mode"] != 0o444:
            fail("step metadata is not frozen mode 0444")
        if (
            set(step_meta) != expected_step_fields
            or step_meta.get("schema_version")
            != "actual-target-foundation-step-meta-v3"
            or os.environ.get("SLURM_JOB_ID") != expected_job_id
            or
            step_meta.get("slurm_job_id") != expected_job_id
            or not isinstance(step_meta.get("slurm_step_id"), str)
            or not step_meta.get("slurm_step_id")
            or step_meta.get("local_rank") != 0
            or step_meta.get("world_size") != 1
            or step_meta.get("candidate_path") != str(candidate_path)
            or step_meta.get("cache_dir") != str(cache_dir)
            or step_meta.get("rank_argv_path") != str(rank_argv_path)
            or step_meta.get("rank_argv_sha256") != rank_argv["sha256"]
            or step_meta.get("rank_argv_argc") != rank_argv["argc"]
            or step_meta.get("snapshot_root") != str(snapshot_root)
            or step_meta.get("snapshot_receipt_digest") != snapshot_receipt["digest"]
            or step_meta.get("snapshot_manifest_file_sha256") != snapshot_receipt["manifest_file_sha256"]
            or step_meta.get("miopen_environment")
            != {
                "MIOPEN_USER_DB_PATH": str(exact_paths["miopen_user"]),
                "MIOPEN_CUSTOM_CACHE_DIR": str(exact_paths["miopen_custom"]),
            }
            or step_meta.get("miopen_disable_cache_present") is not False
            or step_meta.get("miopen_directories_initially_empty") is not True
        ):
            fail("Slurm job/step/rank argv metadata differs")
        initial_scratch_records = step_meta.get("miopen_directory_records")
        if (
            not isinstance(initial_scratch_records, Mapping)
            or set(initial_scratch_records)
            != {"MIOPEN_USER_DB_PATH", "MIOPEN_CUSTOM_CACHE_DIR"}
            or any(
                not isinstance(initial_scratch_records.get(name), Mapping)
                or set(initial_scratch_records[name])
                != {
                    "path",
                    "device",
                    "inode",
                    "mode",
                    "no_symlink_components",
                }
                or initial_scratch_records[name].get("path") != str(path)
                or initial_scratch_records[name].get("mode") != 0o700
                or initial_scratch_records[name].get("no_symlink_components")
                is not True
                for name, path in (
                    ("MIOPEN_USER_DB_PATH", exact_paths["miopen_user"]),
                    (
                        "MIOPEN_CUSTOM_CACHE_DIR",
                        exact_paths["miopen_custom"],
                    ),
                )
            )
        ):
            fail("rank-step initial MIOpen directory records differ")
        launch_evidence["step_meta"] = {"value": step_meta, "file": step_record}
        scratch_closure, scratch_closure_record = _verify_miopen_scratch_closure(
            miopen_scratch_closure_path
        )
        final_scratch_by_root = {
            row["root"]: row for row in scratch_closure["trees"]
        }
        for name, path in (
            ("MIOPEN_USER_DB_PATH", exact_paths["miopen_user"]),
            ("MIOPEN_CUSTOM_CACHE_DIR", exact_paths["miopen_custom"]),
        ):
            initial = initial_scratch_records[name]
            final = final_scratch_by_root.get(str(path), {})
            if (
                initial.get("path") != final.get("root")
                or initial.get("device") != final.get("root_device")
                or initial.get("inode") != final.get("root_inode")
                or final.get("root_mode") != 0o555
            ):
                fail("MIOpen scratch root identity changed between rank and seal")
        launch_evidence["miopen_scratch_closure"] = {
            "value": scratch_closure,
            "file": scratch_closure_record,
        }
        controller_step_value, controller_step_record = strict_json_file(
            controller_step_meta_path
        )
        _verify_digest(controller_step_value, "controller step metadata")
        if step_record["mode"] != 0o444 or controller_step_record["mode"] != 0o444:
            fail("rank/controller step metadata is not frozen mode 0444")
        expected_controller_step = {
            "schema_version": CONTROLLER_STEP_META_SCHEMA,
            "slurm_job_id": expected_job_id,
            "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
            "hostname": socket.gethostname(),
            "cuda_visible_devices": "",
            "rocr_visible_devices": "",
            "hip_visible_devices": "",
            "torch_imported_at_metadata": False,
            "foundation_imports_at_metadata": [],
            "controller_wrapper_path": str(
                snapshot_root
                / "scripts"
                / "auh_actual_target_foundation_canary_controller_wrapper_v3.sh"
            ),
            "controller_wrapper_sha256": stable_file_record(
                snapshot_root
                / "scripts"
                / "auh_actual_target_foundation_canary_controller_wrapper_v3.sh"
            )["sha256"],
            "controller_argv_path": str(controller_argv_path),
            "controller_argv_sha256": controller_argv["sha256"],
            "controller_argv_argc": controller_argv["argc"],
            "snapshot_root": str(snapshot_root),
            "snapshot_receipt_digest": snapshot_receipt["digest"],
            "snapshot_manifest_file_sha256": snapshot_receipt[
                "manifest_file_sha256"
            ],
        }
        if (
            dict(controller_step_value)
            != {**expected_controller_step, "digest": authority.object_sha256(expected_controller_step)}
            or controller_step_meta != controller_step_value
            or controller_step_value["hostname"] != step_meta.get("hostname")
            or controller_step_value["slurm_job_id"]
            != step_meta.get("slurm_job_id")
            or any(os.environ.get(name) != "" for name in CONTROLLER_DEVICE_ENV)
            or _forbidden_foundation_imports()
        ):
            fail("zero-GPU controller Slurm step metadata differs")
        launch_evidence["controller_step_meta"] = {
            "value": controller_step_value,
            "file": controller_step_record,
        }
        if not isinstance(srun_exit_code, int) or isinstance(srun_exit_code, bool) or not isinstance(tee_exit_code, int) or isinstance(tee_exit_code, bool):
            fail("PIPESTATUS exit codes are not integers")
        launch_evidence["pipe_status"] = {"srun": srun_exit_code, "tee": tee_exit_code}
        if srun_exit_code != 0 or tee_exit_code != 0:
            fail("real srun or tee PIPESTATUS was nonzero")
        candidate, candidate_record, cache_records = _validate_candidate(
            candidate_path, cache_dir, expected_contract_digest
        )
        if candidate["device_closure"].get("rocr_visible_devices") != step_meta.get(
            "rocr_visible_devices"
        ):
            fail("candidate device token differs from rank step ROCR token")
        candidate_scratch = candidate["device_closure"].get(
            "miopen_scratch_binding"
        )
        if (
            not isinstance(candidate_scratch, Mapping)
            or candidate_scratch.get("environment")
            != step_meta.get("miopen_environment")
            or candidate_scratch.get("miopen_disable_cache_present") is not False
            or candidate_scratch.get("directories")
            != initial_scratch_records
        ):
            fail("candidate/rank-step MIOpen scratch binding differs")
        launch_evidence["candidate_file"] = candidate_record
        launch_evidence["cache_files"] = list(cache_records)
        launch_evidence["cache_directory"] = {
            "path": str(cache_dir),
            "mode": stat.S_IMODE(cache_dir.stat().st_mode),
        }
        live_contract_digest = runtime.launch_contract()["digest"]
        if _torch_is_imported() or _forbidden_foundation_imports():
            fail("external completion controller imported torch/foundation before sealing")
    except BaseException as error:
        reasons = [f"{type(error).__name__}:{str(error)}"]
        try:
            live_contract_digest = runtime.launch_contract()["digest"]
        except BaseException as contract_error:
            reasons.append(f"LIVE_CONTRACT_UNAVAILABLE:{type(contract_error).__name__}:{str(contract_error)}")
        value = {
            "schema_version": "actual-target-foundation-attempt-ledger-v3",
            "engineering_failure": True,
            "valid_completion_seal": False,
            "completion_outcome": None,
            "failure_reasons": reasons,
            "candidate_file": candidate_record,
            "reviewed_launch_contract_digest": expected_contract_digest,
            "live_launch_contract_digest": live_contract_digest,
            "launch_evidence": launch_evidence,
            "locked_validation_claimed": False,
            "scientific_evidence_claimed": False,
            "representation_admitted": False,
            "gpu_used_by_external_controller": False,
        }
        ledger = {**value, "digest": authority.object_sha256(value)}
        runtime.create_only_json(exact_paths["attempt_ledger"], ledger)
        return ledger

    diagnostic_pass = bool(
        candidate.get("aggregate", {}).get("diagnostic_canary_pass") is True
        and candidate.get("aggregate", {}).get("passed_case_count") == 4
    )
    outcome = "PASS" if diagnostic_pass else "REJECTED"
    reasons = [] if diagnostic_pass else ["DEVELOPMENT_DIAGNOSTIC_DID_NOT_PASS_EXACT_4_OF_4"]
    value = {
        "schema_version": SEAL_SCHEMA,
        "controller_schema_version": SCHEMA,
        "outcome": outcome,
        "valid_completion_seal": True,
        "external_controller_pass": outcome == "PASS",
        "development_diagnostic_pass": diagnostic_pass,
        "locked_validation_claimed": False,
        "scientific_evidence_claimed": False,
        "representation_admitted": False,
        "generator_connection_authorized": False,
        "training_performed": False,
        "generator_loaded": False,
        "rejection_reasons": reasons,
        "candidate_file": candidate_record,
        "candidate_digest": candidate.get("digest") if candidate is not None else None,
        "reviewed_launch_contract_digest": expected_contract_digest,
        "live_launch_contract_digest": live_contract_digest,
        "launch_evidence": launch_evidence,
        "gpu_used_by_external_controller": False,
    }
    seal = {**value, "digest": authority.object_sha256(value)}
    runtime.create_only_json(seal_path, seal)
    return seal


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    nul = subparsers.add_parser("write-nul")
    nul.add_argument("path", type=Path)
    nul.add_argument("argv", nargs=argparse.REMAINDER)
    step = subparsers.add_parser("write-step-meta")
    step.add_argument("--path", type=Path, required=True)
    step.add_argument("--candidate", type=Path, required=True)
    step.add_argument("--cache-dir", type=Path, required=True)
    step.add_argument("--rank-argv", type=Path, required=True)
    step.add_argument("--snapshot-root", type=Path, required=True)
    step.add_argument("--miopen-user-dir", type=Path, required=True)
    step.add_argument("--miopen-custom-cache-dir", type=Path, required=True)
    freeze = subparsers.add_parser("freeze-file")
    freeze.add_argument("path", type=Path)
    freeze_run = subparsers.add_parser("freeze-run-artifacts")
    freeze_run.add_argument("--candidate", type=Path, required=True)
    freeze_run.add_argument("--cache-dir", type=Path, required=True)
    freeze_scratch = subparsers.add_parser("freeze-scratch")
    freeze_scratch.add_argument("--miopen-user-dir", type=Path, required=True)
    freeze_scratch.add_argument(
        "--miopen-custom-cache-dir", type=Path, required=True
    )
    freeze_scratch.add_argument("--closure", type=Path, required=True)
    subparsers.add_parser("verify-prior-closures")
    seal = subparsers.add_parser("seal")
    seal.add_argument("--candidate", type=Path, required=True)
    seal.add_argument("--cache-dir", type=Path, required=True)
    seal.add_argument("--seal", type=Path, required=True)
    seal.add_argument("--expected-contract-digest", required=True)
    seal.add_argument("--srun-exit-code", type=int, required=True)
    seal.add_argument("--tee-exit-code", type=int, required=True)
    seal.add_argument("--expected-job-id", required=True)
    seal.add_argument("--step-meta", type=Path, required=True)
    seal.add_argument("--formal-log", type=Path, required=True)
    seal.add_argument("--srun-argv", type=Path, required=True)
    seal.add_argument("--rank-argv", type=Path, required=True)
    seal.add_argument("--controller-argv", type=Path, required=True)
    seal.add_argument("--controller-step-meta", type=Path, required=True)
    seal.add_argument("--miopen-scratch-closure", type=Path, required=True)
    seal.add_argument("--snapshot-root", type=Path, required=True)
    seal.add_argument("--attempt-ledger", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "write-nul":
        values = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
        result = write_nul_argv(args.path, values)
    elif args.command == "write-step-meta":
        result = write_step_meta(
            args.path,
            args.candidate,
            args.cache_dir,
            args.rank_argv,
            args.snapshot_root,
            args.miopen_user_dir,
            args.miopen_custom_cache_dir,
        )
    elif args.command == "freeze-file":
        result = freeze_existing_file(args.path)
    elif args.command == "freeze-run-artifacts":
        result = freeze_candidate_cache(args.candidate, args.cache_dir)
    elif args.command == "freeze-scratch":
        result = freeze_miopen_scratch(
            args.miopen_user_dir,
            args.miopen_custom_cache_dir,
            args.closure,
        )
    elif args.command == "verify-prior-closures":
        result = _verify_prior_failed_attempt()
    else:
        result = seal_outcome(
            candidate_path=args.candidate,
            cache_dir=args.cache_dir,
            seal_path=args.seal,
            expected_contract_digest=args.expected_contract_digest,
            srun_exit_code=args.srun_exit_code,
            tee_exit_code=args.tee_exit_code,
            expected_job_id=args.expected_job_id,
            step_meta_path=args.step_meta,
            formal_log_path=args.formal_log,
            srun_argv_path=args.srun_argv,
            rank_argv_path=args.rank_argv,
            controller_argv_path=args.controller_argv,
            controller_step_meta_path=args.controller_step_meta,
            miopen_scratch_closure_path=args.miopen_scratch_closure,
            snapshot_root=args.snapshot_root,
            attempt_ledger_path=args.attempt_ledger,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ControllerV3Error",
    "SEAL_SCHEMA",
    "freeze_existing_file",
    "freeze_candidate_cache",
    "freeze_miopen_scratch",
    "seal_outcome",
    "stable_file_record",
    "strict_json_file",
    "write_nul_argv",
    "write_controller_step_meta",
    "write_step_meta",
]
