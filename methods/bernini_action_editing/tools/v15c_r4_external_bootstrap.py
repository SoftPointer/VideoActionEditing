#!/usr/bin/env python3
"""External, stdlib-only trust bootstrap for the v15c-r4 observer package.

This file is deliberately outside the release manifest.  Its SHA256 is pinned
by the external launch template and checked by the shell *before* Python starts.
It imports no workspace code.  It authenticates the complete release closure,
copies that closure through verified file descriptors into a private snapshot,
seals the snapshot read-only, emits a self-hashed bootstrap receipt, and only
then executes the authenticated launcher through an already-open descriptor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Any, Mapping, Sequence


RELEASE_SCHEMA = "bernini-source-object-proposal-role-v15c-r4-release"
BOOTSTRAP_SCHEMA = "bernini-source-object-proposal-role-v15c-r4-bootstrap"
RELEASE_TAG = "v15c-r4"
RELEASE_RELATIVE_PATH = (
    "methods/bernini_action_editing/assets/"
    "e00_source_sam2_proposal_role_probe_v15c_r4_release.json"
)
LAUNCHER_RELATIVE_PATH = (
    "methods/bernini_action_editing/scripts/"
    "auh_launch_e00_source_sam2_proposal_role_probe_v15c_r4_sealed.sh"
)
EXPECTED_MEMBER_COUNT = 8
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
SAFE_GPU_ENV_RE = re.compile(r"^[A-Za-z0-9_,.:-]{1,256}$")
RELEASE_KEYS = (
    "schema_version",
    "tag",
    "member_count",
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
)
SNAPSHOT_POLICY = {
    "construction_root_mode": "0700",
    "sealed_directory_mode": "0500",
    "sealed_member_mode": "0400",
    "exact_tree_required": True,
    "reject_extra_symlink_pyc": True,
}


class BootstrapV15CR4Error(RuntimeError):
    """The externally authenticated release closure differs."""


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
        raise BootstrapV15CR4Error(f"{label} is not lowercase SHA256")
    return value


def _regular_lstat(path: Path, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as error:
        raise BootstrapV15CR4Error(f"{label} is absent") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise BootstrapV15CR4Error(f"{label} is not one regular non-symlink file")
    return info


def file_sha256(path: Path) -> str:
    _regular_lstat(path, str(path))
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise BootstrapV15CR4Error("opened release member is not regular")
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
            raise BootstrapV15CR4Error("release member changed while hashing")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def descriptor_sha256(descriptor: int) -> str:
    """Hash the already-open descriptor authenticated by the external shell."""
    try:
        original_offset = os.lseek(descriptor, 0, os.SEEK_CUR)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise BootstrapV15CR4Error("trusted bootstrap descriptor is not regular")
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
        raise BootstrapV15CR4Error("trusted bootstrap descriptor differs") from error
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
        raise BootstrapV15CR4Error("trusted bootstrap descriptor changed")
    return digest.hexdigest()


def open_beneath(root: Path, relative: str) -> int:
    """Open a release member with an O_NOFOLLOW directory-FD walk."""
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    current = os.open(root, directory_flags)
    try:
        parts = PurePosixPath(relative).parts
        for part in parts[:-1]:
            following = os.open(part, directory_flags, dir_fd=current)
            os.close(current)
            current = following
        result = os.open(parts[-1], file_flags, dir_fd=current)
    except Exception:
        os.close(current)
        raise
    os.close(current)
    if not stat.S_ISREG(os.fstat(result).st_mode):
        os.close(result)
        raise BootstrapV15CR4Error("secure-open release member is not regular")
    return result


def descriptor_bytes(descriptor: int) -> bytes:
    try:
        before = os.fstat(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
    except OSError as error:
        raise BootstrapV15CR4Error("release descriptor read differs") from error
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
        raise BootstrapV15CR4Error("release descriptor changed while reading")
    return b"".join(chunks)


def read_json(path: Path) -> Mapping[str, Any]:
    _regular_lstat(path, "release manifest")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise BootstrapV15CR4Error("release manifest JSON differs") from error
    if type(value) is not dict:
        raise BootstrapV15CR4Error("release manifest is not one object")
    return value


def _relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise BootstrapV15CR4Error(f"{label} path differs")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
        or value.endswith(".pyc")
        or "__pycache__" in path.parts
    ):
        raise BootstrapV15CR4Error(f"{label} path differs")
    return value


def _path_beneath(root: Path, relative: str, *, final_regular: bool) -> Path:
    current = root
    parts = PurePosixPath(relative).parts
    for index, part in enumerate(parts):
        current = current / part
        try:
            info = current.lstat()
        except OSError as error:
            raise BootstrapV15CR4Error("release closure path is absent") from error
        if stat.S_ISLNK(info.st_mode):
            raise BootstrapV15CR4Error("release closure contains a symlink")
        if index + 1 < len(parts):
            if not stat.S_ISDIR(info.st_mode):
                raise BootstrapV15CR4Error("release closure parent is not a directory")
        elif final_regular and not stat.S_ISREG(info.st_mode):
            raise BootstrapV15CR4Error("release closure member is not regular")
    return current


def _verify_manifest_semantics(manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    if set(manifest) != set(RELEASE_KEYS):
        raise BootstrapV15CR4Error("release exact keys differ")
    payload = dict(manifest)
    claimed = require_sha(payload.pop("release_sha256", None), "release self-hash")
    if claimed != object_sha256(payload):
        raise BootstrapV15CR4Error("release self-hash differs")
    members = manifest.get("members")
    if (
        manifest.get("schema_version") != RELEASE_SCHEMA
        or manifest.get("tag") != RELEASE_TAG
        or manifest.get("member_count") != EXPECTED_MEMBER_COUNT
        or type(members) is not list
        or len(members) != EXPECTED_MEMBER_COUNT
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
        raise BootstrapV15CR4Error("release semantics differ")
    normalized: list[Mapping[str, Any]] = []
    paths: list[str] = []
    for row in members:
        if type(row) is not dict or set(row) != {"path", "sha256", "size"}:
            raise BootstrapV15CR4Error("release member exact keys differ")
        relative = _relative_path(row.get("path"), "release member")
        digest = require_sha(row.get("sha256"), "release member hash")
        size = row.get("size")
        if type(size) is not int or size <= 0:
            raise BootstrapV15CR4Error("release member size differs")
        normalized.append({"path": relative, "sha256": digest, "size": size})
        paths.append(relative)
    if paths != sorted(paths) or len(set(paths)) != EXPECTED_MEMBER_COUNT:
        raise BootstrapV15CR4Error("release member registry differs")
    if LAUNCHER_RELATIVE_PATH not in paths or not any(
        path.endswith("finalize_source_sam2_proposal_role_probe_v15c_r4.py")
        for path in paths
    ):
        raise BootstrapV15CR4Error("launcher/finalizer are outside release closure")
    return tuple(normalized)


def verify_release_source(
    source_root: Path, release_path: Path, expected_release_sha256: str
) -> Mapping[str, Any]:
    expected = require_sha(expected_release_sha256, "external release hash")
    source_root = source_root.resolve(strict=True)
    if not source_root.is_dir() or source_root.is_symlink():
        raise BootstrapV15CR4Error("source root differs")
    expected_release = _path_beneath(
        source_root, RELEASE_RELATIVE_PATH, final_regular=True
    )
    if release_path.absolute() != expected_release.absolute():
        raise BootstrapV15CR4Error("release manifest placement differs")
    release_fd = open_beneath(source_root, RELEASE_RELATIVE_PATH)
    try:
        release_bytes = descriptor_bytes(release_fd)
    finally:
        os.close(release_fd)
    if hashlib.sha256(release_bytes).hexdigest() != expected:
        raise BootstrapV15CR4Error("external release hash differs")
    try:
        manifest = json.loads(release_bytes.decode("utf-8"))
    except Exception as error:
        raise BootstrapV15CR4Error("release manifest JSON differs") from error
    if type(manifest) is not dict:
        raise BootstrapV15CR4Error("release manifest is not one object")
    members = _verify_manifest_semantics(manifest)
    for row in members:
        _path_beneath(source_root, row["path"], final_regular=True)
        descriptor = open_beneath(source_root, row["path"])
        try:
            info = os.fstat(descriptor)
            digest = descriptor_sha256(descriptor)
        finally:
            os.close(descriptor)
        if info.st_size != row["size"] or digest != row["sha256"]:
            raise BootstrapV15CR4Error("release member bytes differ")
    return manifest


def _expected_tree(manifest: Mapping[str, Any]) -> set[str]:
    return {RELEASE_RELATIVE_PATH, *(row["path"] for row in manifest["members"])}


def _expected_directories(files: set[str]) -> set[str]:
    directories = {"."}
    for relative in files:
        parent = PurePosixPath(relative).parent
        while parent.as_posix() != ".":
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def verify_snapshot(
    snapshot: Path,
    manifest: Mapping[str, Any],
    expected_release_sha256: str,
    *,
    sealed: bool,
) -> Mapping[str, Mapping[str, Any]]:
    snapshot = snapshot.absolute()
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise BootstrapV15CR4Error("snapshot root differs")
    files_expected = _expected_tree(manifest)
    directories_expected = _expected_directories(files_expected)
    observed_files: set[str] = set()
    observed_directories = {"."}
    for current_text, directory_names, file_names in os.walk(snapshot, topdown=True, followlinks=False):
        current = Path(current_text)
        for name in directory_names:
            path = current / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise BootstrapV15CR4Error("snapshot contains a non-directory or symlink")
            relative = path.relative_to(snapshot).as_posix()
            if name == "__pycache__":
                raise BootstrapV15CR4Error("snapshot contains __pycache__")
            observed_directories.add(relative)
        for name in file_names:
            path = current / name
            info = path.lstat()
            relative = path.relative_to(snapshot).as_posix()
            if (
                stat.S_ISLNK(info.st_mode)
                or not stat.S_ISREG(info.st_mode)
                or name.endswith(".pyc")
            ):
                raise BootstrapV15CR4Error("snapshot contains symlink/non-file/pyc")
            observed_files.add(relative)
    if observed_files != files_expected or observed_directories != directories_expected:
        raise BootstrapV15CR4Error("snapshot exact tree differs")
    expected_file_rows = {
        row["path"]: {"sha256": row["sha256"], "size": row["size"]}
        for row in manifest["members"]
    }
    release_path = snapshot / RELEASE_RELATIVE_PATH
    expected_file_rows[RELEASE_RELATIVE_PATH] = {
        "sha256": require_sha(expected_release_sha256, "snapshot release hash"),
        "size": release_path.stat().st_size,
    }
    observed: dict[str, Mapping[str, Any]] = {}
    expected_file_mode = 0o400 if sealed else 0o400
    expected_directory_mode = 0o500 if sealed else 0o700
    for relative in sorted(files_expected):
        path = _path_beneath(snapshot, relative, final_regular=True)
        info = path.lstat()
        row = expected_file_rows[relative]
        digest = file_sha256(path)
        if (
            info.st_size != row["size"]
            or digest != row["sha256"]
            or stat.S_IMODE(info.st_mode) != expected_file_mode
        ):
            raise BootstrapV15CR4Error("snapshot member bytes/mode differ")
        observed[relative] = {"sha256": digest, "size": info.st_size}
    for relative in directories_expected:
        path = snapshot if relative == "." else snapshot / relative
        if stat.S_IMODE(path.lstat().st_mode) != expected_directory_mode:
            raise BootstrapV15CR4Error("snapshot directory mode differs")
    reopened = read_json(release_path)
    if reopened != manifest:
        raise BootstrapV15CR4Error("snapshot manifest differs")
    return observed


def _copy_one(
    source_root: Path,
    source_relative: str,
    destination: Path,
    expected_sha: str,
    expected_size: int,
) -> None:
    destination_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    source_fd = open_beneath(source_root, source_relative)
    try:
        source_before = os.fstat(source_fd)
        if not stat.S_ISREG(source_before.st_mode) or source_before.st_size != expected_size:
            raise BootstrapV15CR4Error("copy source identity differs")
        destination_fd = os.open(destination, destination_flags, 0o400)
        digest = hashlib.sha256()
        copied = 0
        try:
            while True:
                block = os.read(source_fd, 1024 * 1024)
                if not block:
                    break
                digest.update(block)
                copied += len(block)
                view = memoryview(block)
                while view:
                    written = os.write(destination_fd, view)
                    if written <= 0:
                        raise BootstrapV15CR4Error("snapshot short write")
                    view = view[written:]
            os.fsync(destination_fd)
        finally:
            os.close(destination_fd)
        source_after = os.fstat(source_fd)
        if (
            source_before.st_dev,
            source_before.st_ino,
            source_before.st_size,
            source_before.st_mtime_ns,
            source_before.st_ctime_ns,
        ) != (
            source_after.st_dev,
            source_after.st_ino,
            source_after.st_size,
            source_after.st_mtime_ns,
            source_after.st_ctime_ns,
        ):
            raise BootstrapV15CR4Error("copy source changed during read")
    finally:
        os.close(source_fd)
    if copied != expected_size or digest.hexdigest() != expected_sha:
        raise BootstrapV15CR4Error("copied member bytes differ")


def materialize_snapshot(
    source_root: Path,
    release_path: Path,
    snapshot: Path,
    manifest: Mapping[str, Any],
    expected_release_sha256: str,
) -> Mapping[str, Mapping[str, Any]]:
    if snapshot.exists() or snapshot.is_symlink():
        raise BootstrapV15CR4Error("snapshot destination is not fresh")
    snapshot.mkdir(mode=0o700)
    rows = list(manifest["members"])
    rows.append(
        {
            "path": RELEASE_RELATIVE_PATH,
            "sha256": expected_release_sha256,
            "size": release_path.stat().st_size,
        }
    )
    for row in sorted(rows, key=lambda value: value["path"]):
        relative = row["path"]
        destination = snapshot / relative
        missing_parents: list[Path] = []
        parent = destination.parent
        while parent != snapshot and not parent.exists():
            missing_parents.append(parent)
            parent = parent.parent
        for directory in reversed(missing_parents):
            directory.mkdir(mode=0o700)
        _copy_one(
            source_root,
            relative,
            destination,
            row["sha256"],
            row["size"],
        )
    verify_snapshot(
        snapshot, manifest, expected_release_sha256, sealed=False
    )
    directories = [path for path in snapshot.rglob("*") if path.is_dir()]
    for directory in sorted(directories, key=lambda value: len(value.parts), reverse=True):
        os.chmod(directory, 0o500, follow_symlinks=False)
    os.chmod(snapshot, 0o500, follow_symlinks=False)
    return verify_snapshot(snapshot, manifest, expected_release_sha256, sealed=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_noreplace(path: Path, payload: bytes) -> None:
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}"
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise BootstrapV15CR4Error("bootstrap receipt destination is not fresh")
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
                raise BootstrapV15CR4Error("bootstrap receipt short write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path, follow_symlinks=False)
        _fsync_directory(path.parent)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    temporary.unlink()
    _fsync_directory(path.parent)
    if path.read_bytes() != payload:
        raise BootstrapV15CR4Error("bootstrap receipt reopen differs")


def _clean_environment(python_bin: Path, expected_job: str) -> dict[str, str]:
    environment = {
        "HOME": "/nonexistent/v15c-r4",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": f"{python_bin.parent}:/opt/rocm/bin:/usr/bin:/bin",
        "LD_LIBRARY_PATH": f"/opt/rocm/lib:/opt/rocm/lib64:{python_bin.parent.parent / 'lib'}",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONHASHSEED": "0",
        "TMPDIR": "/tmp",
        "OMP_NUM_THREADS": "1",
        "SLURM_JOB_ID": expected_job,
        "V15C_R4_EXTERNAL_BOOTSTRAP": "1",
    }
    for key in (
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "HSA_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    ):
        value = os.environ.get(key)
        if value is not None:
            if SAFE_GPU_ENV_RE.fullmatch(value) is None:
                raise BootstrapV15CR4Error(f"unsafe inherited {key}")
            environment[key] = value
    return environment


def _exact_keys(value: Mapping[str, Any], keys: Sequence[str], label: str) -> None:
    if set(value) != set(keys):
        raise BootstrapV15CR4Error(f"{label} exact keys differ")


def bootstrap_and_exec(args: argparse.Namespace) -> None:
    source_root = args.source_root.resolve(strict=True)
    release_path = args.release_manifest.absolute()
    bootstrap_path = args.bootstrap_source_path.resolve(strict=True)
    python_bin = args.python_bin.resolve(strict=True)
    expected_bootstrap_sha = require_sha(
        args.expected_bootstrap_sha256, "external bootstrap hash"
    )
    expected_python_sha = require_sha(args.expected_python_sha256, "Python hash")
    if (
        args.trusted_bootstrap_fd < 3
        or descriptor_sha256(args.trusted_bootstrap_fd) != expected_bootstrap_sha
        or file_sha256(bootstrap_path) != expected_bootstrap_sha
    ):
        raise BootstrapV15CR4Error("external bootstrap bytes differ")
    if (
        args.trusted_python_fd < 3
        or descriptor_sha256(args.trusted_python_fd) != expected_python_sha
        or file_sha256(python_bin) != expected_python_sha
    ):
        raise BootstrapV15CR4Error("Python bytes differ")
    if args.expected_job != "143808" or args.expected_node != "auh7-1b-gpu-292":
        raise BootstrapV15CR4Error("execution constants differ")
    if os.environ.get("SLURM_JOB_ID") != args.expected_job:
        raise BootstrapV15CR4Error("Slurm job authority differs")
    node = os.uname().nodename.split(".", 1)[0]
    if node != args.expected_node:
        raise BootstrapV15CR4Error("node authority differs")
    manifest = verify_release_source(
        source_root, release_path, args.expected_release_sha256
    )
    run_root = args.run_root.absolute()
    if not RUN_NAME_RE.fullmatch(run_root.name):
        raise BootstrapV15CR4Error("run id differs")
    run_parent = run_root.parent.resolve(strict=True)
    if run_root.parent.absolute() != run_parent or run_root.exists() or run_root.is_symlink():
        raise BootstrapV15CR4Error("run root is not a fresh direct child")
    run_root.mkdir(mode=0o700)
    snapshot = run_root / "sealed_code_snapshot"
    snapshot_files = materialize_snapshot(
        source_root,
        release_path,
        snapshot,
        manifest,
        args.expected_release_sha256,
    )
    receipt: dict[str, Any] = {
        "schema_version": BOOTSTRAP_SCHEMA,
        "status": "EXTERNAL_BOOTSTRAP_VERIFIED_AND_SNAPSHOT_SEALED",
        "external_bootstrap": {
            "path": str(bootstrap_path),
            "sha256": expected_bootstrap_sha,
            "size": bootstrap_path.stat().st_size,
        },
        "python": {
            "path": str(python_bin),
            "sha256": expected_python_sha,
            "size": python_bin.stat().st_size,
            "startup_flags": ["-I", "-S", "-B"],
        },
        "release": {
            "source_root": str(source_root),
            "manifest_path": str(release_path),
            "manifest_file_sha256": args.expected_release_sha256,
            "manifest_internal_sha256": manifest["release_sha256"],
            "member_count": len(manifest["members"]),
        },
        "snapshot": {
            "root": str(snapshot),
            "construction_root_mode": "0700",
            "sealed_directory_mode": "0500",
            "sealed_member_mode": "0400",
            "exact_tree_verified": True,
            "extras_symlinks_pyc_absent": True,
            "files": snapshot_files,
        },
        "execution": {
            "parent_job_id": int(args.expected_job),
            "node": node,
            "run_root": str(run_root),
        },
        "authority": {
            "observer_only": True,
            "human_audit_action": "reject_only",
            "remote_gpu_status_before_execution": "REMOTE_GPU_UNAUDITED",
            "route_status": "ROUTE_NO_GO",
            "route_authorized": False,
            "decode_authorized": False,
            "training_authorized": False,
        },
    }
    receipt["receipt_sha256"] = object_sha256(receipt)
    receipt_path = run_root / "external_bootstrap_receipt.json"
    publish_noreplace(receipt_path, canonical_bytes(receipt))
    launcher = snapshot / LAUNCHER_RELATIVE_PATH
    launcher_fd = os.open(
        launcher,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    if file_sha256(launcher) != next(
        row["sha256"]
        for row in manifest["members"]
        if row["path"] == LAUNCHER_RELATIVE_PATH
    ):
        os.close(launcher_fd)
        raise BootstrapV15CR4Error("launcher changed before descriptor exec")
    os.set_inheritable(launcher_fd, True)
    os.set_inheritable(args.trusted_python_fd, True)
    environment = _clean_environment(python_bin, args.expected_job)
    argv = [
        "/bin/bash",
        f"/proc/self/fd/{launcher_fd}",
        "--sealed-worker",
        str(run_root),
        str(snapshot),
        args.expected_release_sha256,
        str(receipt_path),
        expected_bootstrap_sha,
        expected_python_sha,
    ]
    os.chdir(run_root)
    os.execve("/bin/bash", argv, environment)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--release-manifest", required=True, type=Path)
    parser.add_argument("--expected-release-sha256", required=True)
    parser.add_argument("--expected-bootstrap-sha256", required=True)
    parser.add_argument("--bootstrap-source-path", required=True, type=Path)
    parser.add_argument("--trusted-bootstrap-fd", required=True, type=int)
    parser.add_argument("--python-bin", required=True, type=Path)
    parser.add_argument("--trusted-python-fd", required=True, type=int)
    parser.add_argument("--expected-python-sha256", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--expected-job", required=True)
    parser.add_argument("--expected-node", required=True)
    args = parser.parse_args()
    bootstrap_and_exec(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
