#!/usr/bin/env python3
"""Stdlib-only external verifier/snapshot builder for v15c-r7 local release.

The package explicitly does not authorize observer execution.  This bootstrap
therefore authenticates through caller-supplied file descriptors and builds an
exact private read-only snapshot, but has no command that enters a worker,
model, GPU, route, decoder, or trainer.
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


RELEASE_SCHEMA = "bernini-source-object-proposal-role-v15c-r7-local-release"
RELEASE_TAG = "v15c-r7-local"
RELEASE_RELATIVE_PATH = (
    "methods/bernini_action_editing/assets/"
    "e00_source_sam2_proposal_role_probe_v15c_r7_release.json"
)
EXPECTED_CORE_MEMBER_COUNT = 8
EXPECTED_SNAPSHOT_FILE_COUNT = 9
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
RELEASE_KEYS = {
    "schema_version",
    "tag",
    "core_member_count",
    "snapshot_file_count",
    "members",
    "manifest_relative_path",
    "snapshot_policy",
    "observer_only",
    "observer_execution_authorized",
    "human_audit_action",
    "remote_gpu_status",
    "local_evidence_status",
    "route_status",
    "scientific_claim_authorized",
    "route_authorized",
    "decode_authorized",
    "training_authorized",
    "release_sha256",
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
EXPECTED_VERIFY_ENVIRONMENT = {
    "HOME": "/nonexistent/v15c-r7-local-verifier",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": (
        "/vast/users/guangyi.chen/anaconda3/envs/vace/bin:/usr/bin:/bin"
    ),
    "LD_LIBRARY_PATH": (
        "/vast/users/guangyi.chen/anaconda3/envs/vace/lib"
    ),
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTHONHASHSEED": "0",
    "TMPDIR": "/tmp",
    "V15C_R7_LOCAL_VERIFY_ONLY": "1",
}
# Bash may export these process bookkeeping keys while replacing itself with
# the verified Python descriptor.  They are not consulted by this bootstrap.
ALLOWED_SHELL_ENVIRONMENT_KEYS = {"PWD", "SHLVL", "_"}


class BootstrapV15CR7Error(RuntimeError):
    """The external trust root, release closure, or snapshot differs."""


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


def verify_clean_environment() -> Mapping[str, str]:
    """Require the exact verification environment and reject injected keys."""

    observed = {}
    for key, expected in EXPECTED_VERIFY_ENVIRONMENT.items():
        value = os.environ.get(key)
        if value != expected:
            raise BootstrapV15CR7Error(f"trusted environment {key} differs")
        observed[key] = value
    extras = set(os.environ) - set(EXPECTED_VERIFY_ENVIRONMENT)
    if not extras.issubset(ALLOWED_SHELL_ENVIRONMENT_KEYS):
        raise BootstrapV15CR7Error("trusted environment contains extra keys")
    return observed


def require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        raise BootstrapV15CR7Error(f"{label} is not lowercase SHA256")
    return value


def require_exact_keys(value: Any, keys: Sequence[str], label: str) -> None:
    if type(value) is not dict or set(value) != set(keys):
        raise BootstrapV15CR7Error(f"{label} exact keys differ")


def _regular(path: Path, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as error:
        raise BootstrapV15CR7Error(f"{label} is absent") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise BootstrapV15CR7Error(f"{label} is not one regular non-symlink file")
    return info


def descriptor_sha256(descriptor: int) -> str:
    try:
        original = os.lseek(descriptor, 0, os.SEEK_CUR)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise BootstrapV15CR7Error("trusted descriptor is not regular")
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
        os.lseek(descriptor, original, os.SEEK_SET)
    except OSError as error:
        raise BootstrapV15CR7Error("trusted descriptor differs") from error
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
        raise BootstrapV15CR7Error("trusted descriptor changed")
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    before = _regular(path, str(path))
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        digest = descriptor_sha256(descriptor)
        opened = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = _regular(path, str(path))
    if (
        (before.st_dev, before.st_ino)
        != (opened.st_dev, opened.st_ino)
        or (current.st_dev, current.st_ino)
        != (opened.st_dev, opened.st_ino)
    ):
        raise BootstrapV15CR7Error("file path changed while hashing")
    return digest


def _relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise BootstrapV15CR7Error(f"{label} path differs")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or value.endswith(".pyc")
        or "__pycache__" in path.parts
    ):
        raise BootstrapV15CR7Error(f"{label} path differs")
    return value


def _parse_json_bytes(raw: bytes) -> Mapping[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception as error:
        raise BootstrapV15CR7Error("release JSON differs") from error
    if type(value) is not dict:
        raise BootstrapV15CR7Error("release JSON is not one object")
    return value


def _verify_manifest(value: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    require_exact_keys(value, RELEASE_KEYS, "release")
    payload = dict(value)
    claimed = require_sha(payload.pop("release_sha256", None), "release self hash")
    if claimed != object_sha256(payload):
        raise BootstrapV15CR7Error("release self hash differs")
    members = value.get("members")
    if (
        value.get("schema_version") != RELEASE_SCHEMA
        or value.get("tag") != RELEASE_TAG
        or value.get("core_member_count") != EXPECTED_CORE_MEMBER_COUNT
        or value.get("snapshot_file_count") != EXPECTED_SNAPSHOT_FILE_COUNT
        or type(members) is not list
        or len(members) != EXPECTED_CORE_MEMBER_COUNT
        or value.get("manifest_relative_path") != RELEASE_RELATIVE_PATH
        or value.get("snapshot_policy") != SNAPSHOT_POLICY
        or value.get("observer_only") is not True
        or value.get("observer_execution_authorized") is not False
        or value.get("human_audit_action") != "reject_only"
        or value.get("remote_gpu_status") != "REMOTE_GPU_UNAUDITED"
        or value.get("local_evidence_status") != "LOCAL_RELEASE_UNAUDITED"
        or value.get("route_status") != "ROUTE_NO_GO"
        or value.get("scientific_claim_authorized") is not False
        or value.get("route_authorized") is not False
        or value.get("decode_authorized") is not False
        or value.get("training_authorized") is not False
    ):
        raise BootstrapV15CR7Error("release authority differs")
    normalized = []
    paths = []
    for row in members:
        require_exact_keys(row, {"path", "sha256", "size"}, "release member")
        relative = _relative(row.get("path"), "release member")
        digest = require_sha(row.get("sha256"), "release member hash")
        size = row.get("size")
        if type(size) is not int or size <= 0:
            raise BootstrapV15CR7Error("release member size differs")
        normalized.append({"path": relative, "sha256": digest, "size": size})
        paths.append(relative)
    if paths != sorted(paths) or len(set(paths)) != EXPECTED_CORE_MEMBER_COUNT:
        raise BootstrapV15CR7Error("release member registry differs")
    return tuple(normalized)


def verify_release_source(
    root: Path, release_path: Path, expected_release_sha256: str
) -> Mapping[str, Any]:
    expected = require_sha(expected_release_sha256, "external release hash")
    root = root.absolute()
    if root.is_symlink() or not root.is_dir():
        raise BootstrapV15CR7Error("release root differs")
    canonical = root / RELEASE_RELATIVE_PATH
    if release_path.absolute() != canonical.absolute():
        raise BootstrapV15CR7Error("release placement differs")
    descriptor = _open_beneath(root, RELEASE_RELATIVE_PATH)
    try:
        raw = _descriptor_bytes(descriptor)
    finally:
        os.close(descriptor)
    if hashlib.sha256(raw).hexdigest() != expected:
        raise BootstrapV15CR7Error("release file hash differs")
    manifest = _parse_json_bytes(raw)
    members = _verify_manifest(manifest)
    for row in members:
        descriptor = _open_beneath(root, row["path"])
        try:
            info = os.fstat(descriptor)
            member_digest = descriptor_sha256(descriptor)
        finally:
            os.close(descriptor)
        if info.st_size != row["size"] or member_digest != row["sha256"]:
            raise BootstrapV15CR7Error("release member bytes differ")
    return manifest


def _open_beneath(root: Path, relative: str) -> int:
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
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
        raise BootstrapV15CR7Error("secure-open member is not regular")
    return result


def _descriptor_bytes(descriptor: int) -> bytes:
    try:
        original = os.lseek(descriptor, 0, os.SEEK_CUR)
        before = os.fstat(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        os.lseek(descriptor, original, os.SEEK_SET)
    except OSError as error:
        raise BootstrapV15CR7Error("release descriptor read differs") from error
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
        raise BootstrapV15CR7Error("release descriptor changed while reading")
    return b"".join(chunks)


def _copy_descriptor(descriptor: int, destination: Path, expected: Mapping[str, Any]) -> None:
    before = os.fstat(descriptor)
    if before.st_size != expected["size"] or descriptor_sha256(descriptor) != expected["sha256"]:
        raise BootstrapV15CR7Error("source descriptor differs before copy")
    os.lseek(descriptor, 0, os.SEEK_SET)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    target = os.open(destination, flags, 0o400)
    try:
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            view = memoryview(block)
            while view:
                written = os.write(target, view)
                if written <= 0:
                    raise BootstrapV15CR7Error("snapshot member short write")
                view = view[written:]
        os.fsync(target)
    finally:
        os.close(target)
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
        raise BootstrapV15CR7Error("source descriptor changed during copy")
    os.chmod(destination, 0o400)
    if file_sha256(destination) != expected["sha256"]:
        raise BootstrapV15CR7Error("snapshot copied bytes differ")


def _scan_tree(root: Path) -> tuple[set[str], set[str]]:
    files = set()
    directories = {"."}
    for current_text, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        current = Path(current_text)
        for name in directory_names:
            path = current / name
            info = path.lstat()
            relative = path.relative_to(root).as_posix()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise BootstrapV15CR7Error("snapshot contains a symlink/non-directory")
            if name == "__pycache__":
                raise BootstrapV15CR7Error("snapshot contains __pycache__")
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
                raise BootstrapV15CR7Error(
                    "snapshot contains symlink/non-file/pyc/hardlink"
                )
            files.add(relative)
    return files, directories


def verify_snapshot(
    snapshot: Path,
    manifest: Mapping[str, Any],
    expected_release_sha256: str,
    *,
    sealed: bool,
) -> Mapping[str, Mapping[str, Any]]:
    expected_files = {
        RELEASE_RELATIVE_PATH,
        *(row["path"] for row in manifest["members"]),
    }
    expected_directories = {"."}
    for relative in expected_files:
        parent = PurePosixPath(relative).parent
        while parent.as_posix() != ".":
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    files, directories = _scan_tree(snapshot)
    if files != expected_files or directories != expected_directories:
        raise BootstrapV15CR7Error("snapshot exact tree differs")
    expected_rows = {
        row["path"]: {"sha256": row["sha256"], "size": row["size"]}
        for row in manifest["members"]
    }
    release = snapshot / RELEASE_RELATIVE_PATH
    expected_rows[RELEASE_RELATIVE_PATH] = {
        "sha256": require_sha(expected_release_sha256, "snapshot release hash"),
        "size": _regular(release, "snapshot release").st_size,
    }
    mode = 0o500 if sealed else 0o700
    observed = {}
    for relative in sorted(expected_files):
        path = snapshot / relative
        info = _regular(path, relative)
        digest = file_sha256(path)
        if (
            info.st_size != expected_rows[relative]["size"]
            or digest != expected_rows[relative]["sha256"]
            or stat.S_IMODE(info.st_mode) != 0o400
            or info.st_nlink != 1
        ):
            raise BootstrapV15CR7Error("snapshot member differs")
        observed[relative] = {"sha256": digest, "size": info.st_size}
    for relative in expected_directories:
        path = snapshot if relative == "." else snapshot / relative
        if stat.S_IMODE(path.lstat().st_mode) != mode:
            raise BootstrapV15CR7Error("snapshot directory mode differs")
    return observed


def _observation(
    snapshot: Path,
    manifest: Mapping[str, Any],
    release_sha256: str,
    *,
    sealed: bool,
) -> Mapping[str, Any]:
    rows = verify_snapshot(snapshot, manifest, release_sha256, sealed=sealed)
    _, directories = _scan_tree(snapshot)
    return {
        "observation_scope": (
            "current_state_reverified_at_runtime"
            if sealed
            else "historical_observation_before_sealing"
        ),
        "directory_mode": "0500" if sealed else "0700",
        "member_mode": "0400",
        "directory_count": len(directories),
        "file_count": len(rows),
        "exact_tree_verified": True,
        "extras_symlinks_pyc_absent": True,
        "single_link_regular_files_verified": True,
        **({"files": rows} if sealed else {}),
    }


def materialize_snapshot(
    root: Path,
    release_path: Path,
    destination: Path,
    manifest: Mapping[str, Any],
    expected_release_sha256: str,
) -> Mapping[str, Any]:
    verified = verify_release_source(root, release_path, expected_release_sha256)
    if verified != manifest:
        raise BootstrapV15CR7Error("release changed before snapshot construction")
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise BootstrapV15CR7Error("snapshot destination is not fresh")
    destination.mkdir(mode=0o700, parents=True)
    rows = list(manifest["members"]) + [
        {
            "path": RELEASE_RELATIVE_PATH,
            "sha256": expected_release_sha256,
            "size": _regular(release_path, "release").st_size,
        }
    ]
    try:
        for row in sorted(rows, key=lambda item: item["path"]):
            target = destination / row["path"]
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            descriptor = _open_beneath(root, row["path"])
            try:
                _copy_descriptor(descriptor, target, row)
            finally:
                os.close(descriptor)
        _, construction_directories = _scan_tree(destination)
        for relative in construction_directories:
            path = destination if relative == "." else destination / relative
            os.chmod(path, 0o700)
        construction = _observation(
            destination, manifest, expected_release_sha256, sealed=False
        )
        _, directories = _scan_tree(destination)
        for relative in sorted(
            directories,
            key=lambda item: len(PurePosixPath(item).parts),
            reverse=True,
        ):
            path = destination if relative == "." else destination / relative
            os.chmod(path, 0o500)
        sealed = _observation(
            destination, manifest, expected_release_sha256, sealed=True
        )
    except Exception:
        # A failed construction remains as explicit forensic evidence.  This
        # bootstrap never recursively removes caller-visible data.
        raise
    return {"construction_phase": construction, "sealed_phase": sealed}


def verify_trusted_invocation(
    *,
    bootstrap_fd: int,
    expected_bootstrap_sha256: str,
    python_fd: int,
    python_authority: Path,
    expected_python_sha256: str,
) -> Mapping[str, Any]:
    environment = verify_clean_environment()
    bootstrap_path = Path(__file__).resolve(strict=True)
    python_path = python_authority.resolve(strict=True)
    bootstrap_info = _regular(bootstrap_path, "bootstrap")
    python_info = _regular(python_path, "Python authority")
    bootstrap_fd_info = os.fstat(bootstrap_fd)
    python_fd_info = os.fstat(python_fd)
    if (
        descriptor_sha256(bootstrap_fd)
        != require_sha(expected_bootstrap_sha256, "external bootstrap hash")
        or file_sha256(bootstrap_path) != expected_bootstrap_sha256
        or (bootstrap_fd_info.st_dev, bootstrap_fd_info.st_ino)
        != (bootstrap_info.st_dev, bootstrap_info.st_ino)
        or descriptor_sha256(python_fd)
        != require_sha(expected_python_sha256, "external Python hash")
        or file_sha256(python_path) != expected_python_sha256
        or (python_fd_info.st_dev, python_fd_info.st_ino)
        != (python_info.st_dev, python_info.st_ino)
        or Path(sys.executable).resolve(strict=True) != python_path
        or sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or sys.flags.no_user_site != 1
        or sys.flags.dont_write_bytecode != 1
    ):
        raise BootstrapV15CR7Error("trusted bootstrap/Python FD authority differs")
    return {
        "bootstrap_path": str(bootstrap_path),
        "bootstrap_sha256": expected_bootstrap_sha256,
        "bootstrap_fd": bootstrap_fd,
        "python_path": str(python_path),
        "python_sha256": expected_python_sha256,
        "python_fd": python_fd,
        "startup_flags": ["-I", "-S", "-B"],
        "environment": environment,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trusted-bootstrap-fd", required=True, type=int)
    parser.add_argument("--expected-bootstrap-sha256", required=True)
    parser.add_argument("--trusted-python-fd", required=True, type=int)
    parser.add_argument("--python-authority", required=True, type=Path)
    parser.add_argument("--expected-python-sha256", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify-release")
    verify.add_argument("--root", required=True, type=Path)
    verify.add_argument("--release-manifest", required=True, type=Path)
    verify.add_argument("--release-sha256", required=True)
    prepare = commands.add_parser("prepare-local-snapshot")
    prepare.add_argument("--root", required=True, type=Path)
    prepare.add_argument("--release-manifest", required=True, type=Path)
    prepare.add_argument("--release-sha256", required=True)
    prepare.add_argument("--snapshot", required=True, type=Path)
    args = parser.parse_args()
    trust = verify_trusted_invocation(
        bootstrap_fd=args.trusted_bootstrap_fd,
        expected_bootstrap_sha256=args.expected_bootstrap_sha256,
        python_fd=args.trusted_python_fd,
        python_authority=args.python_authority,
        expected_python_sha256=args.expected_python_sha256,
    )
    manifest = verify_release_source(
        args.root, args.release_manifest, args.release_sha256
    )
    if args.command == "verify-release":
        result = {
            "status": "LOCAL_RELEASE_VERIFIED_OBSERVER_EXECUTION_NOT_AUTHORIZED",
            "trust": trust,
            "release_file_sha256": args.release_sha256,
            "release_internal_sha256": manifest["release_sha256"],
            "observer_execution_authorized": False,
            "remote_gpu_status": "REMOTE_GPU_UNAUDITED",
            "route_authorized": False,
        }
    elif args.command == "prepare-local-snapshot":
        observations = materialize_snapshot(
            args.root,
            args.release_manifest,
            args.snapshot,
            manifest,
            args.release_sha256,
        )
        result = {
            "status": "LOCAL_SNAPSHOT_SEALED_OBSERVER_EXECUTION_NOT_AUTHORIZED",
            "trust": trust,
            "release_file_sha256": args.release_sha256,
            "release_internal_sha256": manifest["release_sha256"],
            "snapshot": observations,
            "observer_execution_authorized": False,
            "remote_gpu_status": "REMOTE_GPU_UNAUDITED",
            "route_authorized": False,
        }
    else:  # pragma: no cover
        raise BootstrapV15CR7Error("unknown command")
    result["receipt_sha256"] = object_sha256(result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
