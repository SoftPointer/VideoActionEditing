#!/usr/bin/env python3
"""Orchestrate the preregistered ELAL-3 C2 three-holder stage chain.

Run this program on the AUH login node.  It treats the login and each compute
node as different filesystems: source release, exact16 bundle and every later
control receipt are transported through the corresponding holder ``srun``
stdin, then rehashed and sealed inside that compute step.  No login-side
``/vast`` path is passed to a trainer.

The stage order is fixed and fail-closed:

1. stream the same release/bundle to jobs 141620, 141618 and 141619;
2. run exact-three WORLD8 no-update preflights concurrently;
3. collect and redistribute exact-three preflight receipts; generate the A/B gate on
   node226 with the held-FD gate controller;
4. run exact-three fresh one-update engineering tests concurrently;
5. physically replay each fresh1 receipt/checkpoint tree on its origin node,
   collect and redistribute exact-three portable attestations, then generate
   the exact-three fresh1 acceptance gate on node226;
6. run exact-three fresh exact10 jobs concurrently.  No checkpoint from
   fresh1 is accepted or referenced;
7. physically replay each exact10 receipt/checkpoint tree on its origin node.
   Only exact-three portable postflight attestations authorize completion.

Any transport, receipt, validator, process or gate failure stops before the
next stage.  The controller is PENDING until the final trainer/release hashes
are mechanically frozen.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, BinaryIO, Iterator, Mapping, NoReturn, Optional, Sequence


SCHEMA_VERSION = "bernini-elal3-c2-role-binding-training-controller-v1"
NODE_ROOT = Path(
    "/tmp/elal3-c2-role-e6ccc7c5-v10"
)
PYTHON_BIN = Path("/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12")
LAUNCHER_BASENAME = "auh_run_elal3_c2_role_binding_stage_v1.sh"
ORIGIN_VERIFIER_BASENAME = "elal3_c2_origin_receipt_verifier_v1.py"
GATE_CONTROLLER_BASENAME = "elal3_c2_staged_gate_controller_v1.py"

# Final release literals.  A PENDING value prevents even the first srun.
ARCHIVE_SHA256: Optional[str] = (
    "e6ccc7c55c50d03d6df57cb8a9a3d85bb2dc1b0977ef1905105944757b720e61"
)
ARCHIVE_SIZE: Optional[int] = 1_054_720
MANIFEST_SHA256: Optional[str] = (
    "4e95f179a6274bca5611a0532402a71d23db15822e14b85d5111309f59246f15"
)
MANIFEST_SIZE: Optional[int] = 8_830
LAUNCHER_SHA256: Optional[str] = (
    "e1e1bad1581f01952872d8742517f9209f87fd5dfeec69fe0a38b27ba2e0ec98"
)
LAUNCHER_SIZE: Optional[int] = 33_116
TRAINER_SHA256: Optional[str] = (
    "63f35b39e60dbf2c1dd1dcecb29393c04d9f00fd0833054e7d81d40790dfe4ce"
)
TRAINER_SIZE: Optional[int] = 447_559
GATE_CONTROLLER_SHA256: Optional[str] = (
    "f4e931b1f50473a9391aa7e7e68464213aaf43e85cc5a8bee792c380c2035af1"
)
GATE_CONTROLLER_SIZE: Optional[int] = 28_107
ORIGIN_VERIFIER_SHA256: Optional[str] = (
    "07122fd71e8f170b5a50761255a664ac17fc2c66b7b8970a1c113bc8d5e605c1"
)
ORIGIN_VERIFIER_SIZE: Optional[int] = 24_717

LATENT_BUNDLE_SHA256 = "b31d5e1594a112f965a3cebd527d5189a561e2cc2d83cfe94014872ffb94d1b8"
LATENT_BUNDLE_SIZE = 78_277_976
LATENT_RECEIPT_SHA256 = "a1ca0d3c015a54d61c8a71d00bc78688dab20d6592ba30ddf73b0ea18e7d70ee"
LATENT_RECEIPT_SIZE = 52_752
LATENT_RECEIPT_DIGEST = "225255f5ada73848686b240c4a53001c9dd65b1373da2b293c2da8c2ec14f35d"

PLACEMENTS = (
    ("A_duplicate_control", "141620", "auh7-1b-gpu-226", 20260821),
    ("B_paired_role", "141618", "auh7-1b-gpu-249", 20260821),
    ("B_paired_role_replica", "141619", "auh7-1b-gpu-257", 20260822),
)

_SHA = re.compile(r"^[0-9a-f]{64}$")


class ELAL3C2DeploymentError(RuntimeError):
    """The node-local staged deployment failed closed."""


def fail(message: str) -> NoReturn:
    raise ELAL3C2DeploymentError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ELAL3C2DeploymentError(
            "value is not finite canonical ASCII JSON"
        ) from error


def object_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def require_sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        fail(f"{label} is PENDING or invalid")
    return value


def require_release_literals() -> None:
    for value, label in (
        (ARCHIVE_SHA256, "archive SHA"),
        (MANIFEST_SHA256, "manifest SHA"),
        (LAUNCHER_SHA256, "launcher SHA"),
        (TRAINER_SHA256, "trainer SHA"),
        (GATE_CONTROLLER_SHA256, "gate controller SHA"),
        (ORIGIN_VERIFIER_SHA256, "origin verifier SHA"),
    ):
        require_sha(value, label=label)
    for value, label in (
        (ARCHIVE_SIZE, "archive size"),
        (MANIFEST_SIZE, "manifest size"),
        (LAUNCHER_SIZE, "launcher size"),
        (TRAINER_SIZE, "trainer size"),
        (GATE_CONTROLLER_SIZE, "gate controller size"),
        (ORIGIN_VERIFIER_SIZE, "origin verifier size"),
    ):
        if type(value) is not int or value <= 0:
            fail(f"{label} is PENDING")


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_rdev,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _directory_identity(value: os.stat_result) -> tuple[int, ...]:
    """Stable directory identity; member creation may change size/timestamps."""

    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_rdev,
    )


def _hash_held_descriptor(descriptor: int) -> tuple[str, int]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    count = 0
    while True:
        block = os.read(descriptor, 1 << 20)
        if not block:
            return digest.hexdigest(), count
        digest.update(block)
        count += len(block)


@contextmanager
def held_stable_binding(
    path: Path,
    *,
    expected_sha256: str,
    expected_size: int,
    expected_mode: Optional[int],
    label: str,
) -> Iterator[tuple[int, Mapping[str, Any]]]:
    """Hold every parent and one file FD across caller use and final replay."""

    require_sha(expected_sha256, label=f"{label} expected SHA")
    if (
        not path.is_absolute()
        or path == Path(path.anchor)
        or any(part in {"", ".", ".."} for part in path.parts[1:])
    ):
        fail(f"{label} is not an absolute no-follow path")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    root_path = Path(path.anchor)
    root_named_before = root_path.lstat()
    if not stat.S_ISDIR(root_named_before.st_mode):
        fail(f"{label} filesystem root differs")
    root_descriptor = os.open(root_path, directory_flags)
    held: list[int] = [root_descriptor]
    parents: list[tuple[Path, os.stat_result, int]] = [
        (root_path, root_named_before, root_descriptor)
    ]
    try:
        if _directory_identity(root_named_before) != _directory_identity(
            os.fstat(root_descriptor)
        ):
            fail(f"{label} filesystem root identity differs")
        parent_descriptor = root_descriptor
        absolute_parent = root_path
        for component in path.parts[1:-1]:
            named = os.stat(
                component, dir_fd=parent_descriptor, follow_symlinks=False
            )
            child_descriptor = os.open(
                component, directory_flags, dir_fd=parent_descriptor
            )
            held.append(child_descriptor)
            child = os.fstat(child_descriptor)
            absolute_parent = absolute_parent / component
            if (
                not stat.S_ISDIR(named.st_mode)
                or _directory_identity(named) != _directory_identity(child)
                or _directory_identity(absolute_parent.lstat())
                != _directory_identity(child)
            ):
                fail(f"{label} held-openat parent chain differs")
            parents.append((absolute_parent, named, child_descriptor))
            parent_descriptor = child_descriptor
        basename = path.parts[-1]
        named_before = os.stat(
            basename, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if not stat.S_ISREG(named_before.st_mode) or named_before.st_nlink != 1:
            fail(f"{label} is not one nlink1 regular file")
        if named_before.st_size != expected_size:
            fail(f"{label} byte size differs")
        if (
            expected_mode is not None
            and stat.S_IMODE(named_before.st_mode) != expected_mode
        ):
            fail(f"{label} mode differs")
        descriptor = os.open(basename, file_flags, dir_fd=parent_descriptor)
        held.append(descriptor)
        before = os.fstat(descriptor)
        first_sha, first_size = _hash_held_descriptor(descriptor)
        second_sha, second_size = _hash_held_descriptor(descriptor)
        if (
            _identity(named_before) != _identity(before)
            or (first_sha, first_size) != (second_sha, second_size)
            or first_sha != expected_sha256
            or first_size != expected_size
        ):
            fail(f"{label} initial held-FD replay differs")
        binding = {
            "path": str(path),
            "sha256": first_sha,
            "size": first_size,
            "mode": stat.S_IMODE(named_before.st_mode),
            "nlink": named_before.st_nlink,
        }
        yield descriptor, binding
        before_final = os.fstat(descriptor)
        final_first = _hash_held_descriptor(descriptor)
        final_second = _hash_held_descriptor(descriptor)
        after = os.fstat(descriptor)
        named_after = os.stat(
            basename, dir_fd=parent_descriptor, follow_symlinks=False
        )
        absolute_after = path.lstat()
        for absolute, parent_before, parent_fd in parents:
            if (
                _directory_identity(parent_before)
                != _directory_identity(os.fstat(parent_fd))
                or _directory_identity(absolute.lstat())
                != _directory_identity(os.fstat(parent_fd))
            ):
                fail(f"{label} held-openat parent final replay differs")
        if (
            _identity(named_before) != _identity(before_final)
            or _identity(before_final) != _identity(after)
            or _identity(after) != _identity(named_after)
            or _identity(named_after) != _identity(absolute_after)
            or final_first != final_second
            or final_first != (expected_sha256, expected_size)
        ):
            fail(f"{label} final held-FD replay differs")
    finally:
        for held_descriptor in reversed(held):
            os.close(held_descriptor)


def stable_binding(
    path: Path,
    *,
    expected_sha256: str,
    expected_size: int,
    expected_mode: Optional[int],
    label: str,
) -> Mapping[str, Any]:
    with held_stable_binding(
        path,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
        expected_mode=expected_mode,
        label=label,
    ) as (_descriptor, binding):
        result = dict(binding)
    return result


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o444
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.type = tarfile.REGTYPE
    return info


def build_stream_tar(
    output: Path,
    rows: Sequence[tuple[str, Path, str, int]],
) -> Mapping[str, Any]:
    """Build a deterministic transport tar without trusting source names."""

    if not output.is_absolute() or output.exists() or output.is_symlink():
        fail("transport tar output must be fresh and absolute")
    names = [row[0] for row in rows]
    if names != sorted(names, key=lambda item: item.encode("ascii")) or len(set(names)) != len(names):
        fail("transport rows must be uniquely ASCII-sorted")
    with ExitStack() as stack:
        bound_rows: list[tuple[str, int, int]] = []
        for name, path, sha, size in rows:
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts or not pure.parts:
                fail(f"unsafe transport member: {name}")
            descriptor, _binding = stack.enter_context(
                held_stable_binding(
                    path,
                    expected_sha256=sha,
                    expected_size=size,
                    expected_mode=None,
                    label=f"transport input {name}",
                )
            )
            bound_rows.append((name, descriptor, size))
        with output.open("xb") as raw_output:
            with tarfile.open(
                fileobj=raw_output, mode="w", format=tarfile.USTAR_FORMAT
            ) as archive:
                for name, descriptor, size in bound_rows:
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    with os.fdopen(os.dup(descriptor), "rb", closefd=True) as source:
                        archive.addfile(_tar_info(name, size), source)
            raw_output.flush()
            os.fsync(raw_output.fileno())
    os.chmod(output, 0o444)
    raw_sha = hashlib.sha256()
    with output.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            raw_sha.update(block)
    return {
        "path": str(output),
        "sha256": raw_sha.hexdigest(),
        "size": output.stat().st_size,
        "members": [
            {"path": name, "sha256": sha, "size": size, "mode": "0444"}
            for name, _, sha, size in rows
        ],
    }


# The receiver is transported as an argv literal, while the archive itself is
# the step's stdin.  It performs safe create-only extraction and then creates a
# separate persistent 0444 tree solely for held-FD gate generation.
ASSET_RECEIVER = r'''
import hashlib, json, os, stat, sys, tarfile
from pathlib import Path, PurePosixPath
root = Path(sys.argv[1]); expected_rows = json.loads(sys.argv[2])
def reject(message): raise SystemExit("node asset receiver rejected: " + message)
identity = lambda value: (value.st_dev,value.st_ino,value.st_mode,value.st_nlink,value.st_uid,value.st_gid,value.st_rdev,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
def replay_file(path, row, label):
    named = path.lstat()
    if not stat.S_ISREG(named.st_mode) or stat.S_IMODE(named.st_mode) != 0o444 or named.st_nlink != 1 or named.st_size != row["size"]: reject(label + " type/mode/link/size differs")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(fd)
        def one_pass():
            digest = hashlib.sha256(); count = 0
            while True:
                block = os.read(fd, 1 << 20)
                if not block: return digest.hexdigest(), count
                digest.update(block); count += len(block)
        first = one_pass(); os.lseek(fd, 0, os.SEEK_SET); second = one_pass(); after = os.fstat(fd)
    finally: os.close(fd)
    named_after = path.lstat()
    if first != second or first != (row["sha256"], row["size"]) or identity(named) != identity(before) or identity(before) != identity(after) or identity(after) != identity(named_after): reject(label + " held double replay differs")
if not root.is_absolute() or root == Path("/") or root.exists() or root.is_symlink(): reject("node root is not fresh/absolute")
root.mkdir(mode=0o700); assets = root / "assets"; assets.mkdir(mode=0o700)
seen = []
with tarfile.open(fileobj=sys.stdin.buffer, mode="r|*") as archive:
    for member in archive:
        if member.name not in expected_rows or member.name in seen: reject("member closure differs")
        pure = PurePosixPath(member.name)
        if pure.is_absolute() or ".." in pure.parts or not member.isreg() or member.mode != 0o444 or member.uid != 0 or member.gid != 0 or member.mtime != 0: reject("unsafe member")
        row = expected_rows[member.name]; source = archive.extractfile(member)
        target = root.joinpath(*pure.parts); target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(); count = 0
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o400)
        try:
            while True:
                block = source.read(1 << 20)
                if not block: break
                view = memoryview(block)
                while view:
                    written = os.write(fd, view)
                    if written <= 0: reject("write stalled")
                    view = view[written:]
                digest.update(block); count += len(block)
            os.fchmod(fd, 0o444); os.fsync(fd)
        finally: os.close(fd)
        info = target.lstat()
        if digest.hexdigest() != row["sha256"] or count != row["size"] or stat.S_IMODE(info.st_mode) != 0o444 or info.st_nlink != 1: reject("node-local member binding differs")
        seen.append(member.name)
if set(seen) != set(expected_rows): reject("transport exact closure differs")
for name in sorted(seen): replay_file(root.joinpath(*PurePosixPath(name).parts), expected_rows[name], "transport asset " + name)
manifest = assets / "source.manifest.json"; source_archive = assets / "source.tar"
manifest_raw = manifest.read_bytes(); value = json.loads(manifest_raw)
canonical = lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
if manifest_raw != canonical(value) + b"\n": reject("source manifest canonical form differs")
unsigned = dict(value); stored = unsigned.pop("manifest_digest", None)
if stored != hashlib.sha256(canonical(unsigned)).hexdigest(): reject("source manifest digest differs")
archive_raw_sha = hashlib.sha256(); archive_size = 0
with source_archive.open("rb") as stream:
    for block in iter(lambda: stream.read(1 << 20), b""): archive_raw_sha.update(block); archive_size += len(block)
if archive_raw_sha.hexdigest() != value.get("archive_sha256") or archive_size != value.get("archive_size"): reject("source archive/manifest binding differs")
runtime = root / "gate-runtime"; runtime.mkdir(mode=0o700)
rows = value.get("files"); expected = {row.get("path"): row for row in rows if isinstance(row, dict)}
if len(expected) != value.get("file_count") or len(expected) != len(rows): reject("source file closure differs")
with tarfile.open(source_archive, "r:") as archive:
    members = archive.getmembers(); names = [member.name for member in members]
    if names != sorted(names, key=lambda item: item.encode("ascii")) or set(names) != set(expected): reject("source archive closure/order differs")
    for member in members:
        pure = PurePosixPath(member.name); row = expected[member.name]
        if pure.is_absolute() or ".." in pure.parts or not member.isreg() or member.mode != 0o444 or member.uid != 0 or member.gid != 0 or member.mtime != 0: reject("unsafe source member")
        raw = archive.extractfile(member).read()
        if hashlib.sha256(raw).hexdigest() != row["sha256"] or len(raw) != row["size"] or row["mode"] != "0444": reject("source member binding differs")
        target = runtime.joinpath(*pure.parts); target.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o400)
        try:
            view = memoryview(raw)
            while view:
                written = os.write(fd, view)
                if written <= 0: reject("source member write stalled")
                view = view[written:]
            os.fchmod(fd, 0o444); os.fsync(fd)
        finally: os.close(fd)
for walk_root, directories, files in os.walk(runtime, topdown=False):
    for name in directories: os.chmod(Path(walk_root) / name, 0o555)
os.chmod(runtime, 0o555); os.chmod(assets, 0o555)
for name, row in expected.items(): replay_file(runtime.joinpath(*PurePosixPath(name).parts), row, "runtime source " + name)
(root / "controls" / "preflight").mkdir(parents=True, mode=0o700)
(root / "controls" / "fresh1").mkdir(parents=True, mode=0o700)
(root / "controls" / "fresh1-attestations").mkdir(parents=True, mode=0o700)
(root / "controls" / "exact10-attestations").mkdir(parents=True, mode=0o700)
(root / "controls" / "gates").mkdir(parents=True, mode=0o700)
(root / "runs").mkdir(mode=0o700)
print(json.dumps({"status":"NODE_ASSETS_STAGED","node":os.uname().nodename.split(".")[0],"member_count":len(seen)}, sort_keys=True, separators=(",", ":")))
'''


CONTROL_RECEIVER = r'''
import hashlib, json, os, stat, sys, tarfile
from pathlib import Path, PurePosixPath
root = Path(sys.argv[1]); expected = json.loads(sys.argv[2])
def reject(message): raise SystemExit("node control receiver rejected: " + message)
if not root.is_absolute() or not root.is_dir() or root.is_symlink(): reject("node root differs")
identity = lambda value: (value.st_dev,value.st_ino,value.st_mode,value.st_nlink,value.st_uid,value.st_gid,value.st_rdev,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
dir_identity = lambda value: (value.st_dev,value.st_ino,value.st_mode,value.st_uid,value.st_gid,value.st_rdev)
directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_DIRECTORY", 0)
file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
root_named = root.lstat(); root_fd = os.open(root, directory_flags)
if not stat.S_ISDIR(root_named.st_mode) or dir_identity(root_named) != dir_identity(os.fstat(root_fd)): reject("node root held identity differs")
def held_parent(pure):
    held = []; parent_fd = root_fd
    for part in pure.parts[:-1]:
        named = os.stat(part, dir_fd=parent_fd, follow_symlinks=False)
        child_fd = os.open(part, directory_flags, dir_fd=parent_fd); held.append(child_fd)
        if not stat.S_ISDIR(named.st_mode) or dir_identity(named) != dir_identity(os.fstat(child_fd)): reject("control parent held-openat identity differs")
        parent_fd = child_fd
    return held, parent_fd, pure.parts[-1]
def read_pass(fd):
    chunks = []
    while True:
        block = os.read(fd, 1 << 20)
        if not block: return b"".join(chunks)
        chunks.append(block)
seen = []
try:
    with tarfile.open(fileobj=sys.stdin.buffer, mode="r|*") as archive:
        for member in archive:
            if member.name not in expected or member.name in seen: reject("control closure differs")
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts) or not member.isreg() or member.mode != 0o444 or member.uid != 0 or member.gid != 0 or member.mtime != 0: reject("unsafe control member")
            source = archive.extractfile(member)
            if source is None: reject("control payload absent")
            raw = source.read(); row = expected[member.name]
            if set(row) != {"sha256", "size"} or hashlib.sha256(raw).hexdigest() != row["sha256"] or len(raw) != row["size"]: reject("control payload differs")
            held, parent_fd, basename = held_parent(pure)
            try:
                try: named = os.stat(basename, dir_fd=parent_fd, follow_symlinks=False)
                except FileNotFoundError: named = None
                if named is not None:
                    if not stat.S_ISREG(named.st_mode) or stat.S_IMODE(named.st_mode) != 0o444 or named.st_nlink != 1 or named.st_size != len(raw): reject("existing control type/mode/link/size differs")
                    fd = os.open(basename, file_flags, dir_fd=parent_fd)
                    try:
                        before = os.fstat(fd); first = read_pass(fd); os.lseek(fd, 0, os.SEEK_SET); second = read_pass(fd); after = os.fstat(fd); named_after = os.stat(basename, dir_fd=parent_fd, follow_symlinks=False)
                        if first != raw or second != raw or identity(named) != identity(before) or identity(before) != identity(after) or identity(after) != identity(named_after): reject("existing control held replay differs")
                    finally: os.close(fd)
                else:
                    fd = os.open(basename, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0), 0o400, dir_fd=parent_fd)
                    try:
                        view = memoryview(raw)
                        while view:
                            written = os.write(fd, view)
                            if written <= 0: reject("control write stalled")
                            view = view[written:]
                        os.fchmod(fd, 0o444); os.fsync(fd); created = os.fstat(fd)
                    finally: os.close(fd)
                    named_after = os.stat(basename, dir_fd=parent_fd, follow_symlinks=False)
                    if not stat.S_ISREG(created.st_mode) or stat.S_IMODE(created.st_mode) != 0o444 or created.st_nlink != 1 or identity(created) != identity(named_after): reject("created control held replay differs")
            finally:
                for descriptor in reversed(held): os.close(descriptor)
            seen.append(member.name)
finally:
    root_after = root.lstat(); held_root_after = os.fstat(root_fd); os.close(root_fd)
    if dir_identity(root_named) != dir_identity(root_after) or dir_identity(root_after) != dir_identity(held_root_after): reject("node root final identity differs")
if set(seen) != set(expected): reject("control exact closure differs")
print(json.dumps({"status":"NODE_CONTROLS_STAGED","node":os.uname().nodename.split(".")[0],"member_count":len(seen)}, sort_keys=True, separators=(",", ":")))
'''


SEALED_READER = r'''
import json, os, stat, sys
from pathlib import Path
root = Path(sys.argv[1]); path = Path(sys.argv[2]); maximum = int(sys.argv[3])
def reject(message): raise SystemExit("sealed reader rejected: " + message)
if not root.is_absolute() or not path.is_absolute() or root == Path("/"): reject("root/path ABI differs")
try: relative = path.relative_to(root)
except ValueError: reject("path escapes node root")
if not relative.parts: reject("file basename is absent")
identity = lambda value: (value.st_dev,value.st_ino,value.st_mode,value.st_nlink,value.st_uid,value.st_gid,value.st_rdev,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_DIRECTORY", 0)
root_named_before = root.lstat()
if not stat.S_ISDIR(root_named_before.st_mode) or stat.S_ISLNK(root_named_before.st_mode): reject("node root identity differs")
root_fd = os.open(root, directory_flags); held = [root_fd]
try:
    if identity(root_named_before) != identity(os.fstat(root_fd)): reject("node root named/fd identity differs")
    parent_fd = root_fd
    for part in relative.parts[:-1]:
        named = os.stat(part, dir_fd=parent_fd, follow_symlinks=False)
        child_fd = os.open(part, directory_flags, dir_fd=parent_fd); held.append(child_fd)
        child = os.fstat(child_fd)
        if not stat.S_ISDIR(named.st_mode) or identity(named) != identity(child): reject("held openat parent chain differs")
        parent_fd = child_fd
    basename = relative.parts[-1]
    named_before = os.stat(basename, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISREG(named_before.st_mode) or stat.S_IMODE(named_before.st_mode) != 0o444 or named_before.st_nlink != 1 or named_before.st_size > maximum: reject("file type/mode/link/size differs")
    fd = os.open(basename, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd); held.append(fd)
    before = os.fstat(fd)
    def read_pass():
        chunks = []
        while True:
            block = os.read(fd, 1 << 20)
            if not block: return b"".join(chunks)
            chunks.append(block)
    first = read_pass(); os.lseek(fd, 0, os.SEEK_SET); second = read_pass(); after = os.fstat(fd)
    named_after = os.stat(basename, dir_fd=parent_fd, follow_symlinks=False)
    root_named_after = root.lstat()
    if first != second or identity(named_before) != identity(before) or identity(before) != identity(after) or identity(after) != identity(named_after) or identity(root_named_before) != identity(root_named_after) or identity(root_named_after) != identity(os.fstat(root_fd)): reject("held-FD/openat double-read identity differs")
finally:
    for descriptor in reversed(held): os.close(descriptor)
try: value = json.loads(first)
except Exception as error: reject("JSON parse failed: " + str(error))
canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"
if first != canonical: reject("canonical JSON differs")
sys.stdout.buffer.write(first); sys.stdout.buffer.flush()
'''


def srun_prefix(job_id: str, node: str) -> list[str]:
    return [
        "/usr/bin/srun",
        f"--jobid={job_id}",
        "--overlap",
        "--nodes=1",
        "--ntasks=1",
        f"--nodelist={node}",
        "--kill-on-bad-exit=1",
    ]


def run_command(
    command: Sequence[str],
    *,
    stdin_path: Optional[Path] = None,
    timeout: int,
    label: str,
) -> subprocess.CompletedProcess[bytes]:
    with ExitStack() as stack:
        stdin: BinaryIO | int
        if stdin_path is None:
            stdin = subprocess.DEVNULL
        else:
            stdin = stack.enter_context(stdin_path.open("rb"))
        try:
            process_environment = dict(os.environ)
            process_environment.update(
                {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"}
            )
            completed = subprocess.run(
                list(command),
                stdin=stdin,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=timeout,
                env=process_environment,
            )
        except subprocess.TimeoutExpired as error:
            fail(f"{label} timed out: {error}")
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", "replace")[-4000:]
        fail(f"{label} failed rc={completed.returncode}: {stderr}")
    return completed


def stage_transport_all(
    transport: Path,
    manifest_rows: Sequence[Mapping[str, Any]],
    *,
    receiver: str,
    timeout: int,
    label: str,
) -> None:
    expected = {
        str(row["path"]): {"sha256": row["sha256"], "size": row["size"]}
        for row in manifest_rows
    }
    process_environment = {
        **os.environ,
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "LANG": "C",
    }
    processes: list[
        tuple[str, str, subprocess.Popen[bytes], BinaryIO, BinaryIO, BinaryIO]
    ] = []
    try:
        for _, job_id, node, _ in PLACEMENTS:
            stdin_handle = transport.open("rb")
            stdout_handle = tempfile.TemporaryFile(mode="w+b")
            stderr_handle = tempfile.TemporaryFile(mode="w+b")
            command = [
                *srun_prefix(job_id, node),
                str(PYTHON_BIN),
                "-I",
                "-B",
                "-c",
                receiver,
                str(NODE_ROOT),
                canonical_json_bytes(expected).decode("ascii"),
            ]
            process = subprocess.Popen(
                command,
                stdin=stdin_handle,
                stdout=stdout_handle,
                stderr=stderr_handle,
                env=process_environment,
            )
            processes.append(
                (
                    job_id,
                    node,
                    process,
                    stdin_handle,
                    stdout_handle,
                    stderr_handle,
                )
            )
        failures: list[str] = []
        receipts: list[tuple[str, bytes]] = []
        for job_id, node, process, stdin_handle, stdout_handle, stderr_handle in processes:
            try:
                status_code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.terminate()
                status_code = process.wait(timeout=60)
                failures.append(f"{job_id}/{node} timeout rc={status_code}")
            stdin_handle.close()
            stdout_handle.seek(0)
            stderr_handle.seek(0)
            stdout = stdout_handle.read()
            stderr = stderr_handle.read()
            if status_code != 0:
                failures.append(
                    f"{job_id}/{node} rc={status_code}: "
                    + stderr.decode("utf-8", "replace")[-2000:]
                )
            receipts.append((node, stdout))
        if failures:
            fail(f"{label} stopped: {failures}")
        expected_status = (
            "NODE_ASSETS_STAGED"
            if receiver == ASSET_RECEIVER
            else "NODE_CONTROLS_STAGED"
            if receiver == CONTROL_RECEIVER
            else None
        )
        if expected_status is None:
            fail(f"{label} receiver source is not a release literal")
        for node, raw in receipts:
            try:
                status = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                fail(f"{label} node receipt is not JSON: {error}")
            if (
                raw != canonical_json_bytes(status) + b"\n"
                or set(status) != {"status", "node", "member_count"}
                or status.get("status") != expected_status
                or status.get("node") != node
                or status.get("member_count") != len(expected)
            ):
                fail(f"{label} node receipt closure differs for {node}")
    finally:
        for _, _, process, stdin_handle, stdout_handle, stderr_handle in processes:
            if process.poll() is None:
                process.terminate()
            if not stdin_handle.closed:
                stdin_handle.close()
            stdout_handle.close()
            stderr_handle.close()


def launch_stage_parallel(
    stage: str,
    *,
    own_preflight_sha: Optional[Mapping[str, str]] = None,
    cross_gate_sha: Optional[str] = None,
    fresh1_gate_sha: Optional[str] = None,
    log_root: Path,
) -> None:
    processes: list[tuple[str, str, subprocess.Popen[bytes], BinaryIO, BinaryIO]] = []
    try:
        for arm, job_id, node, _ in PLACEMENTS:
            stdout_handle = (log_root / f"{stage}.{arm}.controller.stdout").open("xb")
            stderr_handle = (log_root / f"{stage}.{arm}.controller.stderr").open("xb")
            env_args = [
                f"ELAL3_C2_STAGE={stage}",
                f"ELAL3_C2_LAUNCHER_SHA256={LAUNCHER_SHA256}",
                f"ELAL3_C2_LAUNCHER_SIZE={LAUNCHER_SIZE}",
            ]
            if stage != "preflight":
                if own_preflight_sha is None or cross_gate_sha is None:
                    fail("non-preflight stage lacks predecessor gate pins")
                env_args.extend(
                    [
                        f"ELAL3_C2_OWN_PREFLIGHT_SHA256={own_preflight_sha[arm]}",
                        f"ELAL3_C2_CROSS_GATE_SHA256={cross_gate_sha}",
                    ]
                )
            if stage == "exact10":
                if fresh1_gate_sha is None:
                    fail("exact10 lacks fresh1 acceptance gate pin")
                env_args.append(f"ELAL3_C2_FRESH1_GATE_SHA256={fresh1_gate_sha}")
            command = [
                *srun_prefix(job_id, node),
                "/usr/bin/env",
                *env_args,
                "/bin/bash",
                str(NODE_ROOT / "assets" / LAUNCHER_BASENAME),
            ]
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                env={
                    **os.environ,
                    "PATH": "/usr/bin:/bin",
                    "LC_ALL": "C",
                    "LANG": "C",
                },
            )
            processes.append((arm, node, process, stdout_handle, stderr_handle))
        failures: list[str] = []
        for arm, node, process, stdout_handle, stderr_handle in processes:
            try:
                status = process.wait(timeout=21_600)
            except subprocess.TimeoutExpired:
                process.terminate()
                status = process.wait(timeout=60)
                failures.append(f"{arm}/{node} timeout rc={status}")
            if status != 0:
                failures.append(f"{arm}/{node} rc={status}")
            stdout_handle.close()
            stderr_handle.close()
        if failures:
            fail(f"{stage} stopped; no later stage authorized: {failures}")
    finally:
        for _, _, process, stdout_handle, stderr_handle in processes:
            if process.poll() is None:
                process.terminate()
            if not stdout_handle.closed:
                stdout_handle.close()
            if not stderr_handle.closed:
                stderr_handle.close()


def _receipt_remote_path(arm: str, stage: str) -> Path:
    name = "PRECHECK_RECEIPT.json" if stage == "preflight" else "TRAINING_RECEIPT.json"
    return NODE_ROOT / "runs" / arm / f"elal3_c2_{stage}" / name


def pull_receipts(stage: str, spool: Path) -> Mapping[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for arm, job_id, node, _ in PLACEMENTS:
        remote = _receipt_remote_path(arm, stage)
        completed = run_command(
            [
                *srun_prefix(job_id, node),
                str(PYTHON_BIN),
                "-I",
                "-B",
                "-c",
                SEALED_READER,
                str(NODE_ROOT),
                str(remote),
                str(16 << 20),
            ],
            timeout=300,
            label=f"pull {stage} receipt {arm}",
        )
        raw = completed.stdout
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            fail(f"{stage}/{arm} receipt is not JSON: {error}")
        if raw != canonical_json_bytes(value) + b"\n":
            fail(f"{stage}/{arm} receipt is not canonical JSON+newline")
        unsigned = dict(value)
        stored = unsigned.pop("receipt_digest", None)
        if stored != object_digest(unsigned):
            fail(f"{stage}/{arm} receipt self digest differs")
        if (
            value.get("arm_id") != arm
            or value.get("node") != node
            or value.get("holder_job_id") != job_id
            or value.get("runner_source_sha256") != TRAINER_SHA256
            or value.get("latent_bundle_sha256") != LATENT_BUNDLE_SHA256
        ):
            fail(f"{stage}/{arm} receipt placement/release binding differs")
        target = spool / f"{stage}.{arm}.json"
        with target.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(target, 0o444)
        result[arm] = {
            "path": target,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size": len(raw),
            "receipt_digest": stored,
        }
    return result


def run_origin_verifiers(
    stage: str,
    *,
    receipts: Mapping[str, Mapping[str, Any]],
    preflight: Mapping[str, Mapping[str, Any]],
    cross_gate: Mapping[str, Any],
    fresh1_gate: Optional[Mapping[str, Any]],
    spool: Path,
) -> Mapping[str, Mapping[str, Any]]:
    """Run the release-pinned physical verifier on each receipt's own node."""

    if stage not in {"fresh1", "exact10"}:
        fail("origin verifier stage differs")
    if stage == "exact10" and fresh1_gate is None:
        fail("exact10 origin verification lacks fresh1 gate")
    method_root = NODE_ROOT / "gate-runtime" / "methods" / "bernini_action_editing"
    verifier = method_root / "elal3_c2_origin_receipt_verifier_v1.py"
    processes: list[
        tuple[str, str, str, subprocess.Popen[bytes], BinaryIO, BinaryIO]
    ] = []
    try:
        for arm, job_id, node, _ in PLACEMENTS:
            argv = [
                *srun_prefix(job_id, node),
                str(PYTHON_BIN),
                "-I",
                "-B",
                str(verifier),
                "--method-root",
                str(method_root),
                "--expected-verifier-source-sha256",
                str(ORIGIN_VERIFIER_SHA256),
                "--expected-verifier-source-size",
                str(ORIGIN_VERIFIER_SIZE),
                "--expected-gate-controller-source-sha256",
                str(GATE_CONTROLLER_SHA256),
                "--expected-gate-controller-source-size",
                str(GATE_CONTROLLER_SIZE),
                "--arm-id",
                arm,
                "--receipt",
                str(_receipt_remote_path(arm, stage)),
                "--expected-receipt-sha256",
                str(receipts[arm]["sha256"]),
                "--expected-receipt-size",
                str(receipts[arm]["size"]),
                "--expected-receipt-digest",
                str(receipts[arm]["receipt_digest"]),
                "--own-preflight-receipt",
                str(NODE_ROOT / "controls" / "preflight" / f"{arm}.json"),
                "--expected-own-preflight-receipt-sha256",
                str(preflight[arm]["sha256"]),
                "--cross-arm-gate",
                str(NODE_ROOT / "controls" / "gates" / "cross_arm_preflight_gate.json"),
                "--expected-cross-arm-gate-sha256",
                str(cross_gate["sha256"]),
                "--stage",
                stage,
            ]
            if stage == "exact10":
                argv.extend(
                    [
                        "--fresh1-acceptance-gate",
                        str(NODE_ROOT / "controls" / "gates" / "fresh1_acceptance_gate.json"),
                        "--expected-fresh1-acceptance-gate-sha256",
                        str(fresh1_gate["sha256"]),
                    ]
                )
            stdout_handle = tempfile.TemporaryFile(mode="w+b")
            stderr_handle = tempfile.TemporaryFile(mode="w+b")
            process = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                env={
                    **os.environ,
                    "PATH": "/usr/bin:/bin",
                    "LC_ALL": "C",
                    "LANG": "C",
                },
            )
            processes.append(
                (arm, job_id, node, process, stdout_handle, stderr_handle)
            )
        expected_origin_binding = {
            "name": ORIGIN_VERIFIER_BASENAME,
            "sha256": str(ORIGIN_VERIFIER_SHA256),
            "size": int(ORIGIN_VERIFIER_SIZE),
            "mode": 0o444,
            "nlink": 1,
        }
        expected_gate_binding = {
            "name": GATE_CONTROLLER_BASENAME,
            "sha256": str(GATE_CONTROLLER_SHA256),
            "size": int(GATE_CONTROLLER_SIZE),
            "mode": 0o444,
            "nlink": 1,
        }
        result: dict[str, Mapping[str, Any]] = {}
        failures: list[str] = []
        placement_by_arm = {
            arm: (job_id, node, seed)
            for arm, job_id, node, seed in PLACEMENTS
        }
        for arm, job_id, node, process, stdout_handle, stderr_handle in processes:
            try:
                status_code = process.wait(timeout=7_200)
            except subprocess.TimeoutExpired:
                process.terminate()
                status_code = process.wait(timeout=60)
                failures.append(f"{arm}/{node} timeout rc={status_code}")
            stdout_handle.seek(0)
            stderr_handle.seek(0)
            raw = stdout_handle.read()
            stderr = stderr_handle.read()
            if status_code != 0:
                failures.append(
                    f"{arm}/{node} rc={status_code}: "
                    + stderr.decode("utf-8", "replace")[-2000:]
                )
                continue
            try:
                value = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                failures.append(f"{arm}/{node} attestation JSON error: {error}")
                continue
            unsigned = dict(value)
            attestation_digest = unsigned.pop("attestation_digest", None)
            expected_status = (
                "FRESH1_ORIGIN_PHYSICAL_REPLAY_PASS"
                if stage == "fresh1"
                else "EXACT10_ORIGIN_PHYSICAL_REPLAY_PASS"
            )
            expected_job, expected_node, expected_seed = placement_by_arm[arm]
            if (
                raw != canonical_json_bytes(value) + b"\n"
                or attestation_digest != object_digest(unsigned)
                or value.get("status") != expected_status
                or value.get("arm_id") != arm
                or value.get("holder_job_id") != expected_job
                or value.get("node") != expected_node
                or value.get("seed") != expected_seed
                or value.get("stage") != stage
                or value.get("receipt_sha256") != receipts[arm]["sha256"]
                or value.get("receipt_size") != receipts[arm]["size"]
                or value.get("receipt_digest") != receipts[arm]["receipt_digest"]
                or value.get("physical_origin_replay_passed") is not True
                or value.get("closed_validator_passed") is not True
                or value.get("runner_source_sha256") != TRAINER_SHA256
                or value.get("latent_bundle_sha256") != LATENT_BUNDLE_SHA256
                or value.get("origin_verifier_binding")
                != expected_origin_binding
                or value.get("gate_controller_binding")
                != expected_gate_binding
                or not isinstance(value.get("portable_checkpoint_tree"), Mapping)
                or value["portable_checkpoint_tree"].get(
                    "physical_origin_replay_passed"
                )
                is not True
                or value.get("cross_arm_gate_sha256") != cross_gate["sha256"]
                or value.get("cross_arm_gate_digest")
                != cross_gate["gate_digest"]
            ):
                failures.append(f"{arm}/{node} portable attestation closure differs")
                continue
            if stage == "fresh1":
                if (
                    value.get("cross_arm_recipe_version_digest")
                    != cross_gate.get("recipe_version_digest")
                ):
                    failures.append(
                        f"{arm}/{node} fresh1 attestation cross-gate chain differs"
                    )
                    continue
            else:
                assert fresh1_gate is not None
                if (
                    value.get("fresh1_acceptance_gate_sha256")
                    != fresh1_gate["sha256"]
                    or value.get("fresh1_acceptance_gate_digest")
                    != fresh1_gate["gate_digest"]
                    or value.get("receipt_status")
                    != "EXACT10_LATENT_GATES_PASS_DECODED_REVIEW_PENDING"
                    or value.get("latent_hard_gates_pass") is not True
                    or value.get("decoded_track_effect_gate_pending") is not True
                ):
                    failures.append(
                        f"{arm}/{node} exact10 predecessor/latent gate join differs"
                    )
                    continue
            target = spool / f"{stage}-origin-attestation.{arm}.json"
            with target.open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(target, 0o444)
            result[arm] = {
                "path": target,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size": len(raw),
                "attestation_digest": attestation_digest,
            }
        if failures or set(result) != {row[0] for row in PLACEMENTS}:
            fail(f"{stage} origin physical attestation stopped: {failures}")
        return result
    finally:
        for _, _, _, process, stdout_handle, stderr_handle in processes:
            if process.poll() is None:
                process.terminate()
            stdout_handle.close()
            stderr_handle.close()


def control_rows(
    preflight: Mapping[str, Mapping[str, Any]],
    fresh1: Optional[Mapping[str, Mapping[str, Any]]] = None,
    cross_gate: Optional[Mapping[str, Any]] = None,
    fresh1_gate: Optional[Mapping[str, Any]] = None,
    fresh1_attestations: Optional[Mapping[str, Mapping[str, Any]]] = None,
    exact10_attestations: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> list[tuple[str, Path, str, int]]:
    rows: list[tuple[str, Path, str, int]] = []
    for arm, *_ in PLACEMENTS:
        row = preflight[arm]
        rows.append((f"controls/preflight/{arm}.json", Path(row["path"]), str(row["sha256"]), int(row["size"])))
    if fresh1 is not None:
        for arm, *_ in PLACEMENTS:
            row = fresh1[arm]
            rows.append((f"controls/fresh1/{arm}.json", Path(row["path"]), str(row["sha256"]), int(row["size"])))
    if cross_gate is not None:
        rows.append(("controls/gates/cross_arm_preflight_gate.json", Path(cross_gate["path"]), str(cross_gate["sha256"]), int(cross_gate["size"])))
    if fresh1_gate is not None:
        rows.append(("controls/gates/fresh1_acceptance_gate.json", Path(fresh1_gate["path"]), str(fresh1_gate["sha256"]), int(fresh1_gate["size"])))
    for directory, values in (
        ("fresh1-attestations", fresh1_attestations),
        ("exact10-attestations", exact10_attestations),
    ):
        if values is not None:
            for arm, *_ in PLACEMENTS:
                row = values[arm]
                rows.append(
                    (
                        f"controls/{directory}/{arm}.json",
                        Path(row["path"]),
                        str(row["sha256"]),
                        int(row["size"]),
                    )
                )
    return sorted(rows, key=lambda row: row[0].encode("ascii"))


def run_gate_command(
    command: str,
    *,
    preflight: Mapping[str, Mapping[str, Any]],
    cross_gate: Optional[Mapping[str, Any]] = None,
    fresh1_attestations: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Mapping[str, Any]:
    arm_a, job_id, node, _ = PLACEMENTS[0]
    method_root = NODE_ROOT / "gate-runtime" / "methods" / "bernini_action_editing"
    controller = method_root / "elal3_c2_staged_gate_controller_v1.py"
    base = [
        *srun_prefix(job_id, node),
        str(PYTHON_BIN),
        "-I",
        "-B",
        str(controller),
        "--method-root",
        str(method_root),
        "--expected-controller-source-sha256",
        str(GATE_CONTROLLER_SHA256),
        "--expected-controller-source-size",
        str(GATE_CONTROLLER_SIZE),
    ]
    if command == "cross-arm":
        output = NODE_ROOT / "controls" / "gates" / "cross_arm_preflight_gate.json"
        argv = [
            *base,
            "cross-arm",
            "--a-receipt",
            str(NODE_ROOT / "controls" / "preflight" / f"{arm_a}.json"),
            "--a-receipt-sha256",
            str(preflight[arm_a]["sha256"]),
            "--b-receipt",
            str(NODE_ROOT / "controls" / "preflight" / f"{PLACEMENTS[1][0]}.json"),
            "--b-receipt-sha256",
            str(preflight[PLACEMENTS[1][0]]["sha256"]),
            "--output",
            str(output),
        ]
    else:
        if cross_gate is None or fresh1_attestations is None:
            fail("fresh1 gate command lacks predecessor rows")
        output = NODE_ROOT / "controls" / "gates" / "fresh1_acceptance_gate.json"
        argv = [
            *base,
            "fresh1",
            "--cross-gate",
            str(NODE_ROOT / "controls" / "gates" / "cross_arm_preflight_gate.json"),
            "--cross-gate-sha256",
            str(cross_gate["sha256"]),
        ]
        for prefix, (arm, *_rest) in zip(("a", "b", "replica"), PLACEMENTS):
            argv.extend(
                [
                    f"--{prefix}-origin-attestation",
                    str(
                        NODE_ROOT
                        / "controls"
                        / "fresh1-attestations"
                        / f"{arm}.json"
                    ),
                    f"--{prefix}-origin-attestation-sha256",
                    str(fresh1_attestations[arm]["sha256"]),
                ]
            )
        argv.extend(["--output", str(output)])
    completed = run_command(argv, timeout=600, label=f"generate {command} gate")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        fail(f"{command} gate controller receipt is not JSON: {error}")
    if value.get("path") != str(output):
        fail(f"{command} gate path differs")
    gate_sha = require_sha(value.get("sha256"), label=f"{command} gate SHA")
    pulled = run_command(
        [
            *srun_prefix(job_id, node),
            str(PYTHON_BIN),
            "-I",
            "-B",
            "-c",
            SEALED_READER,
            str(NODE_ROOT),
            str(output),
            str(4 << 20),
        ],
        timeout=300,
        label=f"pull {command} gate",
    ).stdout
    if hashlib.sha256(pulled).hexdigest() != gate_sha:
        fail(f"{command} gate pull SHA differs")
    try:
        gate_value = json.loads(pulled)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        fail(f"{command} pulled gate is not JSON: {error}")
    gate_unsigned = dict(gate_value)
    gate_digest = gate_unsigned.pop("gate_digest", None)
    if (
        pulled != canonical_json_bytes(gate_value) + b"\n"
        or gate_digest != object_digest(gate_unsigned)
        or gate_digest != value.get("gate_digest")
    ):
        fail(f"{command} pulled gate canonical/self digest differs")
    result = {
        "raw": pulled,
        "sha256": gate_sha,
        "size": len(pulled),
        "gate_digest": gate_digest,
    }
    if command == "cross-arm":
        result["recipe_version_digest"] = gate_value.get(
            "recipe_version_digest"
        )
    else:
        result.update(
            {
                "cross_arm_gate_sha256": gate_value.get(
                    "cross_arm_gate_sha256"
                ),
                "cross_arm_gate_digest": gate_value.get(
                    "cross_arm_gate_digest"
                ),
                "cross_arm_recipe_version_digest": gate_value.get(
                    "cross_arm_recipe_version_digest"
                ),
            }
        )
    return result


def publish_local_gate(spool: Path, name: str, gate: Mapping[str, Any]) -> Mapping[str, Any]:
    path = spool / name
    raw = bytes(gate["raw"])
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, 0o444)
    return {**dict(gate), "path": path}


def _write_controller_receipt(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, 0o444)


def execute(args: argparse.Namespace) -> Mapping[str, Any]:
    require_release_literals()
    controller_self = stable_binding(
        Path(__file__).resolve(strict=True),
        expected_sha256=args.expected_controller_sha256,
        expected_size=args.expected_controller_size,
        expected_mode=None,
        label="outer deployment controller self source",
    )
    release = args.release_root.resolve(strict=True)
    archive = release / "source.tar"
    manifest = release / "source.manifest.json"
    launcher = args.launcher.resolve(strict=True)
    bundle = args.latent_bundle.resolve(strict=True)
    stable_binding(archive, expected_sha256=str(ARCHIVE_SHA256), expected_size=int(ARCHIVE_SIZE), expected_mode=0o444, label="training release archive")
    stable_binding(manifest, expected_sha256=str(MANIFEST_SHA256), expected_size=int(MANIFEST_SIZE), expected_mode=0o444, label="training release manifest")
    stable_binding(launcher, expected_sha256=str(LAUNCHER_SHA256), expected_size=int(LAUNCHER_SIZE), expected_mode=None, label="training node launcher")
    stable_binding(bundle, expected_sha256=LATENT_BUNDLE_SHA256, expected_size=LATENT_BUNDLE_SIZE, expected_mode=0o444, label="exact16 bundle retry2")
    spool = args.spool_root
    if not spool.is_absolute() or spool.exists() or spool.is_symlink():
        fail("spool root must be a fresh absolute path")
    spool.mkdir(mode=0o700)
    assets_tar = spool / "node-assets.tar"
    assets_info = build_stream_tar(
        assets_tar,
        sorted(
            (
                ("assets/auh_run_elal3_c2_role_binding_stage_v1.sh", launcher, str(LAUNCHER_SHA256), int(LAUNCHER_SIZE)),
                ("assets/c2-exact16-latents.safetensors", bundle, LATENT_BUNDLE_SHA256, LATENT_BUNDLE_SIZE),
                ("assets/source.manifest.json", manifest, str(MANIFEST_SHA256), int(MANIFEST_SIZE)),
                ("assets/source.tar", archive, str(ARCHIVE_SHA256), int(ARCHIVE_SIZE)),
            ),
            key=lambda row: row[0].encode("ascii"),
        ),
    )
    stage_transport_all(assets_tar, assets_info["members"], receiver=ASSET_RECEIVER, timeout=1800, label="asset stdin stage")

    launch_stage_parallel("preflight", log_root=spool)
    preflight = pull_receipts("preflight", spool)
    preflight_tar = spool / "preflight-controls.tar"
    preflight_info = build_stream_tar(preflight_tar, control_rows(preflight))
    stage_transport_all(preflight_tar, preflight_info["members"], receiver=CONTROL_RECEIVER, timeout=600, label="preflight control stdin stage")

    cross = publish_local_gate(spool, "cross_arm_preflight_gate.json", run_gate_command("cross-arm", preflight=preflight))
    cross_tar = spool / "cross-controls.tar"
    cross_info = build_stream_tar(cross_tar, control_rows(preflight, cross_gate=cross))
    stage_transport_all(cross_tar, cross_info["members"], receiver=CONTROL_RECEIVER, timeout=600, label="cross gate stdin stage")

    own_sha = {arm: str(row["sha256"]) for arm, row in preflight.items()}
    launch_stage_parallel("fresh1", own_preflight_sha=own_sha, cross_gate_sha=str(cross["sha256"]), log_root=spool)
    fresh1 = pull_receipts("fresh1", spool)
    fresh1_attestations = run_origin_verifiers(
        "fresh1",
        receipts=fresh1,
        preflight=preflight,
        cross_gate=cross,
        fresh1_gate=None,
        spool=spool,
    )
    fresh1_attestation_tar = spool / "fresh1-attestation-controls.tar"
    fresh1_attestation_info = build_stream_tar(
        fresh1_attestation_tar,
        control_rows(
            preflight,
            cross_gate=cross,
            fresh1_attestations=fresh1_attestations,
        ),
    )
    stage_transport_all(
        fresh1_attestation_tar,
        fresh1_attestation_info["members"],
        receiver=CONTROL_RECEIVER,
        timeout=600,
        label="fresh1 portable attestation stdin stage",
    )

    fresh_gate = publish_local_gate(
        spool,
        "fresh1_acceptance_gate.json",
        run_gate_command(
            "fresh1",
            preflight=preflight,
            cross_gate=cross,
            fresh1_attestations=fresh1_attestations,
        ),
    )
    final_controls_tar = spool / "exact10-controls.tar"
    final_controls_info = build_stream_tar(
        final_controls_tar,
        control_rows(
            preflight,
            cross_gate=cross,
            fresh1_gate=fresh_gate,
            fresh1_attestations=fresh1_attestations,
        ),
    )
    stage_transport_all(final_controls_tar, final_controls_info["members"], receiver=CONTROL_RECEIVER, timeout=600, label="fresh1 gate stdin stage")

    launch_stage_parallel("exact10", own_preflight_sha=own_sha, cross_gate_sha=str(cross["sha256"]), fresh1_gate_sha=str(fresh_gate["sha256"]), log_root=spool)
    exact10 = pull_receipts("exact10", spool)
    exact10_attestations = run_origin_verifiers(
        "exact10",
        receipts=exact10,
        preflight=preflight,
        cross_gate=cross,
        fresh1_gate=fresh_gate,
        spool=spool,
    )
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "status": "EXACT3_FRESH_EXACT10_LATENT_GATES_COMPLETE",
        "node_local_srun_stdin_transport": True,
        "login_compute_shared_vast_assumed": False,
        "node_root": str(NODE_ROOT),
        "release": {"archive_sha256": ARCHIVE_SHA256, "manifest_sha256": MANIFEST_SHA256, "launcher_sha256": LAUNCHER_SHA256, "trainer_sha256": TRAINER_SHA256, "gate_controller_sha256": GATE_CONTROLLER_SHA256, "origin_verifier_sha256": ORIGIN_VERIFIER_SHA256, "outer_controller_sha256": controller_self["sha256"], "outer_controller_size": controller_self["size"]},
        "exact16": {"bundle_sha256": LATENT_BUNDLE_SHA256, "bundle_size": LATENT_BUNDLE_SIZE, "receipt_sha256": LATENT_RECEIPT_SHA256, "receipt_size": LATENT_RECEIPT_SIZE, "receipt_digest": LATENT_RECEIPT_DIGEST},
        "preflight_receipt_sha256_by_arm": {arm: row["sha256"] for arm, row in preflight.items()},
        "cross_arm_gate_sha256": cross["sha256"],
        "fresh1_receipt_sha256_by_arm": {arm: row["sha256"] for arm, row in fresh1.items()},
        "fresh1_origin_attestation_sha256_by_arm": {
            arm: row["sha256"] for arm, row in fresh1_attestations.items()
        },
        "fresh1_origin_attestation_digest_by_arm": {
            arm: row["attestation_digest"]
            for arm, row in fresh1_attestations.items()
        },
        "fresh1_acceptance_gate_sha256": fresh_gate["sha256"],
        "exact10_receipt_sha256_by_arm": {arm: row["sha256"] for arm, row in exact10.items()},
        "exact10_origin_attestation_sha256_by_arm": {
            arm: row["sha256"] for arm, row in exact10_attestations.items()
        },
        "exact10_origin_attestation_digest_by_arm": {
            arm: row["attestation_digest"]
            for arm, row in exact10_attestations.items()
        },
        "exact3_origin_physical_postflight_pass": True,
        "stage_sequence": [0, 1, 10],
        "fresh_exact10_resume": False,
        "formal_c2_authorized": False,
        "exact160_authorized": False,
        "source_instruction_inference_authorized": False,
        "real_video_generalization_authorized": False,
        "scientific_claim_authorized": False,
    }
    receipt = {**unsigned, "receipt_digest": object_digest(unsigned)}
    _write_controller_receipt(spool / "CONTROLLER_COMPLETE.json", receipt)
    os.chmod(spool, 0o555)
    return receipt


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--release-root", type=Path, required=True)
    value.add_argument("--launcher", type=Path, required=True)
    value.add_argument("--latent-bundle", type=Path, required=True)
    value.add_argument("--spool-root", type=Path, required=True)
    value.add_argument("--expected-controller-sha256", required=True)
    value.add_argument("--expected-controller-size", type=int, required=True)
    value.add_argument("--ack-execute-three-holder-staged-c2", action="store_true")
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    require_sha(args.expected_controller_sha256, label="outer controller expected SHA")
    if args.expected_controller_size <= 0:
        fail("outer controller expected size is invalid")
    if not args.ack_execute_three_holder_staged_c2:
        fail("explicit staged C2 execution acknowledgement is required")
    result = execute(args)
    print(canonical_json_bytes(result).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ELAL3C2DeploymentError, OSError) as error:
        print(f"ELAL3_C2_DEPLOYMENT_ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
