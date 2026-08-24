#!/usr/bin/env python3
"""Build or audit the deterministic ELAL-3 synthetic-C0 source release.

This release is deliberately narrow.  It packages the implementation, CLI,
and its tests so the same bytes can be exercised on AUH nodes 205 and 248.
It grants neither semantic-representation nor optimizer authority.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "bernini-elal3-synthetic-c0-release-v1"
ARCHIVE_FORMAT = "fixed-ustar-ascii-sorted-owner0-mtime0-record10240-v1"
MEMBER_ROOT = PurePosixPath("methods/bernini_action_editing")
RELEASE_FILES = (
    "elal3_c0_v1.py",
    "run_elal3_c0_v1.py",
    "tests/test_elal3_c0_v1.py",
)


class ELAL3ReleaseError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ELAL3ReleaseError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def object_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _stable_plain_file(path: Path, *, maximum_bytes: int = 8 * 1024 * 1024) -> bytes:
    _require(path.is_absolute() and not path.is_symlink(), f"non-canonical source: {path}")
    before_name = path.lstat()
    _require(stat.S_ISREG(before_name.st_mode), f"source is not a regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        _require(before.st_size <= maximum_bytes, f"source is too large: {path}")
        first = bytearray()
        while len(first) < before.st_size:
            block = os.read(descriptor, min(1024 * 1024, before.st_size - len(first)))
            _require(bool(block), f"source was truncated: {path}")
            first.extend(block)
        _require(os.read(descriptor, 1) == b"", f"source grew while reading: {path}")
        os.lseek(descriptor, 0, os.SEEK_SET)
        second = bytearray()
        while len(second) < before.st_size:
            block = os.read(descriptor, min(1024 * 1024, before.st_size - len(second)))
            _require(bool(block), f"source was truncated on replay: {path}")
            second.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_name = path.lstat()
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
        item.st_nlink,
    )
    _require(identity(before_name) == identity(before) == identity(after) == identity(after_name), f"source identity changed: {path}")
    _require(first == second, f"source bytes changed: {path}")
    return bytes(first)


def _member_name(relative: str) -> str:
    return str(MEMBER_ROOT / PurePosixPath(relative))


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


def build_payload(method_root: Path) -> tuple[bytes, dict[str, Any]]:
    root = method_root.resolve(strict=True)
    _require(root.is_dir() and not method_root.is_symlink(), "method root must be a canonical directory")
    _require(tuple(sorted(RELEASE_FILES, key=lambda value: value.encode("ascii"))) == RELEASE_FILES, "release member order differs")
    payloads: dict[str, bytes] = {}
    rows: list[dict[str, Any]] = []
    for relative in RELEASE_FILES:
        raw = _stable_plain_file(root / relative)
        compile(raw, relative, "exec")
        name = _member_name(relative)
        payloads[name] = raw
        rows.append(
            {
                "path": name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size": len(raw),
                "mode": "0444",
            }
        )
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name in sorted(payloads, key=lambda value: value.encode("ascii")):
            archive.addfile(_tar_info(name, len(payloads[name])), io.BytesIO(payloads[name]))
    archive_raw = stream.getvalue()
    _require(len(archive_raw) % 10240 == 0, "archive record size differs")
    unsigned: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "archive_format": ARCHIVE_FORMAT,
        "archive_sha256": hashlib.sha256(archive_raw).hexdigest(),
        "archive_size": len(archive_raw),
        "files": rows,
        "execution_scope": "synthetic_c0_abi_causality_gradient_only",
        "representation_semantics_qualified": False,
        "training_authorized": False,
        "exact160_authorized": False,
    }
    manifest = dict(unsigned)
    manifest["manifest_digest"] = object_digest(unsigned)
    return archive_raw, manifest


def _write_create_only(path: Path, payload: bytes, mode: int) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            _require(written > 0, f"short write made no progress: {path}")
            view = view[written:]
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _require(_stable_plain_file(path, maximum_bytes=max(len(payload), 1)) == payload, f"published bytes differ: {path}")


def publish(method_root: Path, output: Path) -> dict[str, Any]:
    _require(output.is_absolute() and not output.exists() and not output.is_symlink(), "output must be a fresh absolute path")
    archive_raw, manifest = build_payload(method_root)
    os.mkdir(output, 0o700)
    _write_create_only(output / "source.tar", archive_raw, 0o444)
    manifest_raw = canonical_json_bytes(manifest) + b"\n"
    _write_create_only(output / "source.manifest.json", manifest_raw, 0o444)
    os.chmod(output, 0o555)
    return {
        "output": str(output),
        "archive_sha256": manifest["archive_sha256"],
        "archive_size": manifest["archive_size"],
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "manifest_digest": manifest["manifest_digest"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        print(json.dumps(publish(args.method_root, args.output), sort_keys=True, separators=(",", ":")))
    except (ELAL3ReleaseError, OSError) as error:
        print(f"[elal3-c0-release] ERROR: {error}", file=os.sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
