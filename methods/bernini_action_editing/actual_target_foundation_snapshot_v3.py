#!/usr/bin/env python3
"""Create and verify an immutable, no-symlink V3 source snapshot.

The manifest binds both snapshot bytes and the lexical original paths.  Paths
are never ``resolve()``-laundered before component-wise ``lstat`` checks.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, NoReturn, Optional, Sequence

import actual_target_foundation_canary_v3 as authority


MANIFEST_NAME = "snapshot_manifest_v3.json"
SCHEMA = "actual-target-foundation-immutable-snapshot-v3"


class SnapshotV3Error(RuntimeError):
    pass


def fail(message: str) -> NoReturn:
    raise SnapshotV3Error(message)


def _identity(row: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        row.st_dev,
        row.st_ino,
        row.st_mode,
        row.st_size,
        row.st_mtime_ns,
        row.st_ctime_ns,
    )


def _components(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_absolute():
        fail(f"path must be absolute: {path}")
    rows = []
    current = Path(path.anchor)
    root_stat = current.lstat()
    rows.append(
        {
            "path": str(current),
            "kind": "directory",
            "mode": stat.S_IMODE(root_stat.st_mode),
            "device": root_stat.st_dev,
            "inode": root_stat.st_ino,
        }
    )
    for part in path.parts[1:]:
        current = current / part
        try:
            row = current.lstat()
        except OSError as error:
            raise SnapshotV3Error(f"path component unavailable: {current}") from error
        if stat.S_ISLNK(row.st_mode):
            fail(f"symlink component forbidden: {current}")
        kind = "directory" if stat.S_ISDIR(row.st_mode) else "file" if stat.S_ISREG(row.st_mode) else "other"
        rows.append(
            {
                "path": str(current),
                "kind": kind,
                "mode": stat.S_IMODE(row.st_mode),
                "device": row.st_dev,
                "inode": row.st_ino,
            }
        )
    return rows


def _plain_directory(path: Path) -> None:
    rows = _components(path)
    if rows[-1]["kind"] != "directory":
        fail(f"path is not a directory: {path}")


def _plain_regular_file(path: Path) -> None:
    rows = _components(path)
    if rows[-1]["kind"] != "file":
        fail(f"path is not a regular file: {path}")


def _stable_file(path: Path) -> tuple[bytes, os.stat_result]:
    _plain_regular_file(path)
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
        fail(f"file changed during stable read: {path}")
    return b"".join(chunks), before


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_only_bytes(path: Path, payload: bytes, mode: int) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        fail(f"create-only target must be absolute and absent: {path}")
    _plain_directory(path.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, mode)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    else:
        os.close(descriptor)
    os.chmod(path, mode)
    _fsync_directory(path.parent)


def _payload_paths() -> tuple[str, ...]:
    rows = authority.load_authority()["snapshot_payload_relative_paths"]
    if not isinstance(rows, list) or not rows or len(rows) != len(set(rows)):
        fail("snapshot payload authority differs")
    output = []
    for value in rows:
        if not isinstance(value, str):
            fail("snapshot payload path is not text")
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts or value == MANIFEST_NAME:
            fail(f"unsafe snapshot relative path: {value}")
        output.append(value)
    return tuple(output)


def original_source_closure(source_root: Path) -> Mapping[str, Any]:
    _plain_directory(source_root)
    rows = []
    for relative_text in _payload_paths():
        source = source_root / relative_text
        payload, metadata = _stable_file(source)
        rows.append(
            {
                "relative_path": relative_text,
                "original_path": str(source),
                "original_mode": stat.S_IMODE(metadata.st_mode),
                "byte_count": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "original_path_components": _components(source),
            }
        )
    value = {
        "source_root": str(source_root),
        "file_count": len(rows),
        "files": rows,
        "no_symlink_laundering": True,
    }
    return {**value, "digest": authority.object_sha256(value)}


def materialize_snapshot(source_root: Path, snapshot_root: Path) -> Mapping[str, Any]:
    if not source_root.is_absolute() or not snapshot_root.is_absolute():
        fail("source and snapshot roots must be absolute")
    source_closure = original_source_closure(source_root)
    if snapshot_root.exists() or snapshot_root.is_symlink():
        fail("snapshot root must be absent")
    _plain_directory(snapshot_root.parent)
    os.mkdir(snapshot_root, 0o700)
    _fsync_directory(snapshot_root.parent)
    created_directories = {snapshot_root}
    snapshot_rows = []
    for source_row in source_closure["files"]:
        relative = Path(source_row["relative_path"])
        target = snapshot_root / relative
        ancestors = []
        parent = target.parent
        while parent != snapshot_root:
            ancestors.append(parent)
            parent = parent.parent
        for directory in reversed(ancestors):
            if directory not in created_directories:
                os.mkdir(directory, 0o700)
                created_directories.add(directory)
                _fsync_directory(directory.parent)
        payload, source_stat = _stable_file(Path(source_row["original_path"]))
        expected_sha = hashlib.sha256(payload).hexdigest()
        if expected_sha != source_row["sha256"] or stat.S_IMODE(source_stat.st_mode) != source_row["original_mode"]:
            fail("original source changed after closure collection")
        snapshot_mode = 0o555 if source_row["original_mode"] & 0o111 else 0o444
        _create_only_bytes(target, payload, snapshot_mode)
        snapshot_rows.append(
            {
                "relative_path": source_row["relative_path"],
                "sha256": expected_sha,
                "byte_count": len(payload),
                "snapshot_mode": snapshot_mode,
            }
        )
    manifest_body = {
        "schema_version": SCHEMA,
        "source_closure": source_closure,
        "snapshot_root": str(snapshot_root),
        "snapshot_file_count": len(snapshot_rows),
        "snapshot_files": snapshot_rows,
        "snapshot_directory_mode": 0o555,
        "snapshot_file_modes": {"data": 0o444, "executable": 0o555},
        "no_symlinks": True,
        "immutable_permissions_applied": True,
    }
    manifest = {**manifest_body, "manifest_self_sha256": authority.object_sha256(manifest_body)}
    manifest_payload = json.dumps(
        manifest, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False
    ).encode("ascii") + b"\n"
    _create_only_bytes(snapshot_root / MANIFEST_NAME, manifest_payload, 0o444)
    for directory in sorted(created_directories, key=lambda item: len(item.parts), reverse=True):
        os.chmod(directory, 0o555)
        _fsync_directory(directory)
    _fsync_directory(snapshot_root.parent)
    return verify_snapshot(snapshot_root, verify_original=True)


def _load_manifest(path: Path) -> Mapping[str, Any]:
    payload, _ = _stable_file(path)
    value = authority.strict_json_bytes(payload)
    if not isinstance(value, Mapping):
        fail("snapshot manifest is not one object")
    body = dict(value)
    claim = body.pop("manifest_self_sha256", None)
    if claim != authority.object_sha256(body):
        fail("snapshot manifest self hash differs")
    return value


def verify_snapshot(snapshot_root: Path, *, verify_original: bool = True) -> Mapping[str, Any]:
    _plain_directory(snapshot_root)
    if stat.S_IMODE(snapshot_root.stat().st_mode) != 0o555:
        fail("snapshot root is not immutable mode 0555")
    manifest = _load_manifest(snapshot_root / MANIFEST_NAME)
    if manifest.get("schema_version") != SCHEMA or manifest.get("snapshot_root") != str(snapshot_root):
        fail("snapshot manifest identity differs")
    rows = manifest.get("snapshot_files")
    if not isinstance(rows, list) or [row.get("relative_path") for row in rows if isinstance(row, Mapping)] != list(_payload_paths()):
        fail("snapshot payload order/closure differs")
    expected_paths = {MANIFEST_NAME}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"relative_path", "sha256", "byte_count", "snapshot_mode"}:
            fail("snapshot file row schema differs")
        relative = row["relative_path"]
        expected_paths.add(relative)
        path = snapshot_root / relative
        payload, metadata = _stable_file(path)
        if (
            hashlib.sha256(payload).hexdigest() != row["sha256"]
            or len(payload) != row["byte_count"]
            or stat.S_IMODE(metadata.st_mode) != row["snapshot_mode"]
            or row["snapshot_mode"] not in (0o444, 0o555)
        ):
            fail(f"snapshot file bytes/mode differ: {relative}")
    observed_files = set()
    for root, directories, files in os.walk(snapshot_root, followlinks=False):
        root_path = Path(root)
        if stat.S_IMODE(root_path.lstat().st_mode) != 0o555:
            fail(f"snapshot directory mode differs: {root_path}")
        for name in directories:
            child = root_path / name
            if child.is_symlink():
                fail(f"snapshot contains symlink directory: {child}")
        for name in files:
            child = root_path / name
            if child.is_symlink():
                fail(f"snapshot contains symlink file: {child}")
            observed_files.add(str(child.relative_to(snapshot_root)))
    if observed_files != expected_paths:
        fail("snapshot contains missing or extra files")
    source_closure = manifest.get("source_closure")
    if not isinstance(source_closure, Mapping):
        fail("snapshot original source closure missing")
    body = dict(source_closure)
    claim = body.pop("digest", None)
    if claim != authority.object_sha256(body):
        fail("snapshot original source closure digest differs")
    if verify_original:
        rebuilt = original_source_closure(Path(source_closure["source_root"]))
        if rebuilt != source_closure:
            fail("original lexical source closure changed")
    value = {
        "verified": True,
        "snapshot_root": str(snapshot_root),
        "manifest_file_sha256": authority.file_sha256(snapshot_root / MANIFEST_NAME),
        "manifest_self_sha256": manifest["manifest_self_sha256"],
        "source_closure_digest": source_closure["digest"],
        "snapshot_file_count": len(rows),
        "original_reverified": bool(verify_original),
        "no_symlinks": True,
        "immutable_modes": True,
    }
    return {**value, "digest": authority.object_sha256(value)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--materialize", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--print-source-closure", action="store_true")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--skip-original-reverify", action="store_true")
    args = parser.parse_args(argv)
    if args.print_source_closure:
        if args.source_root is None:
            fail("--print-source-closure requires --source-root")
        value = original_source_closure(args.source_root)
    elif args.materialize:
        if args.source_root is None or args.snapshot_root is None:
            fail("--materialize requires source and snapshot roots")
        value = materialize_snapshot(args.source_root, args.snapshot_root)
    else:
        if args.snapshot_root is None:
            fail("--verify requires --snapshot-root")
        value = verify_snapshot(
            args.snapshot_root, verify_original=not args.skip_original_reverify
        )
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MANIFEST_NAME",
    "SCHEMA",
    "SnapshotV3Error",
    "materialize_snapshot",
    "original_source_closure",
    "verify_snapshot",
]
