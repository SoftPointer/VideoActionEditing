#!/usr/bin/env python3
"""Build a stable, complete SHA-256 closure for one local Qwen checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from typing import Any


SCHEMA_VERSION = "motive-qwen-model-closure-v1"


def _sha256_stable(path: Path) -> tuple[int, str]:
    """Hash one resolved regular file and reject concurrent mutation."""

    resolved = path.resolve(strict=True)
    before = resolved.stat()
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"not a regular file: {path}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    after = resolved.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise RuntimeError(f"file changed while hashing: {path}")
    return after.st_size, digest.hexdigest()


def build_closure(model_path: Path, *, model_id: str, revision: str) -> dict[str, Any]:
    root = model_path.expanduser().resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError("model path must be a non-symlink directory")
    if not model_id or model_id != model_id.strip():
        raise RuntimeError("model id must be one non-empty canonical string")
    if not revision or revision != revision.strip():
        raise RuntimeError("revision must be one non-empty canonical string")

    logical_files: list[tuple[str, Path]] = []
    actual_directories: set[str] = set()
    for directory, directory_names, file_names in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        directory_path = Path(directory)
        directory_names.sort()
        file_names.sort()
        for name in directory_names:
            entry = directory_path / name
            relative = entry.relative_to(root).as_posix()
            mode = entry.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise RuntimeError(f"directory symlink is forbidden: {relative}")
            if not stat.S_ISDIR(mode):
                raise RuntimeError(f"special directory entry is forbidden: {relative}")
            actual_directories.add(relative)
        for name in file_names:
            entry = directory_path / name
            relative = entry.relative_to(root).as_posix()
            pure = PurePosixPath(relative)
            if (
                pure.is_absolute()
                or ".." in pure.parts
                or "." in pure.parts
                or pure.as_posix() != relative
            ):
                raise RuntimeError(f"unsafe relative path: {relative}")
            mode = entry.lstat().st_mode
            if not (stat.S_ISREG(mode) or stat.S_ISLNK(mode)):
                raise RuntimeError(f"special file entry is forbidden: {relative}")
            resolved = entry.resolve(strict=True)
            if not resolved.is_file():
                raise RuntimeError(f"file does not resolve to a regular file: {relative}")
            logical_files.append((relative, entry))

    logical_files.sort(key=lambda item: item[0])
    relative_paths = [item[0] for item in logical_files]
    if len(relative_paths) != len(set(relative_paths)):
        raise RuntimeError("duplicate logical file path")
    expected_directories: set[str] = set()
    for relative in relative_paths:
        parent = PurePosixPath(relative).parent
        while parent != PurePosixPath("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    if actual_directories != expected_directories:
        raise RuntimeError(
            "empty or unbound model directory found: "
            f"{sorted(actual_directories - expected_directories)}"
        )

    files: list[dict[str, Any]] = []
    total_bytes = 0
    for index, (relative, path) in enumerate(logical_files, start=1):
        size, digest = _sha256_stable(path)
        if size < 1:
            raise RuntimeError(f"empty model file is forbidden: {relative}")
        total_bytes += size
        files.append(
            {
                "relative_path": relative,
                "bytes": size,
                "sha256": digest,
            }
        )
        print(
            f"[{index}/{len(logical_files)}] {relative} {size} {digest}",
            file=sys.stderr,
            flush=True,
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "model_id": model_id,
        "revision": revision,
        "model_path": str(root),
        "hash_algorithm": "sha256",
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    closure = build_closure(
        args.model_path,
        model_id=args.model_id,
        revision=args.revision,
    )
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(closure, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    print(
        f"wrote={output} sha256={hashlib.sha256(payload).hexdigest()} "
        f"files={closure['file_count']} bytes={closure['total_bytes']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
