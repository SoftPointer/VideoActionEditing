#!/usr/bin/env python3
"""Write one JSON marker through an exclusive temp, fsync, rename, and reread."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import sys
from typing import Any, Mapping, Optional, Sequence


class MarkerError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"


def write_marker(output: Path, value: Mapping[str, Any]) -> Mapping[str, Any]:
    if output.exists() or output.is_symlink():
        raise MarkerError(f"refusing to overwrite marker: {output}")
    parent = output.parent
    if not parent.is_dir() or parent.is_symlink() or not stat.S_ISDIR(parent.lstat().st_mode):
        raise MarkerError("marker parent is not a plain directory")
    payload = _canonical(value)
    temporary = parent / f".{output.name}.exclusive-{os.getpid()}-{secrets.token_hex(8)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(str(temporary), flags, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise MarkerError("marker write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor); descriptor = -1
        if output.exists() or output.is_symlink():
            raise MarkerError("marker appeared before atomic rename")
        os.rename(str(temporary), str(output))
        directory_descriptor = os.open(str(parent), os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        reread = output.read_bytes()
        if reread != payload or json.loads(reread.decode("utf-8")) != value:
            raise MarkerError("marker reread differs after atomic rename")
        return {
            "schema_version": "bernini-e00-r6-atomic-marker-receipt-v6",
            "complete": True,
            "path": str(output),
            "sha256": hashlib.sha256(reread).hexdigest(),
            "bytes": len(reread),
            "exclusive_temp_created": True,
            "file_fsync_completed": True,
            "atomic_rename_completed": True,
            "directory_fsync_completed": True,
            "reread_bit_exact": True,
        }
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("write", choices=("write",))
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        value = json.load(sys.stdin)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise MarkerError("stdin is not JSON") from error
    if not isinstance(value, dict):
        raise MarkerError("marker must be a JSON object")
    receipt = write_marker(Path(args.output), value)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MarkerError as error:
        print(f"R6_MARKER_REJECTED: {error}", file=sys.stderr)
        raise SystemExit(8)
