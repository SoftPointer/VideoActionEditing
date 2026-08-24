#!/usr/bin/env python3
"""Build the frozen, shard-balanced 24-row Qwen3 fast candidate pool.

The parent is the immutable 123-row Goku prefilter manifest.  Selected rows
are copied byte-for-byte, in parent order, so every media and input-row
binding remains unchanged.  This utility never authorizes generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence


PARENT_SHA256 = (
    "824e92112159d559691a039fd949b26e0ca9ff07efe98483814aba2386123a9d"
)
PARENT_ROWS = 123
SHARD_COUNT = 8
IIDS_BY_SHARD: tuple[tuple[str, ...], ...] = (
    (
        "4136aec3248940f1",
        "97e0fbc083c74f7f",
        "2cc5d55bbf384d0f",
    ),
    (
        "5290c7201ab14913",
        "5d948057d82c4f34",
        "87dfcbe218da4d6e",
    ),
    (
        "35567f8c3e9c4183",
        "7e0eabf8ac6e43c6",
        "50e61b33ffad412c",
    ),
    (
        "1dbe39537c984690",
        "2d015c9488054e7e",
        "3c91cad95ae94570",
    ),
    (
        "a4294f3392744bdd",
        "90c34f5320694ae4",
        "43e93e6fd3fd47bb",
    ),
    (
        "10ed90644f81461d",
        "4fb0b85e027e40af",
        "06db761c8e8f43b7",
    ),
    (
        "0ef3b8ef133246f0",
        "7621503f7289478b",
        "b3e73ce8fa574fc9",
    ),
    (
        "42da0dde38394bc7",
        "5907a38e7a8445bc",
        "4e74eeae37744923",
    ),
)


class FastPoolError(RuntimeError):
    """The frozen parent or requested output is unsafe."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _iid_shard(iid: str) -> int:
    return int(hashlib.sha256(iid.encode("utf-8")).hexdigest()[:16], 16) % 8


def build_fast_pool(parent: Path, output: Path) -> dict[str, object]:
    parent = parent.expanduser().resolve(strict=True)
    output = output.expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    raw = parent.read_bytes()
    if _sha256(raw) != PARENT_SHA256:
        raise FastPoolError("parent manifest SHA-256 differs")
    lines = raw.splitlines(keepends=True)
    if len(lines) != PARENT_ROWS or any(not line.endswith(b"\n") for line in lines):
        raise FastPoolError("parent manifest row framing differs")

    expected = {
        iid for shard_iids in IIDS_BY_SHARD for iid in shard_iids
    }
    if len(expected) != 24:
        raise FastPoolError("frozen IID selection is not unique exact-24")
    selected_lines: list[bytes] = []
    selected_iids: list[str] = []
    seen_parent: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FastPoolError(
                f"parent row {line_number} is not strict JSON"
            ) from error
        if not isinstance(row, dict):
            raise FastPoolError(f"parent row {line_number} is not an object")
        iid = row.get("iid")
        if not isinstance(iid, str) or not iid:
            raise FastPoolError(f"parent row {line_number} IID is invalid")
        if iid in seen_parent:
            raise FastPoolError(f"parent IID is duplicated: {iid}")
        seen_parent.add(iid)
        if iid not in expected:
            continue
        if row.get("eligible") is not True or row.get("selected") is not True:
            raise FastPoolError(f"frozen candidate is not eligible: {iid}")
        selected_lines.append(line)
        selected_iids.append(iid)

    if set(selected_iids) != expected or len(selected_iids) != len(expected):
        missing = sorted(expected - set(selected_iids))
        raise FastPoolError(f"frozen candidates are missing: {missing}")
    shard_counts = [0] * SHARD_COUNT
    for iid in selected_iids:
        shard_counts[_iid_shard(iid)] += 1
    if shard_counts != [3] * SHARD_COUNT:
        raise FastPoolError(f"candidate shard geometry differs: {shard_counts}")

    payload = b"".join(selected_lines)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, 0o400)
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        os.replace(temporary_name, output)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return {
        "parent": str(parent),
        "parent_sha256": PARENT_SHA256,
        "output": str(output.resolve(strict=True)),
        "output_sha256": _sha256(payload),
        "rows": len(selected_iids),
        "ordered_iids": selected_iids,
        "shard_counts": shard_counts,
        "generation_authorized": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the frozen shard-balanced Goku fast24 pool."
    )
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_fast_pool(args.parent, args.output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
