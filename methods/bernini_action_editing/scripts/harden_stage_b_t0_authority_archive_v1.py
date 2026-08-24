#!/usr/bin/env python3
"""Turn archived ACTIVE authority JSON into non-executable gzip containers.

The byte-exact originals remain recoverable with gzip, but neither the archive
path nor its ``.gz`` payload is a valid authorization-addendum JSON input.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "bernini-stage-b-t0-authority-archive-hardening-v1"
ACTIVE_STATE = "ACTIVE_CREATE_ONCE_AUTHORITY"
POINTER_STATE = "ARCHIVED_COMPRESSED_NOT_AN_AUTHORITY"


class HardeningError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise HardeningError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require_sha256(value: str, *, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        fail(f"{label} is not one lowercase SHA-256")
    return value


def inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def regular_bytes(path: Path, *, expected_mode: int | None = None) -> bytes:
    metadata = path.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        fail(f"archive member is not one regular non-linked file: {path}")
    if expected_mode is not None and stat.S_IMODE(metadata.st_mode) != expected_mode:
        fail(f"archive member mode differs: {path}")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        fail("archive hardening requires O_NOFOLLOW")
    descriptor = os.open(path, os.O_RDONLY | nofollow)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
            fail(f"archive member changed while opening: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 65536)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def write_create_once(path: Path, payload: bytes, *, mode: int) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                fail(f"create-once write made no progress: {path}")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_plan(
    *, experiment_root: Path, archive_dir: Path, manifest_sha256: str
) -> tuple[Path, Path, Mapping[str, Any], list[dict[str, Any]]]:
    root = experiment_root.resolve(strict=True)
    archive = archive_dir.resolve(strict=True)
    if not inside(archive, root) or archive == root or archive.is_symlink():
        fail("archive directory is not a real descendant of experiment root")
    manifest_path = archive / "manifest.json"
    manifest_payload = regular_bytes(manifest_path, expected_mode=0o400)
    if sha256_bytes(manifest_payload) != require_sha256(
        manifest_sha256, label="revocation manifest"
    ):
        fail("revocation manifest SHA-256 differs")
    try:
        manifest = json.loads(manifest_payload.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise HardeningError("revocation manifest JSON is invalid") from error
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_version") != "bernini-stage-b-t0-authority-revocation-v1"
        or manifest.get("mode") != "applied"
        or manifest.get("experiment_root") != str(root)
        or manifest.get("archive_dir") != str(archive)
        or not isinstance(manifest.get("applied"), list)
    ):
        fail("revocation manifest contract differs")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    originals_root = archive / "originals"
    for raw in manifest["applied"]:
        if not isinstance(raw, Mapping):
            fail("revocation manifest applied row differs")
        original_sha = require_sha256(str(raw.get("original_sha256")), label="original authority")
        archive_path = Path(str(raw.get("archive_path"))).absolute()
        runtime_path = Path(str(raw.get("path"))).absolute()
        if (
            str(archive_path) in seen
            or not inside(archive_path, originals_root)
            or not inside(runtime_path, root)
            or inside(runtime_path, archive)
        ):
            fail("revocation archive/runtime path closure differs")
        seen.add(str(archive_path))
        payload = regular_bytes(archive_path, expected_mode=0o400)
        if sha256_bytes(payload) != original_sha:
            fail(f"raw archived authority SHA-256 differs: {archive_path}")
        try:
            authority = json.loads(payload.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise HardeningError(f"raw archive is not authority JSON: {archive_path}") from error
        if authority.get("activation", {}).get("state") != ACTIVE_STATE:
            fail(f"raw archive is not one ACTIVE authority: {archive_path}")
        runtime = json.loads(regular_bytes(runtime_path).decode("ascii"))
        if runtime.get("activation", {}).get("state") != "REVOKED_SUPERSEDED_BY_RETRY6":
            fail(f"runtime authority was not revoked: {runtime_path}")
        rows.append(
            {
                "runtime_path": str(runtime_path),
                "archive_pointer_path": str(archive_path),
                "gzip_path": f"{archive_path}.gz",
                "original_sha256": original_sha,
                "original_size": len(payload),
                "payload": payload,
            }
        )
    if not rows or len(rows) != manifest.get("superseded_active_count"):
        fail("revocation manifest row count differs")
    return root, archive, manifest, rows


def harden(
    *,
    experiment_root: Path,
    archive_dir: Path,
    revocation_manifest_sha256: str,
    apply: bool,
) -> Mapping[str, Any]:
    root, archive, revocation_manifest, rows = load_plan(
        experiment_root=experiment_root,
        archive_dir=archive_dir,
        manifest_sha256=revocation_manifest_sha256,
    )
    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mode": "apply" if apply else "dry_run",
        "experiment_root": str(root),
        "archive_dir": str(archive),
        "revocation_manifest_sha256": revocation_manifest_sha256,
        "member_count": len(rows),
        "members": [
            {key: row[key] for key in row if key != "payload"} for row in rows
        ],
        "security_property": "no_byte_exact_active_authority_remains_as_parseable_json",
    }
    if not apply:
        return plan
    plan_path = archive / "hardening-plan.json"
    receipt_path = archive / "hardening-manifest.json"
    if plan_path.exists() or plan_path.is_symlink() or receipt_path.exists() or receipt_path.is_symlink():
        fail("create-once hardening evidence already exists")
    plan_payload = canonical_json_bytes(plan)
    write_create_once(plan_path, plan_payload, mode=0o400)
    applied: list[dict[str, Any]] = []
    for row in rows:
        pointer_path = Path(row["archive_pointer_path"])
        gzip_path = Path(row["gzip_path"])
        if gzip_path.exists() or gzip_path.is_symlink():
            fail(f"gzip authority archive already exists: {gzip_path}")
        container = gzip.compress(row["payload"], compresslevel=9, mtime=0)
        write_create_once(gzip_path, container, mode=0o400)
        if (
            gzip.decompress(regular_bytes(gzip_path, expected_mode=0o400))
            != row["payload"]
        ):
            fail(f"gzip authority archive does not round-trip: {gzip_path}")
        pointer = {
            "schema_version": SCHEMA_VERSION,
            "activation": {
                "state": POINTER_STATE,
                "optimizer_creation_authorized": False,
            },
            "original_authority": {
                "sha256": row["original_sha256"],
                "size": row["original_size"],
                "gzip_path": str(gzip_path),
                "gzip_sha256": sha256_bytes(container),
                "compression": "gzip_deterministic_mtime_0_level_9",
            },
            "superseded_by": {
                "active_authority_sha256": revocation_manifest[
                    "kept_active_authority_sha256"
                ]
            },
        }
        pointer_payload = canonical_json_bytes(pointer)
        temporary = pointer_path.with_name(
            f".{pointer_path.name}.archive-pointer-{os.getpid()}"
        )
        write_create_once(temporary, pointer_payload, mode=0o400)
        os.replace(temporary, pointer_path)
        fsync_directory(pointer_path.parent)
        observed_pointer = json.loads(regular_bytes(pointer_path, expected_mode=0o400).decode("ascii"))
        if observed_pointer.get("activation", {}).get("state") != POINTER_STATE:
            fail(f"archive pointer is executable or malformed: {pointer_path}")
        applied.append(
            {
                "archive_pointer_path": str(pointer_path),
                "archive_pointer_sha256": sha256_bytes(pointer_payload),
                "gzip_path": str(gzip_path),
                "gzip_sha256": sha256_bytes(container),
                "decompressed_original_sha256": row["original_sha256"],
            }
        )
    receipt = dict(plan)
    receipt["mode"] = "applied"
    receipt["applied"] = applied
    receipt["plan_sha256"] = sha256_bytes(plan_payload)
    receipt["manifest_digest"] = sha256_bytes(canonical_json_bytes(receipt))
    write_create_once(receipt_path, canonical_json_bytes(receipt), mode=0o400)
    fsync_directory(archive)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--archive-dir", required=True)
    parser.add_argument("--revocation-manifest-sha256", required=True)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = harden(
        experiment_root=Path(args.experiment_root),
        archive_dir=Path(args.archive_dir),
        revocation_manifest_sha256=args.revocation_manifest_sha256,
        apply=args.apply,
    )
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
