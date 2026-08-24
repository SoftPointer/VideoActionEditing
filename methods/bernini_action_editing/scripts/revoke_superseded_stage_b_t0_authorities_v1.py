#!/usr/bin/env python3
"""Archive and revoke superseded Stage-B T0 optimizer authorities.

The operation is deliberately create-once.  A dry run is the default; ``--apply``
requires a fresh archive directory, archives every original byte string, and
then atomically replaces each superseded ACTIVE authority with a fail-closed
REVOKED receipt.  Authorities whose file SHA-256 equals ``--keep-active-sha256``
are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence


ACTIVE_STATE = "ACTIVE_CREATE_ONCE_AUTHORITY"
REVOKED_STATE = "REVOKED_SUPERSEDED_BY_RETRY6"
SCHEMA_VERSION = "bernini-stage-b-t0-authority-revocation-v1"
NAME_GLOB = "stage_b_t0_single_update*authority_addendum.json"


class RevocationError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise RevocationError(message)


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


def read_regular_json(path: Path, *, root: Path) -> tuple[bytes, Mapping[str, Any]]:
    if not inside(path.absolute(), root):
        fail(f"authority escaped experiment root: {path}")
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        fail(f"authority is not one regular non-symlink file: {path}")
    if metadata.st_nlink != 1:
        fail(f"authority hard-link count differs: {path}")
    if metadata.st_size > 1024 * 1024:
        fail(f"authority is unexpectedly large: {path}")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        fail("authority revocation requires O_NOFOLLOW")
    descriptor = os.open(path, os.O_RDONLY | nofollow)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
            fail(f"authority changed while opening: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 65536)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RevocationError(f"authority JSON is invalid: {path}") from error
    if not isinstance(value, Mapping):
        fail(f"authority JSON is not an object: {path}")
    return payload, value


def discover(
    *, experiment_root: Path, keep_active_sha256: str, archive_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = experiment_root.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        fail("experiment root must be one real directory")
    archive_candidate = archive_dir.expanduser().absolute()
    try:
        archive = archive_candidate.parent.resolve(strict=True) / archive_candidate.name
    except (FileNotFoundError, OSError) as error:
        raise RevocationError("archive parent must already be one real directory") from error
    if not inside(archive, root) or archive == root:
        fail("archive directory must be a strict descendant of experiment root")
    revoke: list[dict[str, Any]] = []
    keep: list[dict[str, Any]] = []
    for path in sorted(root.rglob(NAME_GLOB)):
        absolute = path.absolute()
        if inside(absolute, archive):
            continue
        payload, value = read_regular_json(absolute, root=root)
        state = value.get("activation", {}).get("state") if isinstance(value.get("activation"), Mapping) else None
        if state != ACTIVE_STATE:
            continue
        digest = sha256_bytes(payload)
        row = {
            "path": str(absolute),
            "relative_path": str(absolute.relative_to(root)),
            "original_sha256": digest,
            "original_size": len(payload),
            "payload": payload,
        }
        (keep if digest == keep_active_sha256 else revoke).append(row)
    if not keep:
        fail("no ACTIVE authority matches --keep-active-sha256")
    if not revoke:
        fail("no superseded ACTIVE authority remains to revoke")
    return revoke, keep


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


def revoke_authorities(
    *,
    experiment_root: Path,
    archive_dir: Path,
    keep_active_sha256: str,
    reason: str,
    apply: bool,
) -> Mapping[str, Any]:
    keep_sha = require_sha256(keep_active_sha256, label="kept ACTIVE authority")
    root = experiment_root.resolve(strict=True)
    archive_candidate = archive_dir.expanduser().absolute()
    try:
        archive = archive_candidate.parent.resolve(strict=True) / archive_candidate.name
    except (FileNotFoundError, OSError) as error:
        raise RevocationError("archive parent must already be one real directory") from error
    revoke, keep = discover(
        experiment_root=root,
        keep_active_sha256=keep_sha,
        archive_dir=archive,
    )
    public_plan = {
        "schema_version": SCHEMA_VERSION,
        "mode": "apply" if apply else "dry_run",
        "experiment_root": str(root),
        "archive_dir": str(archive),
        "kept_active_authority_sha256": keep_sha,
        "kept_active_paths": [row["path"] for row in keep],
        "superseded_active_count": len(revoke),
        "superseded": [
            {
                key: row[key]
                for key in ("path", "relative_path", "original_sha256", "original_size")
            }
            for row in revoke
        ],
        "reason": reason,
    }
    if not apply:
        return public_plan
    if archive.exists() or archive.is_symlink():
        fail("create-once archive directory already exists")
    archive.mkdir(mode=0o700, parents=False)
    originals = archive / "originals"
    originals.mkdir(mode=0o700)
    fsync_directory(archive.parent)

    planned_payload = canonical_json_bytes(public_plan)
    write_create_once(archive / "plan.json", planned_payload, mode=0o400)
    applied_rows: list[dict[str, Any]] = []
    for row in revoke:
        original_path = Path(row["path"])
        archive_path = originals / row["relative_path"]
        archive_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        write_create_once(archive_path, row["payload"], mode=0o400)
        if sha256_bytes(archive_path.read_bytes()) != row["original_sha256"]:
            fail(f"archived authority SHA-256 differs: {original_path}")
        stub = {
            "schema_version": SCHEMA_VERSION,
            "activation": {
                "state": REVOKED_STATE,
                "optimizer_creation_authorized": False,
                "reversible_from_byte_exact_archive": True,
            },
            "original": {
                "path": row["path"],
                "relative_path": row["relative_path"],
                "sha256": row["original_sha256"],
                "size": row["original_size"],
                "archive_path": str(archive_path),
            },
            "superseded_by": {"active_authority_sha256": keep_sha},
            "reason": reason,
        }
        stub_payload = canonical_json_bytes(stub)
        temporary = original_path.with_name(
            f".{original_path.name}.revoke-retry6-{os.getpid()}"
        )
        write_create_once(temporary, stub_payload, mode=0o600)
        os.replace(temporary, original_path)
        fsync_directory(original_path.parent)
        observed = original_path.read_bytes()
        if observed != stub_payload:
            fail(f"revoked authority bytes differ after atomic replace: {original_path}")
        applied_rows.append(
            {
                "path": row["path"],
                "original_sha256": row["original_sha256"],
                "archive_path": str(archive_path),
                "revoked_sha256": sha256_bytes(stub_payload),
            }
        )

    for row in keep:
        payload, value = read_regular_json(Path(row["path"]), root=root)
        if (
            sha256_bytes(payload) != keep_sha
            or value.get("activation", {}).get("state") != ACTIVE_STATE
        ):
            fail("kept retry6 authority changed during revocation")
    manifest = dict(public_plan)
    manifest["mode"] = "applied"
    manifest["applied"] = applied_rows
    manifest["plan_sha256"] = sha256_bytes(planned_payload)
    manifest["manifest_digest"] = sha256_bytes(canonical_json_bytes(manifest))
    write_create_once(archive / "manifest.json", canonical_json_bytes(manifest), mode=0o400)
    fsync_directory(archive)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--archive-dir", required=True)
    parser.add_argument("--keep-active-sha256", required=True)
    parser.add_argument(
        "--reason",
        default="superseded authorities bypass retry6 permanent preoptimizer claim",
    )
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = revoke_authorities(
        experiment_root=Path(args.experiment_root),
        archive_dir=Path(args.archive_dir),
        keep_active_sha256=args.keep_active_sha256,
        reason=args.reason,
        apply=args.apply,
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
