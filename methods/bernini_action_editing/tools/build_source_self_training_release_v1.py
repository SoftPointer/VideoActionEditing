#!/usr/bin/env python3
"""Build a deterministic exact-closure source-self training release.

The release is intentionally content-addressed.  It can include authorized
workspace bytes that are not in the repository HEAD without mislabelling them
as a Git commit.  The manifest binds every member, an exact USTAR archive, and
a synthetic SHA-1 over the canonical file closure for the trainer's existing
40-hex revision field.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import tempfile
from typing import Any, Mapping, Optional, Sequence


SCHEMA_VERSION = "bernini-source-self-training-release-v1"
ARCHIVE_FORMAT = "ustar-owner0-mtime0-mode0444-v1"
MEMBER_ROOT = "methods/bernini_action_editing"
RELEASE_FILES = (
    "source_self_role_repaint.py",
    "source_self_runtime.py",
    "train_source_self_role_repaint.py",
    "train_lora.py",
    "assets/source_self_role_repaint_canary_spec_v2.json",
    "tools/materialize_source_self_role_repaint.py",
    "tools/materialize_ramp_motion_analogy_vae.py",
    "tools/materialize_vae.py",
    "tools/build_renderer_dataset.py",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ReleaseError(RuntimeError):
    """Raised before a non-exact or mutable release can be published."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _plain_file(path: Path, *, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise ReleaseError(f"{label} must be an absolute non-symlink file")
    try:
        resolved = path.resolve(strict=True)
        mode = resolved.lstat().st_mode
    except OSError as error:
        raise ReleaseError(f"cannot resolve {label}: {error}") from error
    if resolved != path or not stat.S_ISREG(mode):
        raise ReleaseError(f"{label} must be a canonical plain file")
    return resolved


def _read_stable(path: Path) -> bytes:
    path = _plain_file(path, label=str(path))
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
    if identity(before) != identity(after) or len(raw) != before.st_size:
        raise ReleaseError(f"source changed while reading: {path}")
    return raw


def _write_create_only(path: Path, payload: bytes) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise ReleaseError(f"output must be a fresh absolute path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor: Optional[int] = None
    temporary: Optional[Path] = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
        temporary.unlink()
        temporary = None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    member = tarfile.TarInfo(name=name)
    member.size = size
    member.mode = 0o444
    member.uid = 0
    member.gid = 0
    member.uname = ""
    member.gname = ""
    member.mtime = 0
    member.type = tarfile.REGTYPE
    return member


def build_manifest(method_root: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    root = method_root.resolve(strict=True)
    if root != method_root or not root.is_dir() or root.is_symlink():
        raise ReleaseError("method root must be a canonical plain directory")
    payloads: dict[str, bytes] = {}
    rows: list[dict[str, Any]] = []
    for relative in RELEASE_FILES:
        if PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
            raise ReleaseError("release path traversal is forbidden")
        raw = _read_stable(root / relative)
        payloads[relative] = raw
        rows.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size": len(raw),
                "mode": "0444",
            }
        )
    closure_payload = {"member_root": MEMBER_ROOT, "files": rows}
    unsigned: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "archive_format": ARCHIVE_FORMAT,
        "member_root": MEMBER_ROOT,
        "file_count": len(rows),
        "files": rows,
        "revision_kind": "content-closure-sha1",
        "content_closure_sha1": hashlib.sha1(
            canonical_json_bytes(closure_payload)
        ).hexdigest(),
        "git_commit_claimed": False,
        "exact_member_closure": True,
    }
    return {**unsigned, "manifest_digest": object_sha256(unsigned)}, payloads


def build_archive(manifest: Mapping[str, Any], payloads: Mapping[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for row in manifest["files"]:
            relative = str(row["path"])
            raw = payloads[relative]
            archive.addfile(
                _tar_info(f"{MEMBER_ROOT}/{relative}", len(raw)), io.BytesIO(raw)
            )
    return buffer.getvalue()


def verify_archive_bytes(raw: bytes, manifest: Mapping[str, Any]) -> None:
    expected = [f"{MEMBER_ROOT}/{row['path']}" for row in manifest["files"]]
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
            members = archive.getmembers()
            if [item.name for item in members] != expected:
                raise ReleaseError("archive member closure differs")
            for member, row in zip(members, manifest["files"]):
                payload = archive.extractfile(member)
                if (
                    not member.isfile()
                    or member.issym()
                    or member.islnk()
                    or member.uid != 0
                    or member.gid != 0
                    or member.mtime != 0
                    or stat.S_IMODE(member.mode) != 0o444
                    or member.size != row["size"]
                    or payload is None
                    or hashlib.sha256(payload.read()).hexdigest() != row["sha256"]
                ):
                    raise ReleaseError(f"archive member differs: {member.name}")
    except (OSError, tarfile.TarError) as error:
        raise ReleaseError(f"cannot verify archive: {error}") from error


def build(method_root: Path, archive: Path, manifest_path: Path) -> dict[str, Any]:
    manifest, payloads = build_manifest(method_root)
    raw = build_archive(manifest, payloads)
    verify_archive_bytes(raw, manifest)
    if build_archive(manifest, payloads) != raw:
        raise ReleaseError("archive rebuild is not byte-identical")
    manifest_raw = canonical_json_bytes(manifest) + b"\n"
    _write_create_only(archive, raw)
    _write_create_only(manifest_path, manifest_raw)
    if archive.read_bytes() != raw or manifest_path.read_bytes() != manifest_raw:
        raise ReleaseError("published release differs on reread")
    return {
        "schema_version": SCHEMA_VERSION,
        "archive_sha256": hashlib.sha256(raw).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "manifest_digest": manifest["manifest_digest"],
        "content_closure_sha1": manifest["content_closure_sha1"],
        "file_count": manifest["file_count"],
        "git_commit_claimed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-root", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--manifest", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = build(
        Path(args.method_root), Path(args.archive), Path(args.manifest)
    )
    print(canonical_json_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
