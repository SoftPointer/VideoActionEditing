#!/usr/bin/env python3
"""Deterministic exact-member release for packed preservation v2.

The archive is the executable closure: no caller may combine an authenticated
archive with a different method directory.  Python/data-contract sources are
read-only (0444); the two shell entry points are executable/read-only (0555).
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
from typing import Any, Mapping, NoReturn, Optional, Sequence


SCHEMA_VERSION = "bernini-packed-preservation-release-v2"
ARCHIVE_FORMAT = "ustar-owner0-mtime0-exact-modes-v2"
MEMBER_ROOT = "methods/bernini_action_editing"
EXPECTED_CHECKPOINT_CONTENT_FILE_COUNT = 23
CHECKPOINT_TREE_SHA256 = (
    "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
)
FILES_AND_MODES = {
    "packed_preservation_lora_v2.py": 0o444,
    "packed_preservation_release_v2.py": 0o444,
    "train_packed_preservation_lora_v2.py": 0o444,
    "clean_source_visual_context_stage_b_contract_v1.py": 0o444,
    "clean_source_visual_context_training_v1.py": 0o444,
    "inference_sigma_strata.py": 0o444,
    "source_self_runtime.py": 0o444,
    "train_lora.py": 0o444,
    "scripts/auh_packed_preservation_rank_exec_v2.sh": 0o555,
    "scripts/auh_train_packed_preservation_lora_v2_job136140.sh": 0o555,
}
_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class PackedPreservationReleaseError(RuntimeError):
    """Raised when archive bytes and executed bytes are not one closure."""


def fail(message: str) -> NoReturn:
    raise PackedPreservationReleaseError(message)


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
        raise PackedPreservationReleaseError("release JSON is not canonical") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(handle.fileno())
    named = path.stat()
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before) != identity(after) or identity(before) != identity(named):
        fail(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _plain_file(path_value: str | Path, *, label: str) -> Path:
    requested = Path(path_value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail(f"{label} must be an absolute non-symlink file")
    resolved = requested.resolve(strict=True)
    if resolved != requested or not stat.S_ISREG(resolved.lstat().st_mode):
        fail(f"{label} must be one canonical plain file")
    return resolved


def _method_root(path_value: str | Path, *, require_exact: bool) -> Path:
    requested = Path(path_value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail("method root must be an absolute non-symlink directory")
    root = requested.resolve(strict=True)
    if root != requested or not root.is_dir():
        fail("method root differs")
    for relative in FILES_AND_MODES:
        path = root / relative
        if (
            path.resolve(strict=True) != path
            or path.is_symlink()
            or not stat.S_ISREG(path.lstat().st_mode)
        ):
            fail(f"release member is not a canonical plain file: {relative}")
    if require_exact:
        observed: set[str] = set()
        for path in root.rglob("*"):
            relative = path.relative_to(root).as_posix()
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                fail(f"executed method root contains symlink: {relative}")
            if stat.S_ISREG(mode):
                observed.add(relative)
            elif not stat.S_ISDIR(mode):
                fail(f"executed method root contains special entry: {relative}")
        if observed != set(FILES_AND_MODES):
            fail("executed method root is not the exact release file closure")
    return root


def _member_rows(root: Path) -> list[Mapping[str, Any]]:
    rows = []
    for relative in sorted(FILES_AND_MODES):
        path = root / relative
        payload = path.read_bytes()
        rows.append(
            {
                "path": relative,
                "mode": FILES_AND_MODES[relative],
                "size": len(payload),
                "sha256": bytes_sha256(payload),
            }
        )
    return rows


def content_closure_sha1(rows: Sequence[Mapping[str, Any]]) -> str:
    projection = {
        "schema_version": SCHEMA_VERSION,
        "member_root": MEMBER_ROOT,
        "files": list(rows),
    }
    return hashlib.sha1(canonical_json_bytes(projection)).hexdigest()


def build_release(
    *, method_root: str | Path, archive: str | Path, manifest: str | Path,
    revision: Optional[str] = None
) -> Mapping[str, Any]:
    """Create one byte-deterministic ustar archive and canonical manifest."""

    root = _method_root(method_root, require_exact=False)
    archive_path = Path(archive).expanduser()
    manifest_path = Path(manifest).expanduser()
    for path, label in ((archive_path, "archive"), (manifest_path, "manifest")):
        if (
            not path.is_absolute()
            or path.exists()
            or path.is_symlink()
            or path.parent.resolve(strict=True) != path.parent
        ):
            fail(f"release {label} must be a fresh absolute path")
    rows = _member_rows(root)
    computed_revision = content_closure_sha1(rows)
    if revision is not None and revision != computed_revision:
        fail("caller revision differs from deterministic content closure")
    with archive_path.open("xb") as raw:
        with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as bundle:
            for row in rows:
                relative = str(row["path"])
                payload = (root / relative).read_bytes()
                info = tarfile.TarInfo(f"{MEMBER_ROOT}/{relative}")
                info.size = len(payload)
                info.mode = int(row["mode"])
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                info.type = tarfile.REGTYPE
                bundle.addfile(info, io.BytesIO(payload))
        raw.flush()
        os.fsync(raw.fileno())
    archive_sha = file_sha256(archive_path)
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "archive_format": ARCHIVE_FORMAT,
        "member_root": MEMBER_ROOT,
        "revision_kind": "content-closure-sha1",
        "method_revision": computed_revision,
        "archive_sha256": archive_sha,
        "exact_member_closure": True,
        "file_count": len(rows),
        "files": rows,
    }
    value = {**unsigned, "manifest_digest": object_sha256(unsigned)}
    with manifest_path.open("xb") as handle:
        handle.write(canonical_json_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "archive": str(archive_path),
        "archive_sha256": archive_sha,
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "manifest_digest": value["manifest_digest"],
        "method_revision": computed_revision,
    }


def _read_manifest(path: Path) -> Mapping[str, Any]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PackedPreservationReleaseError("cannot decode release manifest") from error
    if not isinstance(value, Mapping) or raw != canonical_json_bytes(value) + b"\n":
        fail("release manifest is not one canonical JSON object")
    return value


def validate_executed_release(
    *, method_root: str | Path, archive: str | Path, manifest: str | Path,
    expected_archive_sha256: str, expected_manifest_sha256: str,
    method_revision: str
) -> Mapping[str, Any]:
    """Bind archive, canonical manifest, and the exact executed filesystem."""

    if (
        _SHA1.fullmatch(method_revision) is None
        or _SHA256.fullmatch(expected_archive_sha256) is None
        or _SHA256.fullmatch(expected_manifest_sha256) is None
    ):
        fail("expected release identities differ")
    root = _method_root(method_root, require_exact=True)
    archive_path = _plain_file(archive, label="method archive")
    manifest_path = _plain_file(manifest, label="method manifest")
    if file_sha256(archive_path) != expected_archive_sha256:
        fail("method archive SHA differs")
    if file_sha256(manifest_path) != expected_manifest_sha256:
        fail("method manifest SHA differs")
    value = _read_manifest(manifest_path)
    required = {
        "schema_version", "archive_format", "member_root", "revision_kind",
        "method_revision", "archive_sha256", "exact_member_closure",
        "file_count", "files", "manifest_digest",
    }
    unsigned = dict(value)
    declared_digest = unsigned.pop("manifest_digest", None)
    rows = value.get("files")
    if (
        set(value) != required
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("archive_format") != ARCHIVE_FORMAT
        or value.get("member_root") != MEMBER_ROOT
        or value.get("revision_kind") != "content-closure-sha1"
        or value.get("method_revision") != method_revision
        or value.get("archive_sha256") != expected_archive_sha256
        or value.get("exact_member_closure") is not True
        or value.get("file_count") != len(FILES_AND_MODES)
        or not isinstance(rows, list)
        or declared_digest != object_sha256(unsigned)
    ):
        fail("release manifest schema/digest differs")
    expected_rows = _member_rows(root)
    if rows != expected_rows:
        fail("executed method-root bytes/modes differ from release manifest")
    if content_closure_sha1(expected_rows) != method_revision:
        fail("method revision differs from recomputed content closure")
    for row in expected_rows:
        mode = stat.S_IMODE((root / str(row["path"])).stat().st_mode)
        if mode != row["mode"]:
            fail(f"executed release mode differs: {row['path']}")
    with tarfile.open(archive_path, mode="r:") as bundle:
        members = bundle.getmembers()
        expected_names = [f"{MEMBER_ROOT}/{row['path']}" for row in expected_rows]
        if [member.name for member in members] != expected_names:
            fail("archive exact member order/set differs")
        for member, row in zip(members, expected_rows):
            if (
                not member.isfile()
                or member.mode != row["mode"]
                or member.uid != 0
                or member.gid != 0
                or member.uname != ""
                or member.gname != ""
                or member.mtime != 0
                or member.size != row["size"]
            ):
                fail(f"archive member metadata differs: {member.name}")
            handle = bundle.extractfile(member)
            if handle is None or bytes_sha256(handle.read()) != row["sha256"]:
                fail(f"archive member bytes differ: {member.name}")
    result = {
        "method_root": str(root),
        "archive": str(archive_path),
        "archive_sha256": expected_archive_sha256,
        "manifest": str(manifest_path),
        "manifest_sha256": expected_manifest_sha256,
        "manifest_digest": declared_digest,
        "method_revision": method_revision,
        "exact_member_count": len(expected_rows),
        "archive_members_verified": True,
        "executed_root_exact_closure_verified": True,
        "executed_modes_verified": True,
    }
    return {**result, "digest": object_sha256(result)}


def validate_checkpoint_content(
    checkpoint: Path, manifest_path: Path, *, expected_manifest_sha256: str
) -> Mapping[str, Any]:
    """Self-contained every-file audit for the pinned Bernini checkpoint."""

    root = checkpoint.resolve(strict=True)
    manifest = _plain_file(manifest_path, label="checkpoint content manifest")
    if root != checkpoint or root.is_symlink() or not root.is_dir():
        fail("base checkpoint root differs")
    if file_sha256(manifest) != expected_manifest_sha256:
        fail("checkpoint content manifest SHA differs")
    lines = manifest.read_text(encoding="utf-8").splitlines()
    if len(lines) != EXPECTED_CHECKPOINT_CONTENT_FILE_COUNT:
        fail("checkpoint content manifest count differs")
    expected: dict[str, str] = {}
    expression = re.compile(r"([0-9a-f]{64})  (\./[^\n]+)")
    for line in lines:
        match = expression.fullmatch(line)
        if match is None:
            fail("checkpoint manifest line differs")
        digest, raw_path = match.groups()
        relative = PurePosixPath(raw_path)
        normalized = PurePosixPath(
            *(part for part in relative.parts if part not in ("", "."))
        ).as_posix()
        if relative.is_absolute() or ".." in relative.parts or not normalized or normalized in expected:
            fail("checkpoint manifest path differs")
        expected[normalized] = digest
    actual: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if ".cache" in relative.parts:
            continue
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            fail("checkpoint contains non-cache symlink")
        if stat.S_ISREG(mode):
            actual.add(relative.as_posix())
        elif not stat.S_ISDIR(mode):
            fail("checkpoint contains special entry")
    if actual != set(expected):
        fail("checkpoint file set differs")
    rows = []
    for relative in sorted(expected):
        path = root / relative
        digest = file_sha256(path)
        if path.resolve(strict=True) != path or path.is_symlink() or digest != expected[relative]:
            fail(f"checkpoint file bytes differ: {relative}")
        rows.append({"path": relative, "sha256": digest})
    result = {
        "checkpoint_root": str(root),
        "tree_sha256": CHECKPOINT_TREE_SHA256,
        "manifest_path": str(manifest),
        "manifest_sha256": expected_manifest_sha256,
        "verified_file_count": len(rows),
        "every_non_cache_file_sha256_verified": True,
        "verified_entries_digest": object_sha256(rows),
    }
    return {**result, "digest": object_sha256(result)}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--method-root", required=True)
    value.add_argument("--archive", required=True)
    value.add_argument("--manifest", required=True)
    value.add_argument("--revision")
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    result = build_release(
        method_root=args.method_root,
        archive=args.archive,
        manifest=args.manifest,
        revision=args.revision,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FILES_AND_MODES",
    "PackedPreservationReleaseError",
    "build_release",
    "content_closure_sha1",
    "validate_checkpoint_content",
    "validate_executed_release",
]
