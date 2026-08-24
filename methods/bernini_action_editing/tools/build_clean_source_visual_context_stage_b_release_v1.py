#!/usr/bin/env python3
"""Build the deterministic exact-member CSVC Stage-B runtime release."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile
import tempfile
from typing import Any, Mapping, NoReturn, Optional, Sequence


SCHEMA_VERSION = "bernini-clean-source-visual-context-stage-b-release-v1"
RELEASE_GENERATION = "r1"
ARCHIVE_FORMAT = "ustar-owner0-mtime0-mode0444-v1"
MEMBER_ROOT = "methods/bernini_action_editing"
RELEASE_FILES = (
    "clean_source_visual_context_adapter_v1.py",
    "clean_source_visual_context_training_v1.py",
    "clean_source_visual_context_stage_b_contract_v1.py",
    "train_clean_source_visual_context_stage_b_v1.py",
    "clean_source_visual_context_pair_controller_v1.py",
    "source_self_runtime.py",
    "train_lora.py",
    "inference_sigma_strata.py",
    "scripts/auh_preservation_rank_cache_exec_v1.sh",
    "scripts/auh_train_clean_source_visual_context_stage_b_holder_v1.sh",
    "scripts/auh_train_clean_source_visual_context_main_holder_v1.sh",
    "scripts/auh_train_clean_source_visual_context_noised_holder_v1.sh",
    "scripts/auh_preflight_clean_source_visual_context_main_holder_v1.sh",
    "scripts/auh_preflight_clean_source_visual_context_noised_holder_v1.sh",
    "scripts/auh_materialize_clean_source_visual_context_source_only_v3_holder_v1.sh",
)


class CleanSourceVisualReleaseError(RuntimeError):
    pass


def fail(message: str) -> NoReturn:
    raise CleanSourceVisualReleaseError(message)


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
        raise CleanSourceVisualReleaseError("release is not canonical JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _stable_plain_bytes(path: Path) -> bytes:
    if not path.is_absolute() or path.is_symlink():
        fail("release input must be absolute/non-symlink")
    resolved = path.resolve(strict=True)
    before = resolved.stat()
    if resolved != path or not stat.S_ISREG(before.st_mode):
        fail("release input must be a canonical plain file")
    raw = resolved.read_bytes()
    after = resolved.stat()
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or len(raw) != before.st_size
        or not raw
    ):
        fail("release input changed while reading")
    return raw


def build_manifest(method_root: Path) -> tuple[Mapping[str, Any], Mapping[str, bytes]]:
    root = method_root.resolve(strict=True)
    if root != method_root or root.is_symlink() or not root.is_dir():
        fail("method root must be canonical")
    rows = []
    payloads = {}
    for relative in RELEASE_FILES:
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
            fail("release member path differs")
        raw = _stable_plain_bytes(root / relative)
        payloads[relative] = raw
        rows.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size": len(raw),
                "mode": "0444",
            }
        )
    closure = {"member_root": MEMBER_ROOT, "files": rows}
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "release_generation": RELEASE_GENERATION,
        "archive_format": ARCHIVE_FORMAT,
        "member_root": MEMBER_ROOT,
        "file_count": len(rows),
        "files": rows,
        "revision_kind": "content-closure-sha1",
        "content_closure_sha1": hashlib.sha1(canonical_json_bytes(closure)).hexdigest(),
        "git_commit_claimed": False,
        "exact_member_closure": True,
    }
    return {**unsigned, "manifest_digest": object_sha256(unsigned)}, payloads


def build_archive(manifest: Mapping[str, Any], payloads: Mapping[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for row in manifest["files"]:
            relative = str(row["path"])
            raw = payloads[relative]
            member = tarfile.TarInfo(f"{MEMBER_ROOT}/{relative}")
            member.size = len(raw)
            member.mode = 0o444
            member.uid = 0
            member.gid = 0
            member.uname = ""
            member.gname = ""
            member.mtime = 0
            member.type = tarfile.REGTYPE
            tar.addfile(member, io.BytesIO(raw))
    return output.getvalue()


def verify_archive(raw: bytes, manifest: Mapping[str, Any]) -> None:
    expected = [f"{MEMBER_ROOT}/{row['path']}" for row in manifest["files"]]
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tar:
            members = tar.getmembers()
            if [member.name for member in members] != expected:
                fail("release archive exact member closure differs")
            for member, row in zip(members, manifest["files"]):
                payload = tar.extractfile(member)
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
                    fail(f"release archive member differs: {member.name}")
    except (OSError, tarfile.TarError) as error:
        raise CleanSourceVisualReleaseError(f"cannot verify release archive: {error}") from error


def _write_create_only(path: Path, raw: bytes) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        fail("release output must be a fresh absolute file")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build(method_root: Path, archive: Path, manifest_path: Path) -> Mapping[str, Any]:
    manifest, payloads = build_manifest(method_root)
    archive_raw = build_archive(manifest, payloads)
    verify_archive(archive_raw, manifest)
    if build_archive(manifest, payloads) != archive_raw:
        fail("release archive rebuild is not byte-identical")
    manifest_raw = canonical_json_bytes(manifest) + b"\n"
    _write_create_only(archive, archive_raw)
    _write_create_only(manifest_path, manifest_raw)
    value = {
        "schema_version": SCHEMA_VERSION,
        "archive": str(archive),
        "archive_sha256": hashlib.sha256(archive_raw).hexdigest(),
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "manifest_digest": manifest["manifest_digest"],
        "content_closure_sha1": manifest["content_closure_sha1"],
        "file_count": len(RELEASE_FILES),
    }
    return {**value, "digest": object_sha256(value)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-root", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)
    value = build(Path(args.method_root), Path(args.archive), Path(args.manifest))
    print(canonical_json_bytes(value).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
