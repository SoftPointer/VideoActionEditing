#!/usr/bin/env python3
"""Build the exact isolated Stage-B inference release.

The twelve training/runtime dependencies are copied from the audited r4
Stage-B archive, never from ambient workspace files.  Only the frozen
inference entry point and its frozen legacy inference helper are added from
the selected method tree.  Publication is deterministic and create-only.
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
import tempfile
from typing import Any, Mapping, Optional, Sequence


SCHEMA_VERSION = "bernini-source-noised-carrier-stage-b-inference-release-v1"
RELEASE_GENERATION = "r1"
ARCHIVE_FORMAT = "ustar-owner0-mtime0-mode0444-v1"
MEMBER_ROOT = "methods/bernini_action_editing"
BASE_ARCHIVE_SHA256 = "637eeaae2b44f40cb3691873a7a3336fcc1b275844bd89a25bab7d75bca150ed"
BASE_MANIFEST_SHA256 = "275c71035f9f1bca1fa3f756d27c30a3da6fa38f15c644f28163c0d6a0e03ded"
BASE_SCHEMA = "bernini-source-noised-carrier-stage-b-release-v1"
BASE_GENERATION = "r4"
BASE_FILES = (
    "source_self_role_repaint.py",
    "source_self_runtime.py",
    "train_source_self_role_repaint.py",
    "train_lora.py",
    "assets/source_self_role_repaint_canary_spec_v2.json",
    "tools/materialize_source_self_role_repaint.py",
    "tools/materialize_ramp_motion_analogy_vae.py",
    "tools/materialize_vae.py",
    "tools/build_renderer_dataset.py",
    "inference_sigma_strata.py",
    "source_noised_ladder_v1.py",
    "train_source_noised_carrier_strata_v1.py",
)
ADDED_FILES = (
    "infer_lora.py",
    "infer_source_noised_carrier_stage_b_v1.py",
)
RELEASE_FILES = BASE_FILES + ADDED_FILES
EXPECTED_FILE_SHA256 = {
    "source_self_role_repaint.py": "bf212ac4effcd5b3975eefc61e01c71cba366969ec92cf2ff186765ddec43f2e",
    "source_self_runtime.py": "62df125ac130697b03aaea167b17a02d7fcb9d766a72f0bef71037924114e59f",
    "train_source_self_role_repaint.py": "357ba5310a297c042e1c1bd10bef35bb69e483e18ff15b5ba4cc2bd65a07c80d",
    "train_lora.py": "630c215240d4547ea0c347b9fb0bf21324ffe5ee229c5f3673d586a4a0eab4d5",
    "assets/source_self_role_repaint_canary_spec_v2.json": "62468b24d4a57ec03d42ce8c006a707cbcf56588ef62d10632089eb5ad457920",
    "tools/materialize_source_self_role_repaint.py": "8065cafc34c15d7e8e6fc8e3abb13551b2cbe20c925ab8415267be5b3993cc80",
    "tools/materialize_ramp_motion_analogy_vae.py": "ca9b4620ad7dc6cd03e70b180f68d83aad05c21cef574fe6467bdaa1202bb93a",
    "tools/materialize_vae.py": "a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0",
    "tools/build_renderer_dataset.py": "afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5",
    "inference_sigma_strata.py": "e3782a22130c09a48dc3ea27fa219af6caca445e1fce2c8f3bca7cde6058afd3",
    "source_noised_ladder_v1.py": "eb8653a5e98d0744c9fd7066f3aefc4c5e0dfcd8f70320e86a2e669a376fef98",
    "train_source_noised_carrier_strata_v1.py": "39c3fad7e8d710eedd453e75b1acf7fb35f30c0ccba4dee71d336efec5274704",
    "infer_lora.py": "babd6d63287723ccd14b2bbe43bd4550c30b4feaa794d17c66f5a5ddefe979fe",
    "infer_source_noised_carrier_stage_b_v1.py": "b21f7f85531fd7f41f1a9741894b26b564b25054da418d7989f2f7a588a6f84f",
}


class ReleaseError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _plain_file(path: Path, *, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise ReleaseError(f"{label} must be an absolute non-symlink file")
    resolved = path.resolve(strict=True)
    if resolved != path or not stat.S_ISREG(resolved.lstat().st_mode):
        raise ReleaseError(f"{label} must be a canonical plain file")
    return resolved


def _stable_bytes(path: Path, *, label: str) -> bytes:
    path = _plain_file(path, label=label)
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
    if identity(before) != identity(after) or len(raw) != before.st_size:
        raise ReleaseError(f"{label} changed while reading")
    return raw


def _validated_base_payloads(archive_path: Path, manifest_path: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    archive_raw = _stable_bytes(archive_path, label="base archive")
    manifest_raw = _stable_bytes(manifest_path, label="base manifest")
    if hashlib.sha256(archive_raw).hexdigest() != BASE_ARCHIVE_SHA256:
        raise ReleaseError("base archive SHA differs")
    if hashlib.sha256(manifest_raw).hexdigest() != BASE_MANIFEST_SHA256:
        raise ReleaseError("base manifest SHA differs")
    try:
        manifest = json.loads(manifest_raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ReleaseError("cannot parse base manifest") from error
    unsigned = dict(manifest)
    declared = unsigned.pop("manifest_digest", None)
    rows = manifest.get("files")
    if (
        manifest.get("schema_version") != BASE_SCHEMA
        or manifest.get("release_generation") != BASE_GENERATION
        or manifest.get("archive_format") != ARCHIVE_FORMAT
        or manifest.get("member_root") != MEMBER_ROOT
        or manifest.get("file_count") != len(BASE_FILES)
        or manifest.get("exact_member_closure") is not True
        or manifest.get("git_commit_claimed") is not False
        or declared != object_sha256(unsigned)
        or not isinstance(rows, list)
        or [row.get("path") for row in rows if isinstance(row, Mapping)] != list(BASE_FILES)
    ):
        raise ReleaseError("base manifest contract differs")
    payloads: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_raw), mode="r:") as archive:
            members = archive.getmembers()
            expected_names = [f"{MEMBER_ROOT}/{name}" for name in BASE_FILES]
            if [member.name for member in members] != expected_names:
                raise ReleaseError("base archive member closure differs")
            for member, row, relative in zip(members, rows, BASE_FILES):
                stream = archive.extractfile(member)
                if (
                    not member.isfile() or member.issym() or member.islnk()
                    or member.uid != 0 or member.gid != 0 or member.mtime != 0
                    or stat.S_IMODE(member.mode) != 0o444 or stream is None
                ):
                    raise ReleaseError(f"base member metadata differs: {relative}")
                raw = stream.read()
                digest = hashlib.sha256(raw).hexdigest()
                if digest != row.get("sha256") or digest != EXPECTED_FILE_SHA256[relative] or len(raw) != row.get("size"):
                    raise ReleaseError(f"base member bytes differ: {relative}")
                payloads[relative] = raw
    except (OSError, tarfile.TarError) as error:
        raise ReleaseError("cannot verify base archive") from error
    return payloads, manifest


def build_manifest(method_root: Path, base_archive: Path, base_manifest: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    root = method_root.resolve(strict=True)
    if root != method_root or not root.is_dir() or root.is_symlink():
        raise ReleaseError("method root must be a canonical plain directory")
    payloads, base = _validated_base_payloads(base_archive, base_manifest)
    for relative in ADDED_FILES:
        if PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
            raise ReleaseError("release path traversal is forbidden")
        raw = _stable_bytes(root / relative, label=relative)
        if hashlib.sha256(raw).hexdigest() != EXPECTED_FILE_SHA256[relative]:
            raise ReleaseError(f"frozen inference source SHA differs: {relative}")
        payloads[relative] = raw
    rows = [
        {"path": name, "sha256": hashlib.sha256(payloads[name]).hexdigest(), "size": len(payloads[name]), "mode": "0444"}
        for name in RELEASE_FILES
    ]
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
        "base_archive_sha256": BASE_ARCHIVE_SHA256,
        "base_manifest_sha256": BASE_MANIFEST_SHA256,
        "base_manifest_digest": base["manifest_digest"],
        "source_self_runtime_from_audited_base_archive": True,
        "git_commit_claimed": False,
        "exact_member_closure": True,
    }
    return {**unsigned, "manifest_digest": object_sha256(unsigned)}, payloads


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    member = tarfile.TarInfo(name)
    member.size = size
    member.mode = 0o444
    member.uid = member.gid = member.mtime = 0
    member.uname = member.gname = ""
    member.type = tarfile.REGTYPE
    return member


def build_archive(manifest: Mapping[str, Any], payloads: Mapping[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for row in manifest["files"]:
            raw = payloads[str(row["path"])]
            archive.addfile(_tar_info(f"{MEMBER_ROOT}/{row['path']}", len(raw)), io.BytesIO(raw))
    return buffer.getvalue()


def verify_archive_bytes(raw: bytes, manifest: Mapping[str, Any]) -> None:
    expected = [f"{MEMBER_ROOT}/{row['path']}" for row in manifest["files"]]
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        members = archive.getmembers()
        if [member.name for member in members] != expected:
            raise ReleaseError("inference archive closure differs")
        for member, row in zip(members, manifest["files"]):
            stream = archive.extractfile(member)
            if (
                not member.isfile() or member.issym() or member.islnk() or stream is None
                or member.uid != 0 or member.gid != 0 or member.mtime != 0
                or stat.S_IMODE(member.mode) != 0o444
                or member.size != row["size"]
                or hashlib.sha256(stream.read()).hexdigest() != row["sha256"]
            ):
                raise ReleaseError(f"inference archive member differs: {member.name}")


def _write_create_only(path: Path, raw: bytes) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise ReleaseError("release outputs must be fresh absolute paths")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build(method_root: Path, base_archive: Path, base_manifest: Path, archive: Path, manifest_path: Path) -> dict[str, Any]:
    manifest, payloads = build_manifest(method_root, base_archive, base_manifest)
    raw = build_archive(manifest, payloads)
    verify_archive_bytes(raw, manifest)
    if build_archive(manifest, payloads) != raw:
        raise ReleaseError("inference archive rebuild is not byte-identical")
    manifest_raw = canonical_json_bytes(manifest) + b"\n"
    _write_create_only(archive, raw)
    _write_create_only(manifest_path, manifest_raw)
    return {
        "schema_version": SCHEMA_VERSION,
        "archive_sha256": hashlib.sha256(raw).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "manifest_digest": manifest["manifest_digest"],
        "content_closure_sha1": manifest["content_closure_sha1"],
        "file_count": manifest["file_count"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-root", required=True)
    parser.add_argument("--base-archive", required=True)
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)
    result = build(*(Path(value) for value in (args.method_root, args.base_archive, args.base_manifest, args.archive, args.manifest)))
    print(canonical_json_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
