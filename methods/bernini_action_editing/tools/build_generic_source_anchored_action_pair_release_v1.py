#!/usr/bin/env python3
"""Build/audit a deterministic exact-member generic action training release."""

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


SCHEMA_VERSION = "bernini-generic-source-anchored-action-pair-release-v1"
RELEASE_GENERATION = "r2"
ARCHIVE_FORMAT = "ustar-owner0-mtime0-exact-modes-v1"
MEMBER_ROOT = "methods/bernini_action_editing"
FILES_AND_MODES: Mapping[str, int] = {
    "generic_source_anchored_action_v1.py": 0o444,
    "train_generic_source_anchored_action_v1.py": 0o444,
    "generic_source_anchored_action_pair_controller_v1.py": 0o444,
    "clean_source_visual_context_adapter_v1.py": 0o444,
    "clean_source_visual_context_training_v1.py": 0o444,
    "clean_source_visual_context_stage_b_contract_v1.py": 0o444,
    "clean_source_visual_context_pair_controller_v1.py": 0o444,
    "train_clean_source_visual_context_stage_b_v1.py": 0o444,
    "source_self_runtime.py": 0o444,
    "train_lora.py": 0o444,
    "inference_sigma_strata.py": 0o444,
    "tools/build_generic_source_anchored_action_pair_release_v1.py": 0o444,
    "scripts/auh_generic_source_anchored_action_rank_exec_v1.sh": 0o555,
    "scripts/auh_smoke_generic_source_anchored_action_r_136309_v1.sh": 0o555,
    "scripts/auh_train_generic_source_anchored_action_stage_r_136309_v1.sh": 0o555,
    "scripts/auh_train_generic_source_anchored_action_world4_holder_v1.sh": 0o555,
}
COMPONENT_FILES: Mapping[str, str] = {
    "trainer_sha256": "train_generic_source_anchored_action_v1.py",
    "core_sha256": "generic_source_anchored_action_v1.py",
    "controller_sha256": "generic_source_anchored_action_pair_controller_v1.py",
    "launcher_sha256": (
        "scripts/auh_train_generic_source_anchored_action_world4_holder_v1.sh"
    ),
}


class GenericActionReleaseError(RuntimeError):
    pass


def fail(message: str) -> NoReturn:
    raise GenericActionReleaseError(message)


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
        raise GenericActionReleaseError("release is not canonical JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _stable_plain_bytes(path: Path) -> bytes:
    if not path.is_absolute() or path.is_symlink():
        fail("release input must be absolute and non-symlink")
    resolved = path.resolve(strict=True)
    before = resolved.stat()
    if resolved != path or not stat.S_ISREG(before.st_mode):
        fail("release input must be one canonical plain file")
    raw = resolved.read_bytes()
    after = resolved.stat()
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or len(raw) != before.st_size
        or not raw
    ):
        fail("release input changed while reading or is empty")
    return raw


def build_manifest(method_root: Path) -> tuple[Mapping[str, Any], Mapping[str, bytes]]:
    root = method_root.resolve(strict=True)
    if root != method_root or root.is_symlink() or not root.is_dir():
        fail("method root must be one canonical directory")
    rows = []
    payloads: dict[str, bytes] = {}
    for relative in sorted(FILES_AND_MODES):
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
            fail("release member path differs")
        raw = _stable_plain_bytes(root / relative)
        payloads[relative] = raw
        rows.append(
            {
                "path": relative,
                "mode": FILES_AND_MODES[relative],
                "size": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    row_by_path = {str(row["path"]): row for row in rows}
    component_pins = {
        label: row_by_path[relative]["sha256"]
        for label, relative in COMPONENT_FILES.items()
    }
    closure = {"member_root": MEMBER_ROOT, "files": rows}
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "release_generation": RELEASE_GENERATION,
        "archive_format": ARCHIVE_FORMAT,
        "member_root": MEMBER_ROOT,
        "file_count": len(rows),
        "files": rows,
        "component_pins": component_pins,
        "revision_kind": "content-closure-sha1",
        "content_closure_sha1": hashlib.sha1(
            canonical_json_bytes(closure)
        ).hexdigest(),
        "exact_member_closure": True,
        "release_scope": "smoke-r-and-stage-r64-only",
        "action_continuation_authorized": False,
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
            member.mode = int(row["mode"])
            member.uid = 0
            member.gid = 0
            member.uname = ""
            member.gname = ""
            member.mtime = 0
            member.type = tarfile.REGTYPE
            tar.addfile(member, io.BytesIO(raw))
    return output.getvalue()


def verify_archive(raw: bytes, manifest: Mapping[str, Any]) -> None:
    rows = manifest.get("files")
    if not isinstance(rows, list):
        fail("release manifest files differ")
    expected = [f"{MEMBER_ROOT}/{row['path']}" for row in rows]
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tar:
            members = tar.getmembers()
            if [member.name for member in members] != expected:
                fail("release archive exact member order or set differs")
            for member, row in zip(members, rows):
                handle = tar.extractfile(member)
                payload = b"" if handle is None else handle.read()
                if (
                    not member.isfile()
                    or member.issym()
                    or member.islnk()
                    or member.uid != 0
                    or member.gid != 0
                    or member.uname != ""
                    or member.gname != ""
                    or member.mtime != 0
                    or member.mode != row["mode"]
                    or member.size != row["size"]
                    or member.pax_headers
                    or len(payload) != row["size"]
                    or hashlib.sha256(payload).hexdigest() != row["sha256"]
                ):
                    fail(f"release archive member differs: {member.name}")
    except (OSError, tarfile.TarError) as error:
        raise GenericActionReleaseError(f"cannot verify release archive: {error}") from error


def validate_manifest(value: Mapping[str, Any]) -> Mapping[str, Any]:
    unsigned = dict(value)
    declared = unsigned.pop("manifest_digest", None)
    rows = value.get("files")
    component_pins = value.get("component_pins")
    expected_paths = sorted(FILES_AND_MODES)
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("release_generation") != RELEASE_GENERATION
        or value.get("archive_format") != ARCHIVE_FORMAT
        or value.get("member_root") != MEMBER_ROOT
        or value.get("file_count") != len(FILES_AND_MODES)
        or value.get("exact_member_closure") is not True
        or value.get("release_scope") != "smoke-r-and-stage-r64-only"
        or value.get("action_continuation_authorized") is not False
        or declared != object_sha256(unsigned)
        or not isinstance(rows, list)
        or [row.get("path") if isinstance(row, Mapping) else None for row in rows]
        != expected_paths
        or not isinstance(component_pins, Mapping)
    ):
        fail("release manifest schema/digest/closure differs")
    row_by_path = {str(row["path"]): row for row in rows}
    for relative, expected_mode in FILES_AND_MODES.items():
        row = row_by_path[relative]
        if (
            set(row) != {"path", "mode", "size", "sha256"}
            or row["mode"] != expected_mode
            or type(row["size"]) is not int
            or row["size"] <= 0
            or not isinstance(row["sha256"], str)
            or len(row["sha256"]) != 64
        ):
            fail(f"release manifest member row differs: {relative}")
    expected_pins = {
        label: row_by_path[relative]["sha256"]
        for label, relative in COMPONENT_FILES.items()
    }
    if dict(component_pins) != expected_pins:
        fail("release component pins differ from exact member rows")
    closure = {"member_root": MEMBER_ROOT, "files": rows}
    if value.get("content_closure_sha1") != hashlib.sha1(
        canonical_json_bytes(closure)
    ).hexdigest():
        fail("release content-closure revision differs")
    return value


def _write_create_only(path: Path, raw: bytes, *, mode: int) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        fail("release output must be one fresh absolute file")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build(method_root: Path, archive: Path, manifest_path: Path) -> Mapping[str, Any]:
    manifest, payloads = build_manifest(method_root)
    validate_manifest(manifest)
    archive_raw = build_archive(manifest, payloads)
    verify_archive(archive_raw, manifest)
    if build_archive(manifest, payloads) != archive_raw:
        fail("release archive rebuild is not byte-identical")
    manifest_raw = canonical_json_bytes(manifest) + b"\n"
    _write_create_only(archive, archive_raw, mode=0o444)
    _write_create_only(manifest_path, manifest_raw, mode=0o444)
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "archive": str(archive),
        "archive_sha256": hashlib.sha256(archive_raw).hexdigest(),
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "manifest_digest": manifest["manifest_digest"],
        "content_closure_sha1": manifest["content_closure_sha1"],
        "component_pins": manifest["component_pins"],
        "file_count": len(FILES_AND_MODES),
    }
    return {**unsigned, "receipt_digest": object_sha256(unsigned)}


def audit(
    archive: Path,
    manifest_path: Path,
    *,
    expected_archive_sha256: str,
    expected_manifest_sha256: str,
) -> Mapping[str, Any]:
    archive_raw = _stable_plain_bytes(archive)
    manifest_raw = _stable_plain_bytes(manifest_path)
    if hashlib.sha256(archive_raw).hexdigest() != expected_archive_sha256:
        fail("release archive SHA-256 differs")
    if hashlib.sha256(manifest_raw).hexdigest() != expected_manifest_sha256:
        fail("release manifest SHA-256 differs")
    try:
        manifest = json.loads(manifest_raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise GenericActionReleaseError(f"cannot decode release manifest: {error}") from error
    if (
        not isinstance(manifest, Mapping)
        or manifest_raw != canonical_json_bytes(manifest) + b"\n"
    ):
        fail("release manifest bytes are not canonical JSON")
    validate_manifest(manifest)
    verify_archive(archive_raw, manifest)
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--method-root", required=True)
    build_parser.add_argument("--archive", required=True)
    build_parser.add_argument("--manifest", required=True)
    audit_parser = commands.add_parser("audit")
    audit_parser.add_argument("--archive", required=True)
    audit_parser.add_argument("--manifest", required=True)
    audit_parser.add_argument("--expected-archive-sha256", required=True)
    audit_parser.add_argument("--expected-manifest-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.command == "build":
        value = build(
            Path(args.method_root), Path(args.archive), Path(args.manifest)
        )
    else:
        value = audit(
            Path(args.archive),
            Path(args.manifest),
            expected_archive_sha256=args.expected_archive_sha256,
            expected_manifest_sha256=args.expected_manifest_sha256,
        )
    print(canonical_json_bytes(value).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
