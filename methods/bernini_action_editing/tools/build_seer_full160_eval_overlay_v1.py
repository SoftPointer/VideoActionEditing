#!/usr/bin/env python3
"""Build and verify the deterministic three-file SEER full160 eval overlay.

The overlay is an inference-runtime artifact, not the training method source
archive.  Its paths are relative to ``methods/bernini_action_editing`` and its
manifest has exact member closure.  Tar headers are normalized so identical
input bytes produce identical archive bytes on every host.
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
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "bernini-seer-full160-eval-overlay-v1"
ARCHIVE_FORMAT = "posix-ustar-owner0-mtime0-mode0444-sorted-v1"
OVERLAY_FILES = (
    "infer_seer_same_state_full160_lora.py",
    "infer_seer_same_state_lora.py",
    "run_self_generated_action_lora_heldout_core4_v1.py",
)
ADDED_FILES = ("infer_seer_same_state_full160_lora.py",)
REPLACED_FILES = tuple(path for path in OVERLAY_FILES if path not in ADDED_FILES)
TRAINING_METHOD_SOURCE_REVISION = "6bae2f9a70fb851fd1bf87f5a01a1064a8fe8a6a"
TRAINING_METHOD_SOURCE_ARCHIVE_SHA256 = (
    "ec4064949b1cc8f4dba5d3c15fab375c9392a1407f9610ad01cee24c173bf822"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class OverlayError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise OverlayError(f"value is not canonical JSON: {error}") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise OverlayError(f"{label} must be absolute")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise OverlayError(f"{label} is unavailable: {error}") from error
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise OverlayError(f"{label} must be a plain file")
    return path.resolve(strict=True)


def _fresh_output(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path == Path("/"):
        raise OverlayError(f"{label} must be an absolute non-root path")
    if path.exists() or path.is_symlink():
        raise OverlayError(f"{label} must be fresh: {path}")
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir():
        raise OverlayError(f"{label} parent is not a directory")
    return parent / path.name


def _write_create_only(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, mode)
    try:
        view = memoryview(payload)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                raise OverlayError(f"short write: {path}")
            view = view[count:]
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)


def _source_files(method_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative in OVERLAY_FILES:
        path = _plain_file(method_root / relative, label=f"overlay source {relative}")
        payload = path.read_bytes()
        rows.append(
            {
                "path": relative,
                "sha256": bytes_sha256(payload),
                "size": len(payload),
                "mode": "0444",
            }
        )
    return rows


def build_manifest(method_root: Path) -> dict[str, Any]:
    method_root = method_root.resolve(strict=True)
    if not method_root.is_dir() or method_root.is_symlink():
        raise OverlayError("method root must be a plain directory")
    unsigned: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "archive_format": ARCHIVE_FORMAT,
        "training_method_source": {
            "revision": TRAINING_METHOD_SOURCE_REVISION,
            "archive_sha256": TRAINING_METHOD_SOURCE_ARCHIVE_SHA256,
            "overlay_is_training_archive": False,
            "overlay_sha_must_not_be_passed_as_method_source_archive_sha256": True,
        },
        "overlay_member_root": "methods/bernini_action_editing",
        "file_count": len(OVERLAY_FILES),
        "added_paths": list(ADDED_FILES),
        "replaced_paths": list(REPLACED_FILES),
        "files": _source_files(method_root),
        "exact_member_closure": True,
        "training_receipt_mutated": False,
        "training_provenance_replaced": False,
        "inference_runtime_overlay_only": True,
    }
    return {**unsigned, "manifest_digest": object_sha256(unsigned)}


def _manifest_payload(manifest: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(manifest) + b"\n"


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


def build_archive_bytes(method_root: Path, manifest: Mapping[str, Any]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for row in manifest["files"]:
            relative = str(row["path"])
            payload = (method_root / relative).read_bytes()
            if len(payload) != row["size"] or bytes_sha256(payload) != row["sha256"]:
                raise OverlayError(f"overlay source changed during build: {relative}")
            name = f"methods/bernini_action_editing/{relative}"
            archive.addfile(_tar_info(name, len(payload)), io.BytesIO(payload))
    return buffer.getvalue()


def validate_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    candidate = dict(value)
    declared = candidate.pop("manifest_digest", None)
    if not isinstance(declared, str) or _SHA256.fullmatch(declared) is None:
        raise OverlayError("overlay manifest digest is invalid")
    if object_sha256(candidate) != declared:
        raise OverlayError("overlay manifest digest differs")
    if set(value) != {
        "schema_version",
        "archive_format",
        "training_method_source",
        "overlay_member_root",
        "file_count",
        "added_paths",
        "replaced_paths",
        "files",
        "exact_member_closure",
        "training_receipt_mutated",
        "training_provenance_replaced",
        "inference_runtime_overlay_only",
        "manifest_digest",
    }:
        raise OverlayError("overlay manifest root closure differs")
    training = value.get("training_method_source")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("archive_format") != ARCHIVE_FORMAT
        or value.get("overlay_member_root") != "methods/bernini_action_editing"
        or value.get("file_count") != len(OVERLAY_FILES)
        or value.get("added_paths") != list(ADDED_FILES)
        or value.get("replaced_paths") != list(REPLACED_FILES)
        or value.get("exact_member_closure") is not True
        or value.get("training_receipt_mutated") is not False
        or value.get("training_provenance_replaced") is not False
        or value.get("inference_runtime_overlay_only") is not True
        or not isinstance(training, Mapping)
        or training
        != {
            "revision": TRAINING_METHOD_SOURCE_REVISION,
            "archive_sha256": TRAINING_METHOD_SOURCE_ARCHIVE_SHA256,
            "overlay_is_training_archive": False,
            "overlay_sha_must_not_be_passed_as_method_source_archive_sha256": True,
        }
    ):
        raise OverlayError("overlay/training provenance contract differs")
    files = value.get("files")
    if not isinstance(files, list) or [row.get("path") for row in files if isinstance(row, Mapping)] != list(OVERLAY_FILES):
        raise OverlayError("overlay file closure or order differs")
    for index, row in enumerate(files):
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256", "size", "mode"}:
            raise OverlayError(f"overlay file row {index} closure differs")
        relative = PurePosixPath(str(row["path"]))
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
            raise OverlayError(f"overlay file row {index} path differs")
        if _SHA256.fullmatch(str(row["sha256"])) is None:
            raise OverlayError(f"overlay file row {index} SHA differs")
        if type(row["size"]) is not int or row["size"] <= 0 or row["mode"] != "0444":
            raise OverlayError(f"overlay file row {index} metadata differs")
    return dict(value)


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OverlayError(f"cannot read overlay manifest: {error}") from error
    if not isinstance(value, dict):
        raise OverlayError("overlay manifest root must be an object")
    return validate_manifest(value)


def validate_archive(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    path = _plain_file(path, label="overlay archive")
    expected_names = [
        f"methods/bernini_action_editing/{row['path']}" for row in manifest["files"]
    ]
    try:
        with tarfile.open(path, mode="r:") as archive:
            members = archive.getmembers()
            if [member.name for member in members] != expected_names:
                raise OverlayError("overlay archive exact member closure differs")
            for member, row in zip(members, manifest["files"]):
                if (
                    not member.isfile()
                    or member.issym()
                    or member.islnk()
                    or member.uid != 0
                    or member.gid != 0
                    or member.uname != ""
                    or member.gname != ""
                    or member.mtime != 0
                    or stat.S_IMODE(member.mode) != 0o444
                    or member.size != row["size"]
                ):
                    raise OverlayError(f"overlay tar header differs: {member.name}")
                extracted = archive.extractfile(member)
                if extracted is None or bytes_sha256(extracted.read()) != row["sha256"]:
                    raise OverlayError(f"overlay tar member bytes differ: {member.name}")
    except (OSError, tarfile.TarError) as error:
        raise OverlayError(f"cannot validate overlay archive: {error}") from error
    archive_sha = file_sha256(path)
    if archive_sha == TRAINING_METHOD_SOURCE_ARCHIVE_SHA256:
        raise OverlayError("overlay archive must not equal the training archive")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "overlay_exact_closure_verified",
        "archive_path": str(path),
        "archive_sha256": archive_sha,
        "manifest_digest": manifest["manifest_digest"],
        "file_count": len(expected_names),
        "training_method_source_archive_sha256": TRAINING_METHOD_SOURCE_ARCHIVE_SHA256,
        "overlay_is_training_archive": False,
    }


def build(method_root: Path, archive_path: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = build_manifest(method_root)
    archive_payload = build_archive_bytes(method_root, manifest)
    _write_create_only(archive_path, archive_payload)
    _write_create_only(manifest_path, _manifest_payload(manifest))
    reopened = _read_manifest(manifest_path)
    result = validate_archive(archive_path, reopened)
    # Determinism is verified independently by rebuilding the byte stream.
    if build_archive_bytes(method_root, reopened) != archive_payload:
        raise OverlayError("overlay archive rebuild is not byte deterministic")
    return {
        **result,
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("build")
    make.add_argument("--method-root", required=True)
    make.add_argument("--archive", required=True)
    make.add_argument("--manifest", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--archive", required=True)
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--expected-archive-sha256", required=True)
    verify.add_argument("--expected-manifest-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        method_root = Path(args.method_root)
        if not method_root.is_absolute():
            raise OverlayError("method root must be absolute")
        result = build(
            method_root.resolve(strict=True),
            _fresh_output(args.archive, label="overlay archive"),
            _fresh_output(args.manifest, label="overlay manifest"),
        )
    else:
        archive = _plain_file(args.archive, label="overlay archive")
        manifest_path = _plain_file(args.manifest, label="overlay manifest")
        if file_sha256(archive) != args.expected_archive_sha256:
            raise OverlayError("overlay archive SHA differs")
        if file_sha256(manifest_path) != args.expected_manifest_sha256:
            raise OverlayError("overlay manifest raw SHA differs")
        result = validate_archive(archive, _read_manifest(manifest_path))
        result["manifest_path"] = str(manifest_path)
        result["manifest_sha256"] = file_sha256(manifest_path)
    print(canonical_json_bytes(result).decode("utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except OverlayError as error:
        print(f"[seer-full160-overlay] ERROR: {error}", file=os.sys.stderr)
        raise SystemExit(2)
