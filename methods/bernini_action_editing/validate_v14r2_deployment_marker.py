#!/usr/bin/env python3
"""Validate an immutable v14r2 source/archive/test deployment closure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional, Sequence


SCHEMA = "bernini-v14r2-immutable-deployment-v1"
CONTENT_SCHEMA = "bernini-v14r2-source-content-manifest-v1"
HEX = set("0123456789abcdef")


class V14R2DeploymentValidationError(RuntimeError):
    pass


def _fail(message: str) -> None:
    raise V14R2DeploymentValidationError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} is not an object")
    return value


def _plain_file(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        _fail(f"{label} is not a plain file: {path}")


def _plain_directory(path: Path, label: str) -> None:
    if not path.is_dir() or path.is_symlink():
        _fail(f"{label} is not a plain directory: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    _plain_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V14R2DeploymentValidationError(f"{label} is unreadable") from error
    return _mapping(value, label)


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in HEX for character in value)
    ):
        _fail(f"{label} is not a SHA-256")
    return value


def _relative(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        _fail(f"unsafe required-file path: {value!r}")
    return path.as_posix()


def _source_file(source_tree: Path, relative: str) -> Path:
    relative = _relative(relative)
    path = source_tree.joinpath(*PurePosixPath(relative).parts)
    current = source_tree
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            _fail(f"source closure contains a symlink: {relative}")
    _plain_file(path, f"source file {relative}")
    try:
        path.resolve(strict=True).relative_to(source_tree.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise V14R2DeploymentValidationError(
            f"source file escapes source tree: {relative}"
        ) from error
    return path


def _source_tree_file_set(source_tree: Path) -> set[str]:
    """Return the exact plain-file closure and reject shadowing symlinks."""

    files: set[str] = set()
    for path in source_tree.rglob("*"):
        relative = path.relative_to(source_tree).as_posix()
        if path.is_symlink():
            _fail(f"source closure contains a symlink: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            _fail(f"source closure contains a non-file entry: {relative}")
        files.add(_relative(relative))
    return files


def _revision_value(path: Path) -> str:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise V14R2DeploymentValidationError("revision is unreadable") from error
    if len(lines) != 1 or not lines[0] or len(lines[0]) > 256:
        _fail("revision must contain exactly one bounded non-empty line")
    return lines[0]


def _validate_core(
    *,
    marker_path: Path,
    expected_role: str,
    source_tree: Optional[Path] = None,
    archive: Optional[Path] = None,
    revision: Optional[Path] = None,
    content_manifest: Optional[Path] = None,
    min_test_count: int = 1,
    required_files: Sequence[str] = (),
) -> tuple[Mapping[str, Any], Mapping[str, Any], Path]:
    marker = _read_json(marker_path, "deployment marker")
    if marker.get("schema_version") != SCHEMA or marker.get("complete") is not True:
        _fail("deployment marker schema/completion differs")
    if marker.get("role") != expected_role:
        _fail("deployment marker role differs")

    declared_source = Path(str(marker.get("source_tree", "")))
    archive_row = _mapping(marker.get("archive"), "archive binding")
    revision_row = _mapping(marker.get("revision"), "revision binding")
    content_row = _mapping(marker.get("content_manifest"), "content binding")
    declared_archive = Path(str(archive_row.get("path", "")))
    declared_revision = Path(str(revision_row.get("path", "")))
    declared_content = Path(str(content_row.get("path", "")))
    for declared, expected, label in (
        (declared_source, source_tree, "source tree"),
        (declared_archive, archive, "archive"),
        (declared_revision, revision, "revision"),
        (declared_content, content_manifest, "content manifest"),
    ):
        if expected is not None and declared != expected:
            _fail(f"deployment {label} path differs")

    _plain_directory(declared_source, "source tree")
    for path, label in (
        (declared_archive, "archive"),
        (declared_revision, "revision"),
        (declared_content, "content manifest"),
    ):
        _plain_file(path, label)
    if _digest(archive_row.get("sha256"), "archive SHA-256") != _sha256(declared_archive):
        _fail("archive SHA-256 differs")
    if _digest(revision_row.get("sha256"), "revision SHA-256") != _sha256(declared_revision):
        _fail("revision SHA-256 differs")
    if revision_row.get("value") != _revision_value(declared_revision):
        _fail("revision value differs")
    if _digest(content_row.get("sha256"), "content SHA-256") != _sha256(declared_content):
        _fail("content-manifest SHA-256 differs")

    tests = _mapping(marker.get("tests"), "test receipt")
    total_passed = tests.get("total_passed")
    if (
        tests.get("passed") is not True
        or isinstance(total_passed, bool)
        or not isinstance(total_passed, int)
        or total_passed < min_test_count
    ):
        _fail("deployment test receipt differs")

    content = _read_json(declared_content, "content manifest")
    if (
        content.get("schema_version") != CONTENT_SCHEMA
        or content.get("complete") is not True
        or content.get("source_tree") != str(declared_source)
    ):
        _fail("content-manifest contract differs")
    raw_files = _mapping(content.get("files"), "content files")
    raw_marker_files = _mapping(marker.get("required_files"), "required files")
    files: dict[str, str] = {}
    marker_files: dict[str, str] = {}
    for raw_relative, raw_digest in raw_files.items():
        if not isinstance(raw_relative, str):
            _fail("content file path is not a string")
        relative = _relative(raw_relative)
        if relative != raw_relative or relative in files:
            _fail(f"content file path is not canonical: {raw_relative!r}")
        files[relative] = _digest(raw_digest, f"manifest {relative}")
    for raw_relative, raw_digest in raw_marker_files.items():
        if not isinstance(raw_relative, str):
            _fail("required-file path is not a string")
        relative = _relative(raw_relative)
        if relative != raw_relative or relative in marker_files:
            _fail(f"required-file path is not canonical: {raw_relative!r}")
        marker_files[relative] = _digest(raw_digest, f"marker {relative}")
    if not files or set(marker_files) != set(files):
        _fail("required files do not exactly cover the content manifest")
    if _source_tree_file_set(declared_source) != set(files):
        _fail("active source-tree file closure differs from the content manifest")
    requested = {_relative(value) for value in required_files}
    if not requested.issubset(marker_files):
        _fail("deployment marker omits a required file")
    for relative, manifest_digest in files.items():
        actual = _sha256(_source_file(declared_source, relative))
        if marker_files[relative] != actual or manifest_digest != actual:
            _fail(f"source file SHA-256 differs: {relative}")
    return marker, content, declared_source


def validate(
    *,
    marker_path: Path,
    role: str,
    source_tree: Path,
    archive: Path,
    revision: Path,
    content_manifest: Path,
    min_test_count: int,
    required_files: Sequence[str],
    training_marker_path: Optional[Path] = None,
    shared_core: Sequence[str] = (),
) -> None:
    marker, content, decode_source_tree = _validate_core(
        marker_path=marker_path,
        expected_role=role,
        source_tree=source_tree,
        archive=archive,
        revision=revision,
        content_manifest=content_manifest,
        min_test_count=min_test_count,
        required_files=required_files,
    )
    if role == "decode":
        if training_marker_path is None or not shared_core:
            _fail("decode deployment requires a training marker and shared core")
        training_marker, training_content, training_source_tree = _validate_core(
            marker_path=training_marker_path,
            expected_role="training",
        )
        compatibility = _mapping(
            marker.get("training_compatibility"), "training compatibility"
        )
        if (
            compatibility.get("training_marker_path") != str(training_marker_path)
            or _digest(
                compatibility.get("training_marker_sha256"),
                "training marker SHA-256",
            )
            != _sha256(training_marker_path)
        ):
            _fail("decode/training marker binding differs")
        shared = _mapping(compatibility.get("shared_core"), "shared core")
        decode_files = _mapping(content.get("files"), "decode content files")
        training_files = _mapping(
            training_content.get("files"), "training content files"
        )
        decode_required = _mapping(marker.get("required_files"), "decode required files")
        training_required = _mapping(
            training_marker.get("required_files"), "training required files"
        )
        for relative in shared_core:
            relative = _relative(relative)
            digest = _digest(shared.get(relative), f"shared core {relative}")
            decode_actual = _sha256(_source_file(decode_source_tree, relative))
            training_actual = _sha256(
                _source_file(training_source_tree, relative)
            )
            if (
                _digest(decode_files.get(relative), f"decode core {relative}") != digest
                or _digest(training_files.get(relative), f"training core {relative}")
                != digest
                or _digest(
                    decode_required.get(relative),
                    f"decode required core {relative}",
                )
                != digest
                or _digest(
                    training_required.get(relative),
                    f"training required core {relative}",
                )
                != digest
                or decode_actual != digest
                or training_actual != digest
            ):
                _fail(f"decode/training shared-core SHA differs: {relative}")
        if training_marker.get("complete") is not True:
            _fail("training deployment is incomplete")
    elif training_marker_path is not None or shared_core:
        _fail("training deployment cannot claim decode compatibility")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--role", required=True, choices=("training", "decode"))
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--content-manifest", required=True)
    parser.add_argument("--min-test-count", required=True, type=int)
    parser.add_argument("--required-file", action="append", default=[])
    parser.add_argument("--training-marker")
    parser.add_argument("--shared-core", action="append", default=[])
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.min_test_count <= 0:
        _fail("minimum test count must be positive")
    validate(
        marker_path=Path(args.marker),
        role=args.role,
        source_tree=Path(args.source_tree),
        archive=Path(args.archive),
        revision=Path(args.revision),
        content_manifest=Path(args.content_manifest),
        min_test_count=args.min_test_count,
        required_files=args.required_file,
        training_marker_path=(
            Path(args.training_marker) if args.training_marker else None
        ),
        shared_core=args.shared_core,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
