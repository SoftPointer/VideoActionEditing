#!/usr/bin/env python3
"""Stdlib-only verified extraction and execution for preservation-v2 releases.

The manifest and archive are captured through retained descriptors before any
release member is trusted.  Extraction never delegates path handling to
``tarfile`` and execution imports release-local Python modules only from the
captured byte closure.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import importlib.machinery
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile
from types import ModuleType
from typing import Any, Dict, Iterable, Mapping, NoReturn, Optional, Sequence, Tuple


SCHEMA_VERSION = "bernini-self-generated-action-preservation-v2-release-v1"
RELEASE_GENERATION = "preservation-v2-seed20260818-four-holder-r1"
ARCHIVE_FORMAT = "fixed-ustar-ascii-zero-dev-sorted-owner0-mtime0-record10240-v1"
FIXED_USTAR_BLOCK_SIZE = 512
FIXED_USTAR_RECORD_SIZE = 10240
MEMBER_ROOT = "methods/bernini_action_editing"
ALLOWED_PYTHON_TARGETS = frozenset(
    {
        "train_self_generated_action_quotient_v1.py",
        "audit_self_generated_action_preservation_v2.py",
        "action_preservation_completion_publisher_v1.py",
    }
)
ALLOWED_SHELL_TARGETS = frozenset(
    {"scripts/auh_run_self_generated_action_preservation_v2.sh"}
)
BASH_PATH = Path("/usr/bin/bash")
BASH_MODE = 0o755
FROZEN_SITE_PACKAGES_LITERAL = (
    "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages"
)
FROZEN_SITE_PACKAGES = Path(FROZEN_SITE_PACKAGES_LITERAL)
EXECVE_SUPPORTS_FD = os.execve in os.supports_fd
SHA1_RE = re.compile(r"[0-9a-f]{40}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "release_generation",
        "archive_format",
        "member_root",
        "exact_member_closure",
        "file_count",
        "files",
        "content_revision",
        "allowed_entrypoints",
        "authority",
        "component_sha256",
        "manifest_digest",
    }
)


class ActionPreservationVerifiedReleaseError(RuntimeError):
    """Raised before unverified bytes can be extracted or executed."""


def fail(message: str) -> NoReturn:
    raise ActionPreservationVerifiedReleaseError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ActionPreservationVerifiedReleaseError(
            "release JSON is not canonical finite UTF-8"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def content_revision(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha1(canonical_json_bytes(list(rows))).hexdigest()


def _identity(value: os.stat_result) -> Tuple[int, ...]:
    """Stable physical identity fields (atime is deliberately excluded)."""

    return (
        value.st_dev,
        value.st_ino,
        value.st_uid,
        value.st_gid,
        value.st_mode,
        value.st_nlink,
        value.st_rdev,
        value.st_size,
        getattr(value, "st_blocks", 0),
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_fd(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _stable_capture(
    path_value: Path,
    *,
    label: str,
    expected_sha256: Optional[str] = None,
    expected_mode: Optional[int] = None,
) -> Tuple[bytes, os.stat_result]:
    """Capture one canonical single-link file twice through the same fd."""

    path = Path(path_value)
    if expected_sha256 is not None and SHA256_RE.fullmatch(expected_sha256) is None:
        fail(f"{label} expected SHA-256 differs")
    if not path.is_absolute() or path.is_symlink():
        fail(f"{label} must be an absolute non-symlink path")
    try:
        if path.resolve(strict=True) != path:
            fail(f"{label} must be a canonical path")
    except OSError as error:
        raise ActionPreservationVerifiedReleaseError(
            f"{label} is unavailable"
        ) from error
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        before = os.fstat(descriptor)
        first = _read_fd(descriptor)
        middle = os.fstat(descriptor)
        second = _read_fd(descriptor)
        after = os.fstat(descriptor)
        named = path.lstat()
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or _identity(before) != _identity(middle)
        or _identity(before) != _identity(after)
        or _identity(before) != _identity(named)
        or first != second
        or len(first) != before.st_size
        or (expected_mode is not None and stat.S_IMODE(before.st_mode) != expected_mode)
    ):
        fail(f"{label} physical identity changed or differs")
    digest = hashlib.sha256(first).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        fail(f"{label} SHA-256 differs")
    return first, before


def _stable_executable_fd(
    path: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> int:
    """Open, double-read, validate, and retain one exact root executable fd."""

    if (
        path != BASH_PATH
        or SHA256_RE.fullmatch(expected_sha256) is None
        or type(expected_size) is not int
        or expected_size <= 0
        or not path.is_absolute()
        or path.is_symlink()
    ):
        fail("held /usr/bin/bash declaration differs")
    try:
        if path.resolve(strict=True) != path:
            fail("held /usr/bin/bash path is not literal canonical /usr/bin/bash")
    except OSError as error:
        raise ActionPreservationVerifiedReleaseError(
            "held /usr/bin/bash is unavailable"
        ) from error
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        before = os.fstat(descriptor)
        first = _read_fd(descriptor)
        middle = os.fstat(descriptor)
        second = _read_fd(descriptor)
        after = os.fstat(descriptor)
        named = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != BASH_MODE
            or before.st_uid != 0
            or before.st_gid != 0
            or before.st_nlink != 1
            or before.st_size != expected_size
            or _identity(before) != _identity(middle)
            or _identity(before) != _identity(after)
            or _identity(before) != _identity(named)
            or first != second
            or len(first) != expected_size
            or hashlib.sha256(first).hexdigest() != expected_sha256
        ):
            fail("held /usr/bin/bash physical identity/SHA differs")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _unique_pairs(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            fail(f"duplicate manifest JSON key: {key!r}")
        value[key] = item
    return value


def _decode_manifest(raw: bytes) -> Mapping[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ActionPreservationVerifiedReleaseError(
            "release manifest is not valid strict JSON"
        ) from error
    if type(value) is not dict or canonical_json_bytes(value) + b"\n" != raw:
        fail("release manifest is not one canonical JSON object")
    return value


def _validate_member_path(relative: Any) -> str:
    if not isinstance(relative, str):
        fail("release member path is not text")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or not pure.parts
        or ".." in pure.parts
        or "." in pure.parts
        or pure.as_posix() != relative
        or relative.startswith("._")
        or "/._" in relative
    ):
        fail("release member path is unsafe or non-canonical")
    return relative


def validate_manifest(
    value: Mapping[str, Any],
    *,
    expected_manifest_sha256: str,
    expected_content_revision: str,
    expected_archive_sha256: Optional[str] = None,
) -> Mapping[str, Any]:
    if (
        SHA256_RE.fullmatch(expected_manifest_sha256) is None
        or SHA1_RE.fullmatch(expected_content_revision) is None
        or (
            expected_archive_sha256 is not None
            and SHA256_RE.fullmatch(expected_archive_sha256) is None
        )
    ):
        fail("expected release identities differ")
    unsigned = dict(value)
    declared_digest = unsigned.pop("manifest_digest", None)
    rows = value.get("files")
    if (
        type(value) is not dict
        or set(value) != MANIFEST_FIELDS
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("release_generation") != RELEASE_GENERATION
        or value.get("archive_format") != ARCHIVE_FORMAT
        or value.get("member_root") != MEMBER_ROOT
        or value.get("content_revision") != expected_content_revision
        or value.get("exact_member_closure") is not True
        or type(value.get("file_count")) is not int
        or not isinstance(rows, list)
        or value.get("file_count") != len(rows)
        or not rows
        or not isinstance(declared_digest, str)
        or SHA256_RE.fullmatch(declared_digest) is None
        or object_sha256(unsigned) != declared_digest
    ):
        fail("release manifest schema or digest differs")
    allowed_entrypoints = value.get("allowed_entrypoints")
    authority = value.get("authority")
    components = value.get("component_sha256")
    if (
        not isinstance(allowed_entrypoints, list)
        or not allowed_entrypoints
        or any(not isinstance(item, str) or not item for item in allowed_entrypoints)
        or len(set(allowed_entrypoints)) != len(allowed_entrypoints)
        or type(authority) is not dict
        or not authority
        or type(components) is not dict
        or not components
        or any(
            not isinstance(name, str)
            or not isinstance(digest, str)
            or SHA256_RE.fullmatch(digest) is None
            for name, digest in components.items()
        )
    ):
        fail("release manifest authority/entrypoint/component closure differs")
    observed_paths = []
    for row in rows:
        if type(row) is not dict or set(row) != {"path", "mode", "size", "sha256"}:
            fail("release manifest member row schema differs")
        relative = _validate_member_path(row.get("path"))
        mode = row.get("mode")
        size = row.get("size")
        digest = row.get("sha256")
        if (
            type(mode) is not int
            or mode not in (0o444, 0o555)
            or (relative.endswith(".py") and mode != 0o444)
            or (relative.endswith(".sh") and mode != 0o555)
            or type(size) is not int
            or size <= 0
            or not isinstance(digest, str)
            or SHA256_RE.fullmatch(digest) is None
        ):
            fail(f"release manifest member row differs: {relative}")
        observed_paths.append(relative)
    if observed_paths != sorted(observed_paths) or len(set(observed_paths)) != len(rows):
        fail("release manifest exact member order or uniqueness differs")
    if content_revision(rows) != expected_content_revision:
        fail("release content revision differs from exact member rows")
    return value


def capture_manifest(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_content_revision: str,
    expected_archive_sha256: Optional[str] = None,
) -> Tuple[Mapping[str, Any], bytes]:
    raw, _ = _stable_capture(
        Path(manifest_path),
        label="release manifest",
        expected_sha256=expected_manifest_sha256,
    )
    value = _decode_manifest(raw)
    validate_manifest(
        value,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_content_revision=expected_content_revision,
        expected_archive_sha256=expected_archive_sha256,
    )
    return value, raw


def _ustar_text(value: str, width: int, label: str) -> bytes:
    if type(value) is not str or "\0" in value:
        fail(f"{label} differs")
    try:
        raw = value.encode("ascii", "strict")
    except UnicodeEncodeError as error:
        raise ActionPreservationVerifiedReleaseError(
            f"{label} is not canonical USTAR ASCII"
        ) from error
    if len(raw) > width:
        fail(f"{label} exceeds canonical USTAR width")
    return raw + b"\0" * (width - len(raw))


def _ustar_octal(value: int, width: int, label: str) -> bytes:
    if type(value) is not int or value < 0:
        fail(f"{label} differs")
    digits = width - 1
    if value >= 8**digits:
        fail(f"{label} exceeds canonical USTAR octal width")
    raw = f"{value:0{digits}o}".encode("ascii") + b"\0"
    if len(raw) != width:
        fail(f"{label} canonical USTAR width differs")
    return raw


def _ustar_name_fields(value: str) -> Tuple[bytes, bytes]:
    if type(value) is not str or not value or "\0" in value:
        fail("USTAR member name differs")
    try:
        encoded = value.encode("ascii", "strict")
    except UnicodeEncodeError as error:
        raise ActionPreservationVerifiedReleaseError(
            "USTAR member name is not ASCII"
        ) from error
    if len(encoded) <= 100:
        return _ustar_text(value, 100, "USTAR name"), b"\0" * 155
    for index in range(len(value) - 1, -1, -1):
        if value[index] != "/":
            continue
        prefix, basename = value[:index], value[index + 1 :]
        if not prefix or not basename:
            continue
        try:
            prefix_raw = prefix.encode("ascii", "strict")
            basename_raw = basename.encode("ascii", "strict")
        except UnicodeEncodeError:
            continue
        if len(prefix_raw) <= 155 and len(basename_raw) <= 100:
            return (
                _ustar_text(basename, 100, "USTAR name"),
                _ustar_text(prefix, 155, "USTAR prefix"),
            )
    fail("USTAR member name cannot be represented without extensions")


def fixed_ustar_header(name: str, *, size: int, mode: int) -> bytes:
    """Serialize one fixed regular USTAR header without ``TarInfo.tobuf``."""

    name_field, prefix_field = _ustar_name_fields(name)
    header = bytearray(FIXED_USTAR_BLOCK_SIZE)
    header[0:100] = name_field
    header[100:108] = _ustar_octal(mode, 8, "USTAR mode")
    header[108:116] = _ustar_octal(0, 8, "USTAR uid")
    header[116:124] = _ustar_octal(0, 8, "USTAR gid")
    header[124:136] = _ustar_octal(size, 12, "USTAR size")
    header[136:148] = _ustar_octal(0, 12, "USTAR mtime")
    header[148:156] = b" " * 8
    header[156:157] = b"0"
    header[157:257] = b"\0" * 100
    header[257:263] = b"ustar\0"
    header[263:265] = b"00"
    header[265:297] = b"\0" * 32
    header[297:329] = b"\0" * 32
    header[329:337] = _ustar_octal(0, 8, "USTAR devmajor")
    header[337:345] = _ustar_octal(0, 8, "USTAR devminor")
    header[345:500] = prefix_field
    header[500:512] = b"\0" * 12
    checksum = sum(header)
    if checksum >= 8**6:
        fail("USTAR checksum exceeds field width")
    header[148:156] = f"{checksum:06o}\0 ".encode("ascii")
    if len(header) != FIXED_USTAR_BLOCK_SIZE:
        fail("USTAR header size differs")
    return bytes(header)


def _canonical_ustar_header(name: str, row: Mapping[str, Any]) -> bytes:
    return fixed_ustar_header(name, size=row["size"], mode=row["mode"])


def fixed_ustar_archive(
    rows: Sequence[Mapping[str, Any]], payloads: Mapping[str, bytes]
) -> bytes:
    paths = [row.get("path") for row in rows]
    if any(type(path) is not str for path in paths) or set(payloads) != set(paths):
        fail("fixed USTAR payload closure differs")
    output = bytearray()
    for row in rows:
        relative = _validate_member_path(row["path"])
        payload = payloads[relative]
        if (
            type(payload) is not bytes
            or len(payload) != row["size"]
            or hashlib.sha256(payload).hexdigest() != row["sha256"]
        ):
            fail(f"fixed USTAR payload differs: {relative}")
        output.extend(
            fixed_ustar_header(
                f"{MEMBER_ROOT}/{relative}",
                size=row["size"],
                mode=row["mode"],
            )
        )
        output.extend(payload)
        output.extend(b"\0" * (-len(payload) % FIXED_USTAR_BLOCK_SIZE))
    output.extend(b"\0" * (2 * FIXED_USTAR_BLOCK_SIZE))
    output.extend(b"\0" * (-len(output) % FIXED_USTAR_RECORD_SIZE))
    if len(output) % FIXED_USTAR_RECORD_SIZE != 0:
        fail("fixed USTAR record boundary differs")
    return bytes(output)


def verify_archive_snapshot(
    raw: bytes, manifest: Mapping[str, Any]
) -> Mapping[str, bytes]:
    rows = manifest["files"]
    expected_names = [f"{MEMBER_ROOT}/{row['path']}" for row in rows]
    payloads: Dict[str, bytes] = {}
    expected_offset = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as bundle:
            members = bundle.getmembers()
            if [member.name for member in members] != expected_names:
                fail("archive exact regular member closure/order differs")
            for member, row, expected_name in zip(members, rows, expected_names):
                if (
                    member.type != tarfile.REGTYPE
                    or not member.isfile()
                    or member.issym()
                    or member.islnk()
                    or member.linkname != ""
                    or member.pax_headers
                    or member.mode != row["mode"]
                    or member.uid != 0
                    or member.gid != 0
                    or member.uname != ""
                    or member.gname != ""
                    or member.mtime != 0
                    or member.size != row["size"]
                    or member.offset != expected_offset
                    or member.offset_data != expected_offset + FIXED_USTAR_BLOCK_SIZE
                ):
                    fail(f"archive member metadata differs: {expected_name}")
                header = raw[member.offset : member.offset + FIXED_USTAR_BLOCK_SIZE]
                if header != _canonical_ustar_header(expected_name, row):
                    fail(f"archive member is not canonical USTAR: {expected_name}")
                handle = bundle.extractfile(member)
                payload = b"" if handle is None else handle.read()
                if (
                    len(payload) != row["size"]
                    or hashlib.sha256(payload).hexdigest() != row["sha256"]
                ):
                    fail(f"archive member bytes differ: {expected_name}")
                payloads[row["path"]] = payload
                blocks = (
                    member.size + FIXED_USTAR_BLOCK_SIZE - 1
                ) // FIXED_USTAR_BLOCK_SIZE
                expected_offset = member.offset_data + blocks * FIXED_USTAR_BLOCK_SIZE
    except (tarfile.TarError, OSError) as error:
        raise ActionPreservationVerifiedReleaseError(
            "release archive is not a readable USTAR"
        ) from error
    trailer = raw[expected_offset:]
    if (
        len(raw) % FIXED_USTAR_RECORD_SIZE != 0
        or len(trailer) < 2 * FIXED_USTAR_BLOCK_SIZE
        or any(trailer)
    ):
        fail("archive zero trailer or USTAR record boundary differs")
    if raw != fixed_ustar_archive(rows, payloads):
        fail("archive fixed canonical USTAR byte closure differs")
    return payloads


def _canonical_fresh_output(path_value: Path) -> Tuple[Path, Path]:
    path = Path(path_value)
    if not path.is_absolute() or path.exists() or path.is_symlink() or path.name in ("", ".", ".."):
        fail("release extraction root must be one fresh absolute path")
    try:
        parent = path.parent.resolve(strict=True)
        metadata = path.parent.lstat()
    except OSError as error:
        raise ActionPreservationVerifiedReleaseError(
            "release extraction parent is unavailable"
        ) from error
    if parent != path.parent or not stat.S_ISDIR(metadata.st_mode) or path.parent.is_symlink():
        fail("release extraction parent must be canonical")
    return path, parent


def _directory_flags() -> int:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        fail("safe extraction requires O_DIRECTORY and O_NOFOLLOW")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _expected_tree(manifest: Mapping[str, Any]) -> Tuple[set, set]:
    files = set()
    directories = {"."}
    root = PurePosixPath(MEMBER_ROOT)
    for row in manifest["files"]:
        path = root / row["path"]
        files.add(path.as_posix())
        parent = path.parent
        while parent != PurePosixPath("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return files, directories


def _scan_tree(root: Path) -> Tuple[Dict[str, os.stat_result], Dict[str, os.stat_result]]:
    files: Dict[str, os.stat_result] = {}
    directories: Dict[str, os.stat_result] = {}
    stack = [root]
    while stack:
        current = stack.pop()
        relative = current.relative_to(root)
        key = "." if relative == Path(".") else relative.as_posix()
        before = current.lstat()
        if current.is_symlink() or not stat.S_ISDIR(before.st_mode):
            fail(f"materialized release directory differs: {key}")
        if stat.S_IMODE(before.st_mode) != 0o555:
            fail(f"materialized release directory mode differs: {key}")
        try:
            entries = sorted(os.scandir(current), key=lambda item: item.name)
        except OSError as error:
            raise ActionPreservationVerifiedReleaseError(
                f"cannot scan materialized release directory: {key}"
            ) from error
        after = current.lstat()
        if _identity(before) != _identity(after):
            fail(f"materialized release directory changed while scanning: {key}")
        directories[key] = after
        child_directories = []
        for entry in entries:
            child = current / entry.name
            child_relative = child.relative_to(root).as_posix()
            metadata = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                fail(f"materialized release symlink is forbidden: {child_relative}")
            if stat.S_ISDIR(metadata.st_mode):
                child_directories.append(child)
            elif stat.S_ISREG(metadata.st_mode):
                files[child_relative] = metadata
            else:
                fail(f"materialized release special entry is forbidden: {child_relative}")
        stack.extend(reversed(child_directories))
    return files, directories


def capture_materialized_release(
    release_root: Path, manifest: Mapping[str, Any]
) -> Mapping[str, bytes]:
    root = Path(release_root)
    if not root.is_absolute() or root.is_symlink():
        fail("materialized release root must be absolute and non-symlink")
    try:
        if root.resolve(strict=True) != root:
            fail("materialized release root must be canonical")
    except OSError as error:
        raise ActionPreservationVerifiedReleaseError(
            "materialized release root is unavailable"
        ) from error
    expected_files, expected_directories = _expected_tree(manifest)
    files_before, directories_before = _scan_tree(root)
    if set(files_before) != expected_files or set(directories_before) != expected_directories:
        fail("materialized release has links, extras, or missing entries")
    row_by_full_path = {
        f"{MEMBER_ROOT}/{row['path']}": row for row in manifest["files"]
    }
    payloads: Dict[str, bytes] = {}
    captured_identities: Dict[str, Tuple[int, ...]] = {}
    for full_path in sorted(expected_files):
        row = row_by_full_path[full_path]
        raw, metadata = _stable_capture(
            root / full_path,
            label=f"materialized release member {full_path}",
            expected_sha256=row["sha256"],
            expected_mode=row["mode"],
        )
        if len(raw) != row["size"]:
            fail(f"materialized release member size differs: {full_path}")
        payloads[row["path"]] = raw
        captured_identities[full_path] = _identity(metadata)
    files_after, directories_after = _scan_tree(root)
    if set(files_after) != expected_files or set(directories_after) != expected_directories:
        fail("materialized release tree changed during capture")
    for path in expected_files:
        if captured_identities[path] != _identity(files_after[path]):
            fail(f"materialized release member changed after capture: {path}")
    for path in expected_directories:
        if _identity(directories_before[path]) != _identity(directories_after[path]):
            fail(f"materialized release directory changed during capture: {path}")
    return payloads


def extract_verified_release(
    *,
    archive: Path,
    expected_archive_sha256: str,
    manifest: Path,
    expected_manifest_sha256: str,
    expected_content_revision: str,
    output_root: Path,
) -> Mapping[str, Any]:
    """Verify stable snapshots, then create and seal one exact release tree."""

    if SHA256_RE.fullmatch(expected_archive_sha256) is None:
        fail("expected archive SHA-256 differs")
    manifest_value, _ = capture_manifest(
        Path(manifest),
        expected_manifest_sha256=expected_manifest_sha256,
        expected_content_revision=expected_content_revision,
        expected_archive_sha256=expected_archive_sha256,
    )
    archive_raw, _ = _stable_capture(
        Path(archive),
        label="release archive",
        expected_sha256=expected_archive_sha256,
    )
    payloads = verify_archive_snapshot(archive_raw, manifest_value)
    destination, parent = _canonical_fresh_output(Path(output_root))
    parent_fd = os.open(parent, _directory_flags())
    directory_fds: Dict[str, int] = {}
    try:
        os.mkdir(destination.name, 0o700, dir_fd=parent_fd)
        os.fsync(parent_fd)
        root_fd = os.open(destination.name, _directory_flags(), dir_fd=parent_fd)
        directory_fds["."] = root_fd
        os.fchmod(root_fd, 0o700)
        os.fsync(root_fd)
        _, expected_directories = _expected_tree(manifest_value)
        for directory in sorted(
            expected_directories - {"."}, key=lambda item: (item.count("/"), item)
        ):
            pure = PurePosixPath(directory)
            parent_key = "." if pure.parent == PurePosixPath(".") else pure.parent.as_posix()
            parent_directory_fd = directory_fds[parent_key]
            os.mkdir(pure.name, 0o700, dir_fd=parent_directory_fd)
            child_fd = os.open(pure.name, _directory_flags(), dir_fd=parent_directory_fd)
            os.fchmod(child_fd, 0o700)
            os.fsync(child_fd)
            os.fsync(parent_directory_fd)
            directory_fds[directory] = child_fd
        for row in manifest_value["files"]:
            relative = PurePosixPath(row["path"])
            full_parent = PurePosixPath(MEMBER_ROOT) / relative.parent
            parent_key = full_parent.as_posix()
            descriptor = os.open(
                relative.name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                row["mode"],
                dir_fd=directory_fds[parent_key],
            )
            try:
                raw = payloads[row["path"]]
                offset = 0
                while offset < len(raw):
                    count = os.write(descriptor, raw[offset:])
                    if count <= 0:
                        fail("release member write made no progress")
                    offset += count
                os.fchmod(descriptor, row["mode"])
                os.fsync(descriptor)
                opened = os.fstat(descriptor)
                named = os.stat(
                    relative.name,
                    dir_fd=directory_fds[parent_key],
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                    or opened.st_size != row["size"]
                    or stat.S_IMODE(opened.st_mode) != row["mode"]
                    or _identity(opened) != _identity(named)
                ):
                    fail(f"created release member identity differs: {row['path']}")
            finally:
                os.close(descriptor)
            os.fsync(directory_fds[parent_key])
        for directory in sorted(
            directory_fds, key=lambda item: (item.count("/"), item), reverse=True
        ):
            descriptor = directory_fds[directory]
            os.fchmod(descriptor, 0o555)
            os.fsync(descriptor)
            if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o555:
                fail(f"release directory seal differs: {directory}")
        os.fsync(parent_fd)
    except FileExistsError as error:
        raise ActionPreservationVerifiedReleaseError(
            "create-only release extraction collided with an existing entry"
        ) from error
    finally:
        for descriptor in directory_fds.values():
            try:
                os.close(descriptor)
            except OSError:
                pass
        os.close(parent_fd)
    captured = capture_materialized_release(destination, manifest_value)
    if set(captured) != {row["path"] for row in manifest_value["files"]}:
        fail("post-extraction captured member closure differs")
    result = {
        "release_root": str(destination),
        "method_root": str(destination / MEMBER_ROOT),
        "archive_sha256": expected_archive_sha256,
        "manifest_sha256": expected_manifest_sha256,
        "content_revision": expected_content_revision,
        "file_count": len(captured),
        "exact_tree_verified": True,
        "directories_sealed_mode": "0555",
    }
    return {**result, "receipt_digest": object_sha256(result)}


def _path_is_within(candidate: Path, root: Path) -> bool:
    return candidate == root or root in candidate.parents


def _strip_forbidden_sys_path(forbidden_roots: Sequence[Path]) -> None:
    kept = []
    cwd = Path.cwd().resolve()
    roots = tuple(root.resolve() for root in forbidden_roots)
    for entry in sys.path:
        if entry == "":
            continue
        try:
            candidate = Path(entry).expanduser()
            if not candidate.is_absolute():
                candidate = cwd / candidate
            candidate = candidate.resolve(strict=False)
        except (OSError, RuntimeError):
            kept.append(entry)
            continue
        if candidate == cwd or any(
            _path_is_within(candidate, root) for root in roots
        ):
            continue
        kept.append(entry)
    sys.path[:] = kept


class _CapturedModuleLoader(importlib.abc.Loader):
    def __init__(self, fullname: str, source: Path, raw: bytes) -> None:
        self.fullname = fullname
        self.source = source
        self.raw = raw

    def create_module(self, spec: Any) -> Optional[ModuleType]:
        return None

    def exec_module(self, module: ModuleType) -> None:
        module.__file__ = str(self.source)
        module.__package__ = ""
        module.__cached__ = None
        exec(
            compile(self.raw, str(self.source), "exec", dont_inherit=True),
            module.__dict__,
        )


class _CapturedReleaseFinder(importlib.abc.MetaPathFinder):
    def __init__(
        self,
        *,
        modules: Mapping[str, Tuple[Path, bytes]],
        forbidden_roots: Sequence[Path],
    ) -> None:
        self.modules = modules
        self.forbidden_roots = tuple(forbidden_roots)

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        _strip_forbidden_sys_path(self.forbidden_roots)
        if "." in fullname:
            return None
        item = self.modules.get(fullname)
        if item is None:
            return None
        source, raw = item
        loader = _CapturedModuleLoader(fullname, source, raw)
        return importlib.machinery.ModuleSpec(
            fullname, loader, origin=str(source), is_package=False
        )


def verified_python_run(
    *,
    release_root: Path,
    manifest: Path,
    expected_manifest_sha256: str,
    expected_content_revision: str,
    target: str,
    target_arguments: Sequence[str],
) -> int:
    """Compile/exec an allowed target and imports from captured release bytes."""

    manifest_value, _ = capture_manifest(
        Path(manifest),
        expected_manifest_sha256=expected_manifest_sha256,
        expected_content_revision=expected_content_revision,
    )
    if target not in ALLOWED_PYTHON_TARGETS:
        fail("verified Python target is not allowed")
    row_paths = {row["path"] for row in manifest_value["files"]}
    if target not in row_paths or not target.endswith(".py"):
        fail("verified Python target is absent from the release closure")
    root = Path(release_root)
    payloads = capture_materialized_release(root, manifest_value)
    method_root = root / MEMBER_ROOT
    modules: Dict[str, Tuple[Path, bytes]] = {}
    module_directories = {method_root}
    for relative, raw in payloads.items():
        pure = PurePosixPath(relative)
        if pure.suffix != ".py":
            continue
        module_name = pure.stem
        if not module_name.isidentifier() or module_name in modules:
            fail("release top-level Python module name is invalid or ambiguous")
        source = method_root / relative
        modules[module_name] = (source, raw)
        module_directories.add(source.parent)
    if any(name in sys.modules for name in modules):
        fail("release-local Python module was imported before verified capture")
    for name, module in tuple(sys.modules.items()):
        source_value = getattr(module, "__file__", None)
        if not isinstance(source_value, str):
            continue
        try:
            source_path = Path(source_value).resolve(strict=False)
        except (OSError, RuntimeError):
            continue
        if _path_is_within(source_path, root):
            fail(f"unknown release-root module was imported before capture: {name}")
    forbidden_roots = tuple(sorted({root, *module_directories}, key=str))
    old_path = list(sys.path)
    old_meta_path = list(sys.meta_path)
    old_argv = list(sys.argv)
    old_dont_write = sys.dont_write_bytecode
    loaded_before = set(sys.modules)
    finder = _CapturedReleaseFinder(
        modules=modules,
        forbidden_roots=forbidden_roots,
    )
    target_path = method_root / target
    arguments = list(target_arguments)
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    try:
        _strip_forbidden_sys_path(forbidden_roots)
        # Frozen Python is entered with -S, so mutable .pth/sitecustomize bytes
        # cannot run before the complete release has been captured above.  Add
        # the one fixed dependency root as a plain sys.path entry: unlike
        # site.addsitedir(), this does not evaluate .pth files.
        if (
            not FROZEN_SITE_PACKAGES.is_absolute()
            or FROZEN_SITE_PACKAGES.as_posix()
            != FROZEN_SITE_PACKAGES_LITERAL
        ):
            fail("frozen site-packages authority differs")
        sys.path.append(str(FROZEN_SITE_PACKAGES))
        sys.meta_path.insert(0, finder)
        sys.dont_write_bytecode = True
        sys.argv = [str(target_path), *arguments]
        globals_value = {
            "__name__": "__main__",
            "__file__": str(target_path),
            "__package__": None,
            "__loader__": _CapturedModuleLoader("__main__", target_path, payloads[target]),
            "__spec__": None,
            "__cached__": None,
            "__builtins__": __builtins__,
        }
        exec(
            compile(payloads[target], str(target_path), "exec", dont_inherit=True),
            globals_value,
        )
    finally:
        sys.path[:] = old_path
        sys.meta_path[:] = old_meta_path
        sys.argv[:] = old_argv
        sys.dont_write_bytecode = old_dont_write
        for name in set(sys.modules) - loaded_before:
            module = sys.modules.get(name)
            loader = getattr(module, "__loader__", None)
            if isinstance(loader, _CapturedModuleLoader):
                sys.modules.pop(name, None)
    return 0


def _clean_shell_environment() -> Dict[str, str]:
    forbidden = {
        "BASH_ENV",
        "ENV",
        "SHELLOPTS",
        "BASHOPTS",
        "CDPATH",
        "GLOBIGNORE",
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONINSPECT",
        "PYTHONUSERBASE",
        "GCONV_PATH",
        "LOCPATH",
        "NLSPATH",
        "HOSTALIASES",
        "RES_OPTIONS",
        "LOCALDOMAIN",
        "IFS",
        "PS4",
        "PROMPT_COMMAND",
        "BASH_XTRACEFD",
    }
    value = {
        name: item
        for name, item in os.environ.items()
        if name not in forbidden
        and not name.startswith(
            ("LD_", "DYLD_", "BASH_FUNC_", "GLIBC_", "MALLOC_")
        )
    }
    value["PATH"] = "/usr/bin:/bin"
    value["LC_ALL"] = "C"
    value["LANG"] = "C"
    return value


def verified_shell_run(
    *,
    release_root: Path,
    manifest: Path,
    expected_manifest_sha256: str,
    expected_content_revision: str,
    target: str,
    target_arguments: Sequence[str],
    expected_bash_sha256: str,
    expected_bash_size: int,
) -> NoReturn:
    """Exec captured shell bytes with a retained, physically verified bash fd."""

    manifest_value, _ = capture_manifest(
        Path(manifest),
        expected_manifest_sha256=expected_manifest_sha256,
        expected_content_revision=expected_content_revision,
    )
    if target not in ALLOWED_SHELL_TARGETS:
        fail("verified shell target is not allowed")
    row_by_path = {row["path"]: row for row in manifest_value["files"]}
    row = row_by_path.get(target)
    if type(row) is not dict or row.get("mode") != 0o555:
        fail("verified shell target is absent or non-executable in the closure")
    root = Path(release_root)
    payloads = capture_materialized_release(root, manifest_value)
    source = payloads[target]
    if b"\x00" in source:
        fail("verified shell target contains a NUL byte")
    try:
        source_text = source.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ActionPreservationVerifiedReleaseError(
            "verified shell target is not strict UTF-8"
        ) from error
    arguments = list(target_arguments)
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    target_display = str(root / MEMBER_ROOT / target)
    if not EXECVE_SUPPORTS_FD:
        fail("held-fd os.execve is unavailable")
    bash_descriptor = _stable_executable_fd(
        BASH_PATH,
        expected_sha256=expected_bash_sha256,
        expected_size=expected_bash_size,
    )
    try:
        os.execve(
            bash_descriptor,
            [
                str(BASH_PATH),
                "--noprofile",
                "--norc",
                "-p",
                "-c",
                source_text,
                target_display,
                *arguments,
            ],
            _clean_shell_environment(),
        )
    finally:
        os.close(bash_descriptor)
    fail("held-fd /usr/bin/bash execve unexpectedly returned")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract")
    extract.add_argument("--archive", required=True)
    extract.add_argument("--expected-archive-sha256", required=True)
    extract.add_argument("--manifest", required=True)
    extract.add_argument("--expected-manifest-sha256", required=True)
    extract.add_argument(
        "--expected-content-revision", "--method-revision", dest="revision", required=True
    )
    extract.add_argument("--output-root", required=True)
    run = commands.add_parser("verified-run")
    run.add_argument("--release-root", required=True)
    run.add_argument("--manifest", required=True)
    run.add_argument("--expected-manifest-sha256", required=True)
    run.add_argument(
        "--expected-content-revision", "--method-revision", dest="revision", required=True
    )
    run.add_argument("--target", required=True, choices=sorted(ALLOWED_PYTHON_TARGETS))
    run.add_argument("target_arguments", nargs=argparse.REMAINDER)
    shell = commands.add_parser("verified-shell-run")
    shell.add_argument("--release-root", required=True)
    shell.add_argument("--manifest", required=True)
    shell.add_argument("--expected-manifest-sha256", required=True)
    shell.add_argument(
        "--expected-content-revision", "--method-revision", dest="revision", required=True
    )
    shell.add_argument("--target", required=True, choices=sorted(ALLOWED_SHELL_TARGETS))
    shell.add_argument("--expected-bash-sha256", required=True)
    shell.add_argument("--expected-bash-size", required=True, type=int)
    shell.add_argument("target_arguments", nargs=argparse.REMAINDER)
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "extract":
        result = extract_verified_release(
            archive=Path(args.archive),
            expected_archive_sha256=args.expected_archive_sha256,
            manifest=Path(args.manifest),
            expected_manifest_sha256=args.expected_manifest_sha256,
            expected_content_revision=args.revision,
            output_root=Path(args.output_root),
        )
        print(canonical_json_bytes(result).decode("ascii"), flush=True)
        return 0
    if args.command == "verified-run":
        return verified_python_run(
            release_root=Path(args.release_root),
            manifest=Path(args.manifest),
            expected_manifest_sha256=args.expected_manifest_sha256,
            expected_content_revision=args.revision,
            target=args.target,
            target_arguments=args.target_arguments,
        )
    verified_shell_run(
        release_root=Path(args.release_root),
        manifest=Path(args.manifest),
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_content_revision=args.revision,
        target=args.target,
        target_arguments=args.target_arguments,
        expected_bash_sha256=args.expected_bash_sha256,
        expected_bash_size=args.expected_bash_size,
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALLOWED_PYTHON_TARGETS",
    "ALLOWED_SHELL_TARGETS",
    "ActionPreservationVerifiedReleaseError",
    "capture_manifest",
    "capture_materialized_release",
    "content_revision",
    "extract_verified_release",
    "fixed_ustar_archive",
    "fixed_ustar_header",
    "validate_manifest",
    "verified_python_run",
    "verified_shell_run",
    "verify_archive_snapshot",
]
