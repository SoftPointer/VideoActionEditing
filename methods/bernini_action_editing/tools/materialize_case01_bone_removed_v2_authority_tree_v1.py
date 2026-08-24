#!/usr/bin/env python3
"""Create one exact, reviewable authority-tree manifest without approval claims.

The output schema is consumed by the frozen case01 bone-removed-v2 producer
and acceptance gate.  This tool only records current bytes.  Creating a
manifest does not preapprove a runtime, checkpoint, or source tree and does
not authorize generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from typing import Any, Dict, List, Mapping, Sequence, Tuple


TREE_MANIFEST_SCHEMA = (
    "bernini-case01-bone-removed-v2-authority-tree-manifest-v1"
)
INVENTORY_POLICY = "exact_recursive_regular_nonsymlink_nlink1"
AUTHORITY_ROLES = (
    "python_runtime_tree",
    "vace_checkpoint_tree",
    "vace_source_tree",
)
CHUNK_BYTES = 8 * 1024 * 1024


class AuthorityTreeError(RuntimeError):
    """Raised before an ambiguous tree manifest can be published."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise AuthorityTreeError("manifest value is not canonical JSON") from error


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _canonical_absolute(value: str, label: str) -> Path:
    if type(value) is not str or not value or "\x00" in value:
        raise AuthorityTreeError("%s path differs" % label)
    path = Path(value)
    if not path.is_absolute():
        raise AuthorityTreeError("%s path is not absolute" % label)
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise AuthorityTreeError("%s path is absent" % label) from error
    if resolved != path or path.is_symlink():
        raise AuthorityTreeError("%s path is not canonical/plain" % label)
    return path


def _identity(row: os.stat_result) -> Tuple[int, ...]:
    return (
        int(row.st_dev),
        int(row.st_ino),
        int(row.st_uid),
        int(row.st_gid),
        int(row.st_mode),
        int(row.st_nlink),
        int(row.st_size),
        int(row.st_mtime_ns),
        int(row.st_ctime_ns),
    )


def _ownership_identity(row: os.stat_result) -> Tuple[int, int, int, int]:
    return (
        int(row.st_dev),
        int(row.st_ino),
        int(row.st_uid),
        int(row.st_gid),
    )


def _plain_directory(path: Path, label: str) -> Tuple[int, ...]:
    row = path.lstat()
    if path.is_symlink() or not stat.S_ISDIR(row.st_mode):
        raise AuthorityTreeError("%s is not a plain directory" % label)
    return _identity(row)


def _walk_error(error: OSError) -> None:
    raise AuthorityTreeError("authority tree enumeration failed") from error


def _relative_text(path: Path, root: Path) -> str:
    relative = PurePosixPath(path.relative_to(root).as_posix())
    text = relative.as_posix()
    if (
        relative.is_absolute()
        or not text
        or "\\" in text
        or any(part in ("", ".", "..") for part in relative.parts)
    ):
        raise AuthorityTreeError("tree relative path differs")
    return text


def _held_file_row(path: Path, root: Path) -> Tuple[Dict[str, Any], Tuple[int, ...]]:
    before = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(before.st_mode)
        or int(before.st_nlink) != 1
    ):
        raise AuthorityTreeError(
            "tree leaf is not regular nonsymlink nlink1: %s" % path
        )
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(str(path), flags)
    try:
        held_before = os.fstat(descriptor)
        if _identity(before) != _identity(held_before):
            raise AuthorityTreeError("tree leaf named/held identity differs: %s" % path)
        digest = hashlib.sha256()
        offset = 0
        while True:
            if hasattr(os, "pread"):
                block = os.pread(descriptor, CHUNK_BYTES, offset)
            else:  # pragma: no cover - supported production Python has pread
                os.lseek(descriptor, offset, os.SEEK_SET)
                block = os.read(descriptor, CHUNK_BYTES)
            if not block:
                break
            digest.update(block)
            offset += len(block)
        held_after = os.fstat(descriptor)
        named_after = path.lstat()
        if (
            _identity(held_before) != _identity(held_after)
            or _identity(held_after) != _identity(named_after)
            or offset != int(held_after.st_size)
        ):
            raise AuthorityTreeError("tree leaf changed while hashing: %s" % path)
        return (
            {
                "relative_path": _relative_text(path, root),
                "sha256": digest.hexdigest(),
                "size": offset,
            },
            _identity(held_after),
        )
    finally:
        os.close(descriptor)


def _scan_tree(
    root: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Tuple[int, ...]], Dict[str, Tuple[int, ...]]]:
    file_identities: Dict[str, Tuple[int, ...]] = {}
    directory_identities: Dict[str, Tuple[int, ...]] = {
        "": _plain_directory(root, "tree root")
    }
    rows: List[Dict[str, Any]] = []
    for directory_text, dirnames, filenames in os.walk(
        str(root), topdown=True, onerror=_walk_error, followlinks=False
    ):
        dirnames.sort()
        filenames.sort()
        directory = Path(directory_text)
        for dirname in dirnames:
            child = directory / dirname
            relative = _relative_text(child, root)
            directory_identities[relative] = _plain_directory(
                child, "tree directory"
            )
        for filename in filenames:
            child = directory / filename
            row, identity = _held_file_row(child, root)
            relative = row["relative_path"]
            if relative in file_identities:
                raise AuthorityTreeError("tree relative path repeats")
            file_identities[relative] = identity
            rows.append(row)
    rows.sort(key=lambda row: row["relative_path"])
    if not rows:
        raise AuthorityTreeError("authority tree is empty")
    return rows, file_identities, directory_identities


def _replay_inventory(
    root: Path,
    file_identities: Mapping[str, Tuple[int, ...]],
    directory_identities: Mapping[str, Tuple[int, ...]],
) -> None:
    observed_files: Dict[str, Tuple[int, ...]] = {}
    observed_directories: Dict[str, Tuple[int, ...]] = {
        "": _plain_directory(root, "tree root replay")
    }
    for directory_text, dirnames, filenames in os.walk(
        str(root), topdown=True, onerror=_walk_error, followlinks=False
    ):
        dirnames.sort()
        filenames.sort()
        directory = Path(directory_text)
        for dirname in dirnames:
            child = directory / dirname
            relative = _relative_text(child, root)
            observed_directories[relative] = _plain_directory(
                child, "tree directory replay"
            )
        for filename in filenames:
            child = directory / filename
            before = child.lstat()
            if (
                child.is_symlink()
                or not stat.S_ISREG(before.st_mode)
                or int(before.st_nlink) != 1
            ):
                raise AuthorityTreeError("tree replay contains non-plain leaf")
            observed_files[_relative_text(child, root)] = _identity(before)
    if observed_files != dict(file_identities):
        raise AuthorityTreeError("tree file inventory/identity changed after hashing")
    if observed_directories != dict(directory_identities):
        raise AuthorityTreeError("tree directory inventory/identity changed after hashing")


def build_manifest(role: str, tree_root: str) -> Dict[str, Any]:
    if type(role) is not str or role not in AUTHORITY_ROLES:
        raise AuthorityTreeError("authority role differs")
    root = _canonical_absolute(tree_root, "tree root")
    _plain_directory(root, "tree root")
    entries, file_identities, directory_identities = _scan_tree(root)
    _replay_inventory(root, file_identities, directory_identities)
    payload: Dict[str, Any] = {
        "schema_version": TREE_MANIFEST_SCHEMA,
        "authority_role": role,
        "inventory_policy": INVENTORY_POLICY,
        "tree_root": str(root),
        "entries": entries,
        "file_count": len(entries),
        "total_bytes": sum(int(row["size"]) for row in entries),
        "tree_digest": _object_sha256(entries),
    }
    return {**payload, "manifest_digest": _object_sha256(payload)}


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if type(written) is not int or written <= 0:
            raise AuthorityTreeError("manifest write made no progress")
        offset += written


def _write_create_only(output_value: str, payload: bytes) -> None:
    if type(output_value) is not str or not output_value or "\x00" in output_value:
        raise AuthorityTreeError("output path differs")
    output = Path(output_value)
    if not output.is_absolute():
        raise AuthorityTreeError("output path is not absolute")
    parent = _canonical_absolute(str(output.parent), "output parent")
    _plain_directory(parent, "output parent")
    if output.exists() or output.is_symlink():
        raise AuthorityTreeError("output already exists")
    parent_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        parent_flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        parent_flags |= os.O_CLOEXEC
    descriptor = -1
    parent_descriptor = -1
    output_created = False
    owned_identity = None  # type: Any
    parent_identity = None  # type: Any
    try:
        parent_descriptor = os.open(str(parent), parent_flags)
        held_parent = os.fstat(parent_descriptor)
        named_parent = parent.lstat()
        if (
            not stat.S_ISDIR(held_parent.st_mode)
            or _ownership_identity(held_parent) != _ownership_identity(named_parent)
        ):
            raise AuthorityTreeError("output parent held/named identity differs")
        parent_identity = _ownership_identity(held_parent)

        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        descriptor = os.open(output.name, flags, 0o600, dir_fd=parent_descriptor)
        output_created = True
        owned_identity = _ownership_identity(os.fstat(descriptor))
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        held = os.fstat(descriptor)
        named = output.lstat()
        if (
            _identity(held) != _identity(named)
            or not stat.S_ISREG(held.st_mode)
            or int(held.st_nlink) != 1
            or stat.S_IMODE(held.st_mode) != 0o400
            or int(held.st_size) != len(payload)
        ):
            raise AuthorityTreeError("published manifest identity differs")
        if hasattr(os, "pread"):
            replay = os.pread(descriptor, len(payload) + 1, 0)
        else:  # pragma: no cover
            os.lseek(descriptor, 0, os.SEEK_SET)
            replay = os.read(descriptor, len(payload) + 1)
        if replay != payload:
            raise AuthorityTreeError("published manifest bytes differ")
        published_identity = _identity(held)

        try:
            os.close(descriptor)
        finally:
            descriptor = -1
        os.fsync(parent_descriptor)
        final_held = os.stat(
            output.name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        final_named = output.lstat()
        final_parent = parent.lstat()
        if (
            _identity(final_held) != published_identity
            or _identity(final_named) != published_identity
            or parent_identity != _ownership_identity(final_parent)
        ):
            raise AuthorityTreeError("published manifest final replay differs")
        try:
            os.close(parent_descriptor)
        finally:
            parent_descriptor = -1
        return
    except BaseException as primary_error:
        cleanup_errors: List[BaseException] = []
        if descriptor >= 0:
            if output_created and owned_identity is None:
                try:
                    # The first fstat after O_EXCL may itself be the failing
                    # operation.  Retry while the original descriptor is held
                    # so cleanup can still authenticate the newly created name.
                    owned_identity = _ownership_identity(os.fstat(descriptor))
                except BaseException as error:
                    cleanup_errors.append(error)
            try:
                os.close(descriptor)
            except BaseException as error:
                cleanup_errors.append(error)
            finally:
                descriptor = -1

        removed = False
        try:
            if owned_identity is not None:
                if parent_descriptor >= 0:
                    try:
                        named = os.stat(
                            output.name,
                            dir_fd=parent_descriptor,
                            follow_symlinks=False,
                        )
                    except FileNotFoundError:
                        named = None
                    if named is not None:
                        if _ownership_identity(named) != owned_identity:
                            raise AuthorityTreeError(
                                "refusing cleanup of changed manifest identity"
                            )
                        os.unlink(output.name, dir_fd=parent_descriptor)
                        removed = True
                else:
                    named_parent = parent.lstat()
                    if parent_identity != _ownership_identity(named_parent):
                        raise AuthorityTreeError(
                            "refusing cleanup through changed output parent"
                        )
                    try:
                        named = output.lstat()
                    except FileNotFoundError:
                        named = None
                    if named is not None:
                        if _ownership_identity(named) != owned_identity:
                            raise AuthorityTreeError(
                                "refusing cleanup of changed manifest identity"
                            )
                        output.unlink()
                        removed = True
        except BaseException as error:
            cleanup_errors.append(error)

        cleanup_parent_descriptor = parent_descriptor
        cleanup_parent_is_fresh = False
        if cleanup_parent_descriptor < 0 and parent_identity is not None:
            try:
                named_parent = parent.lstat()
                if parent_identity != _ownership_identity(named_parent):
                    raise AuthorityTreeError(
                        "refusing cleanup fsync through changed output parent"
                    )
                cleanup_parent_descriptor = os.open(str(parent), parent_flags)
                cleanup_parent_is_fresh = True
            except BaseException as error:
                cleanup_errors.append(error)
        if cleanup_parent_descriptor >= 0:
            try:
                if removed:
                    os.fsync(cleanup_parent_descriptor)
            except BaseException as error:
                cleanup_errors.append(error)
            if cleanup_parent_is_fresh:
                try:
                    os.close(cleanup_parent_descriptor)
                except BaseException as error:
                    cleanup_errors.append(error)
                cleanup_parent_descriptor = -1

        if parent_descriptor >= 0:
            try:
                os.close(parent_descriptor)
            except BaseException as error:
                cleanup_errors.append(error)
            finally:
                parent_descriptor = -1

        try:
            if output.exists() or output.is_symlink():
                if output_created and owned_identity is None:
                    raise AuthorityTreeError(
                        "unidentified create-only output remains after failed publication"
                    )
                if owned_identity is not None:
                    named = output.lstat()
                    if _ownership_identity(named) == owned_identity:
                        raise AuthorityTreeError(
                            "owned manifest remains after failed publication"
                        )
        except FileNotFoundError:
            pass
        except BaseException as error:
            cleanup_errors.append(error)

        if cleanup_errors:
            raise AuthorityTreeError(
                "manifest publication failed and owned-output cleanup differs"
            ) from primary_error
        if isinstance(primary_error, AuthorityTreeError):
            raise primary_error
        if not isinstance(primary_error, Exception):
            raise primary_error
        raise AuthorityTreeError("manifest publication failed") from primary_error


def materialize(role: str, tree_root: str, output: str) -> Dict[str, Any]:
    root = _canonical_absolute(tree_root, "tree root")
    output_path = Path(output)
    try:
        output_path.relative_to(root)
    except ValueError:
        pass
    else:
        raise AuthorityTreeError("manifest output must be outside its tree")
    manifest = build_manifest(role, str(root))
    payload = _canonical_bytes(manifest) + b"\n"
    _write_create_only(output, payload)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("materialize", choices=("materialize",))
    parser.add_argument("--role", required=True, choices=AUTHORITY_ROLES)
    parser.add_argument("--tree-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] = ()) -> int:
    args = _parser().parse_args(list(argv) if argv else None)
    manifest = materialize(args.role, args.tree_root, args.output)
    sys.stdout.buffer.write(_canonical_bytes(manifest) + b"\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuthorityTreeError as error:
        print("HOLD: %s" % error, file=sys.stderr)
        raise SystemExit(96)
