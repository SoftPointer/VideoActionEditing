"""Shared sealing rules for immutable R7 artifact directories.

Publishers build in private sibling staging directories.  This module seals
every regular file and directory before the atomic rename, verifies the
committed modes, and makes an unpublished sealed staging tree removable on a
failure path.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import stat
from typing import Any, Mapping


FILE_MODE = 0o444
DIRECTORY_MODE = 0o555
PERMISSION_CONTRACT_SCHEMA = (
    "motive-r7-immutable-artifact-permission-contract-v1"
)


def permission_contract() -> dict[str, Any]:
    """Return the canonical permission marker embedded by new publishers."""

    return {
        "schema_version": PERMISSION_CONTRACT_SCHEMA,
        "regular_file_mode_octal": "0444",
        "directory_mode_octal": "0555",
    }


def validate_permission_contract(value: Any) -> dict[str, Any]:
    """Validate and normalize one embedded permission marker."""

    expected = permission_contract()
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise ValueError("artifact permission contract differs")
    return expected


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _tree_entries(root: Path) -> tuple[list[Path], list[Path]]:
    try:
        root_status = root.lstat()
    except FileNotFoundError as error:
        raise ValueError("artifact tree root is missing") from error
    if stat.S_ISLNK(root_status.st_mode) or not stat.S_ISDIR(
        root_status.st_mode
    ):
        raise ValueError("artifact tree root is not a real directory")

    directories = [root]
    files: list[Path] = []
    for directory, child_directories, child_files in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        current = Path(directory)
        for name in sorted(child_directories):
            path = current / name
            status = path.lstat()
            if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(
                status.st_mode
            ):
                raise ValueError(
                    "artifact tree contains a symlink/non-directory child"
                )
            directories.append(path)
        for name in sorted(child_files):
            path = current / name
            status = path.lstat()
            if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(
                status.st_mode
            ):
                raise ValueError(
                    "artifact tree contains a symlink/non-regular file"
                )
            if status.st_nlink != 1:
                raise ValueError("artifact tree contains a hard-linked file")
            files.append(path)
    return directories, files


def seal_staging_tree(
    root: Path,
    *,
    leave_root_writable: bool = False,
) -> None:
    """Seal a private, fully validated staging tree before publication.

    macOS refuses to rename a directory whose own mode is ``0555``.  A
    cross-platform publisher can therefore seal all files and descendant
    directories while retaining the private staging root as exact ``0700``,
    rename it, and immediately call :func:`seal_published_root`.
    """

    directories, files = _tree_entries(root)
    for path in files:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        try:
            status = os.fstat(descriptor)
            if (
                not stat.S_ISREG(status.st_mode)
                or status.st_nlink != 1
            ):
                raise ValueError(
                    "artifact file changed type/link count while sealing"
                )
            os.fchmod(descriptor, FILE_MODE)
            # Persist both the payload already written by the publisher and
            # the final inode mode before the directory becomes visible.
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    sealed_directories = (
        [path for path in directories if path != root]
        if leave_root_writable
        else directories
    )
    for path in sorted(
        sealed_directories,
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        os.chmod(path, DIRECTORY_MODE)
        _fsync_directory(path)
    if leave_root_writable:
        os.chmod(root, 0o700)
        _fsync_directory(root)


def assert_sealed_tree(
    root: Path,
    *,
    allow_writable_root: bool = False,
) -> None:
    """Require exact immutable modes throughout a committed artifact tree."""

    directories, files = _tree_entries(root)
    for path in directories:
        mode = stat.S_IMODE(path.lstat().st_mode)
        expected = (
            0o700
            if allow_writable_root and path == root
            else DIRECTORY_MODE
        )
        if mode != expected:
            raise ValueError(
                "artifact directory mode differs from "
                f"{expected:04o}: {path}"
            )
    for path in files:
        mode = stat.S_IMODE(path.lstat().st_mode)
        if mode != FILE_MODE:
            raise ValueError(
                f"artifact file mode differs from 0444: {path}"
            )


def seal_published_root(root: Path) -> None:
    """Finish the root mode immediately after a cross-platform rename."""

    status = root.lstat()
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise ValueError("published artifact root is not a real directory")
    os.chmod(root, DIRECTORY_MODE)
    _fsync_directory(root)
    assert_sealed_tree(root)


def make_staging_tree_removable(root: Path) -> None:
    """Thaw only an unpublished private staging tree for failure cleanup."""

    if not root.exists() or root.is_symlink():
        return
    os.chmod(root, 0o700)
    for directory, child_directories, child_files in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        current = Path(directory)
        os.chmod(current, 0o700)
        for name in child_directories:
            path = current / name
            if not path.is_symlink():
                os.chmod(path, 0o700)
        for name in child_files:
            path = current / name
            if not path.is_symlink() and path.is_file():
                os.chmod(path, 0o600)


def remove_staging_tree(root: Path) -> None:
    """Remove an unpublished staging tree even if sealing already occurred."""

    if root.exists() and not root.is_symlink():
        make_staging_tree_removable(root)
        shutil.rmtree(root)


__all__ = [
    "DIRECTORY_MODE",
    "FILE_MODE",
    "PERMISSION_CONTRACT_SCHEMA",
    "assert_sealed_tree",
    "make_staging_tree_removable",
    "permission_contract",
    "remove_staging_tree",
    "seal_published_root",
    "seal_staging_tree",
    "validate_permission_contract",
]
