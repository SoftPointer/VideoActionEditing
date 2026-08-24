"""Canonical path contracts for the immutable R10 job lifecycle.

The R10 controller accepts paths through Slurm environment variables and later
reads identities back from JSON receipts.  Content hashes are insufficient if
one of the directories below the experiment root is a symlink: a self-consistent
receipt could otherwise bind an artifact outside the intended experiment.

These helpers compare the unresolved lexical path with one exact expected
location and reject symlinks at every component from the experiment root to the
target.  Callers still enforce the file-specific content and permission seal.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import stat
from typing import Literal


class R10PathContractError(ValueError):
    """A lifecycle path is ambiguous, redirected, or outside its root."""


PathKind = Literal["dir", "file", "any"]


def _normalized_absolute(value: str | os.PathLike[str], context: str) -> Path:
    raw = os.fspath(value)
    if not isinstance(raw, str) or not os.path.isabs(raw):
        raise R10PathContractError(f"{context} must be absolute")
    if raw.startswith("//") or raw != os.path.normpath(raw):
        raise R10PathContractError(f"{context} must be lexically normalized")
    return Path(raw)


def canonical_experiment_root(
    value: str | os.PathLike[str],
) -> Path:
    """Return an existing absolute experiment root with no symlink indirection."""

    root = _normalized_absolute(value, "experiment root")
    if root.is_symlink() or not root.is_dir():
        raise R10PathContractError(
            "experiment root must be one existing non-symlink directory"
        )
    if root.resolve(strict=True) != root:
        raise R10PathContractError(
            "experiment root must equal its canonical resolved path"
        )
    return root


def _relative_parts(relative: str) -> tuple[str, ...]:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise R10PathContractError(
            "expected experiment-relative path must be non-empty POSIX text"
        )
    path = PurePosixPath(relative)
    if (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or path.as_posix() != relative
    ):
        raise R10PathContractError(
            "expected experiment-relative path must be normalized"
        )
    return tuple(path.parts)


def require_experiment_path(
    value: str | os.PathLike[str],
    experiment_root: str | os.PathLike[str],
    relative: str,
    *,
    kind: PathKind = "any",
    allow_missing: bool = False,
) -> Path:
    """Require one exact path and reject symlinks in its rooted ancestry."""

    if kind not in {"dir", "file", "any"}:
        raise R10PathContractError(f"unsupported path kind: {kind}")
    root = canonical_experiment_root(experiment_root)
    parts = _relative_parts(relative)
    expected = root.joinpath(*parts)
    observed = _normalized_absolute(value, relative)
    if observed != expected:
        raise R10PathContractError(
            f"path differs from canonical experiment location: {relative}"
        )

    current = root
    missing = False
    for index, part in enumerate(parts):
        current = current / part
        is_leaf = index == len(parts) - 1
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            missing = True
            if not (allow_missing and is_leaf):
                raise R10PathContractError(
                    f"required experiment path is missing: {relative}"
                ) from None
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise R10PathContractError(
                f"experiment path contains a symlink: {current}"
            )
        if not is_leaf and not stat.S_ISDIR(metadata.st_mode):
            raise R10PathContractError(
                f"experiment path ancestor is not a directory: {current}"
            )

    if missing:
        return expected
    metadata = expected.lstat()
    if kind == "dir" and not stat.S_ISDIR(metadata.st_mode):
        raise R10PathContractError(f"expected directory differs: {relative}")
    if kind == "file" and not stat.S_ISREG(metadata.st_mode):
        raise R10PathContractError(f"expected regular file differs: {relative}")
    if expected.resolve(strict=True) != expected:
        raise R10PathContractError(
            f"experiment path does not resolve canonically: {relative}"
        )
    return expected


def ensure_experiment_directory(
    value: str | os.PathLike[str],
    experiment_root: str | os.PathLike[str],
    relative: str,
    *,
    mode: int = 0o755,
) -> Path:
    """Create a rooted directory one component at a time without following links."""

    root = canonical_experiment_root(experiment_root)
    parts = _relative_parts(relative)
    expected = root.joinpath(*parts)
    observed = _normalized_absolute(value, relative)
    if observed != expected:
        raise R10PathContractError(
            f"directory differs from canonical experiment location: {relative}"
        )
    current = root
    for part in parts:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            try:
                os.mkdir(current, mode)
            except FileExistsError:
                pass
            metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise R10PathContractError(
                f"experiment directory is redirected or non-directory: {current}"
            )
    if expected.resolve(strict=True) != expected:
        raise R10PathContractError(
            f"experiment directory is not canonical: {relative}"
        )
    return expected


def attempt_receipt_relative(seed: int, attempt: int) -> str:
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or isinstance(attempt, bool)
        or not isinstance(attempt, int)
        or attempt < 1
    ):
        raise R10PathContractError("seed/attempt identity is invalid")
    return (
        f"provenance/job_attempts/seed_{seed}/attempt_{attempt}.json"
    )


def require_attempt_receipt_path(
    value: str | os.PathLike[str],
    experiment_root: str | os.PathLike[str],
    seed: int,
    attempt: int,
    *,
    allow_missing: bool = False,
) -> Path:
    return require_experiment_path(
        value,
        experiment_root,
        attempt_receipt_relative(seed, attempt),
        kind="file",
        allow_missing=allow_missing,
    )
