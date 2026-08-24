#!/usr/bin/env python3
"""Retained-FD consumption authority for preservation-v2 decoded evaluation.

The authority is intentionally independent of Slurm and inference libraries.
One holder process captures the exact 23-file Bernini checkpoint once, keeps
every file and directory descriptor alive for the whole shard, and exposes a
fresh private tree whose leaves resolve to ``/proc/self/fd/<fd>``.  The exact
descriptor allowlist is inherited only across the executor -> decoder ->
captured torchrun -> rank spawn chain; unrelated children receive no model
or adapter descriptors.
Every task has a pre-use and post-use replay.  The shard closes only after a
full final rehash of the retained model descriptors.

Adapter checkpoints use the same protocol per task.  Their exact authority
files (training receipt, non-executable PEFT README, PEFT config, and
safetensors weights) remain held until the decoder exits, are fully rehashed
after use, and receive an explicit publication gate.  A caller must not
publish staging media before that gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Iterable, Mapping, Sequence


MODEL_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)
MODEL_FILE_COUNT = 23
MODEL_DIRECTORY_COUNT = 7
MODEL_CAPTURE_SCHEMA = "bernini-action-preservation-model-held-fd-capture-v3"
MODEL_REPLAY_SCHEMA = "bernini-action-preservation-model-held-fd-replay-v3"
MODEL_FINAL_SCHEMA = "bernini-action-preservation-model-held-fd-final-v3"
ADAPTER_CAPTURE_SCHEMA = "bernini-action-preservation-adapter-held-fd-capture-v3"
ADAPTER_FINAL_SCHEMA = "bernini-action-preservation-adapter-held-fd-final-v3"
PUBLICATION_GATE_SCHEMA = "bernini-action-preservation-consumption-publication-gate-v2"
CONSUMPTION_CHAIN_SCHEMA = "bernini-action-preservation-consumption-chain-v2"
CONSUMPTION_INPUT_SCHEMA = "bernini-action-preservation-consumption-input-v3"
INHERITED_FD_BINDING_SCHEMA = (
    "bernini-action-preservation-inherited-fd-binding-v3"
)
INHERITED_FD_BINDING_ENV = "APV2_EVAL_INHERITED_AUTHORITY_FDS"

MODEL_RELATIVE_DIRECTORIES = (
    ".",
    "assets",
    "scheduler",
    "text_encoder",
    "tokenizer",
    "transformer",
    "vae",
)
MODEL_RELATIVE_FILES = (
    ".gitattributes",
    "README.md",
    "assets/arena.png",
    "assets/bernini-icon.png",
    "config.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00005.safetensors",
    "text_encoder/model-00002-of-00005.safetensors",
    "text_encoder/model-00003-of-00005.safetensors",
    "text_encoder/model-00004-of-00005.safetensors",
    "text_encoder/model-00005-of-00005.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/spiece.model",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)
ADAPTER_RELATIVE_FILES = (
    "receipt.json",
    "adapter/README.md",
    "adapter/adapter_config.json",
    "adapter/adapter_model.safetensors",
)
ADAPTER_RELATIVE_DIRECTORIES = (".", "adapter")

_MANIFEST_LINE = re.compile(r"([0-9a-f]{64})  (\./[^\n]+)")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}")


class ModelConsumptionAuthorityError(RuntimeError):
    """A retained descriptor, named path, view, or digest chain differed."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ModelConsumptionAuthorityError("value is not canonical JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ModelConsumptionAuthorityError(f"{label} is not a lowercase SHA-256")
    return value


def _task_id(value: Any) -> str:
    if not isinstance(value, str) or _SAFE_TASK_ID.fullmatch(value) is None:
        raise ModelConsumptionAuthorityError("task id differs")
    return value


def _identity(info: os.stat_result) -> dict[str, int]:
    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "uid": int(info.st_uid),
        "gid": int(info.st_gid),
        "mode": int(info.st_mode),
        "nlink": int(info.st_nlink),
        "rdev": int(info.st_rdev),
        "size": int(info.st_size),
        "blocks": int(getattr(info, "st_blocks", 0)),
        "mtime_ns": int(info.st_mtime_ns),
        "ctime_ns": int(info.st_ctime_ns),
    }


def _directory_flags() -> int:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise ModelConsumptionAuthorityError("safe directory flags are unavailable")
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )


def _file_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise ModelConsumptionAuthorityError("O_NOFOLLOW is unavailable")
    return os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _read_fd(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        block = os.read(descriptor, 16 * 1024 * 1024)
        if not block:
            return b"".join(chunks)
        chunks.append(block)


def _hash_fd(descriptor: int) -> tuple[str, int]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    size = 0
    while True:
        block = os.read(descriptor, 16 * 1024 * 1024)
        if not block:
            return digest.hexdigest(), size
        digest.update(block)
        size += len(block)


def _canonical_absolute(path: Path, *, label: str, directory: bool) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise ModelConsumptionAuthorityError(f"{label} path differs")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ModelConsumptionAuthorityError(f"{label} is unavailable") from error
    if resolved != path:
        raise ModelConsumptionAuthorityError(f"{label} is not canonical")
    if directory and not path.is_dir():
        raise ModelConsumptionAuthorityError(f"{label} is not a directory")
    if not directory and not path.is_file():
        raise ModelConsumptionAuthorityError(f"{label} is not a file")
    return path


def _named_identity(path: Path, *, label: str) -> dict[str, int]:
    try:
        info = path.lstat()
    except OSError as error:
        raise ModelConsumptionAuthorityError(f"{label} named path is unavailable") from error
    return _identity(info)


def _stable_small_file(
    path: Path, *, label: str, expected_sha256: str | None = None
) -> tuple[bytes, dict[str, int]]:
    path = _canonical_absolute(path, label=label, directory=False)
    descriptor = os.open(path, _file_flags())
    try:
        before = _identity(os.fstat(descriptor))
        first = _read_fd(descriptor)
        middle = _identity(os.fstat(descriptor))
        second = _read_fd(descriptor)
        after = _identity(os.fstat(descriptor))
        named = _named_identity(path, label=label)
    finally:
        os.close(descriptor)
    digest = bytes_sha256(first)
    if (
        not stat.S_ISREG(before["mode"])
        or before["nlink"] != 1
        or before != middle
        or before != after
        or before != named
        or first != second
        or len(first) != before["size"]
        or (expected_sha256 is not None and digest != expected_sha256)
    ):
        raise ModelConsumptionAuthorityError(f"{label} stable capture differs")
    return first, before


def parse_exact23_manifest(
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str = MODEL_MANIFEST_SHA256,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Load the independently pinned exact-23 sha256sum manifest."""

    _sha(expected_manifest_sha256, label="model manifest SHA")
    path = Path(manifest_path)
    raw, identity = _stable_small_file(
        path,
        label="model content manifest",
        expected_sha256=expected_manifest_sha256,
    )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ModelConsumptionAuthorityError("model manifest is not UTF-8") from error
    if not text.endswith("\n") or "\r" in text:
        raise ModelConsumptionAuthorityError("model manifest byte framing differs")
    rows: dict[str, str] = {}
    for line in text.splitlines():
        match = _MANIFEST_LINE.fullmatch(line)
        if match is None:
            raise ModelConsumptionAuthorityError("model manifest line differs")
        digest, raw_relative = match.groups()
        relative = PurePosixPath(raw_relative)
        normalized = PurePosixPath(
            *(part for part in relative.parts if part not in ("", "."))
        ).as_posix()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not normalized
            or normalized in rows
        ):
            raise ModelConsumptionAuthorityError("model manifest path differs")
        rows[normalized] = digest
    if len(rows) != MODEL_FILE_COUNT:
        raise ModelConsumptionAuthorityError("model manifest is not exact-23")
    expected_directories = {
        ".",
        *(str(PurePosixPath(path).parent) for path in rows),
    }
    if expected_directories != set(MODEL_RELATIVE_DIRECTORIES):
        raise ModelConsumptionAuthorityError("model manifest directory closure differs")
    if tuple(rows) != MODEL_RELATIVE_FILES:
        raise ModelConsumptionAuthorityError("model manifest file order differs")
    receipt = {
        "path": str(path),
        "sha256": expected_manifest_sha256,
        "identity": identity,
        "row_count": len(rows),
        "ordered_rows_digest": object_sha256(
            [{"relative_path": key, "sha256": value} for key, value in rows.items()]
        ),
    }
    return rows, receipt


@dataclass
class _HeldFile:
    relative_path: str
    path: Path
    expected_sha256: str
    descriptor: int
    identity: dict[str, int]

    def row(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "path": str(self.path),
            "sha256": self.expected_sha256,
            "identity": dict(self.identity),
            "authority_fd": self.descriptor,
            "proc_fd_path": f"/proc/self/fd/{self.descriptor}",
        }


@dataclass
class _HeldDirectory:
    relative_path: str
    path: Path
    descriptor: int
    identity: dict[str, int]
    named_parent_descriptor: int | None = None
    named_basename: str | None = None

    def row(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "path": str(self.path),
            "authority_fd": self.descriptor,
            "identity": dict(self.identity),
        }

    def named_identity(self, *, label: str) -> dict[str, int]:
        try:
            if self.named_parent_descriptor is None:
                return _named_identity(self.path, label=label)
            if self.named_basename is None:
                raise ModelConsumptionAuthorityError(
                    f"{label} relative directory name is absent"
                )
            return _identity(os.stat(
                self.named_basename,
                dir_fd=self.named_parent_descriptor,
                follow_symlinks=False,
            ))
        except OSError as error:
            raise ModelConsumptionAuthorityError(
                f"{label} named directory is unavailable"
            ) from error


def _open_directory(path: Path, *, relative_path: str, label: str) -> _HeldDirectory:
    _canonical_absolute(path, label=label, directory=True)
    descriptor = os.open(path, _directory_flags())
    identity = _identity(os.fstat(descriptor))
    named = _named_identity(path, label=label)
    if (
        not stat.S_ISDIR(identity["mode"])
        or identity != named
        or stat.S_IMODE(identity["mode"]) & 0o022
    ):
        os.close(descriptor)
        raise ModelConsumptionAuthorityError(f"{label} physical identity differs")
    return _HeldDirectory(relative_path, path, descriptor, identity)


def _open_directory_at(
    *, parent_descriptor: int, basename: str, path: Path,
    relative_path: str, label: str,
) -> _HeldDirectory:
    if (
        type(parent_descriptor) is not int
        or parent_descriptor < 3
        or not basename
        or basename in (".", "..")
        or "/" in basename
        or "\x00" in basename
    ):
        raise ModelConsumptionAuthorityError(f"{label} relative binding differs")
    try:
        descriptor = os.open(basename, _directory_flags(), dir_fd=parent_descriptor)
        os.set_inheritable(descriptor, False)
        identity = _identity(os.fstat(descriptor))
        named = _identity(os.stat(
            basename, dir_fd=parent_descriptor, follow_symlinks=False
        ))
    except OSError as error:
        raise ModelConsumptionAuthorityError(
            f"{label} relative open is unavailable"
        ) from error
    if (
        not stat.S_ISDIR(identity["mode"])
        or identity != named
        or stat.S_IMODE(identity["mode"]) & 0o022
    ):
        os.close(descriptor)
        raise ModelConsumptionAuthorityError(f"{label} physical identity differs")
    return _HeldDirectory(
        relative_path,
        path,
        descriptor,
        identity,
        named_parent_descriptor=parent_descriptor,
        named_basename=basename,
    )


def _open_double_hashed_file(
    path: Path,
    *,
    relative_path: str,
    expected_sha256: str,
    expected_uid: int,
    expected_gid: int,
    expected_mode: int,
) -> _HeldFile:
    _sha(expected_sha256, label=f"{relative_path} SHA")
    _canonical_absolute(path, label=relative_path, directory=False)
    descriptor = os.open(path, _file_flags())
    try:
        before = _identity(os.fstat(descriptor))
        first_digest, first_size = _hash_fd(descriptor)
        middle = _identity(os.fstat(descriptor))
        second_digest, second_size = _hash_fd(descriptor)
        after = _identity(os.fstat(descriptor))
        named = _named_identity(path, label=relative_path)
        if (
            not stat.S_ISREG(before["mode"])
            or before["uid"] != expected_uid
            or before["gid"] != expected_gid
            or stat.S_IMODE(before["mode"]) != expected_mode
            or before["nlink"] != 1
            or before != middle
            or before != after
            or before != named
            or first_digest != expected_sha256
            or second_digest != expected_sha256
            or first_size != before["size"]
            or second_size != before["size"]
        ):
            raise ModelConsumptionAuthorityError(
                f"{relative_path} double-hash physical identity differs"
            )
        return _HeldFile(relative_path, path, expected_sha256, descriptor, before)
    except Exception:
        os.close(descriptor)
        raise


def _scan_model_tree(root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories = {"."}
    stack = [root]
    while stack:
        parent = stack.pop()
        for entry in os.scandir(parent):
            relative = Path(entry.path).relative_to(root).as_posix()
            if relative == ".cache" or relative.startswith(".cache/"):
                continue
            info = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode):
                raise ModelConsumptionAuthorityError("model tree contains a symlink")
            if stat.S_ISDIR(info.st_mode):
                directories.add(relative)
                stack.append(Path(entry.path))
            elif stat.S_ISREG(info.st_mode):
                files.add(relative)
            else:
                raise ModelConsumptionAuthorityError("model tree contains a special entry")
    return files, directories


def _held_private_parent(
    parent: Path, *, private_parent_fd: int,
) -> _HeldDirectory:
    parent = _canonical_absolute(
        parent, label="private view parent", directory=True
    )
    if type(private_parent_fd) is not int or private_parent_fd < 3:
        raise ModelConsumptionAuthorityError("private parent FD differs")
    try:
        descriptor = os.dup(private_parent_fd)
        os.set_inheritable(descriptor, False)
        identity = _identity(os.fstat(descriptor))
        named = _named_identity(parent, label="private view parent")
    except OSError as error:
        raise ModelConsumptionAuthorityError(
            "private parent FD is unavailable"
        ) from error
    if (
        not stat.S_ISDIR(identity["mode"])
        or identity != named
        or stat.S_IMODE(identity["mode"]) & 0o022
        or os.get_inheritable(descriptor)
    ):
        os.close(descriptor)
        raise ModelConsumptionAuthorityError(
            "private parent FD identity differs"
        )
    return _HeldDirectory(".", parent, descriptor, identity)


def _fresh_private_root(
    parent: Path, name: str, *, private_parent_fd: int,
) -> tuple[Path, _HeldDirectory, _HeldDirectory]:
    if not name or name in (".", "..") or "/" in name or "\x00" in name:
        raise ModelConsumptionAuthorityError("private view name differs")
    held_parent = _held_private_parent(
        parent, private_parent_fd=private_parent_fd
    )
    root = held_parent.path / name
    try:
        before_entries = sorted(os.listdir(held_parent.descriptor))
        if name in before_entries:
            raise ModelConsumptionAuthorityError("private view root is not fresh")
        os.mkdir(name, 0o700, dir_fd=held_parent.descriptor)
        root_directory = _open_directory_at(
            parent_descriptor=held_parent.descriptor,
            basename=name,
            path=root,
            relative_path=".",
            label="private view root",
        )
        parent_after = _identity(os.fstat(held_parent.descriptor))
        parent_named = _named_identity(
            held_parent.path, label="private view parent"
        )
        after_entries = sorted(os.listdir(held_parent.descriptor))
        if (
            parent_after != parent_named
            or after_entries != sorted([*before_entries, name])
        ):
            raise ModelConsumptionAuthorityError(
                "private view parent changed during root creation"
            )
        held_parent.identity = parent_after
        return root, held_parent, root_directory
    except Exception:
        try:
            os.rmdir(name, dir_fd=held_parent.descriptor)
        except OSError:
            pass
        os.close(held_parent.descriptor)
        raise


def _make_fd_view(
    *,
    parent: Path,
    name: str,
    files: Sequence[_HeldFile],
    directories: Sequence[str],
    proc_fd_prefix: str,
    private_parent_fd: int,
) -> tuple[
    Path, _HeldDirectory, list[_HeldDirectory], dict[str, str]
]:
    root, held_parent, root_directory = _fresh_private_root(
        parent, name, private_parent_fd=private_parent_fd
    )
    held_directories: list[_HeldDirectory] = [root_directory]
    directory_by_relative: dict[str, _HeldDirectory] = {".": root_directory}
    created_links: list[tuple[int, str]] = []
    try:
        for relative in directories:
            if relative == ".":
                continue
            pure = PurePosixPath(relative)
            if (
                pure.is_absolute()
                or ".." in pure.parts
                or len(pure.parts) != 1
            ):
                raise ModelConsumptionAuthorityError(
                    "private view directory topology differs"
                )
            os.mkdir(pure.name, 0o700, dir_fd=root_directory.descriptor)
            child = _open_directory_at(
                parent_descriptor=root_directory.descriptor,
                basename=pure.name,
                path=root / relative,
                relative_path=relative,
                label=f"private view directory {relative}",
            )
            held_directories.append(child)
            directory_by_relative[relative] = child
        links: dict[str, str] = {}
        for item in files:
            pure = PurePosixPath(item.relative_path)
            parent_relative = pure.parent.as_posix()
            if parent_relative == "":
                parent_relative = "."
            parent_directory = directory_by_relative.get(parent_relative)
            if parent_directory is None or pure.name in ("", ".", ".."):
                raise ModelConsumptionAuthorityError(
                    "private FD-view leaf topology differs"
                )
            proc_path = f"{proc_fd_prefix}/{item.descriptor}"
            os.symlink(
                proc_path,
                pure.name,
                dir_fd=parent_directory.descriptor,
            )
            created_links.append((parent_directory.descriptor, pure.name))
            links[item.relative_path] = proc_path
        if [item.relative_path for item in held_directories] != list(directories):
            raise ModelConsumptionAuthorityError(
                "private view held-directory order differs"
            )
        for item in held_directories:
            refreshed = _identity(os.fstat(item.descriptor))
            if refreshed != item.named_identity(
                label=f"private view directory {item.relative_path}"
            ):
                raise ModelConsumptionAuthorityError(
                    "private view directory changed during construction"
                )
            item.identity = refreshed
        held_parent.identity = _identity(os.fstat(held_parent.descriptor))
        if held_parent.identity != held_parent.named_identity(
            label="private view parent"
        ):
            raise ModelConsumptionAuthorityError(
                "private view parent identity differs after construction"
            )
        return root, held_parent, held_directories, links
    except Exception:
        for descriptor, basename in reversed(created_links):
            try:
                os.unlink(basename, dir_fd=descriptor)
            except OSError:
                pass
        for item in reversed(held_directories[1:]):
            try:
                os.rmdir(
                    item.named_basename,
                    dir_fd=item.named_parent_descriptor,
                )
            except OSError:
                pass
        try:
            os.rmdir(name, dir_fd=held_parent.descriptor)
        except OSError:
            pass
        _close_all(held_directories)
        os.close(held_parent.descriptor)
        raise


def _close_all(items: Iterable[_HeldFile | _HeldDirectory]) -> None:
    for item in items:
        try:
            os.close(item.descriptor)
        except OSError:
            pass


class _RetainedFDTree:
    """Shared implementation for holder-wide model and per-task adapter trees."""

    def __init__(
        self,
        *,
        source_root: Path,
        files: list[_HeldFile],
        source_directories: list[_HeldDirectory],
        view_root: Path,
        private_parent: _HeldDirectory,
        view_directories: list[_HeldDirectory],
        view_links: dict[str, str],
    ) -> None:
        self.source_root = source_root
        self.files = files
        self.source_directories = source_directories
        self.view_root = view_root
        self.private_parent = private_parent
        self.view_directories = view_directories
        self.view_links = view_links
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise ModelConsumptionAuthorityError("retained authority is closed")

    def replay(self, *, stage: str) -> dict[str, Any]:
        self._ensure_open()
        view_directory_by_relative = {
            item.relative_path: item for item in self.view_directories
        }
        file_rows: list[dict[str, Any]] = []
        for item in self.files:
            held = _identity(os.fstat(item.descriptor))
            named = _named_identity(item.path, label=f"{stage} {item.relative_path}")
            pure = PurePosixPath(item.relative_path)
            parent_relative = pure.parent.as_posix() or "."
            view_parent = view_directory_by_relative.get(parent_relative)
            if view_parent is None:
                raise ModelConsumptionAuthorityError(
                    f"{stage} FD-view parent differs: {item.relative_path}"
                )
            try:
                view_info = os.stat(
                    pure.name,
                    dir_fd=view_parent.descriptor,
                    follow_symlinks=False,
                )
                view_target = os.readlink(
                    pure.name, dir_fd=view_parent.descriptor
                )
                view_descriptor = os.open(
                    pure.name,
                    os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=view_parent.descriptor,
                )
                try:
                    resolved = _identity(os.fstat(view_descriptor))
                finally:
                    os.close(view_descriptor)
            except OSError as error:
                raise ModelConsumptionAuthorityError(
                    f"{stage} FD-view leaf differs: {item.relative_path}"
                ) from error
            if (
                held != item.identity
                or named != item.identity
                or not stat.S_ISLNK(view_info.st_mode)
                or view_target != self.view_links[item.relative_path]
                or resolved != item.identity
            ):
                raise ModelConsumptionAuthorityError(
                    f"{stage} file identity/view replay differs: {item.relative_path}"
                )
            file_rows.append(
                {
                    "relative_path": item.relative_path,
                    "sha256": item.expected_sha256,
                    "identity": held,
                    "view_target": view_target,
                }
            )
        directory_rows: list[dict[str, Any]] = []
        private_parent_current_identity: dict[str, int] | None = None
        for scope, items in (
            ("source", self.source_directories),
            ("view_parent", [self.private_parent]),
            ("view", self.view_directories),
        ):
            for item in items:
                held = _identity(os.fstat(item.descriptor))
                named = item.named_identity(
                    label=f"{stage} {scope} directory {item.relative_path}"
                )
                immutable_fields = {
                    "device", "inode", "uid", "gid", "mode", "rdev",
                }
                private_parent_differs = (
                    scope == "view_parent"
                    and (
                        held != named
                        or {
                            field: held[field] for field in immutable_fields
                        } != {
                            field: item.identity[field]
                            for field in immutable_fields
                        }
                    )
                )
                if (
                    (scope != "view_parent" and (
                        held != item.identity or named != item.identity
                    ))
                    or private_parent_differs
                ):
                    raise ModelConsumptionAuthorityError(
                        f"{stage} {scope} directory identity differs: {item.relative_path}"
                    )
                directory_rows.append(
                    {
                        "scope": scope,
                        "relative_path": item.relative_path,
                        "identity": held,
                    }
                )
                if scope == "view_parent":
                    private_parent_current_identity = held
        if private_parent_current_identity is None:
            raise ModelConsumptionAuthorityError(
                f"{stage} private parent replay is absent"
            )
        value: dict[str, Any] = {
            "schema_version": MODEL_REPLAY_SCHEMA,
            "stage": stage,
            "source_root": str(self.source_root),
            "view_root": str(self.view_root),
            "file_count": len(file_rows),
            "directory_count": len(directory_rows),
            "files_digest": object_sha256(file_rows),
            "directories_digest": object_sha256(directory_rows),
            "private_parent_current_identity": private_parent_current_identity,
            "all_retained_fds_still_open": True,
            "named_paths_replayed": True,
            "fd_view_replayed": True,
        }
        value["replay_digest"] = object_sha256(value)
        return value

    def final_rehash(self, *, schema_version: str, stage: str) -> dict[str, Any]:
        fast = self.replay(stage=stage)
        rows: list[dict[str, Any]] = []
        for item in self.files:
            before = _identity(os.fstat(item.descriptor))
            digest, size = _hash_fd(item.descriptor)
            after = _identity(os.fstat(item.descriptor))
            named = _named_identity(item.path, label=f"{stage} {item.relative_path}")
            if (
                before != item.identity
                or after != item.identity
                or named != item.identity
                or digest != item.expected_sha256
                or size != item.identity["size"]
            ):
                raise ModelConsumptionAuthorityError(
                    f"{stage} full rehash differs: {item.relative_path}"
                )
            rows.append(
                {
                    "relative_path": item.relative_path,
                    "sha256": digest,
                    "size": size,
                    "identity": before,
                }
            )
        value: dict[str, Any] = {
            "schema_version": schema_version,
            "stage": stage,
            "replay_digest": fast["replay_digest"],
            "private_parent_current_identity": (
                fast["private_parent_current_identity"]
            ),
            "file_count": len(rows),
            "fully_rehashed_rows_digest": object_sha256(rows),
            "all_expected_sha256_matched": True,
            "all_full_identities_matched": True,
        }
        value["final_digest"] = object_sha256(value)
        return value

    def close(self, *, strict: bool = True) -> None:
        if self._closed:
            return
        failure: Exception | None = None
        view_directory_by_relative = {
            item.relative_path: item for item in self.view_directories
        }
        try:
            for item in self.files:
                pure = PurePosixPath(item.relative_path)
                parent_relative = pure.parent.as_posix() or "."
                parent = view_directory_by_relative[parent_relative]
                os.unlink(pure.name, dir_fd=parent.descriptor)
            for item in reversed(self.view_directories[1:]):
                assert item.named_parent_descriptor is not None
                assert item.named_basename is not None
                os.rmdir(
                    item.named_basename,
                    dir_fd=item.named_parent_descriptor,
                )
            root = self.view_directories[0]
            assert root.named_basename is not None
            os.rmdir(
                root.named_basename,
                dir_fd=self.private_parent.descriptor,
            )
        except (KeyError, OSError, AssertionError) as error:
            failure = error
        _close_all(self.view_directories)
        try:
            os.close(self.private_parent.descriptor)
        except OSError:
            pass
        _close_all(self.source_directories)
        _close_all(self.files)
        self._closed = True
        if strict and failure is not None:
            raise ModelConsumptionAuthorityError(
                "private FD-view relative cleanup differs"
            ) from failure


class ModelAuthority:
    """Holder-lifetime exact-23 model authority."""

    def __init__(
        self,
        tree: _RetainedFDTree,
        *,
        manifest: Mapping[str, Any],
        capture_receipt: Mapping[str, Any],
    ) -> None:
        self._tree = tree
        self.manifest = dict(manifest)
        self.capture_receipt = dict(capture_receipt)
        self._active_task: str | None = None
        self._active_pre_digest: str | None = None
        self._task_consumption_digests: list[str] = []
        self._final_receipt: dict[str, Any] | None = None
        self._closed = False

    @property
    def view_root(self) -> Path:
        return self._tree.view_root

    @property
    def capture_digest(self) -> str:
        return self.capture_receipt["capture_digest"]

    @classmethod
    def capture(
        cls,
        *,
        model_root: str | Path,
        manifest_path: str | Path,
        private_parent: str | Path,
        private_parent_fd: int,
        view_name: str,
        expected_uid: int,
        expected_gid: int,
        expected_device: int | None,
        expected_manifest_sha256: str = MODEL_MANIFEST_SHA256,
        expected_file_mode: int = 0o644,
        proc_fd_prefix: str | None = None,
    ) -> "ModelAuthority":
        root = _canonical_absolute(Path(model_root), label="model root", directory=True)
        expected, manifest = parse_exact23_manifest(
            manifest_path, expected_manifest_sha256=expected_manifest_sha256
        )
        actual_files, actual_directories = _scan_model_tree(root)
        if actual_files != set(expected) or actual_directories != set(
            MODEL_RELATIVE_DIRECTORIES
        ):
            raise ModelConsumptionAuthorityError("model non-cache tree closure differs")
        source_directories: list[_HeldDirectory] = []
        files: list[_HeldFile] = []
        tree: _RetainedFDTree | None = None
        try:
            source_directories = [
                _open_directory(
                    root if relative == "." else root / relative,
                    relative_path=relative,
                    label=f"model source directory {relative}",
                )
                for relative in MODEL_RELATIVE_DIRECTORIES
            ]
            files = [
                _open_double_hashed_file(
                    root / relative,
                    relative_path=relative,
                    expected_sha256=digest,
                    expected_uid=expected_uid,
                    expected_gid=expected_gid,
                    expected_mode=expected_file_mode,
                )
                for relative, digest in expected.items()
            ]
            if expected_device is not None and any(
                item.identity["device"] != expected_device for item in files
            ):
                raise ModelConsumptionAuthorityError("model device differs")
            if any(
                item.identity["uid"] != expected_uid
                or item.identity["gid"] != expected_gid
                for item in source_directories
            ):
                raise ModelConsumptionAuthorityError("model directory owner differs")
            fd_prefix = (
                "/proc/self/fd"
                if proc_fd_prefix is None
                else proc_fd_prefix
            )
            if not fd_prefix.startswith("/") or fd_prefix.endswith("/"):
                raise ModelConsumptionAuthorityError("proc FD prefix differs")
            (
                view_root,
                held_private_parent,
                view_directories,
                links,
            ) = _make_fd_view(
                parent=Path(private_parent),
                name=view_name,
                files=files,
                directories=MODEL_RELATIVE_DIRECTORIES,
                proc_fd_prefix=fd_prefix,
                private_parent_fd=private_parent_fd,
            )
            tree = _RetainedFDTree(
                source_root=root,
                files=files,
                source_directories=source_directories,
                view_root=view_root,
                private_parent=held_private_parent,
                view_directories=view_directories,
                view_links=links,
            )
            initial_replay = tree.replay(stage="holder_capture")
            rows = [
                {
                    **item.row(),
                    "proc_fd_path": links[item.relative_path],
                }
                for item in files
            ]
            directories = [item.row() for item in source_directories]
            view_directory_rows = [item.row() for item in view_directories]
            value: dict[str, Any] = {
                "schema_version": MODEL_CAPTURE_SCHEMA,
                "model_root": str(root),
                "model_view_root": str(view_root),
                "executor_pid": os.getpid(),
                "manifest": dict(manifest),
                "expected_uid": expected_uid,
                "expected_gid": expected_gid,
                "expected_device": expected_device,
                "expected_file_mode": expected_file_mode,
                "file_count": len(rows),
                "source_directory_count": len(directories),
                "view_directory_count": len(view_directories),
                "private_parent": held_private_parent.row(),
                "private_root_name": view_name,
                "view_created_only_via_held_parent_fd": True,
                "files": rows,
                "source_directories": directories,
                "view_directories": view_directory_rows,
                "view_links": links,
                "files_digest": object_sha256(rows),
                "source_directories_digest": object_sha256(directories),
                "view_directories_digest": object_sha256(
                    view_directory_rows
                ),
                "view_links_digest": object_sha256(links),
                "initial_replay_digest": initial_replay["replay_digest"],
                "same_fd_double_hash_complete": True,
                "full_identity_captured": True,
                "file_and_directory_fds_retained": True,
                "fd_view_leaf_target_kind": (
                    "inherited_proc_self_fd"
                    if proc_fd_prefix is None
                    else "injected_test_fd_prefix"
                ),
            }
            value["capture_digest"] = object_sha256(value)
            return cls(tree, manifest=manifest, capture_receipt=value)
        except Exception:
            if tree is not None:
                tree.close(strict=False)
            else:
                _close_all(files)
                _close_all(source_directories)
            raise

    def begin_task(self, task_id: str) -> dict[str, Any]:
        if self._closed or self._final_receipt is not None:
            raise ModelConsumptionAuthorityError("model authority is not active")
        task_id = _task_id(task_id)
        if self._active_task is not None:
            raise ModelConsumptionAuthorityError("another model use is active")
        replay = self._tree.replay(stage=f"task_pre:{task_id}")
        value = {
            "schema_version": MODEL_REPLAY_SCHEMA,
            "task_id": task_id,
            "model_capture_digest": self.capture_digest,
            "phase": "pre_use",
            "replay_digest": replay["replay_digest"],
            "private_parent_current_identity": (
                replay["private_parent_current_identity"]
            ),
        }
        value["use_digest"] = object_sha256(value)
        self._active_task = task_id
        self._active_pre_digest = value["use_digest"]
        return value

    def end_task(self, task_id: str) -> dict[str, Any]:
        task_id = _task_id(task_id)
        if self._active_task != task_id or self._active_pre_digest is None:
            raise ModelConsumptionAuthorityError("model use close differs")
        replay = self._tree.replay(stage=f"task_post:{task_id}")
        value = {
            "schema_version": MODEL_REPLAY_SCHEMA,
            "task_id": task_id,
            "model_capture_digest": self.capture_digest,
            "phase": "post_use",
            "pre_use_digest": self._active_pre_digest,
            "replay_digest": replay["replay_digest"],
            "private_parent_current_identity": (
                replay["private_parent_current_identity"]
            ),
        }
        value["use_digest"] = object_sha256(value)
        self._active_task = None
        self._active_pre_digest = None
        return value

    def record_task_consumption(self, consumption_digest: str) -> None:
        _sha(consumption_digest, label="task consumption digest")
        if self._active_task is not None or self._final_receipt is not None:
            raise ModelConsumptionAuthorityError("task consumption record timing differs")
        self._task_consumption_digests.append(consumption_digest)

    def finalize(self, *, expected_task_count: int) -> dict[str, Any]:
        if self._closed or self._active_task is not None:
            raise ModelConsumptionAuthorityError("model finalization timing differs")
        if self._final_receipt is not None:
            raise ModelConsumptionAuthorityError("model finalization may run only once")
        if (
            type(expected_task_count) is not int
            or expected_task_count < 0
            or len(self._task_consumption_digests) != expected_task_count
        ):
            raise ModelConsumptionAuthorityError("model task digest count differs")
        rehash = self._tree.final_rehash(
            schema_version=MODEL_FINAL_SCHEMA, stage="holder_final"
        )
        value: dict[str, Any] = {
            "schema_version": MODEL_FINAL_SCHEMA,
            "model_capture_digest": self.capture_digest,
            "task_count": expected_task_count,
            "task_consumption_digests": list(self._task_consumption_digests),
            "task_consumption_set_digest": object_sha256(
                self._task_consumption_digests
            ),
            "final_rehash_digest": rehash["final_digest"],
            "private_parent_current_identity": (
                rehash["private_parent_current_identity"]
            ),
            "all_model_bytes_rehashed_after_last_task": True,
            "all_model_file_and_directory_fds_retained_through_final_rehash": True,
        }
        value["model_final_digest"] = object_sha256(value)
        self._final_receipt = value
        return value

    def close(self) -> None:
        if self._closed:
            return
        if self._final_receipt is None or self._active_task is not None:
            raise ModelConsumptionAuthorityError(
                "model authority cannot close before final rehash"
            )
        self._tree.close(strict=True)
        self._closed = True

    def abort(self, *, reason: str) -> dict[str, Any]:
        """Close retained resources after a fail-closed shard termination."""

        if self._closed or not isinstance(reason, str) or not reason:
            raise ModelConsumptionAuthorityError("model abort differs")
        value = {
            "schema_version": MODEL_FINAL_SCHEMA,
            "model_capture_digest": self.capture_digest,
            "active_task": self._active_task,
            "completed_task_count": len(self._task_consumption_digests),
            "failure_reason": reason,
            "publication_authorized": False,
            "final_success_rehash_complete": False,
        }
        value["abort_digest"] = object_sha256(value)
        self._tree.close(strict=False)
        self._closed = True
        return value


class AdapterAuthority:
    """Per-task retained-FD authority for the exact adapter checkpoint files."""

    def __init__(
        self,
        tree: _RetainedFDTree,
        *,
        task_id: str,
        capture_receipt: Mapping[str, Any],
    ) -> None:
        self._tree = tree
        self.task_id = task_id
        self.capture_receipt = dict(capture_receipt)
        self._pre: dict[str, Any] | None = None
        self._post: dict[str, Any] | None = None
        self._final: dict[str, Any] | None = None
        self._closed = False

    @property
    def view_root(self) -> Path:
        return self._tree.view_root

    @property
    def capture_digest(self) -> str:
        return self.capture_receipt["capture_digest"]

    @classmethod
    def capture(
        cls,
        *,
        task_id: str,
        checkpoint_root: str | Path,
        expected_sha256: Mapping[str, str],
        private_parent: str | Path,
        private_parent_fd: int,
        view_name: str,
        expected_uid: int,
        expected_gid: int,
        expected_file_mode: int = 0o444,
        proc_fd_prefix: str | None = None,
    ) -> "AdapterAuthority":
        task_id = _task_id(task_id)
        if set(expected_sha256) != set(ADAPTER_RELATIVE_FILES):
            raise ModelConsumptionAuthorityError("adapter SHA closure differs")
        root = _canonical_absolute(
            Path(checkpoint_root), label="adapter checkpoint root", directory=True
        )
        root_entries = {entry.name for entry in os.scandir(root)}
        allowed_root_entries = (
            {"adapter", "optimizer.pt", "receipt.json"},
            {
                "adapter",
                "checkpoint_manifest.json",
                "optimizer.pt",
                "receipt.json",
            },
        )
        if root_entries not in allowed_root_entries or {
            entry.name for entry in os.scandir(root / "adapter")
        } != {
            "README.md",
            "adapter_config.json",
            "adapter_model.safetensors",
        }:
            raise ModelConsumptionAuthorityError("adapter checkpoint closure differs")
        source_directories: list[_HeldDirectory] = []
        files: list[_HeldFile] = []
        tree: _RetainedFDTree | None = None
        try:
            source_directories = [
                _open_directory(
                    root if relative == "." else root / relative,
                    relative_path=relative,
                    label=f"adapter source directory {relative}",
                )
                for relative in ADAPTER_RELATIVE_DIRECTORIES
            ]
            files = [
                _open_double_hashed_file(
                    root / relative,
                    relative_path=relative,
                    expected_sha256=expected_sha256[relative],
                    expected_uid=expected_uid,
                    expected_gid=expected_gid,
                    expected_mode=expected_file_mode,
                )
                for relative in ADAPTER_RELATIVE_FILES
            ]
            fd_prefix = (
                "/proc/self/fd"
                if proc_fd_prefix is None
                else proc_fd_prefix
            )
            if not fd_prefix.startswith("/") or fd_prefix.endswith("/"):
                raise ModelConsumptionAuthorityError("proc FD prefix differs")
            (
                view_root,
                held_private_parent,
                view_directories,
                links,
            ) = _make_fd_view(
                parent=Path(private_parent),
                name=view_name,
                files=files,
                directories=ADAPTER_RELATIVE_DIRECTORIES,
                proc_fd_prefix=fd_prefix,
                private_parent_fd=private_parent_fd,
            )
            tree = _RetainedFDTree(
                source_root=root,
                files=files,
                source_directories=source_directories,
                view_root=view_root,
                private_parent=held_private_parent,
                view_directories=view_directories,
                view_links=links,
            )
            initial = tree.replay(stage=f"adapter_capture:{task_id}")
            rows = [
                {
                    **item.row(),
                    "proc_fd_path": links[item.relative_path],
                }
                for item in files
            ]
            source_directory_rows = [
                item.row() for item in source_directories
            ]
            view_directory_rows = [item.row() for item in view_directories]
            value: dict[str, Any] = {
                "schema_version": ADAPTER_CAPTURE_SCHEMA,
                "task_id": task_id,
                "checkpoint_root": str(root),
                "adapter_view_root": str(view_root),
                "executor_pid": os.getpid(),
                "file_count": len(rows),
                "source_directory_count": len(source_directories),
                "view_directory_count": len(view_directories),
                "private_parent": held_private_parent.row(),
                "private_root_name": view_name,
                "view_created_only_via_held_parent_fd": True,
                "files": rows,
                "source_directories": source_directory_rows,
                "view_directories": view_directory_rows,
                "view_links": links,
                "files_digest": object_sha256(rows),
                "source_directories_digest": object_sha256(
                    source_directory_rows
                ),
                "view_directories_digest": object_sha256(
                    view_directory_rows
                ),
                "view_links_digest": object_sha256(links),
                "initial_replay_digest": initial["replay_digest"],
                "same_fd_double_hash_complete": True,
                "full_identity_captured": True,
                "file_and_directory_fds_retained": True,
                "fd_view_leaf_target_kind": (
                    "inherited_proc_self_fd"
                    if proc_fd_prefix is None
                    else "injected_test_fd_prefix"
                ),
                "safetensors_consumption_path": str(
                    view_root / "adapter/adapter_model.safetensors"
                ),
                "safetensors_consumption_is_explicit_executor_proc_fd_view": (
                    proc_fd_prefix is None
                ),
            }
            value["capture_digest"] = object_sha256(value)
            return cls(tree, task_id=task_id, capture_receipt=value)
        except Exception:
            if tree is not None:
                tree.close(strict=False)
            else:
                _close_all(files)
                _close_all(source_directories)
            raise

    def begin_use(self) -> dict[str, Any]:
        if self._closed or self._pre is not None:
            raise ModelConsumptionAuthorityError("adapter use begin differs")
        replay = self._tree.replay(stage=f"adapter_pre:{self.task_id}")
        value = {
            "schema_version": MODEL_REPLAY_SCHEMA,
            "task_id": self.task_id,
            "phase": "adapter_pre_use",
            "adapter_capture_digest": self.capture_digest,
            "replay_digest": replay["replay_digest"],
            "private_parent_current_identity": (
                replay["private_parent_current_identity"]
            ),
        }
        value["use_digest"] = object_sha256(value)
        self._pre = value
        return value

    def end_use(self) -> dict[str, Any]:
        if self._pre is None or self._post is not None or self._closed:
            raise ModelConsumptionAuthorityError("adapter use close differs")
        replay = self._tree.replay(stage=f"adapter_post:{self.task_id}")
        value = {
            "schema_version": MODEL_REPLAY_SCHEMA,
            "task_id": self.task_id,
            "phase": "adapter_post_use",
            "adapter_capture_digest": self.capture_digest,
            "pre_use_digest": self._pre["use_digest"],
            "replay_digest": replay["replay_digest"],
            "private_parent_current_identity": (
                replay["private_parent_current_identity"]
            ),
        }
        value["use_digest"] = object_sha256(value)
        self._post = value
        return value

    def finalize_and_close(self) -> dict[str, Any]:
        if self._post is None or self._final is not None or self._closed:
            raise ModelConsumptionAuthorityError("adapter finalization timing differs")
        rehash = self._tree.final_rehash(
            schema_version=ADAPTER_FINAL_SCHEMA,
            stage=f"adapter_final:{self.task_id}",
        )
        value: dict[str, Any] = {
            "schema_version": ADAPTER_FINAL_SCHEMA,
            "task_id": self.task_id,
            "adapter_capture_digest": self.capture_digest,
            "post_use_digest": self._post["use_digest"],
            "final_rehash_digest": rehash["final_digest"],
            "private_parent_current_identity": (
                rehash["private_parent_current_identity"]
            ),
            "all_adapter_bytes_rehashed_after_decoder_exit": True,
            "all_adapter_file_and_directory_fds_retained_through_rehash": True,
        }
        value["adapter_final_digest"] = object_sha256(value)
        self._final = value
        self._tree.close(strict=True)
        self._closed = True
        return value

    def abort(self, *, reason: str) -> dict[str, Any]:
        """Close an adapter authority without granting publication."""

        if self._closed or not isinstance(reason, str) or not reason:
            raise ModelConsumptionAuthorityError("adapter abort differs")
        value = {
            "schema_version": ADAPTER_FINAL_SCHEMA,
            "task_id": self.task_id,
            "adapter_capture_digest": self.capture_digest,
            "failure_reason": reason,
            "publication_authorized": False,
            "final_success_rehash_complete": False,
        }
        value["abort_digest"] = object_sha256(value)
        self._tree.close(strict=False)
        self._closed = True
        return value


def _capture_fd_rows(
    capture: Mapping[str, Any], *, scope: str
) -> list[dict[str, Any]]:
    return [
        {
            "fd": item["authority_fd"],
            "scope": scope,
            "role": "file",
            "relative_path": item["relative_path"],
            "source_path": item["path"],
            "identity": item["identity"],
        }
        for item in capture["files"]
    ]


def _capture_namespace_root_row(
    capture: Mapping[str, Any], *, scope: str
) -> dict[str, Any]:
    rows = [
        item for item in capture["view_directories"]
        if item["relative_path"] == "."
    ]
    if len(rows) != 1:
        raise ModelConsumptionAuthorityError(
            f"{scope} namespace-root descriptor closure differs"
        )
    item = rows[0]
    return {
        "fd": item["authority_fd"],
        "scope": scope,
        "role": "namespace_root",
        "relative_path": ".",
        "source_path": item["path"],
        "identity": item["identity"],
    }


def validate_task_publication_root(
    value: Any,
    *,
    verify_open_fd: bool,
    expected_inheritable: bool | None = None,
) -> dict[str, Any]:
    fields = {"fd", "path", "identity"}
    identity_fields = {
        "device", "inode", "uid", "gid", "mode", "nlink", "rdev",
        "size", "blocks", "mtime_ns", "ctime_ns",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ModelConsumptionAuthorityError(
            "task publication-root binding closure differs"
        )
    row = dict(value)
    path = Path(row["path"]) if isinstance(row.get("path"), str) else Path()
    if (
        type(row.get("fd")) is not int
        or row["fd"] < 3
        or not path.is_absolute()
        or os.path.normpath(str(path)) != str(path)
        or not isinstance(row.get("identity"), Mapping)
        or set(row["identity"]) != identity_fields
        or any(type(item) is not int for item in row["identity"].values())
        or not stat.S_ISDIR(row["identity"]["mode"])
    ):
        raise ModelConsumptionAuthorityError(
            "task publication-root binding value differs"
        )
    row["identity"] = dict(row["identity"])
    if verify_open_fd:
        try:
            before = _identity(os.fstat(row["fd"]))
            named = _named_identity(path, label="task publication root")
            inheritable = os.get_inheritable(row["fd"])
        except OSError as error:
            raise ModelConsumptionAuthorityError(
                "task publication-root descriptor is unavailable"
            ) from error
        if (
            before != row["identity"]
            or named != row["identity"]
            or (
                expected_inheritable is not None
                and inheritable is not expected_inheritable
            )
        ):
            raise ModelConsumptionAuthorityError(
                "task publication-root descriptor identity differs"
            )
    return row


def task_publication_root_binding(
    *, descriptor: int, path: str | Path
) -> dict[str, Any]:
    value = {
        "fd": descriptor,
        "path": str(Path(path)),
        "identity": _identity(os.fstat(descriptor)),
    }
    return validate_task_publication_root(
        value, verify_open_fd=True, expected_inheritable=False
    )


def build_inherited_fd_binding(
    *,
    task_id: str,
    model_capture: Mapping[str, Any],
    adapter_capture: Mapping[str, Any] | None,
    task_publication_root: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the one descriptor allowlist inherited by decoder and ranks."""

    task_id = _task_id(task_id)
    model = _validate_capture_receipt(model_capture, adapter=False)
    adapter = (
        None
        if adapter_capture is None
        else _validate_capture_receipt(adapter_capture, adapter=True)
    )
    if adapter is not None and adapter["task_id"] != task_id:
        raise ModelConsumptionAuthorityError(
            "adapter inherited-FD task differs"
        )
    publication_root = validate_task_publication_root(
        task_publication_root,
        verify_open_fd=True,
        expected_inheritable=False,
    )
    publication_row = {
        "fd": publication_root["fd"],
        "scope": "task",
        "role": "publication_root",
        "relative_path": ".",
        "source_path": publication_root["path"],
        "identity": publication_root["identity"],
    }
    fd_rows = sorted(
        [
            *_capture_fd_rows(model, scope="model"),
            _capture_namespace_root_row(model, scope="model"),
            publication_row,
            *(
                []
                if adapter is None
                else [
                    *_capture_fd_rows(adapter, scope="adapter"),
                    _capture_namespace_root_row(adapter, scope="adapter"),
                ]
            ),
        ],
        key=lambda row: row["fd"],
    )
    fds = [row["fd"] for row in fd_rows]
    if (
        len(set(fds)) != len(fds)
        or any(type(fd) is not int or fd < 3 for fd in fds)
    ):
        raise ModelConsumptionAuthorityError(
            "inherited FD allowlist differs"
        )
    value: dict[str, Any] = {
        "schema_version": INHERITED_FD_BINDING_SCHEMA,
        "task_id": task_id,
        "model_capture_digest": model["capture_digest"],
        "adapter_capture_digest": (
            None if adapter is None else adapter["capture_digest"]
        ),
        "fd_count": len(fd_rows),
        "fd_rows": fd_rows,
        "fd_rows_digest": object_sha256(fd_rows),
        "namespace_root_count": 1 if adapter is None else 2,
        "publication_root_count": 1,
        "exact_allowlist_only": True,
        "proc_self_fd_consumption_required": True,
        "cross_process_proc_fd_access_forbidden": True,
        "ptrace_authorization_used": False,
    }
    value["fd_binding_digest"] = object_sha256(value)
    return value


def validate_inherited_fd_binding(
    value: Any,
    *,
    model_capture: Mapping[str, Any] | None = None,
    adapter_capture: Mapping[str, Any] | None = None,
    task_publication_root: Mapping[str, Any] | None = None,
    verify_open_fds: bool,
    expected_inheritable: bool | None = None,
) -> dict[str, Any]:
    fields = {
        "schema_version", "task_id", "model_capture_digest",
        "adapter_capture_digest", "fd_count", "fd_rows", "fd_rows_digest",
        "namespace_root_count", "publication_root_count",
        "exact_allowlist_only", "proc_self_fd_consumption_required",
        "cross_process_proc_fd_access_forbidden", "ptrace_authorization_used",
        "fd_binding_digest",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ModelConsumptionAuthorityError(
            "inherited FD binding field closure differs"
        )
    row = dict(value)
    unsigned = dict(row)
    claimed = unsigned.pop("fd_binding_digest")
    if (
        row["schema_version"] != INHERITED_FD_BINDING_SCHEMA
        or row["exact_allowlist_only"] is not True
        or row["proc_self_fd_consumption_required"] is not True
        or row["cross_process_proc_fd_access_forbidden"] is not True
        or row["ptrace_authorization_used"] is not False
        or claimed != object_sha256(unsigned)
    ):
        raise ModelConsumptionAuthorityError(
            "inherited FD binding policy or digest differs"
        )
    _task_id(row["task_id"])
    _sha(row["model_capture_digest"], label="inherited model capture")
    if row["adapter_capture_digest"] is not None:
        _sha(row["adapter_capture_digest"], label="inherited adapter capture")
    if (
        not isinstance(row["fd_rows"], list)
        or row["fd_count"] != len(row["fd_rows"])
        or row["fd_rows_digest"] != object_sha256(row["fd_rows"])
        or type(row["namespace_root_count"]) is not int
        or type(row["publication_root_count"]) is not int
    ):
        raise ModelConsumptionAuthorityError("inherited FD rows differ")
    fds: list[int] = []
    scope_roles_relatives: list[tuple[str, str, str]] = []
    identity_fields = {
        "device", "inode", "uid", "gid", "mode", "nlink", "rdev",
        "size", "blocks", "mtime_ns", "ctime_ns",
    }
    for item in row["fd_rows"]:
        if not isinstance(item, Mapping) or set(item) != {
            "fd", "scope", "role", "relative_path", "source_path", "identity"
        }:
            raise ModelConsumptionAuthorityError("inherited FD row differs")
        fd = item["fd"]
        if (
            type(fd) is not int
            or fd < 3
            or item["scope"] not in {"model", "adapter", "task"}
            or item["role"] not in {"file", "namespace_root", "publication_root"}
            or not isinstance(item["relative_path"], str)
            or not isinstance(item["source_path"], str)
            or not Path(item["source_path"]).is_absolute()
            or os.path.normpath(item["source_path"]) != item["source_path"]
            or not isinstance(item["identity"], Mapping)
            or set(item["identity"]) != identity_fields
            or any(type(value) is not int for value in item["identity"].values())
        ):
            raise ModelConsumptionAuthorityError("inherited FD row value differs")
        if (
            (item["role"] == "file" and not stat.S_ISREG(item["identity"]["mode"]))
            or (
                item["role"] in {"namespace_root", "publication_root"}
                and not stat.S_ISDIR(item["identity"]["mode"])
            )
        ):
            raise ModelConsumptionAuthorityError("inherited FD row type differs")
        fds.append(fd)
        scope_roles_relatives.append(
            (item["scope"], item["role"], item["relative_path"])
        )
    if fds != sorted(fds) or len(fds) != len(set(fds)):
        raise ModelConsumptionAuthorityError("inherited FD order differs")
    expected_scope_roles_relatives = {
        *(("model", "file", relative) for relative in MODEL_RELATIVE_FILES),
        ("model", "namespace_root", "."),
        ("task", "publication_root", "."),
        *(
            ()
            if row["adapter_capture_digest"] is None
            else tuple(
                ("adapter", "file", relative)
                for relative in ADAPTER_RELATIVE_FILES
            )
        ),
        *(
            ()
            if row["adapter_capture_digest"] is None
            else (("adapter", "namespace_root", "."),)
        ),
    }
    if (
        len(scope_roles_relatives) != len(expected_scope_roles_relatives)
        or set(scope_roles_relatives) != expected_scope_roles_relatives
        or row["fd_count"] not in {MODEL_FILE_COUNT + 2, MODEL_FILE_COUNT + 7}
        or (row["adapter_capture_digest"] is None)
        is not (row["fd_count"] == MODEL_FILE_COUNT + 2)
        or row["namespace_root_count"]
        != (1 if row["adapter_capture_digest"] is None else 2)
        or row["publication_root_count"] != 1
    ):
        raise ModelConsumptionAuthorityError(
            "inherited FD intrinsic file/directory closure differs"
        )
    if model_capture is not None:
        model = _validate_capture_receipt(model_capture, adapter=False)
        adapter = (
            None
            if adapter_capture is None
            else _validate_capture_receipt(adapter_capture, adapter=True)
        )
        expected_capture_rows = [
            *_capture_fd_rows(model, scope="model"),
            _capture_namespace_root_row(model, scope="model"),
            *(
                []
                if adapter is None
                else [
                    *_capture_fd_rows(adapter, scope="adapter"),
                    _capture_namespace_root_row(adapter, scope="adapter"),
                ]
            ),
        ]
        observed_capture_rows = [
            item for item in row["fd_rows"] if item["scope"] != "task"
        ]
        if sorted(
            observed_capture_rows, key=lambda item: item["fd"]
        ) != sorted(expected_capture_rows, key=lambda item: item["fd"]):
            raise ModelConsumptionAuthorityError(
                "inherited FD/capture binding differs"
            )
    elif adapter_capture is not None:
        raise ModelConsumptionAuthorityError(
            "adapter capture supplied without model capture"
        )
    if task_publication_root is not None:
        task = validate_task_publication_root(
            task_publication_root,
            verify_open_fd=verify_open_fds,
            expected_inheritable=expected_inheritable,
        )
        task_rows = [
            item for item in row["fd_rows"]
            if item["role"] == "publication_root"
        ]
        if task_rows != [{
            "fd": task["fd"],
            "scope": "task",
            "role": "publication_root",
            "relative_path": ".",
            "source_path": task["path"],
            "identity": task["identity"],
        }]:
            raise ModelConsumptionAuthorityError(
                "inherited FD/task publication-root binding differs"
            )
    if verify_open_fds:
        for item in row["fd_rows"]:
            try:
                observed = _identity(os.fstat(item["fd"]))
                named = _named_identity(
                    Path(item["source_path"]),
                    label="inherited authority FD source path",
                )
                inheritable = os.get_inheritable(item["fd"])
            except OSError as error:
                raise ModelConsumptionAuthorityError(
                    "inherited authority FD is unavailable"
                ) from error
            task_root_mutable = (
                item["scope"] == "task"
                and item["role"] == "publication_root"
            )
            immutable_fields = {
                "device", "inode", "uid", "gid", "mode", "rdev",
            }
            identity_differs = (
                (
                    not task_root_mutable
                    and (observed != item["identity"] or named != item["identity"])
                )
                or (
                    task_root_mutable
                    and (
                        observed != named
                        or {
                            field: observed[field] for field in immutable_fields
                        } != {
                            field: item["identity"][field]
                            for field in immutable_fields
                        }
                    )
                )
            )
            if identity_differs or (
                expected_inheritable is not None
                and inheritable is not expected_inheritable
            ):
                raise ModelConsumptionAuthorityError(
                    "inherited authority FD identity differs"
                )
    return row


def inherited_fd_numbers(value: Mapping[str, Any]) -> tuple[int, ...]:
    row = validate_inherited_fd_binding(value, verify_open_fds=False)
    return tuple(item["fd"] for item in row["fd_rows"])


def inherited_fd_row(
    value: Mapping[str, Any], *, scope: str, role: str
) -> dict[str, Any]:
    row = validate_inherited_fd_binding(value, verify_open_fds=False)
    matches = [
        dict(item) for item in row["fd_rows"]
        if item["scope"] == scope and item["role"] == role
    ]
    if len(matches) != 1:
        raise ModelConsumptionAuthorityError(
            f"inherited {scope} {role} descriptor closure differs"
        )
    return matches[0]


def inherited_proc_root(
    value: Mapping[str, Any], *, scope: str, role: str
) -> str:
    item = inherited_fd_row(value, scope=scope, role=role)
    return f"/proc/self/fd/{item['fd']}"


def inherited_task_member_path(
    value: Mapping[str, Any], basename: str
) -> str:
    if (
        not isinstance(basename, str)
        or basename in ("", ".", "..")
        or "/" in basename
        or "\x00" in basename
    ):
        raise ModelConsumptionAuthorityError(
            "inherited task member basename differs"
        )
    return str(
        Path(inherited_proc_root(
            value, scope="task", role="publication_root"
        )) / basename
    )


def seal_inherited_fds(value: Mapping[str, Any]) -> dict[str, Any]:
    """Restore CLOEXEC immediately after one sanctioned spawn boundary."""

    row = validate_inherited_fd_binding(
        value, verify_open_fds=True, expected_inheritable=True
    )
    for fd in inherited_fd_numbers(row):
        os.set_inheritable(fd, False)
    validate_inherited_fd_binding(
        row, verify_open_fds=True, expected_inheritable=False
    )
    return row


def inherited_fd_environment_value(value: Mapping[str, Any]) -> str:
    row = validate_inherited_fd_binding(value, verify_open_fds=False)
    return canonical_json_bytes(row).decode("utf-8")


def load_inherited_fd_environment(
    *,
    model_capture: Mapping[str, Any] | None = None,
    adapter_capture: Mapping[str, Any] | None = None,
    verify_open_fds: bool,
    expected_inheritable: bool | None = None,
) -> dict[str, Any]:
    raw = os.environ.get(INHERITED_FD_BINDING_ENV)
    if raw is None:
        raise ModelConsumptionAuthorityError(
            "inherited authority FD environment is absent"
        )
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError, UnicodeError) as error:
        raise ModelConsumptionAuthorityError(
            "inherited authority FD environment is not JSON"
        ) from error
    if not isinstance(value, dict) or canonical_json_bytes(value).decode(
        "utf-8"
    ) != raw:
        raise ModelConsumptionAuthorityError(
            "inherited authority FD environment is not canonical"
        )
    return validate_inherited_fd_binding(
        value,
        model_capture=model_capture,
        adapter_capture=adapter_capture,
        verify_open_fds=verify_open_fds,
        expected_inheritable=expected_inheritable,
    )


def build_consumption_input(
    *,
    task_id: str,
    physical_bindings_digest: str,
    model_capture: Mapping[str, Any],
    model_pre_use: Mapping[str, Any],
    model_capture_receipt_path: str | Path,
    model_capture_receipt_sha256: str,
    adapter_capture: Mapping[str, Any] | None,
    adapter_pre_use: Mapping[str, Any] | None,
    adapter_capture_receipt_path: str | Path | None,
    adapter_capture_receipt_sha256: str | None,
    inherited_fd_binding: Mapping[str, Any],
    task_publication_root: Mapping[str, Any],
    production_mode: bool = True,
    task_member_path_prefix: str | Path | None = None,
) -> dict[str, Any]:
    """Bind the exact FD views available before the decoder starts."""

    task_id = _task_id(task_id)
    _sha(physical_bindings_digest, label="physical bindings digest")
    if (
        model_capture.get("schema_version") != MODEL_CAPTURE_SCHEMA
        or model_pre_use.get("schema_version") != MODEL_REPLAY_SCHEMA
        or model_pre_use.get("task_id") != task_id
        or model_pre_use.get("phase") != "pre_use"
        or model_pre_use.get("model_capture_digest")
        != model_capture.get("capture_digest")
    ):
        raise ModelConsumptionAuthorityError("model consumption input differs")
    model_unsigned = dict(model_capture)
    model_claim = model_unsigned.pop("capture_digest", None)
    if model_claim != object_sha256(model_unsigned):
        raise ModelConsumptionAuthorityError("model capture receipt digest differs")
    model_path = Path(model_capture_receipt_path)
    if not model_path.is_absolute():
        raise ModelConsumptionAuthorityError("model capture receipt path differs")
    _sha(model_capture_receipt_sha256, label="model capture receipt file")
    adapter_values = (
        adapter_capture,
        adapter_pre_use,
        adapter_capture_receipt_path,
        adapter_capture_receipt_sha256,
    )
    if any(value is None for value in adapter_values) and not all(
        value is None for value in adapter_values
    ):
        raise ModelConsumptionAuthorityError("adapter consumption input is mixed")
    adapter_binding: dict[str, Any] | None
    if adapter_capture is None:
        adapter_binding = None
    else:
        assert adapter_pre_use is not None
        assert adapter_capture_receipt_path is not None
        assert adapter_capture_receipt_sha256 is not None
        if (
            adapter_capture.get("schema_version") != ADAPTER_CAPTURE_SCHEMA
            or adapter_capture.get("task_id") != task_id
            or adapter_pre_use.get("schema_version") != MODEL_REPLAY_SCHEMA
            or adapter_pre_use.get("task_id") != task_id
            or adapter_pre_use.get("phase") != "adapter_pre_use"
            or adapter_pre_use.get("adapter_capture_digest")
            != adapter_capture.get("capture_digest")
        ):
            raise ModelConsumptionAuthorityError("adapter consumption input differs")
        adapter_unsigned = dict(adapter_capture)
        adapter_claim = adapter_unsigned.pop("capture_digest", None)
        if adapter_claim != object_sha256(adapter_unsigned):
            raise ModelConsumptionAuthorityError("adapter capture receipt digest differs")
        adapter_path = Path(adapter_capture_receipt_path)
        if not adapter_path.is_absolute():
            raise ModelConsumptionAuthorityError("adapter capture receipt path differs")
        _sha(adapter_capture_receipt_sha256, label="adapter capture receipt file")
        adapter_binding = {
            "capture_receipt_path": str(adapter_path),
            "capture_receipt_sha256": adapter_capture_receipt_sha256,
            "capture_digest": adapter_capture["capture_digest"],
            "pre_use_digest": adapter_pre_use["use_digest"],
            "view_root": adapter_capture["adapter_view_root"],
        }
    inherited_fds = validate_inherited_fd_binding(
        inherited_fd_binding,
        model_capture=model_capture,
        adapter_capture=adapter_capture,
        task_publication_root=task_publication_root,
        verify_open_fds=True,
        expected_inheritable=False,
    )
    if type(production_mode) is not bool:
        raise ModelConsumptionAuthorityError(
            "consumption input production mode differs"
        )
    inherited_task_root = Path(inherited_proc_root(
        inherited_fds, scope="task", role="publication_root"
    ))
    if production_mode:
        if task_member_path_prefix is not None:
            raise ModelConsumptionAuthorityError(
                "production task member prefix must be inherited proc-FD root"
            )
        task_member_root = inherited_task_root
        model_view_root = inherited_proc_root(
            inherited_fds, scope="model", role="namespace_root"
        )
        path_kind = "inherited_proc_self_fd"
    else:
        prefix = (
            Path(task_member_path_prefix)
            if task_member_path_prefix is not None else Path()
        )
        if (
            not prefix.is_absolute()
            or os.path.normpath(str(prefix)) != str(prefix)
            or str(prefix) != task_publication_root["path"]
        ):
            raise ModelConsumptionAuthorityError(
                "injected task member prefix differs"
            )
        task_member_root = prefix
        model_view_root = model_capture["model_view_root"]
        path_kind = "injected_named_test_root"
    adapter_view_root = (
        None
        if adapter_capture is None
        else (
            inherited_proc_root(
                inherited_fds, scope="adapter", role="namespace_root"
            )
            if production_mode
            else adapter_capture["adapter_view_root"]
        )
    )
    if model_path.parent != task_member_root:
        raise ModelConsumptionAuthorityError(
            "model capture receipt is outside inherited task root"
        )
    if adapter_binding is not None:
        adapter_receipt = Path(adapter_binding["capture_receipt_path"])
        if adapter_receipt.parent != task_member_root:
            raise ModelConsumptionAuthorityError(
                "adapter capture receipt is outside inherited task root"
            )
        adapter_binding["view_root"] = adapter_view_root
    value: dict[str, Any] = {
        "schema_version": CONSUMPTION_INPUT_SCHEMA,
        "task_id": task_id,
        "physical_bindings_digest": physical_bindings_digest,
        "model": {
            "capture_receipt_path": str(model_path),
            "capture_receipt_sha256": model_capture_receipt_sha256,
            "capture_digest": model_capture["capture_digest"],
            "pre_use_digest": model_pre_use["use_digest"],
            "view_root": model_view_root,
        },
        "adapter": adapter_binding,
        "inherited_fds": inherited_fds,
        "production_mode": production_mode,
        "task_member_path_kind": path_kind,
        "base_model_and_adapter_consumed_only_from_fd_views": True,
        "training_loss_read_or_used": False,
    }
    value["consumption_input_digest"] = object_sha256(value)
    return value


def validate_consumption_input(value: Any) -> dict[str, Any]:
    fields = {
        "schema_version", "task_id", "physical_bindings_digest", "model",
        "adapter", "inherited_fds",
        "production_mode", "task_member_path_kind",
        "base_model_and_adapter_consumed_only_from_fd_views",
        "training_loss_read_or_used", "consumption_input_digest",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ModelConsumptionAuthorityError("consumption input field closure differs")
    row = dict(value)
    if (
        row["schema_version"] != CONSUMPTION_INPUT_SCHEMA
        or row["base_model_and_adapter_consumed_only_from_fd_views"] is not True
        or row["training_loss_read_or_used"] is not False
    ):
        raise ModelConsumptionAuthorityError("consumption input policy differs")
    _task_id(row["task_id"])
    _sha(row["physical_bindings_digest"], label="physical bindings digest")
    model_fields = {
        "capture_receipt_path", "capture_receipt_sha256", "capture_digest",
        "pre_use_digest", "view_root",
    }
    for label, binding in (("model", row["model"]), ("adapter", row["adapter"])):
        if binding is None:
            if label != "adapter":
                raise ModelConsumptionAuthorityError("model input binding is absent")
            continue
        if not isinstance(binding, Mapping) or set(binding) != model_fields:
            raise ModelConsumptionAuthorityError(f"{label} input binding differs")
        if not Path(binding["capture_receipt_path"]).is_absolute() or not Path(
            binding["view_root"]
        ).is_absolute():
            raise ModelConsumptionAuthorityError(f"{label} input paths differ")
        for field in ("capture_receipt_sha256", "capture_digest", "pre_use_digest"):
            _sha(binding[field], label=f"{label} {field}")
    inherited = validate_inherited_fd_binding(
        row["inherited_fds"], verify_open_fds=False
    )
    inherited_task_root = Path(inherited_proc_root(
        inherited, scope="task", role="publication_root"
    ))
    task_row = inherited_fd_row(
        inherited, scope="task", role="publication_root"
    )
    if row.get("production_mode") is True:
        if row.get("task_member_path_kind") != "inherited_proc_self_fd":
            raise ModelConsumptionAuthorityError(
                "production consumption path kind differs"
            )
        task_root = inherited_task_root
        model_root = inherited_proc_root(
            inherited, scope="model", role="namespace_root"
        )
    elif row.get("production_mode") is False:
        if row.get("task_member_path_kind") != "injected_named_test_root":
            raise ModelConsumptionAuthorityError(
                "injected consumption path kind differs"
            )
        task_root = Path(task_row["source_path"])
        model_root = inherited_fd_row(
            inherited, scope="model", role="namespace_root"
        )["source_path"]
    else:
        raise ModelConsumptionAuthorityError(
            "consumption production mode differs"
        )
    if (
        Path(row["model"]["capture_receipt_path"]).parent != task_root
        or row["model"]["view_root"] != model_root
    ):
        raise ModelConsumptionAuthorityError(
            "model input inherited-root paths differ"
        )
    if row["adapter"] is not None:
        adapter_root = (
            inherited_proc_root(
                inherited, scope="adapter", role="namespace_root"
            )
            if row["production_mode"]
            else inherited_fd_row(
                inherited, scope="adapter", role="namespace_root"
            )["source_path"]
        )
        if (
            Path(row["adapter"]["capture_receipt_path"]).parent != task_root
            or row["adapter"]["view_root"] != adapter_root
        ):
            raise ModelConsumptionAuthorityError(
                "adapter input inherited-root paths differ"
            )
    unsigned = dict(row)
    claimed = unsigned.pop("consumption_input_digest")
    if claimed != object_sha256(unsigned):
        raise ModelConsumptionAuthorityError("consumption input digest differs")
    return row


def stable_inherited_task_file(
    path: str | Path,
    *,
    inherited_fd_binding: Mapping[str, Any],
    label: str,
    expected_sha256: str | None = None,
) -> tuple[bytes, dict[str, int]]:
    if expected_sha256 is not None:
        _sha(expected_sha256, label=f"{label} SHA")
    binding = validate_inherited_fd_binding(
        inherited_fd_binding,
        verify_open_fds=True,
        expected_inheritable=False,
    )
    task = inherited_fd_row(
        binding, scope="task", role="publication_root"
    )
    path_value = Path(path)
    expected_parent = Path(f"/proc/self/fd/{task['fd']}")
    if (
        not path_value.is_absolute()
        or path_value.parent != expected_parent
        or path_value.name in ("", ".", "..")
        or "/" in path_value.name
    ):
        raise ModelConsumptionAuthorityError(
            f"{label} path is outside inherited task root"
        )
    descriptor = os.open(
        path_value.name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        dir_fd=task["fd"],
    )
    try:
        before = _identity(os.fstat(descriptor))
        first = _read_fd(descriptor)
        middle = _identity(os.fstat(descriptor))
        second = _read_fd(descriptor)
        after = _identity(os.fstat(descriptor))
        named = _identity(os.stat(
            path_value.name, dir_fd=task["fd"], follow_symlinks=False
        ))
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISREG(before["mode"])
        or before["nlink"] != 1
        or before != middle
        or before != after
        or before != named
        or first != second
        or len(first) != before["size"]
        or (
            expected_sha256 is not None
            and bytes_sha256(first) != expected_sha256
        )
    ):
        raise ModelConsumptionAuthorityError(
            f"{label} inherited task-root capture differs"
        )
    return first, before


def stable_inherited_view_file(
    path: str | Path,
    *,
    inherited_fd_binding: Mapping[str, Any],
    label: str,
    expected_sha256: str | None = None,
) -> tuple[bytes, dict[str, int]]:
    """Double-read one exact leaf through its inherited namespace-root FD."""

    if expected_sha256 is not None:
        _sha(expected_sha256, label=f"{label} SHA")
    binding = validate_inherited_fd_binding(
        inherited_fd_binding,
        verify_open_fds=True,
        expected_inheritable=False,
    )
    path_value = Path(path)
    matched: tuple[dict[str, Any], dict[str, Any]] | None = None
    for scope in ("model", "adapter"):
        namespace_rows = [
            dict(item) for item in binding["fd_rows"]
            if item["scope"] == scope and item["role"] == "namespace_root"
        ]
        if not namespace_rows:
            continue
        if len(namespace_rows) != 1:
            raise ModelConsumptionAuthorityError(
                f"{label} namespace-root closure differs"
            )
        namespace = namespace_rows[0]
        root = Path(f"/proc/self/fd/{namespace['fd']}")
        try:
            relative = path_value.relative_to(root).as_posix()
        except ValueError:
            continue
        file_rows = [
            dict(item) for item in binding["fd_rows"]
            if item["scope"] == scope
            and item["role"] == "file"
            and item["relative_path"] == relative
        ]
        if len(file_rows) != 1:
            raise ModelConsumptionAuthorityError(
                f"{label} inherited namespace leaf differs"
            )
        matched = namespace, file_rows[0]
        break
    if matched is None:
        raise ModelConsumptionAuthorityError(
            f"{label} is outside inherited namespace roots"
        )
    _, file_row = matched
    try:
        before = _identity(os.fstat(file_row["fd"]))
        first = _read_fd(file_row["fd"])
        middle = _identity(os.fstat(file_row["fd"]))
        second = _read_fd(file_row["fd"])
        after = _identity(os.fstat(file_row["fd"]))
        link_info = path_value.lstat()
        link_target = os.readlink(path_value)
        resolved_fd = os.open(
            path_value, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            resolved = _identity(os.fstat(resolved_fd))
        finally:
            os.close(resolved_fd)
    except OSError as error:
        raise ModelConsumptionAuthorityError(
            f"{label} inherited namespace replay is unavailable"
        ) from error
    digest = bytes_sha256(first)
    if (
        before != file_row["identity"]
        or middle != before
        or after != before
        or resolved != before
        or not stat.S_ISLNK(link_info.st_mode)
        or link_target != f"/proc/self/fd/{file_row['fd']}"
        or first != second
        or len(first) != before["size"]
        or (expected_sha256 is not None and digest != expected_sha256)
    ):
        raise ModelConsumptionAuthorityError(
            f"{label} inherited namespace capture differs"
        )
    return first, before


def _load_canonical_receipt(
    path: Path, *, expected_sha256: str | None, label: str,
    inherited_fd_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if inherited_fd_binding is None:
        raw, _ = _stable_small_file(
            path, label=label, expected_sha256=expected_sha256
        )
    else:
        raw, _ = stable_inherited_task_file(
            path,
            inherited_fd_binding=inherited_fd_binding,
            label=label,
            expected_sha256=expected_sha256,
        )
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelConsumptionAuthorityError(f"{label} is not JSON") from error
    if not isinstance(value, dict) or raw != canonical_json_bytes(value) + b"\n":
        raise ModelConsumptionAuthorityError(f"{label} is not canonical JSON")
    return value


def _validate_capture_receipt(
    value: Any, *, adapter: bool, verify_views: bool = True
) -> dict[str, Any]:
    model_fields = {
        "schema_version", "model_root", "model_view_root", "executor_pid",
        "manifest", "expected_uid", "expected_gid", "expected_device",
        "expected_file_mode", "file_count", "source_directory_count",
        "view_directory_count", "files", "source_directories",
        "view_directories", "view_links", "files_digest",
        "private_parent", "private_root_name",
        "view_created_only_via_held_parent_fd",
        "source_directories_digest", "view_directories_digest",
        "view_links_digest",
        "initial_replay_digest", "same_fd_double_hash_complete",
        "full_identity_captured", "file_and_directory_fds_retained",
        "fd_view_leaf_target_kind", "capture_digest",
    }
    adapter_fields = {
        "schema_version", "task_id", "checkpoint_root", "adapter_view_root",
        "executor_pid", "file_count", "source_directory_count",
        "view_directory_count", "files", "source_directories",
        "view_directories", "view_links", "files_digest",
        "private_parent", "private_root_name",
        "view_created_only_via_held_parent_fd",
        "source_directories_digest", "view_directories_digest",
        "view_links_digest",
        "initial_replay_digest",
        "same_fd_double_hash_complete", "full_identity_captured",
        "file_and_directory_fds_retained", "fd_view_leaf_target_kind",
        "safetensors_consumption_path",
        "safetensors_consumption_is_explicit_executor_proc_fd_view",
        "capture_digest",
    }
    fields = adapter_fields if adapter else model_fields
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ModelConsumptionAuthorityError("capture receipt field closure differs")
    row = dict(value)
    expected_schema = ADAPTER_CAPTURE_SCHEMA if adapter else MODEL_CAPTURE_SCHEMA
    if row["schema_version"] != expected_schema:
        raise ModelConsumptionAuthorityError("capture receipt schema differs")
    unsigned = dict(row)
    claimed = unsigned.pop("capture_digest")
    if claimed != object_sha256(unsigned):
        raise ModelConsumptionAuthorityError("capture receipt digest differs")
    view_root_key = "adapter_view_root" if adapter else "model_view_root"
    view_root = Path(row[view_root_key])
    if not view_root.is_absolute() or (
        verify_views
        and (not view_root.is_dir() or view_root.is_symlink())
    ):
        raise ModelConsumptionAuthorityError("capture view root differs")
    private_parent = row["private_parent"]
    identity_fields = {
        "device", "inode", "uid", "gid", "mode", "nlink", "rdev",
        "size", "blocks", "mtime_ns", "ctime_ns",
    }
    if (
        not isinstance(private_parent, Mapping)
        or set(private_parent) != {
            "relative_path", "path", "authority_fd", "identity",
        }
        or private_parent.get("relative_path") != "."
        or private_parent.get("path") != str(view_root.parent)
        or type(private_parent.get("authority_fd")) is not int
        or private_parent["authority_fd"] < 3
        or not isinstance(private_parent.get("identity"), Mapping)
        or set(private_parent["identity"]) != identity_fields
        or any(type(value) is not int for value in private_parent["identity"].values())
        or not stat.S_ISDIR(private_parent["identity"]["mode"])
        or type(row.get("private_root_name")) is not str
        or row["private_root_name"] in ("", ".", "..")
        or "/" in row["private_root_name"]
        or "\x00" in row["private_root_name"]
        or view_root.name != row["private_root_name"]
        or row.get("view_created_only_via_held_parent_fd") is not True
    ):
        raise ModelConsumptionAuthorityError(
            "capture private-parent authority differs"
        )
    if verify_views:
        try:
            parent_held = _identity(os.fstat(private_parent["authority_fd"]))
            parent_named = _named_identity(
                Path(private_parent["path"]), label="capture private parent"
            )
        except OSError as error:
            raise ModelConsumptionAuthorityError(
                "capture private-parent FD is unavailable"
            ) from error
        immutable_fields = {
            "device", "inode", "uid", "gid", "mode", "rdev",
        }
        if (
            parent_held != parent_named
            or {
                field: parent_held[field] for field in immutable_fields
            } != {
                field: private_parent["identity"][field]
                for field in immutable_fields
            }
            or os.get_inheritable(private_parent["authority_fd"])
        ):
            raise ModelConsumptionAuthorityError(
                "capture private-parent FD replay differs"
            )
    expected_count = len(ADAPTER_RELATIVE_FILES) if adapter else MODEL_FILE_COUNT
    if row["file_count"] != expected_count or not isinstance(row["files"], list):
        raise ModelConsumptionAuthorityError("capture file count differs")
    if row["files_digest"] != object_sha256(row["files"]):
        raise ModelConsumptionAuthorityError("capture file rows digest differs")
    if row["source_directories_digest"] != object_sha256(
        row["source_directories"]
    ):
        raise ModelConsumptionAuthorityError("capture source directories digest differs")
    if row["view_directories_digest"] != object_sha256(
        row["view_directories"]
    ):
        raise ModelConsumptionAuthorityError("capture view directories digest differs")
    if row["view_links_digest"] != object_sha256(row["view_links"]):
        raise ModelConsumptionAuthorityError("capture view links digest differs")
    if not isinstance(row["view_links"], Mapping):
        raise ModelConsumptionAuthorityError("capture view links differ")
    expected_directory_count = (
        len(ADAPTER_RELATIVE_DIRECTORIES)
        if adapter
        else MODEL_DIRECTORY_COUNT
    )
    for label, directories in (
        ("source", row["source_directories"]),
        ("view", row["view_directories"]),
    ):
        if (
            not isinstance(directories, list)
            or len(directories) != expected_directory_count
        ):
            raise ModelConsumptionAuthorityError(
                f"capture {label} directory count differs"
            )
        for item in directories:
            if not isinstance(item, Mapping) or set(item) != {
                "relative_path", "path", "authority_fd", "identity"
            }:
                raise ModelConsumptionAuthorityError(
                    f"capture {label} directory row differs"
                )
            if (
                type(item["authority_fd"]) is not int
                or item["authority_fd"] < 3
                or not isinstance(item["identity"], Mapping)
                or set(item["identity"]) != identity_fields
                or any(type(value) is not int for value in item["identity"].values())
            ):
                raise ModelConsumptionAuthorityError(
                    f"capture {label} directory identity differs"
                )
    if (
        row["source_directory_count"] != expected_directory_count
        or row["view_directory_count"] != expected_directory_count
        or row["file_and_directory_fds_retained"] is not True
        or row["fd_view_leaf_target_kind"]
        not in {"inherited_proc_self_fd", "injected_test_fd_prefix"}
    ):
        raise ModelConsumptionAuthorityError(
            "capture retained directory/FD policy differs"
        )
    relative_paths = [item.get("relative_path") for item in row["files"]]
    if (
        len(set(relative_paths)) != expected_count
        or set(relative_paths) != set(row["view_links"])
    ):
        raise ModelConsumptionAuthorityError("capture relative path closure differs")
    for item in row["files"]:
        if not isinstance(item, Mapping) or set(item) != {
            "relative_path", "path", "sha256", "identity", "authority_fd",
            "proc_fd_path",
        }:
            raise ModelConsumptionAuthorityError("capture file row differs")
        relative = item["relative_path"]
        _sha(item["sha256"], label="captured file SHA")
        if (
            type(item["authority_fd"]) is not int
            or item["authority_fd"] < 3
            or item["proc_fd_path"] != row["view_links"][relative]
            or not item["proc_fd_path"].endswith(
                f"/fd/{item['authority_fd']}"
            )
        ):
            raise ModelConsumptionAuthorityError("capture proc-FD binding differs")
        if not verify_views:
            continue
        leaf = view_root / relative
        try:
            leaf_info = leaf.lstat()
            target = os.readlink(leaf)
            descriptor = os.open(
                leaf, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            )
            try:
                resolved = _identity(os.fstat(descriptor))
            finally:
                os.close(descriptor)
        except OSError as error:
            raise ModelConsumptionAuthorityError("capture FD-view leaf unavailable") from error
        if (
            not stat.S_ISLNK(leaf_info.st_mode)
            or target != item["proc_fd_path"]
            or resolved != item["identity"]
        ):
            raise ModelConsumptionAuthorityError("capture FD-view identity differs")
    return row


def validate_model_capture_receipt(
    value: Any, *, verify_views: bool
) -> dict[str, Any]:
    return _validate_capture_receipt(
        value, adapter=False, verify_views=verify_views
    )


def validate_adapter_capture_receipt(
    value: Any, *, verify_views: bool
) -> dict[str, Any]:
    return _validate_capture_receipt(
        value, adapter=True, verify_views=verify_views
    )


def load_consumption_input(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_digest: str,
    verify_views: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    """Load a pre-use receipt and replay the model/adapter FD views in a rank."""

    _sha(expected_sha256, label="consumption input file")
    _sha(expected_digest, label="consumption input digest")
    environment_fds = load_inherited_fd_environment(
        verify_open_fds=True, expected_inheritable=False
    )
    task_row = inherited_fd_row(
        environment_fds, scope="task", role="publication_root"
    )
    proc_task_root = Path(f"/proc/self/fd/{task_row['fd']}")
    injected_task_root = Path(task_row["source_path"])
    input_path = Path(path)
    if input_path.parent == proc_task_root:
        input_via_inherited_root = True
        input_value = _load_canonical_receipt(
            input_path,
            expected_sha256=expected_sha256,
            label="consumption input",
            inherited_fd_binding=environment_fds,
        )
    elif input_path.parent == injected_task_root:
        input_via_inherited_root = False
        input_value = _load_canonical_receipt(
            input_path,
            expected_sha256=expected_sha256,
            label="consumption input",
        )
    else:
        raise ModelConsumptionAuthorityError(
            "consumption input is outside task authority roots"
        )
    input_row = validate_consumption_input(input_value)
    if (
        input_row["consumption_input_digest"] != expected_digest
        or input_row["production_mode"] is not input_via_inherited_root
    ):
        raise ModelConsumptionAuthorityError("consumption input literal digest differs")
    model_binding = input_row["model"]
    model_value = _load_canonical_receipt(
        Path(model_binding["capture_receipt_path"]),
        expected_sha256=model_binding["capture_receipt_sha256"],
        label="model capture receipt",
        inherited_fd_binding=(
            environment_fds if input_row["production_mode"] else None
        ),
    )
    model = _validate_capture_receipt(
        model_value,
        adapter=False,
        verify_views=(verify_views and not input_row["production_mode"]),
    )
    model_namespace = inherited_fd_row(
        environment_fds, scope="model", role="namespace_root"
    )
    if (
        model["capture_digest"] != model_binding["capture_digest"]
        or model["model_view_root"] != model_namespace["source_path"]
        or model_binding["view_root"]
        != (
            f"/proc/self/fd/{model_namespace['fd']}"
            if input_row["production_mode"]
            else model_namespace["source_path"]
        )
    ):
        raise ModelConsumptionAuthorityError("model capture/input binding differs")
    adapter_binding = input_row["adapter"]
    adapter_row: dict[str, Any] | None = None
    if adapter_binding is not None:
        adapter_value = _load_canonical_receipt(
            Path(adapter_binding["capture_receipt_path"]),
            expected_sha256=adapter_binding["capture_receipt_sha256"],
            label="adapter capture receipt",
            inherited_fd_binding=(
                environment_fds if input_row["production_mode"] else None
            ),
        )
        adapter_row = _validate_capture_receipt(
            adapter_value,
            adapter=True,
            verify_views=(verify_views and not input_row["production_mode"]),
        )
        adapter_namespace = inherited_fd_row(
            environment_fds, scope="adapter", role="namespace_root"
        )
        if (
            adapter_row["capture_digest"] != adapter_binding["capture_digest"]
            or adapter_row["adapter_view_root"]
            != adapter_namespace["source_path"]
            or adapter_binding["view_root"]
            != (
                f"/proc/self/fd/{adapter_namespace['fd']}"
                if input_row["production_mode"]
                else adapter_namespace["source_path"]
            )
            or adapter_row["task_id"] != input_row["task_id"]
        ):
            raise ModelConsumptionAuthorityError("adapter capture/input binding differs")
    expected_inherited_fds = validate_inherited_fd_binding(
        input_row["inherited_fds"],
        model_capture=model,
        adapter_capture=adapter_row,
        verify_open_fds=True,
    )
    if (
        expected_inherited_fds != input_row["inherited_fds"]
        or environment_fds != input_row["inherited_fds"]
    ):
        raise ModelConsumptionAuthorityError(
            "consumption input inherited FD binding differs"
        )
    if not verify_views:
        # The exact same validation runs above because capture receipt identity
        # cannot be trusted without resolving its FD leaves.  The flag exists
        # only to make the API's production requirement explicit.
        raise ModelConsumptionAuthorityError("FD-view verification cannot be disabled")
    return input_row, model, adapter_row


def build_consumption_chain(
    *,
    task_id: str,
    model_capture_digest: str,
    model_pre_use_digest: str,
    model_post_use_digest: str,
    adapter_capture_digest: str | None,
    adapter_pre_use_digest: str | None,
    adapter_post_use_digest: str | None,
    adapter_final_digest: str | None,
    native_inference_receipt_digest: str,
    consumption_input_digest: str | None = None,
) -> dict[str, Any]:
    task_id = _task_id(task_id)
    for label, value in (
        ("model capture", model_capture_digest),
        ("model pre-use", model_pre_use_digest),
        ("model post-use", model_post_use_digest),
        ("native inference receipt", native_inference_receipt_digest),
    ):
        _sha(value, label=label)
    if consumption_input_digest is None:
        # Unit-fixture compatibility is explicit and cannot be mistaken for a
        # production pre-use receipt because the sentinel is digest-bound.
        consumption_input_digest = object_sha256(
            {"task_id": task_id, "injected_fixture_without_input_receipt": True}
        )
    _sha(consumption_input_digest, label="consumption input")
    optional = (
        adapter_capture_digest,
        adapter_pre_use_digest,
        adapter_post_use_digest,
        adapter_final_digest,
    )
    if any(value is None for value in optional) and not all(
        value is None for value in optional
    ):
        raise ModelConsumptionAuthorityError("adapter consumption closure is mixed")
    for value in optional:
        if value is not None:
            _sha(value, label="adapter authority digest")
    row: dict[str, Any] = {
        "schema_version": CONSUMPTION_CHAIN_SCHEMA,
        "task_id": task_id,
        "consumption_input_digest": consumption_input_digest,
        "model_capture_digest": model_capture_digest,
        "model_pre_use_digest": model_pre_use_digest,
        "model_post_use_digest": model_post_use_digest,
        "adapter_capture_digest": adapter_capture_digest,
        "adapter_pre_use_digest": adapter_pre_use_digest,
        "adapter_post_use_digest": adapter_post_use_digest,
        "adapter_final_digest": adapter_final_digest,
        "native_inference_receipt_digest": native_inference_receipt_digest,
        "post_use_closed_before_publication": True,
        "training_loss_read_or_used": False,
    }
    row["consumption_digest"] = object_sha256(row)
    return row


def validate_consumption_chain(value: Any) -> dict[str, Any]:
    fields = {
        "schema_version",
        "task_id",
        "consumption_input_digest",
        "model_capture_digest",
        "model_pre_use_digest",
        "model_post_use_digest",
        "adapter_capture_digest",
        "adapter_pre_use_digest",
        "adapter_post_use_digest",
        "adapter_final_digest",
        "native_inference_receipt_digest",
        "post_use_closed_before_publication",
        "training_loss_read_or_used",
        "consumption_digest",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ModelConsumptionAuthorityError("consumption chain field closure differs")
    row = dict(value)
    expected = build_consumption_chain(
        task_id=row["task_id"],
        model_capture_digest=row["model_capture_digest"],
        model_pre_use_digest=row["model_pre_use_digest"],
        model_post_use_digest=row["model_post_use_digest"],
        adapter_capture_digest=row["adapter_capture_digest"],
        adapter_pre_use_digest=row["adapter_pre_use_digest"],
        adapter_post_use_digest=row["adapter_post_use_digest"],
        adapter_final_digest=row["adapter_final_digest"],
        native_inference_receipt_digest=row["native_inference_receipt_digest"],
        consumption_input_digest=row["consumption_input_digest"],
    )
    if row != expected:
        raise ModelConsumptionAuthorityError("consumption chain digest differs")
    return row


def build_publication_gate(
    *,
    consumption_chain: Mapping[str, Any],
    staging_path: str | Path,
    staging_sha256: str,
    staging_size: int,
) -> dict[str, Any]:
    chain = validate_consumption_chain(consumption_chain)
    staging = Path(staging_path)
    if not staging.is_absolute():
        raise ModelConsumptionAuthorityError("staging path is not absolute")
    _sha(staging_sha256, label="staging SHA")
    if type(staging_size) is not int or staging_size <= 0:
        raise ModelConsumptionAuthorityError("staging size differs")
    _canonical_absolute(staging, label="staging output", directory=False)
    descriptor = os.open(staging, _file_flags())
    try:
        before = _identity(os.fstat(descriptor))
        first_sha, first_size = _hash_fd(descriptor)
        middle = _identity(os.fstat(descriptor))
        second_sha, second_size = _hash_fd(descriptor)
        after = _identity(os.fstat(descriptor))
        named = _named_identity(staging, label="staging output")
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISREG(before["mode"])
        or before["nlink"] != 1
        or before != middle
        or before != after
        or before != named
        or first_sha != staging_sha256
        or second_sha != staging_sha256
        or first_size != staging_size
        or second_size != staging_size
    ):
        raise ModelConsumptionAuthorityError("staging output stable identity differs")
    value: dict[str, Any] = {
        "schema_version": PUBLICATION_GATE_SCHEMA,
        "task_id": chain["task_id"],
        "consumption_digest": chain["consumption_digest"],
        "staging_path": str(staging),
        "staging_sha256": staging_sha256,
        "staging_size": staging_size,
        "model_post_use_verified": True,
        "adapter_post_use_verified_or_base_control": True,
        "adapter_fds_closed_or_base_control": True,
        "publication_authorized": True,
        "publication_has_occurred": False,
    }
    value["publication_gate_digest"] = object_sha256(value)
    return value


def propagate_consumption_digest(
    *,
    input_digest: str,
    native_digest: str,
    process_digest: str,
    output_digest: str,
    result_digest: str,
    shard_digest: str,
    aggregate_digest: str,
    consumption_digest: str,
) -> dict[str, Any]:
    """Build the closed digest projection required at every receipt layer."""

    for label, digest in (
        ("input", input_digest),
        ("native", native_digest),
        ("process", process_digest),
        ("output", output_digest),
        ("result", result_digest),
        ("shard", shard_digest),
        ("aggregate", aggregate_digest),
        ("consumption", consumption_digest),
    ):
        _sha(digest, label=f"{label} digest")
    value = {
        "input": {"receipt_digest": input_digest, "consumption_digest": consumption_digest},
        "native": {"receipt_digest": native_digest, "consumption_digest": consumption_digest},
        "process": {"receipt_digest": process_digest, "consumption_digest": consumption_digest},
        "output": {"receipt_digest": output_digest, "consumption_digest": consumption_digest},
        "result": {"receipt_digest": result_digest, "consumption_digest": consumption_digest},
        "shard": {"receipt_digest": shard_digest, "consumption_digest": consumption_digest},
        "aggregate": {"receipt_digest": aggregate_digest, "consumption_digest": consumption_digest},
    }
    value["propagation_digest"] = object_sha256(value)
    return value


def validate_consumption_propagation(value: Any) -> dict[str, Any]:
    layers = ("input", "native", "process", "output", "result", "shard", "aggregate")
    if not isinstance(value, Mapping) or set(value) != {*layers, "propagation_digest"}:
        raise ModelConsumptionAuthorityError("consumption propagation closure differs")
    row = dict(value)
    projected: dict[str, Any] = {}
    shared: str | None = None
    for layer in layers:
        item = row[layer]
        if not isinstance(item, Mapping) or set(item) != {
            "receipt_digest", "consumption_digest"
        }:
            raise ModelConsumptionAuthorityError(
                f"consumption propagation {layer} closure differs"
            )
        receipt_digest = _sha(item["receipt_digest"], label=f"{layer} receipt")
        consumption_digest = _sha(
            item["consumption_digest"], label=f"{layer} consumption"
        )
        if shared is None:
            shared = consumption_digest
        elif consumption_digest != shared:
            raise ModelConsumptionAuthorityError("mixed consumption digest propagation")
        projected[layer] = {
            "receipt_digest": receipt_digest,
            "consumption_digest": consumption_digest,
        }
    expected = {**projected, "propagation_digest": object_sha256(projected)}
    if row != expected:
        raise ModelConsumptionAuthorityError("consumption propagation digest differs")
    return row


__all__ = [
    "ADAPTER_RELATIVE_FILES",
    "AdapterAuthority",
    "CONSUMPTION_CHAIN_SCHEMA",
    "CONSUMPTION_INPUT_SCHEMA",
    "INHERITED_FD_BINDING_ENV",
    "INHERITED_FD_BINDING_SCHEMA",
    "MODEL_DIRECTORY_COUNT",
    "MODEL_FILE_COUNT",
    "MODEL_MANIFEST_SHA256",
    "MODEL_RELATIVE_FILES",
    "ModelAuthority",
    "ModelConsumptionAuthorityError",
    "build_consumption_chain",
    "build_consumption_input",
    "build_inherited_fd_binding",
    "build_publication_gate",
    "canonical_json_bytes",
    "object_sha256",
    "parse_exact23_manifest",
    "propagate_consumption_digest",
    "validate_consumption_propagation",
    "validate_consumption_chain",
    "validate_consumption_input",
    "validate_model_capture_receipt",
    "validate_adapter_capture_receipt",
    "load_consumption_input",
    "load_inherited_fd_environment",
    "inherited_fd_environment_value",
    "inherited_fd_numbers",
    "inherited_fd_row",
    "inherited_proc_root",
    "inherited_task_member_path",
    "seal_inherited_fds",
    "stable_inherited_task_file",
    "stable_inherited_view_file",
    "task_publication_root_binding",
    "validate_task_publication_root",
    "validate_inherited_fd_binding",
]
