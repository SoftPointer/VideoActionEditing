#!/usr/bin/env python3
"""Final held-FD publisher for the preservation-v2 training authority.

All expensive byte/tree validation happens while normal signals are active.
The final critical section only replays held descriptors, publishes one
create-only marker, seals the experiment root, and fsyncs the directory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import signal
import stat
import sys
from typing import Any, Mapping, Sequence


ARMS = (
    "v2_onset_all",
    "v2_noop020_all",
    "v2_func010_all",
    "v2_func025_all",
    "v2_func050_all",
    "v2_onset_cross_qo",
    "v2_func010_cross_qo",
    "v2_func025_cross_qo",
)
CHECKPOINT_STEPS = (0, 5, 10, 20)
CACHE_BASENAME = "teacher-cache-preservation-v2-seed20260818-row4-sigma5.pt"
MARKER_BASENAME = "TRAINING_COMPLETE.json"


class CompletionPublicationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CompletionPublicationError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require_sha(value: Any, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} is not a SHA-256",
    )
    return value


def _unique_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        require(key not in value, f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _reject_constant(token: str) -> Any:
    raise CompletionPublicationError(f"non-finite JSON constant: {token}")


def strict_json(raw: bytes, *, label: str, canonical_newline: bool = False) -> Any:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, CompletionPublicationError) as error:
        raise CompletionPublicationError(f"{label} is not strict JSON") from error
    if canonical_newline:
        require(canonical(value) + b"\n" == raw, f"{label} is not canonical JSON")
    return value


def identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_uid,
        value.st_gid,
        stat.S_IMODE(value.st_mode),
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def read_all(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            return b"".join(chunks)
        chunks.append(block)


@dataclass
class HeldFile:
    descriptor: int
    expected_identity: tuple[int, ...]
    expected_sha256: str
    absolute_path: Path | None
    parent_descriptor: int | None
    basename: str | None
    label: str


@dataclass
class HeldDirectory:
    descriptor: int
    expected_identity: tuple[int, ...]
    expected_names: tuple[str, ...]
    absolute_path: Path | None
    parent_descriptor: int | None
    basename: str | None
    label: str


def stable_open_absolute(
    path: Path,
    *,
    expected_sha256: str | None,
    expected_mode: int | None,
    label: str,
) -> tuple[HeldFile, bytes]:
    if expected_sha256 is not None:
        require_sha(expected_sha256, f"{label} SHA-256")
    require(path.is_absolute() and Path(os.path.realpath(path)) == path, f"{label} path differs")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        require(
            stat.S_ISREG(before.st_mode)
            and before.st_uid == os.getuid()
            and before.st_nlink == 1
            and (expected_mode is None or stat.S_IMODE(before.st_mode) == expected_mode),
            f"{label} physical topology differs",
        )
        first = read_all(descriptor)
        middle = os.fstat(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        second = read_all(descriptor)
        after = os.fstat(descriptor)
        named = os.lstat(path)
        expected_identity = identity(before)
        require(
            expected_identity
            == identity(middle)
            == identity(after)
            == identity(named),
            f"{label} changed during stable capture",
        )
        require(first == second and len(first) == before.st_size, f"{label} bytes changed")
        observed_sha = sha256(first)
        require(
            expected_sha256 is None or observed_sha == expected_sha256,
            f"{label} SHA-256 differs",
        )
        os.lseek(descriptor, 0, os.SEEK_SET)
        return (
            HeldFile(
                descriptor=descriptor,
                expected_identity=expected_identity,
                expected_sha256=observed_sha,
                absolute_path=path,
                parent_descriptor=None,
                basename=None,
                label=label,
            ),
            first,
        )
    except BaseException:
        os.close(descriptor)
        raise


def safe_relative(value: str, *, label: str) -> PurePosixPath:
    relative = PurePosixPath(value)
    require(
        value == str(relative)
        and not relative.is_absolute()
        and relative.parts
        and all(part not in ("", ".", "..") for part in relative.parts),
        f"{label} path differs",
    )
    return relative


def release_rows(manifest: Any, *, content_revision: str) -> list[dict[str, Any]]:
    require(isinstance(manifest, dict), "release manifest shape differs")
    unsigned = dict(manifest)
    declared = unsigned.pop("manifest_digest", None)
    require(declared == sha256(canonical(unsigned)), "release manifest digest differs")
    require(manifest.get("content_revision") == content_revision, "release revision differs")
    member_root = manifest.get("member_root")
    safe_relative(member_root, label="release member root")
    rows = manifest.get("files")
    require(isinstance(rows, list) and rows, "release manifest rows differ")
    require(
        type(manifest.get("file_count")) is int
        and manifest["file_count"] == len(rows),
        "release manifest file count differs",
    )
    observed: set[str] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        require(
            isinstance(row, dict)
            and set(row) == {"path", "mode", "size", "sha256"},
            "release manifest row fields differ",
        )
        relative = safe_relative(row["path"], label="release member")
        require(str(relative) not in observed, "duplicate release member")
        observed.add(str(relative))
        require(row["mode"] in (0o444, 0o555), "release member mode differs")
        require(type(row["size"]) is int and row["size"] > 0, "release member size differs")
        require_sha(row["sha256"], "release member")
        result.append(dict(row))
    require([row["path"] for row in result] == sorted(observed), "release member order differs")
    require(
        hashlib.sha1(canonical(result)).hexdigest()
        == manifest["content_revision"]
        == content_revision,
        "release content revision differs from exact member rows",
    )
    return result


def training_rows(value: Any) -> list[dict[str, Any]]:
    require(isinstance(value, dict) and value.get("training_audit_go") is True, "training audit gate differs")
    require(
        value.get("arm_count") == len(ARMS)
        and value.get("checkpoint_count") == len(ARMS) * len(CHECKPOINT_STEPS)
        and value.get("checkpoint_steps") == list(CHECKPOINT_STEPS)
        and value.get("decoded_evaluation_complete") is False
        and value.get("scientific_promotion_authorized") is False,
        "training audit authority differs",
    )
    rows = value.get("receipt_rows")
    required = {
        "arm",
        "step",
        "receipt_sha256",
        "adapter_sha256",
        "adapter_config_sha256",
        "optimizer_sha256",
        "loss",
        "preclip_gradient_norm",
    }
    require(isinstance(rows, list) and len(rows) == 32, "training audit row count differs")
    expected_keys = {(arm, step) for arm in ARMS for step in CHECKPOINT_STEPS}
    observed: set[tuple[str, int]] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        require(isinstance(row, dict) and set(row) == required, "training audit row fields differ")
        key = (row["arm"], row["step"])
        require(key in expected_keys and key not in observed, "training audit row key differs")
        observed.add(key)
        for field in (
            "receipt_sha256",
            "adapter_sha256",
            "adapter_config_sha256",
            "optimizer_sha256",
        ):
            require_sha(row[field], f"training audit {field}")
        for field in ("loss", "preclip_gradient_norm"):
            require(
                not isinstance(row[field], bool)
                and isinstance(row[field], (int, float))
                and math.isfinite(float(row[field])),
                f"training audit {field} differs",
            )
        result.append(dict(row))
    require(observed == expected_keys, "training audit row closure differs")
    return result


def expected_tree_files(
    *, manifest: Mapping[str, Any], manifest_rows: Sequence[Mapping[str, Any]],
    training: Mapping[str, Any], training_receipt_rows: Sequence[Mapping[str, Any]],
    cache_sha256: str, cache_receipt_sha256: str, cache_audit_sha256: str,
    training_audit_sha256: str, materialization_sha256: str,
) -> dict[str, tuple[int, int | None, str | None]]:
    expected: dict[str, tuple[int, int | None, str | None]] = {
        CACHE_BASENAME: (0o444, None, cache_sha256),
        CACHE_BASENAME + ".receipt.json": (0o444, None, cache_receipt_sha256),
        "logs/cache-audit.json": (0o444, None, cache_audit_sha256),
        "logs/training-audit.json": (0o444, None, training_audit_sha256),
        "logs/materialization.json": (0o444, None, materialization_sha256),
        "logs/cache-full.log": (0o444, None, None),
    }
    for arm in ARMS:
        expected[f"logs/train-{arm}.log"] = (0o444, None, None)
    member_root = manifest["member_root"]
    for row in manifest_rows:
        relative = str(PurePosixPath("materialized") / member_root / row["path"])
        expected[relative] = (
            int(row["mode"]),
            int(row["size"]),
            str(row["sha256"]),
        )
    for row in training_receipt_rows:
        prefix = PurePosixPath("runs") / row["arm"] / f"checkpoint-{row['step']:08d}"
        expected[str(prefix / "receipt.json")] = (0o444, None, row["receipt_sha256"])
        expected[str(prefix / "optimizer.pt")] = (0o444, None, row["optimizer_sha256"])
        expected[str(prefix / "adapter" / "adapter_model.safetensors")] = (
            0o444,
            None,
            row["adapter_sha256"],
        )
        expected[str(prefix / "adapter" / "adapter_config.json")] = (
            0o444,
            None,
            row["adapter_config_sha256"],
        )
    require(len(expected) == 2 + 4 + len(ARMS) + len(manifest_rows) + 4 * 32, "tree file count differs")
    return expected


def expected_tree_directories(files: Mapping[str, Any]) -> dict[str, set[str]]:
    children: dict[str, set[str]] = {"": set()}
    for raw_relative in files:
        relative = safe_relative(raw_relative, label="tree file")
        parent_parts: tuple[str, ...] = ()
        for part in relative.parts[:-1]:
            parent = "/".join(parent_parts)
            children.setdefault(parent, set()).add(part)
            parent_parts = (*parent_parts, part)
            children.setdefault("/".join(parent_parts), set())
        children["/".join(parent_parts)].add(relative.name)
    return children


def capture_experiment_tree(
    root: Path,
    expected_files: Mapping[str, tuple[int, int | None, str | None]],
) -> tuple[list[HeldDirectory], list[HeldFile], list[dict[str, Any]]]:
    require(root.is_absolute() and Path(os.path.realpath(root)) == root, "experiment root path differs")
    expected_children = expected_tree_directories(expected_files)
    directories: list[HeldDirectory] = []
    files: list[HeldFile] = []
    digest_rows: list[dict[str, Any]] = []
    root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        root_stat = os.fstat(root_fd)
        root_identity = identity(root_stat)
        require(
            stat.S_ISDIR(root_stat.st_mode)
            and root_stat.st_uid == os.getuid()
            and stat.S_IMODE(root_stat.st_mode) == 0o700
            and identity(os.lstat(root)) == root_identity,
            "experiment root topology differs",
        )
        directories.append(
            HeldDirectory(
                descriptor=root_fd,
                expected_identity=root_identity,
                expected_names=tuple(sorted(expected_children[""])),
                absolute_path=root,
                parent_descriptor=None,
                basename=None,
                label="experiment root",
            )
        )

        def descend(parent_fd: int, relative: str) -> None:
            for name in sorted(expected_children[relative]):
                child_relative = f"{relative}/{name}" if relative else name
                if child_relative in expected_children:
                    descriptor = os.open(
                        name,
                        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=parent_fd,
                    )
                    details = os.fstat(descriptor)
                    named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                    expected_identity = identity(details)
                    try:
                        require(
                            stat.S_ISDIR(details.st_mode)
                            and details.st_uid == os.getuid()
                            and stat.S_IMODE(details.st_mode) == 0o555
                            and expected_identity == identity(named),
                            f"tree directory differs: {child_relative}",
                        )
                        directories.append(
                            HeldDirectory(
                                descriptor=descriptor,
                                expected_identity=expected_identity,
                                expected_names=tuple(sorted(expected_children[child_relative])),
                                absolute_path=None,
                                parent_descriptor=parent_fd,
                                basename=name,
                                label=child_relative,
                            )
                        )
                        descend(descriptor, child_relative)
                    except BaseException:
                        os.close(descriptor)
                        raise
                    continue
                mode, expected_size, expected_sha = expected_files[child_relative]
                descriptor = os.open(
                    name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent_fd,
                )
                try:
                    before = os.fstat(descriptor)
                    require(
                        stat.S_ISREG(before.st_mode)
                        and before.st_uid == os.getuid()
                        and before.st_nlink == 1
                        and stat.S_IMODE(before.st_mode) == mode,
                        f"tree file topology differs: {child_relative}",
                    )
                    require(
                        expected_size is None or before.st_size == expected_size,
                        f"tree file size differs: {child_relative}",
                    )
                    first = read_all(descriptor)
                    middle = os.fstat(descriptor)
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    second = read_all(descriptor)
                    after = os.fstat(descriptor)
                    named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                    expected_identity = identity(before)
                    observed_sha = sha256(first)
                    require(
                        first == second
                        and expected_identity == identity(middle) == identity(after) == identity(named),
                        f"tree file changed during capture: {child_relative}",
                    )
                    require(expected_sha is None or observed_sha == expected_sha, f"tree file SHA differs: {child_relative}")
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    files.append(
                        HeldFile(
                            descriptor=descriptor,
                            expected_identity=expected_identity,
                            expected_sha256=observed_sha,
                            absolute_path=None,
                            parent_descriptor=parent_fd,
                            basename=name,
                            label=child_relative,
                        )
                    )
                    digest_rows.append(
                        {
                            "path": child_relative,
                            "mode": mode,
                            "size": before.st_size,
                            "sha256": observed_sha,
                        }
                    )
                except BaseException:
                    os.close(descriptor)
                    raise

        descend(root_fd, "")
        for directory in directories:
            require(
                tuple(sorted(os.listdir(directory.descriptor))) == directory.expected_names,
                f"tree directory entry closure differs: {directory.label}",
            )
        return directories, files, sorted(digest_rows, key=lambda row: row["path"])
    except BaseException:
        for item in reversed(files):
            os.close(item.descriptor)
        for item in reversed(directories):
            try:
                os.close(item.descriptor)
            except OSError:
                pass
        raise


def revalidate_file(item: HeldFile) -> None:
    current = os.fstat(item.descriptor)
    require(identity(current) == item.expected_identity, f"held file changed: {item.label}")
    named = (
        os.lstat(item.absolute_path)
        if item.absolute_path is not None
        else os.stat(item.basename, dir_fd=item.parent_descriptor, follow_symlinks=False)
    )
    require(identity(named) == item.expected_identity, f"held file pathname changed: {item.label}")


def revalidate_directory(item: HeldDirectory, *, root_final: bool = False) -> None:
    current = os.fstat(item.descriptor)
    if root_final:
        expected = list(item.expected_identity)
        expected[4] = 0o555
        require(identity(current) == tuple(expected), f"held directory changed: {item.label}")
    else:
        require(identity(current) == item.expected_identity, f"held directory changed: {item.label}")
    named = (
        os.lstat(item.absolute_path)
        if item.absolute_path is not None
        else os.stat(item.basename, dir_fd=item.parent_descriptor, follow_symlinks=False)
    )
    require(identity(named) == identity(current), f"held directory pathname changed: {item.label}")
    expected_names = item.expected_names
    if root_final:
        expected_names = tuple(sorted((*expected_names, MARKER_BASENAME)))
    require(tuple(sorted(os.listdir(item.descriptor))) == expected_names, f"held directory entries changed: {item.label}")


def completion_value(args: argparse.Namespace, *, tree_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tree_digest = sha256(canonical(list(tree_rows)))
    value: dict[str, Any] = {
        "schema_version": "bernini-action-preservation-v2-training-complete-v3",
        "seed": 20260818,
        "cache_sha256": args.cache_sha256,
        "source_archive_sha256": args.source_archive_sha256,
        "source_revision": args.source_revision,
        "source_data_manifest_sha256": args.source_data_manifest_sha256,
        "source_data_manifest_digest": args.source_data_manifest_digest,
        "release_manifest_sha256": args.release_manifest_sha256,
        "controller_sha256": args.controller_sha256,
        "deployment_envelope_sha256": args.deployment_envelope_sha256,
        "cache_audit_sha256": args.cache_audit_sha256,
        "training_audit_sha256": args.training_audit_sha256,
        "cache_receipt_sha256": args.cache_receipt_sha256,
        "retained_tree_digest": tree_digest,
        "retained_tree_file_count": len(tree_rows),
        "retained_tree_stable_double_read_before_commit": True,
        "retained_tree_held_fd_identity_replay": True,
        "optimizer_updates_per_arm": 20,
        "arm_count": 8,
        "decoded_evaluation_complete": False,
        "scientific_promotion_authorized": False,
        "parent_allocations_cancelled": False,
        "automatic_retry": False,
    }
    value["completion_digest"] = sha256(canonical(value))
    return value


def publish(args: argparse.Namespace) -> str:
    hash_fields = (
        "cache_sha256",
        "cache_receipt_sha256",
        "cache_audit_sha256",
        "training_audit_sha256",
        "source_archive_sha256",
        "release_manifest_sha256",
        "controller_sha256",
        "deployment_envelope_sha256",
        "source_data_manifest_sha256",
        "source_data_manifest_digest",
    )
    for field in hash_fields:
        require_sha(getattr(args, field), field)
    require(
        isinstance(args.source_revision, str)
        and len(args.source_revision) == 40
        and all(character in "0123456789abcdef" for character in args.source_revision),
        "source revision differs",
    )
    experiment_root = Path(args.experiment_root)
    external: list[HeldFile] = []
    directories: list[HeldDirectory] = []
    files: list[HeldFile] = []
    try:
        release_manifest_file, manifest_raw = stable_open_absolute(
            Path(args.release_manifest),
            expected_sha256=args.release_manifest_sha256,
            expected_mode=0o444,
            label="release manifest",
        )
        external.append(release_manifest_file)
        manifest = strict_json(manifest_raw, label="release manifest", canonical_newline=True)
        manifest_rows = release_rows(manifest, content_revision=args.source_revision)
        training_audit_file, training_raw = stable_open_absolute(
            experiment_root / "logs" / "training-audit.json",
            expected_sha256=args.training_audit_sha256,
            expected_mode=0o444,
            label="training audit",
        )
        external.append(training_audit_file)
        training = strict_json(training_raw, label="training audit")
        receipt_rows = training_rows(training)
        cache_audit_file, cache_audit_raw = stable_open_absolute(
            experiment_root / "logs" / "cache-audit.json",
            expected_sha256=args.cache_audit_sha256,
            expected_mode=0o444,
            label="cache audit",
        )
        external.append(cache_audit_file)
        cache_audit = strict_json(cache_audit_raw, label="cache audit")
        require(
            isinstance(cache_audit, dict)
            and cache_audit.get("cache_audit_go") is True
            and cache_audit.get("cache_sha256") == args.cache_sha256
            and cache_audit.get("cache_receipt_sha256") == args.cache_receipt_sha256,
            "cache audit authority differs",
        )
        for path, digest, mode, label in (
            (Path(args.source_archive), args.source_archive_sha256, 0o444, "source archive"),
            (Path(args.controller), args.controller_sha256, 0o555, "controller"),
            (Path(args.deployment_envelope), args.deployment_envelope_sha256, 0o444, "deployment envelope"),
        ):
            held, _ = stable_open_absolute(
                path,
                expected_sha256=digest,
                expected_mode=mode,
                label=label,
            )
            external.append(held)
        source_manifest_file, source_manifest_raw = stable_open_absolute(
            Path(args.source_data_manifest),
            expected_sha256=args.source_data_manifest_sha256,
            expected_mode=None,
            label="source data manifest",
        )
        external.append(source_manifest_file)
        source_manifest_value = strict_json(
            source_manifest_raw, label="source data manifest"
        )
        require(
            isinstance(source_manifest_value, dict),
            "source data manifest shape differs",
        )
        source_manifest_unsigned = dict(source_manifest_value)
        source_manifest_declared = source_manifest_unsigned.pop(
            "manifest_digest", None
        )
        require(
            source_manifest_declared == args.source_data_manifest_digest
            and sha256(canonical(source_manifest_unsigned))
            == args.source_data_manifest_digest,
            "source data manifest digest differs",
        )
        materialization_file, materialization_raw = stable_open_absolute(
            experiment_root / "logs" / "materialization.json",
            expected_sha256=None,
            expected_mode=0o444,
            label="materialization receipt",
        )
        external.append(materialization_file)
        materialization = strict_json(
            materialization_raw,
            label="materialization receipt",
            canonical_newline=True,
        )
        require(
            isinstance(materialization, dict)
            and set(materialization)
            == {
                "release_root",
                "method_root",
                "archive_sha256",
                "manifest_sha256",
                "content_revision",
                "file_count",
                "exact_tree_verified",
                "directories_sealed_mode",
                "receipt_digest",
            },
            "materialization receipt fields differ",
        )
        materialization_unsigned = dict(materialization)
        materialization_digest = materialization_unsigned.pop(
            "receipt_digest", None
        )
        expected_materialized = experiment_root / "materialized"
        require(
            materialization_digest == sha256(canonical(materialization_unsigned))
            and materialization.get("release_root") == str(expected_materialized)
            and materialization.get("method_root")
            == str(expected_materialized / manifest["member_root"])
            and materialization.get("archive_sha256")
            == args.source_archive_sha256
            and materialization.get("manifest_sha256")
            == args.release_manifest_sha256
            and materialization.get("content_revision") == args.source_revision
            and materialization.get("file_count") == len(manifest_rows)
            and materialization.get("exact_tree_verified") is True
            and materialization.get("directories_sealed_mode") == "0555",
            "materialization receipt authority differs",
        )
        materialization_sha256 = sha256(materialization_raw)
        expected_files = expected_tree_files(
            manifest=manifest,
            manifest_rows=manifest_rows,
            training=training,
            training_receipt_rows=receipt_rows,
            cache_sha256=args.cache_sha256,
            cache_receipt_sha256=args.cache_receipt_sha256,
            cache_audit_sha256=args.cache_audit_sha256,
            training_audit_sha256=args.training_audit_sha256,
            materialization_sha256=materialization_sha256,
        )
        directories, files, tree_rows = capture_experiment_tree(
            experiment_root, expected_files
        )
        value = completion_value(args, tree_rows=tree_rows)
        raw = canonical(value) + b"\n"
        completion_sha = sha256(raw)
        root = directories[0]
        rollback_inode: tuple[int, int] | None = None
        marker_identity: tuple[int, ...] | None = None
        pending_name = (
            f".{MARKER_BASENAME}.pending-{os.getpid()}-{os.urandom(16).hex()}"
        )
        committed = False

        def timeout(_signum: int, _frame: Any) -> None:
            raise TimeoutError("completion publication timed out")

        try:
            for item in external:
                revalidate_file(item)
            for item in files:
                revalidate_file(item)
            for item in directories:
                revalidate_directory(item)
            # Prepare and fsync an unambiguous inode while ordinary signals
            # are still active.  The canonical basename is published later
            # with linkat, so even an interruption immediately after O_EXCL
            # cannot leave a false completion authority.
            pending_fd = os.open(
                pending_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o444,
                dir_fd=root.descriptor,
            )
            try:
                opened = os.fstat(pending_fd)
                rollback_inode = (opened.st_dev, opened.st_ino)
                require(
                    stat.S_ISREG(opened.st_mode)
                    and opened.st_uid == os.getuid()
                    and opened.st_nlink == 1,
                    "new pending completion inode topology differs",
                )
                view = memoryview(raw)
                while view:
                    count = os.write(pending_fd, view)
                    require(count > 0, "short pending completion write")
                    view = view[count:]
                os.fchmod(pending_fd, 0o444)
                os.fsync(pending_fd)
                sealed = os.fstat(pending_fd)
                require(
                    stat.S_ISREG(sealed.st_mode)
                    and sealed.st_uid == os.getuid()
                    and sealed.st_nlink == 1
                    and stat.S_IMODE(sealed.st_mode) == 0o444
                    and sealed.st_size == len(raw),
                    "sealed pending completion inode topology differs",
                )
            finally:
                os.close(pending_fd)
            root.expected_identity = identity(os.fstat(root.descriptor))
            require(
                identity(os.lstat(experiment_root)) == root.expected_identity
                and tuple(sorted(os.listdir(root.descriptor)))
                == tuple(sorted((*root.expected_names, pending_name))),
                "pending completion root closure differs",
            )
            signal.signal(signal.SIGALRM, timeout)
            signal.alarm(30)
            signal.pthread_sigmask(
                signal.SIG_BLOCK,
                {signal.SIGINT, signal.SIGTERM, signal.SIGHUP},
            )
            for item in external:
                revalidate_file(item)
            for item in files:
                revalidate_file(item)
            for item in directories[1:]:
                revalidate_directory(item)
            require(
                identity(os.fstat(root.descriptor)) == root.expected_identity
                and identity(os.lstat(experiment_root)) == root.expected_identity
                and tuple(sorted(os.listdir(root.descriptor)))
                == tuple(sorted((*root.expected_names, pending_name))),
                "held pending root changed before publish",
            )
            os.link(
                pending_name,
                MARKER_BASENAME,
                src_dir_fd=root.descriptor,
                dst_dir_fd=root.descriptor,
                follow_symlinks=False,
            )
            published = os.stat(
                MARKER_BASENAME,
                dir_fd=root.descriptor,
                follow_symlinks=False,
            )
            require(
                stat.S_ISREG(published.st_mode)
                and (published.st_dev, published.st_ino) == rollback_inode
                and published.st_nlink == 2,
                "linked completion inode topology differs",
            )
            os.unlink(pending_name, dir_fd=root.descriptor)
            os.fsync(root.descriptor)
            published = os.stat(
                MARKER_BASENAME,
                dir_fd=root.descriptor,
                follow_symlinks=False,
            )
            require(
                stat.S_ISREG(published.st_mode)
                and (published.st_dev, published.st_ino) == rollback_inode
                and published.st_uid == os.getuid()
                and published.st_nlink == 1
                and stat.S_IMODE(published.st_mode) == 0o444
                and published.st_size == len(raw),
                "published completion inode topology differs",
            )
            marker_identity = identity(published)
            for item in external:
                revalidate_file(item)
            for item in files:
                revalidate_file(item)
            for item in directories[1:]:
                revalidate_directory(item)
            require(
                tuple(sorted(os.listdir(root.descriptor)))
                == tuple(sorted((*root.expected_names, MARKER_BASENAME))),
                "root changed during completion write",
            )
            replay = os.open(
                MARKER_BASENAME,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=root.descriptor,
            )
            try:
                before = os.fstat(replay)
                first = read_all(replay)
                middle = os.fstat(replay)
                os.lseek(replay, 0, os.SEEK_SET)
                second = read_all(replay)
                after = os.fstat(replay)
                named = os.stat(
                    MARKER_BASENAME,
                    dir_fd=root.descriptor,
                    follow_symlinks=False,
                )
                require(
                    marker_identity
                    == identity(before)
                    == identity(middle)
                    == identity(after)
                    == identity(named),
                    "completion inode identity differs",
                )
                require(
                    first == raw
                    and second == raw
                    and sha256(first) == completion_sha,
                    "completion byte replay differs",
                )
            finally:
                os.close(replay)
            os.fchmod(root.descriptor, 0o555)
            os.fsync(root.descriptor)
            root.expected_identity = identity(os.fstat(root.descriptor))
            for item in external:
                revalidate_file(item)
            for item in files:
                revalidate_file(item)
            for item in directories[1:]:
                revalidate_directory(item)
            revalidate_directory(root, root_final=True)
            committed = True
            signal.alarm(0)
        except BaseException:
            signal.alarm(0)
            if rollback_inode is not None:
                try:
                    os.fchmod(root.descriptor, 0o700)
                    for rollback_name in (MARKER_BASENAME, pending_name):
                        try:
                            named = os.stat(
                                rollback_name,
                                dir_fd=root.descriptor,
                                follow_symlinks=False,
                            )
                        except FileNotFoundError:
                            continue
                        if stat.S_ISREG(named.st_mode) and (named.st_dev, named.st_ino) == rollback_inode:
                            os.unlink(rollback_name, dir_fd=root.descriptor)
                    os.fsync(root.descriptor)
                except BaseException:
                    pass
            raise
        require(committed, "completion transaction did not commit")
        return completion_sha
    finally:
        for item in reversed(files):
            try:
                os.close(item.descriptor)
            except OSError:
                pass
        for item in reversed(directories):
            try:
                os.close(item.descriptor)
            except OSError:
                pass
        for item in reversed(external):
            try:
                os.close(item.descriptor)
            except OSError:
                pass


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--experiment-root", required=True)
    value.add_argument("--cache-sha256", required=True)
    value.add_argument("--cache-receipt-sha256", required=True)
    value.add_argument("--cache-audit-sha256", required=True)
    value.add_argument("--training-audit-sha256", required=True)
    value.add_argument("--source-archive", required=True)
    value.add_argument("--source-archive-sha256", required=True)
    value.add_argument("--release-manifest", required=True)
    value.add_argument("--release-manifest-sha256", required=True)
    value.add_argument("--controller", required=True)
    value.add_argument("--controller-sha256", required=True)
    value.add_argument("--deployment-envelope", required=True)
    value.add_argument("--deployment-envelope-sha256", required=True)
    value.add_argument("--source-data-manifest", required=True)
    value.add_argument("--source-data-manifest-sha256", required=True)
    value.add_argument("--source-data-manifest-digest", required=True)
    value.add_argument("--source-revision", required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    completion_sha = publish(args)
    line = (
        "ALL_ACTION_PRESERVATION_V2_TRAINING_COMPLETE "
        f"completion_sha256={completion_sha} "
        "evaluation_pending=true scientific_promotion=false\n"
    ).encode("utf-8")
    try:
        os.write(1, line)
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
