#!/usr/bin/env python3
"""Executable v2 verifier for the frozen full644 matched-eval plan.

The v1 plan and all of its checkpoint, pairing, sampling, receipt, and MP4
closure remain authoritative.  V2 fixes representation mismatches exposed by
the real terminal run: source-file ``mode`` is permission bits rather than a
complete ``st_mode`` value, and the audited Bernini base has no root
``model_index.json`` so the terminal training receipt truthfully records only
the transformer and VAE config hashes.  Both compatibilities are scoped,
fail-closed, and pinned to the exact terminal cp644 manifest.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Iterator, Mapping, Sequence

import full644_exploratory_matched_eval_v1 as v1


REPORT_SCHEMA = "bernini-full644-exploratory-matched-eval-report-v2"
FFPROBE_AUTHORITY_SCHEMA = (
    "bernini-full644-exploratory-matched-ffprobe-exec-authority-v1"
)
PUBLICATION_AUTHORITY_SCHEMA = (
    "bernini-full644-exploratory-matched-publication-authority-v1"
)
TERMINAL_CP644_MANIFEST_SHA256 = (
    "7a4864a3ffa50c12af91f8d2b88610a6cd8f994aa68eef8d27b95bcc2d73d3b2"
)
TERMINAL_CP644_MANIFEST_DIGEST = (
    "7bae23da51a3c5a67adb41ee85dd026c374d2581bd3409e868e18b2f6f4dffc4"
)
TERMINAL_CP644_RECEIPT_SHA256 = (
    "3402c8c93c092bfc4490bf86790ab6429b4cbaad38358956cb0beeb5df7d4c4c"
)
TERMINAL_CP644_RECEIPT_DIGEST = (
    "aaf348a7daa6c5ca2fe721771857287125ee02eb2c9a499f45b11a2e113d15d7"
)
TERMINAL_CP644_ADAPTER_CONFIG_SHA256 = (
    "94bfaf73d714d7e77095ff68ce57e24932e0c05bde324263f5fe321660b95f62"
)
TERMINAL_CP644_ADAPTER_MODEL_SHA256 = (
    "44efdc5a0501238250b1d32ae2859abe248ffc37b152cd8db86ff84b378d6b22"
)
TERMINAL_CP644_OPTIMIZER_SHA256 = (
    "77b7b22db4da92f28f23b4ae91c7271f55ab6a92353bfc8b0bbeb30529a7af63"
)
_V1_LEGACY_CHECKPOINT_CONFIG_FIELDS = {
    "model_index.json",
    "transformer/config.json",
    "vae/config.json",
}
_TERMINAL_CHECKPOINT_CONFIG_FIELDS = {
    "transformer/config.json",
    "vae/config.json",
}
_RECEIPT_PATCH_ACTIVE = False
_CHECKPOINT_COMPAT_ACTIVE = False
_EXEC_IDENTITY_FIELDS = {
    "device",
    "inode",
    "uid",
    "gid",
    "mode",
    "nlink",
    "rdev",
    "size",
    "blocks",
    "mtime_ns",
    "ctime_ns",
}


class MatchedEvalV2Error(v1.MatchedEvalContractError):
    """A real-receipt v2 invariant differs."""


@contextmanager
def _v1_terminal_cp644_config_compatibility() -> Iterator[None]:
    """Run one v1 checkpoint replay with the exact real base config closure."""

    global _CHECKPOINT_COMPAT_ACTIVE
    missing = object()
    original = getattr(v1, "_TRAINING_CHECKPOINT_CONFIG_FIELDS", missing)
    if (
        _CHECKPOINT_COMPAT_ACTIVE
        or type(original) is not set
        or original != _V1_LEGACY_CHECKPOINT_CONFIG_FIELDS
    ):
        raise MatchedEvalV2Error("v1 checkpoint config compatibility origin differs")
    original_contents = frozenset(original)
    patched = set(_TERMINAL_CHECKPOINT_CONFIG_FIELDS)
    _CHECKPOINT_COMPAT_ACTIVE = True
    v1._TRAINING_CHECKPOINT_CONFIG_FIELDS = patched
    try:
        yield
    finally:
        current = getattr(v1, "_TRAINING_CHECKPOINT_CONFIG_FIELDS", missing)
        scoped_state_changed = not (
            current is patched
            and type(current) is set
            and current == _TERMINAL_CHECKPOINT_CONFIG_FIELDS
        )
        original_state_changed = not (
            type(original) is set and frozenset(original) == original_contents
        )
        try:
            if type(original) is set:
                original.clear()
                original.update(original_contents)
            v1._TRAINING_CHECKPOINT_CONFIG_FIELDS = original
        finally:
            _CHECKPOINT_COMPAT_ACTIVE = False
        restored = getattr(v1, "_TRAINING_CHECKPOINT_CONFIG_FIELDS", missing)
        if (
            scoped_state_changed
            or original_state_changed
            or restored is not original
            or type(restored) is not set
            or frozenset(restored) != original_contents
        ):
            raise MatchedEvalV2Error(
                "v1 checkpoint config compatibility was not restored"
            )


def _expected_terminal_checkpoint_identity(path_value: str | Path) -> dict[str, Any]:
    path = Path(path_value).expanduser()
    if (
        not path.is_absolute()
        or os.path.normpath(str(path)) != str(path)
        or path.is_symlink()
        or path.resolve(strict=True) != path
    ):
        raise MatchedEvalV2Error("terminal cp644 manifest path differs")
    return {
        "path": str(path),
        "sha256": TERMINAL_CP644_MANIFEST_SHA256,
        "manifest_digest": TERMINAL_CP644_MANIFEST_DIGEST,
        "global_step": 644,
        "receipt_digest": TERMINAL_CP644_RECEIPT_DIGEST,
        "file_count": 5,
        "adapter_config_sha256": TERMINAL_CP644_ADAPTER_CONFIG_SHA256,
        "adapter_model_sha256": TERMINAL_CP644_ADAPTER_MODEL_SHA256,
        "training_receipt_sha256": TERMINAL_CP644_RECEIPT_SHA256,
        "optimizer_sha256": TERMINAL_CP644_OPTIMIZER_SHA256,
    }


def validate_terminal_checkpoint_manifest(
    path_value: str | Path, expected_sha256: str
) -> dict[str, Any]:
    if expected_sha256 != TERMINAL_CP644_MANIFEST_SHA256:
        raise MatchedEvalV2Error("terminal cp644 manifest SHA differs")
    expected = _expected_terminal_checkpoint_identity(path_value)
    with _v1_terminal_cp644_config_compatibility():
        observed = v1.validate_terminal_checkpoint_manifest(
            path_value, expected_sha256
        )
    if observed != expected:
        raise MatchedEvalV2Error("terminal cp644 identity differs")
    return observed


def validate_plan(plan: Mapping[str, Any]) -> None:
    with _v1_terminal_cp644_config_compatibility():
        v1.validate_plan(plan)
    checkpoint = plan.get("checkpoint_manifest")
    if (
        not isinstance(checkpoint, Mapping)
        or dict(checkpoint)
        != _expected_terminal_checkpoint_identity(checkpoint.get("path", ""))
    ):
        raise MatchedEvalV2Error("plan terminal cp644 identity differs")


def load_plan(path_value: str | Path, expected_sha256: str) -> dict[str, Any]:
    with _v1_terminal_cp644_config_compatibility():
        plan = v1._load_plan(path_value, expected_sha256)
    checkpoint = plan.get("checkpoint_manifest")
    if (
        not isinstance(checkpoint, Mapping)
        or dict(checkpoint)
        != _expected_terminal_checkpoint_identity(checkpoint.get("path", ""))
    ):
        raise MatchedEvalV2Error("loaded plan terminal cp644 identity differs")
    return plan


def _directory_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_uid,
        info.st_gid,
        info.st_mode,
        info.st_rdev,
    )


def _exec_identity(info: os.stat_result) -> dict[str, int]:
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "uid": info.st_uid,
        "gid": info.st_gid,
        "mode": info.st_mode,
        "nlink": info.st_nlink,
        "rdev": info.st_rdev,
        "size": info.st_size,
        "blocks": getattr(info, "st_blocks", 0),
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }


def _pread_exact(descriptor: int, size: int) -> bytes:
    if type(size) is not int or size <= 0 or not hasattr(os, "pread"):
        raise MatchedEvalV2Error("retained ffprobe pread is unavailable")
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        block = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
        if not block:
            break
        chunks.append(block)
        offset += len(block)
    raw = b"".join(chunks)
    if len(raw) != size:
        raise MatchedEvalV2Error("retained ffprobe read is incomplete")
    return raw


def validate_retained_ffprobe_authority(
    value: Mapping[str, Any], producer: Mapping[str, Any]
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "fd",
        "source_path",
        "sha256",
        "identity",
        "authority_digest",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise MatchedEvalV2Error("retained ffprobe authority closure differs")
    row = dict(value)
    unsigned = dict(row)
    claimed = unsigned.pop("authority_digest", None)
    identity = row.get("identity")
    source = Path(row.get("source_path", ""))
    if (
        row.get("schema_version") != FFPROBE_AUTHORITY_SCHEMA
        or claimed != v1.object_sha256(unsigned)
        or type(row.get("fd")) is not int
        or row["fd"] < 3
        or not source.is_absolute()
        or os.path.normpath(str(source)) != str(source)
        or source.is_symlink()
        or source.resolve(strict=True) != source
        or row.get("source_path") != producer.get("ffprobe_path")
        or row.get("sha256") != producer.get("ffprobe_sha256")
        or not isinstance(identity, Mapping)
        or set(identity) != _EXEC_IDENTITY_FIELDS
        or any(type(item) is not int for item in identity.values())
    ):
        raise MatchedEvalV2Error("retained ffprobe authority value differs")
    try:
        before = os.fstat(row["fd"])
        raw = _pread_exact(row["fd"], before.st_size)
        middle = os.fstat(row["fd"])
        named = source.lstat()
        after = os.fstat(row["fd"])
        inheritable = os.get_inheritable(row["fd"])
    except OSError as error:
        raise MatchedEvalV2Error("retained ffprobe FD is unavailable") from error
    expected = dict(identity)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or not before.st_mode & 0o111
        or _exec_identity(before) != expected
        or _exec_identity(middle) != expected
        or _exec_identity(named) != expected
        or _exec_identity(after) != expected
        or hashlib.sha256(raw).hexdigest() != row["sha256"]
        or inheritable
    ):
        raise MatchedEvalV2Error("retained ffprobe FD replay differs")
    row["identity"] = expected
    return row


def validate_retained_publication_authority(
    value: Mapping[str, Any], task: Mapping[str, Any]
) -> dict[str, Any]:
    """Replay one receipt/output pair through the exact retained leaf FDs."""

    fields = {
        "schema_version",
        "task_id",
        "output_path",
        "output_fd",
        "output_identity",
        "output_sha256",
        "output_size",
        "receipt_path",
        "receipt_fd",
        "receipt_identity",
        "receipt_sha256",
        "receipt_size",
        "authority_digest",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise MatchedEvalV2Error("retained publication authority closure differs")
    row = dict(value)
    unsigned = dict(row)
    claimed = unsigned.pop("authority_digest", None)
    output = Path(row.get("output_path", ""))
    receipt = Path(row.get("receipt_path", ""))
    output_identity = row.get("output_identity")
    receipt_identity = row.get("receipt_identity")
    if (
        row.get("schema_version") != PUBLICATION_AUTHORITY_SCHEMA
        or claimed != v1.object_sha256(unsigned)
        or row.get("task_id") != task.get("task_id")
        or output != Path(task.get("output", {}).get("video_path", ""))
        or receipt != Path(task.get("output", {}).get("receipt_path", ""))
        or not output.is_absolute()
        or not receipt.is_absolute()
        or os.path.normpath(str(output)) != str(output)
        or os.path.normpath(str(receipt)) != str(receipt)
        or output.parent != receipt.parent
        or type(row.get("output_fd")) is not int
        or type(row.get("receipt_fd")) is not int
        or row["output_fd"] < 3
        or row["receipt_fd"] < 3
        or row["output_fd"] == row["receipt_fd"]
        or not isinstance(output_identity, Mapping)
        or not isinstance(receipt_identity, Mapping)
        or set(output_identity) != _EXEC_IDENTITY_FIELDS
        or set(receipt_identity) != _EXEC_IDENTITY_FIELDS
        or any(type(item) is not int for item in output_identity.values())
        or any(type(item) is not int for item in receipt_identity.values())
        or type(row.get("output_sha256")) is not str
        or v1.SHA256_RE.fullmatch(row["output_sha256"]) is None
        or type(row.get("receipt_sha256")) is not str
        or v1.SHA256_RE.fullmatch(row["receipt_sha256"]) is None
        or type(row.get("output_size")) is not int
        or row["output_size"] <= 0
        or type(row.get("receipt_size")) is not int
        or row["receipt_size"] <= 0
    ):
        raise MatchedEvalV2Error("retained publication authority value differs")
    try:
        output_before = os.fstat(row["output_fd"])
        receipt_before = os.fstat(row["receipt_fd"])
        output_raw = _pread_exact(row["output_fd"], output_before.st_size)
        receipt_raw = _pread_exact(row["receipt_fd"], receipt_before.st_size)
        output_middle = os.fstat(row["output_fd"])
        receipt_middle = os.fstat(row["receipt_fd"])
        output_named = output.lstat()
        receipt_named = receipt.lstat()
        output_after = os.fstat(row["output_fd"])
        receipt_after = os.fstat(row["receipt_fd"])
        output_inheritable = os.get_inheritable(row["output_fd"])
        receipt_inheritable = os.get_inheritable(row["receipt_fd"])
    except OSError as error:
        raise MatchedEvalV2Error("retained publication FD is unavailable") from error
    expected_output = dict(output_identity)
    expected_receipt = dict(receipt_identity)
    if (
        not stat.S_ISREG(output_before.st_mode)
        or output_before.st_nlink != 1
        or stat.S_IMODE(output_before.st_mode) != 0o444
        or not stat.S_ISREG(receipt_before.st_mode)
        or receipt_before.st_nlink != 1
        or stat.S_IMODE(receipt_before.st_mode) != 0o400
        or _exec_identity(output_before) != expected_output
        or _exec_identity(output_middle) != expected_output
        or _exec_identity(output_named) != expected_output
        or _exec_identity(output_after) != expected_output
        or _exec_identity(receipt_before) != expected_receipt
        or _exec_identity(receipt_middle) != expected_receipt
        or _exec_identity(receipt_named) != expected_receipt
        or _exec_identity(receipt_after) != expected_receipt
        or len(output_raw) != row["output_size"]
        or len(receipt_raw) != row["receipt_size"]
        or hashlib.sha256(output_raw).hexdigest() != row["output_sha256"]
        or hashlib.sha256(receipt_raw).hexdigest() != row["receipt_sha256"]
        or output_inheritable
        or receipt_inheritable
    ):
        raise MatchedEvalV2Error("retained publication FD replay differs")
    row["output_identity"] = expected_output
    row["receipt_identity"] = expected_receipt
    return row


def _validate_publication_root_fd(root: Path, descriptor: int) -> None:
    try:
        held = os.fstat(descriptor)
        named = root.lstat()
    except OSError as error:
        raise MatchedEvalV2Error("publication-root FD is unavailable") from error
    if (
        type(descriptor) is not int
        or descriptor < 0
        or not root.is_absolute()
        or os.path.normpath(str(root)) != str(root)
        or root.is_symlink()
        or not stat.S_ISDIR(held.st_mode)
        or _directory_identity(held) != _directory_identity(named)
        or os.get_inheritable(descriptor)
    ):
        raise MatchedEvalV2Error("publication-root FD identity differs")


@contextmanager
def _v1_output_fd_compatibility(
    logical_output: Path,
    publication_root: Path,
    publication_root_fd: int,
    producer: Mapping[str, Any],
    ffprobe_authority: Mapping[str, Any],
    publication_authority: Mapping[str, Any],
    task: Mapping[str, Any],
) -> Iterator[None]:
    """Route v1 media reads and ffprobe through retained authority FDs."""

    if logical_output.parent != publication_root:
        raise MatchedEvalV2Error("logical output root differs")
    _validate_publication_root_fd(publication_root, publication_root_fd)
    ffprobe = validate_retained_ffprobe_authority(ffprobe_authority, producer)
    ffprobe_fd = ffprobe["fd"]
    publication = validate_retained_publication_authority(
        publication_authority, task
    )
    output_fd = publication["output_fd"]
    proc_output = Path(f"/proc/self/fd/{output_fd}")
    original_stable = v1._stable_file
    original_publication = v1._publication_identity
    original_probe = v1._probe_mp4
    original_run = v1.subprocess.run
    probe_calls = [0]

    def stable_file(path_value: str | Path, **kwargs: Any) -> Any:
        requested = Path(path_value)
        if requested == logical_output:
            validated = validate_retained_publication_authority(
                publication, task
            )
            raw = _pread_exact(output_fd, validated["output_size"])
            expected_sha256 = kwargs.get("expected_sha256")
            if (
                set(kwargs) - {"expected_sha256", "return_bytes"}
                or (
                    expected_sha256 is not None
                    and expected_sha256 != validated["output_sha256"]
                )
            ):
                raise MatchedEvalV2Error("retained output stable-read differs")
            return (
                raw if kwargs.get("return_bytes", True) else None,
                validated["output_sha256"],
                validated["output_size"],
            )
        return original_stable(
            requested,
            **kwargs,
        )

    def publication_identity(path_value: Path) -> dict[str, int]:
        requested = Path(path_value)
        if requested == logical_output:
            validated = validate_retained_publication_authority(
                publication, task
            )
            return _exec_identity(os.fstat(validated["output_fd"]))
        return original_publication(requested)

    def probe_run(arguments: Sequence[str], **kwargs: Any) -> Any:
        values = list(arguments)
        expected_arguments = [
            ffprobe["source_path"],
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=codec_type,width,height,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(proc_output),
        ]
        if (
            probe_calls[0] != 0
            or values != expected_arguments
            or set(kwargs)
            != {"check", "stdout", "stderr", "timeout", "env"}
            or kwargs.get("check") is not False
            or kwargs.get("stdout") != v1.subprocess.PIPE
            or kwargs.get("stderr") != v1.subprocess.PIPE
            or kwargs.get("timeout") != 60
            or kwargs.get("env") != {"LC_ALL": "C", "LANG": "C"}
            or "pass_fds" in kwargs
            or kwargs.get("shell") not in (None, False)
        ):
            raise MatchedEvalV2Error("ffprobe retained-exec launch differs")
        probe_calls[0] += 1
        validate_retained_ffprobe_authority(ffprobe, producer)
        kwargs["close_fds"] = True
        kwargs["pass_fds"] = tuple(sorted((output_fd, ffprobe_fd)))
        kwargs["executable"] = f"/proc/self/fd/{ffprobe_fd}"
        try:
            return original_run(values, **kwargs)
        finally:
            validate_retained_ffprobe_authority(ffprobe, producer)

    def probe_mp4(path_value: Path, producer: Mapping[str, Any]) -> dict[str, Any]:
        if Path(path_value) != logical_output:
            return original_probe(path_value, producer)
        v1.subprocess.run = probe_run
        try:
            result = original_probe(proc_output, producer)
        finally:
            hook_changed = v1.subprocess.run is not probe_run
            v1.subprocess.run = original_run
            if hook_changed or v1.subprocess.run is not original_run:
                raise MatchedEvalV2Error("ffprobe subprocess origin was not restored")
        if probe_calls != [1]:
            raise MatchedEvalV2Error("ffprobe retained-root call count differs")
        return result

    v1._stable_file = stable_file
    v1._publication_identity = publication_identity
    v1._probe_mp4 = probe_mp4
    try:
        yield
    finally:
        hook_changed = (
            v1._stable_file is not stable_file
            or v1._publication_identity is not publication_identity
            or v1._probe_mp4 is not probe_mp4
            or v1.subprocess.run is not original_run
        )
        v1._stable_file = original_stable
        v1._publication_identity = original_publication
        v1._probe_mp4 = original_probe
        v1.subprocess.run = original_run
        if hook_changed:
            raise MatchedEvalV2Error("v1 output-FD origins were not restored")
    _validate_publication_root_fd(publication_root, publication_root_fd)
    validate_retained_ffprobe_authority(ffprobe, producer)
    validate_retained_publication_authority(publication, task)


def _source_stat_projection(path: Path, info: os.stat_result, sha256: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256,
        "size": int(info.st_size),
        "mode": stat.S_IMODE(info.st_mode),
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "uid": int(info.st_uid),
        "gid": int(info.st_gid),
        "nlink": int(info.st_nlink),
        "rdev": int(info.st_rdev),
        "blocks": int(getattr(info, "st_blocks", 0)),
        "mtime_ns": int(info.st_mtime_ns),
        "ctime_ns": int(info.st_ctime_ns),
    }


def validate_real_source_authority(
    task: Mapping[str, Any], receipt: Mapping[str, Any]
) -> dict[str, Any]:
    input_value = receipt.get("input")
    if not isinstance(input_value, Mapping):
        raise MatchedEvalV2Error("receipt input differs")
    authority = input_value.get("source_video_physical_authority")
    if not isinstance(authority, dict) or set(authority) != v1._SOURCE_AUTHORITY_FIELDS:
        raise MatchedEvalV2Error("source physical authority schema differs")
    permission_mode = authority.get("mode")
    if (
        type(permission_mode) is not int
        or permission_mode < 0
        or permission_mode > 0o7777
    ):
        raise MatchedEvalV2Error("source authority mode is not permission bits")
    path = Path(task.get("source_video", ""))
    if (
        not path.is_absolute()
        or os.path.normpath(str(path)) != str(path)
        or path.resolve(strict=True) != path
    ):
        raise MatchedEvalV2Error("source path is not canonical")
    try:
        before = path.lstat()
        _, observed_sha, observed_size = v1._stable_file(
            path,
            expected_sha256=task.get("source_video_sha256"),
            return_bytes=False,
        )
        after = path.lstat()
    except (OSError, v1.MatchedEvalContractError) as error:
        raise MatchedEvalV2Error(f"source stable replay failed: {error}") from error
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or v1._identity(before) != v1._identity(after)
        or observed_size != before.st_size
    ):
        raise MatchedEvalV2Error("source is not one stable regular inode")
    observed = _source_stat_projection(path, before, observed_sha)
    if authority != observed:
        raise MatchedEvalV2Error("source authority differs from stable named inode")
    authority_digest = v1.object_sha256(authority)
    consumption = receipt.get("model_consumption")
    if (
        input_value.get("source_video_physical_authority_digest")
        != authority_digest
        or not isinstance(consumption, Mapping)
        or consumption.get("source_video_physical_authority_digest")
        != authority_digest
        or input_value.get("retained_source_fd_consumed") is not True
        or input_value.get("source_video_pre_and_post_decode_rehashed") is not True
        or consumption.get("all_ranks_use_retained_source_fd") is not True
    ):
        raise MatchedEvalV2Error("retained source replay evidence differs")
    return observed


class _V1StatCompatibility:
    """Local facade used only after v2 has verified real source inodes."""

    def __init__(self, original: Any) -> None:
        self._original = original

    def S_ISREG(self, value: int) -> bool:
        if type(value) is int and 0 <= value <= 0o7777:
            return True
        return bool(self._original.S_ISREG(value))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


@contextmanager
def _v1_permission_mode_compatibility() -> Iterator[None]:
    original = v1.stat
    facade = _V1StatCompatibility(original)
    v1.stat = facade
    try:
        yield
    finally:
        hook_changed = v1.stat is not facade
        v1.stat = original
        if hook_changed or v1.stat is not original:
            raise MatchedEvalV2Error("v1 stat origin was not restored")


@contextmanager
def _v1_exact_receipt_replay(
    path_value: str | Path,
    receipt: dict[str, Any],
    receipt_sha256: str,
) -> Iterator[list[int]]:
    """Give v1 the already-stably-read receipt exactly once.

    V2 must not validate source authority from one receipt inode and then let
    v1 reopen a substituted inode.  The rest of v1 verification therefore
    consumes this exact object/SHA under a task-local, always-restored hook.
    """

    global _RECEIPT_PATCH_ACTIVE
    if _RECEIPT_PATCH_ACTIVE:
        raise MatchedEvalV2Error("receipt replay hook is already active")
    expected_path = Path(path_value)
    original = v1._load_receipt
    calls = [0]

    def exact_load(requested: str | Path) -> tuple[dict[str, Any], str]:
        if Path(requested) != expected_path or calls[0] != 0:
            raise MatchedEvalV2Error("v1 receipt replay call differs")
        calls[0] += 1
        return receipt, receipt_sha256

    _RECEIPT_PATCH_ACTIVE = True
    v1._load_receipt = exact_load
    try:
        yield calls
    finally:
        hook_changed = v1._load_receipt is not exact_load
        v1._load_receipt = original
        _RECEIPT_PATCH_ACTIVE = False
        if hook_changed or v1._load_receipt is not original:
            raise MatchedEvalV2Error("v1 receipt origin was not restored")


def verify_arm(
    task: Mapping[str, Any],
    producer: Mapping[str, Any],
    *,
    publication_root: Path | None = None,
    publication_root_fd: int | None = None,
    ffprobe_authority: Mapping[str, Any] | None = None,
    publication_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    receipt_path = Path(task["output"]["receipt_path"])
    logical_output = Path(task["output"]["video_path"])
    held = (
        publication_root is not None
        or publication_root_fd is not None
        or ffprobe_authority is not None
        or publication_authority is not None
    )
    if held and (
        publication_root is None
        or publication_root_fd is None
        or ffprobe_authority is None
        or publication_authority is None
        or receipt_path.parent != publication_root
        or logical_output.parent != publication_root
    ):
        raise MatchedEvalV2Error("held publication-root arguments differ")
    retained_publication = (
        None
        if not held
        else validate_retained_publication_authority(
            publication_authority, task
        )
    )
    try:
        if not held:
            receipt_before = receipt_path.lstat()
            receipt, receipt_sha256 = v1._load_receipt(receipt_path)
            receipt_after = receipt_path.lstat()
        else:
            retained_publication = validate_retained_publication_authority(
                retained_publication, task
            )
            receipt_fd = retained_publication["receipt_fd"]
            receipt_before = os.fstat(receipt_fd)
            receipt_raw = _pread_exact(
                receipt_fd, retained_publication["receipt_size"]
            )
            receipt = v1._json(receipt_raw, label="inference receipt")
            if (
                not isinstance(receipt, dict)
                or receipt_raw != v1.canonical_json_bytes(receipt) + b"\n"
            ):
                raise MatchedEvalV2Error(
                    "inference receipt is not canonical JSON plus LF"
                )
            v1._strict_digest(receipt, "receipt_digest", label="inference receipt")
            receipt_sha256 = retained_publication["receipt_sha256"]
            if hashlib.sha256(receipt_raw).hexdigest() != receipt_sha256:
                raise MatchedEvalV2Error("retained receipt SHA differs")
            receipt_after = os.fstat(receipt_fd)
    except OSError as error:
        raise MatchedEvalV2Error("native receipt leaf is unavailable") from error
    if (
        not stat.S_ISREG(receipt_before.st_mode)
        or receipt_before.st_nlink != 1
        or stat.S_IMODE(receipt_before.st_mode) != 0o400
        or v1._identity(receipt_before) != v1._identity(receipt_after)
    ):
        raise MatchedEvalV2Error("native receipt leaf must be stable 0400/single-link")
    validate_real_source_authority(task, receipt)
    output_context = (
        _v1_output_fd_compatibility(
            logical_output,
            publication_root,
            publication_root_fd,
            producer,
            ffprobe_authority,
            retained_publication,
            task,
        )
        if held
        else nullcontext()
    )
    with _v1_exact_receipt_replay(
        receipt_path, receipt, receipt_sha256
    ) as receipt_calls:
        with _v1_permission_mode_compatibility():
            with output_context:
                result = v1._verify_arm(task, producer)
    if receipt_calls != [1]:
        raise MatchedEvalV2Error("v1 did not consume the exact receipt once")
    return result


def verify_results(
    plan: Mapping[str, Any],
    *,
    publication_root_fd: int | None = None,
    ffprobe_authority: Mapping[str, Any] | None = None,
    publication_authorities: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    validate_plan(plan)
    if plan.get("production_ready") is not True:
        raise MatchedEvalV2Error(
            "result verification requires a production source-byte-authorized plan"
        )
    checkpoint = plan["checkpoint_manifest"]
    output_roots = {
        Path(task["output"]["video_path"]).parent for task in plan["tasks"]
    }
    if len(output_roots) != 1:
        raise MatchedEvalV2Error("publication root differs")
    publication_root = next(iter(output_roots))
    held_values = (
        publication_root_fd,
        ffprobe_authority,
        publication_authorities,
    )
    if any(value is not None for value in held_values) and any(
        value is None for value in held_values
    ):
        raise MatchedEvalV2Error("retained verifier authority is incomplete")
    if publication_root_fd is not None:
        _validate_publication_root_fd(publication_root, publication_root_fd)
        validate_retained_ffprobe_authority(
            ffprobe_authority,
            plan["producer"],
        )
        if (
            not isinstance(publication_authorities, Mapping)
            or set(publication_authorities)
            != {task["task_id"] for task in plan["tasks"]}
        ):
            raise MatchedEvalV2Error("retained publication task closure differs")
        for task in plan["tasks"]:
            validate_retained_publication_authority(
                publication_authorities[task["task_id"]], task
            )
    if validate_terminal_checkpoint_manifest(
        checkpoint["path"], checkpoint["sha256"]
    ) != checkpoint:
        raise MatchedEvalV2Error("terminal checkpoint changed before result verification")
    verified: list[dict[str, Any]] = []
    for index in range(8):
        pair = [task for task in plan["tasks"] if task["case_index"] == index]
        base_task = next(task for task in pair if task["arm"] == "base")
        adapted_task = next(task for task in pair if task["arm"] == "full644")
        base = verify_arm(
            base_task,
            plan["producer"],
            publication_root=(publication_root if publication_root_fd is not None else None),
            publication_root_fd=publication_root_fd,
            ffprobe_authority=ffprobe_authority,
            publication_authority=(
                publication_authorities[base_task["task_id"]]
                if publication_authorities is not None
                else None
            ),
        )
        adapted = verify_arm(
            adapted_task,
            plan["producer"],
            publication_root=(publication_root if publication_root_fd is not None else None),
            publication_root_fd=publication_root_fd,
            ffprobe_authority=ffprobe_authority,
            publication_authority=(
                publication_authorities[adapted_task["task_id"]]
                if publication_authorities is not None
                else None
            ),
        )
        if not v1._same_exact_json_value(
            base["receipt"]["model_consumption"]["model_capture_digest"],
            adapted["receipt"]["model_consumption"]["model_capture_digest"],
        ):
            raise MatchedEvalV2Error(
                f"case {index} frozen-base model capture differs between arms"
            )
        for key in ("input", "preprocessing", "prompt_contract", "sampling"):
            if not v1._same_exact_json_value(
                base["receipt"].get(key), adapted["receipt"].get(key)
            ):
                raise MatchedEvalV2Error(f"case {index} pair differs on {key}")
        for key in (
            "method_source_revision",
            "method_source_archive_sha256",
            "bernini_commit",
            "infer_lora_source_sha256",
            "veomni_commit",
            "bernini_inference_files",
            "checkpoint_tree_sha256",
            "runtime_versions",
        ):
            if not v1._same_exact_json_value(
                base["receipt"].get(key), adapted["receipt"].get(key)
            ):
                raise MatchedEvalV2Error(f"case {index} runtime differs on {key}")
        for item in (base, adapted):
            item.pop("receipt")
            verified.append(item)
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "plan_schema_version": plan["schema_version"],
        "plan_digest": plan["plan_digest"],
        "pair_count": 8,
        "verified_task_count": 16,
        "all_16_tasks_verified_no_cherry_pick": True,
        "source_mode_contract": "permission_bits_plus_stable_regular_inode_replay",
        "retained_publication_root_fd_replayed": publication_root_fd is not None,
        "retained_ffprobe_executable_fd_replayed": ffprobe_authority is not None,
        "retained_publication_leaf_fds_replayed": (
            publication_authorities is not None
        ),
        "producer_execution_proven_by_receipt_contract": False,
        "external_frozen_runner_attestation_still_required": True,
        "results": verified,
        "claim_limits": dict(v1.CLAIM_LIMITS),
    }
    report["report_digest"] = v1.object_sha256(report)
    return report


def write_create_only(path: str | Path, value: Mapping[str, Any]) -> str:
    return v1.write_create_only(path, value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("authority-check", help="delegate unchanged to frozen v1")
    sub.add_parser("build-plan", help="delegate unchanged to frozen v1")
    verify = sub.add_parser("verify-results")
    verify.add_argument("--plan", required=True)
    verify.add_argument("--plan-sha256", required=True)
    verify.add_argument("--output-report", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == "authority-check":
        return v1.main(raw)
    if raw and raw[0] == "build-plan":
        args = v1.build_parser().parse_args(raw)
        authority = v1.validate_shared8_authority(
            args.input_manifest,
            args.exposure_audit,
            require_source_bytes=True,
            source_root=args.source_root,
        )
        checkpoint = validate_terminal_checkpoint_manifest(
            args.checkpoint_manifest, args.checkpoint_manifest_sha256
        )
        plan = v1.build_plan(
            authority,
            checkpoint,
            args.output_root,
            production=True,
            producer={
                "inference_receipt_schema": v1.INFERENCE_RECEIPT_SCHEMA,
                "infer_lora_path": str(Path(args.infer_lora_source).expanduser()),
                "infer_lora_sha256": args.infer_lora_source_sha256,
                "method_source_revision": args.method_source_revision,
                "method_source_archive_sha256": (
                    args.method_source_archive_sha256
                ),
                "ffprobe_path": str(Path(args.ffprobe).expanduser()),
                "ffprobe_sha256": args.ffprobe_sha256,
            },
        )
        validate_plan(plan)
        print(v1.write_create_only(args.output_plan, plan))
        return 0
    args = build_parser().parse_args(raw)
    plan = load_plan(args.plan, args.plan_sha256)
    report = verify_results(plan)
    print(write_create_only(args.output_report, report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
