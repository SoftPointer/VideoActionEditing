#!/usr/bin/env python3
"""Exploratory direct T2V-action -> RV2V-LoRA quotient training.

Unlike the historical reward-selected SFT branch, this runner cannot access an
RV2V chosen target.  A pure T2V action video is used only to build detached
``[21,32]`` action/camera/appearance codes at the frozen post-head boundary.
The trainable RV2V path always noised the source itself and learns a velocity
delta whose quotient matches that detached action code.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import full30_action_learning_v1 as action_core
import self_generated_action_quotient_v1 as quotient
import self_generated_action_preservation_v2 as preservation_v2
import train_lora as legacy


METHOD = "bernini-self-generated-action-quotient-v1"
CACHE_SCHEMA = "bernini-self-generated-action-quotient-cache-v1"
CACHE_SCHEMA_V2 = "bernini-self-generated-action-preservation-cache-v2"
METHOD_V2 = "bernini-self-generated-action-preservation-canary-v2"
RECEIPT_SCHEMA = legacy.RECEIPT_SCHEMA
RECEIPT_SCHEMA_V2 = "bernini-r-1p3b-action-preservation-lora-receipt-v2"
SOURCE_MANIFEST_SCHEMA = "bernini-self-generated-action-quotient-source-manifest-v1"
NOOP_INSTRUCTION = legacy.IDENTITY_PRESERVATION_INSTRUCTION
SAVE_STEPS = (10, 20, 40, 80, 160)
V2_SAVE_STEPS = (0, 5, 10, 20)
OBJECTIVE_FAMILIES = ("legacy_v1", "preservation_v2")
V2_SIGMA_BINS = (
    (0.0, 0.20),
    (0.20, 0.40),
    (0.40, 0.60),
    (0.60, 0.80),
    (0.80, 1.0001),
)
V2_MAX_SIGMA_TRIALS = 512
REPLICATION_SEED = 20260817
V2_CANARY_SEED = 20260818
V2_WEIGHT_DECAY = 0.0
LEGACY_CACHE_SEED = 20260816
CHECKPOINT_ENTRY_NAMES = frozenset({"adapter", "optimizer.pt", "receipt.json"})
CHECKPOINT_ADAPTER_ENTRY_NAMES = frozenset(
    {"adapter_config.json", "adapter_model.safetensors"}
)
PEFT_REMOVABLE_ADAPTER_ENTRY_NAMES = frozenset({"README.md"})
PEFT_SERIALIZED_TARGET_MODULES_BY_ROUTE_SCOPE = {
    "all_attention": ("to_k", "to_out.0", "to_q", "to_v"),
    "cross_attn2_qo": ("attn2.to_out.0", "attn2.to_q"),
}


class QuotientTrainingError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise QuotientTrainingError(message)


@contextmanager
def serialized_model_load():
    """Serialize both SP4 islands in one holder during checkpoint loading."""

    job_id = os.environ.get("SLURM_JOB_ID")
    if not isinstance(job_id, str) or not job_id.isdigit():
        fail("shared model-load lock requires a numeric Slurm job identity")
    path = Path(f"/tmp/action-preservation-v2-{job_id}.model-load.lock")
    descriptor = os.open(
        path,
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        details = os.fstat(descriptor)
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_uid != os.getuid()
            or details.st_nlink != 1
            or stat.S_IMODE(details.st_mode) != 0o600
        ):
            fail("shared model-load lock topology differs")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def object_sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_plain_file_bytes(
    path: Path, *, expected_sha256: str, label: str,
) -> bytes:
    """Capture exactly the bytes whose physical identity and SHA are trusted."""

    require_sha256(expected_sha256, f"{label} SHA-256")
    if not path.is_absolute() or Path(os.path.realpath(path)) != path:
        fail(f"{label} path is not canonical")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
        ):
            fail(f"{label} physical topology differs")
        first = bytearray()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            first.extend(block)
        middle = os.fstat(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        second = bytearray()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            second.extend(block)
        after = os.fstat(descriptor)
        identity = lambda value: (
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
        if identity(before) != identity(middle) or identity(before) != identity(after):
            fail(f"{label} changed during stable capture")
        named = path.lstat()
        if identity(named) != identity(before) or path.is_symlink():
            fail(f"{label} pathname identity differs")
        raw = bytes(first)
        if raw != bytes(second) or len(raw) != before.st_size:
            fail(f"{label} bytes changed during stable capture")
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            fail(f"{label} SHA-256 differs")
        return raw
    finally:
        os.close(descriptor)


def _checkpoint_identity(details: os.stat_result) -> tuple[int, ...]:
    """Return the physical identity fields used for staged artifact capture."""

    return (
        details.st_dev,
        details.st_ino,
        details.st_uid,
        details.st_gid,
        details.st_mode,
        details.st_nlink,
        details.st_rdev,
        details.st_size,
        getattr(details, "st_blocks", -1),
        details.st_mtime_ns,
        details.st_ctime_ns,
    )


def _checkpoint_directory_anchor(details: os.stat_result) -> tuple[int, ...]:
    """Identity fields that must survive an intentional directory mutation."""

    return (
        details.st_dev,
        details.st_ino,
        details.st_uid,
        details.st_gid,
        details.st_mode,
        details.st_rdev,
    )


def _checkpoint_content_identity(details: os.stat_result) -> tuple[int, ...]:
    """Identity fields unaffected by this code's deliberate link/unlink calls."""

    return (
        details.st_dev,
        details.st_ino,
        details.st_uid,
        details.st_gid,
        details.st_mode,
        details.st_rdev,
        details.st_size,
        getattr(details, "st_blocks", -1),
        details.st_mtime_ns,
    )


def _checkpoint_directory_names(
    descriptor: int, *, label: str,
) -> frozenset[str]:
    try:
        return frozenset(os.listdir(descriptor))
    except OSError as error:
        raise QuotientTrainingError(f"{label} entry capture failed") from error


def _capture_checkpoint_member_before_unlink(
    adapter_descriptor: int,
    name: str,
    *,
    label: str,
    expected_raw: bytes,
    expected_content_identity: tuple[int, ...],
    allowed_nlinks: frozenset[int],
) -> os.stat_result:
    """Authorize a staged member, then close its FD before unlinking on NFS.

    VAST uses NFS-style silly renames when a pathname whose inode is open is
    unlinked.  The caller therefore cannot retain this descriptor across the
    unlink.  It instead obtains a same-FD double capture immediately before
    the operation; create-only publication and exact directory closure remain
    responsible for detecting any intervening pathname race.
    """

    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=adapter_descriptor)
    except OSError as error:
        raise QuotientTrainingError(f"{label} topology differs") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_nlink not in allowed_nlinks
            or before.st_size != len(expected_raw)
            or _checkpoint_content_identity(before) != expected_content_identity
        ):
            fail(f"{label} topology differs")
        first = bytearray()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            first.extend(block)
        middle = os.fstat(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        second = bytearray()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            second.extend(block)
        after = os.fstat(descriptor)
        try:
            named = os.stat(
                name,
                dir_fd=adapter_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise QuotientTrainingError(f"{label} pathname differs") from error
        if not (
            _checkpoint_identity(before)
            == _checkpoint_identity(middle)
            == _checkpoint_identity(after)
            == _checkpoint_identity(named)
        ):
            fail(f"{label} changed during stable capture")
        if bytes(first) != expected_raw or bytes(second) != expected_raw:
            fail(f"{label} bytes differ")
        return after
    finally:
        os.close(descriptor)


def expected_peft_serialized_target_modules(route_scope: str) -> tuple[str, ...]:
    """Return PEFT's exact, deterministic serialized target suffix list."""

    try:
        targets = PEFT_SERIALIZED_TARGET_MODULES_BY_ROUTE_SCOPE[route_scope]
    except (KeyError, TypeError) as error:
        raise QuotientTrainingError(
            "checkpoint PEFT route scope differs"
        ) from error
    if targets != tuple(sorted(targets)) or len(targets) != len(set(targets)):
        fail("checkpoint PEFT canonical target authority differs")
    return targets


def _unique_checkpoint_json_pairs(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            fail(f"checkpoint adapter config has duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _reject_checkpoint_json_constant(token: str) -> Any:
    raise ValueError(f"non-finite JSON constant: {token}")


def _canonicalize_peft_adapter_config(adapter: Path, route_scope: str) -> None:
    """Canonically replace PEFT's nondeterministic ``target_modules`` list.

    PEFT 0.19.1 accepts a set and serializes it through process-dependent set
    iteration order.  The checkpoint is still private staging at this point.
    Capture the generated config through an anchored directory descriptor and
    one ``O_NOFOLLOW`` file descriptor, reject any scope change, then publish a
    canonical replacement through create-only names in the same private staging
    directory.  VAST silly-renames unlinked open files, so each deletion is
    preceded by same-FD authorization and descriptor closure.  This never
    overwrites an unverified pathname; a crash can only leave an unpublished
    temporary checkpoint.  The formal checkpoint closure is checked afterward.
    """

    expected_targets = expected_peft_serialized_target_modules(route_scope)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        adapter_descriptor = os.open(adapter, directory_flags)
    except OSError as error:
        raise QuotientTrainingError(
            "checkpoint adapter directory topology differs during config canonicalization"
        ) from error
    config_authority_descriptor: int | None = None
    try:
        directory_before = os.fstat(adapter_descriptor)
        try:
            directory_named = adapter.lstat()
        except OSError as error:
            raise QuotientTrainingError(
                "checkpoint adapter directory pathname differs during config canonicalization"
            ) from error
        if (
            not stat.S_ISDIR(directory_before.st_mode)
            or stat.S_ISLNK(directory_named.st_mode)
            or directory_before.st_uid != os.getuid()
            or _checkpoint_identity(directory_before)
            != _checkpoint_identity(directory_named)
        ):
            fail(
                "checkpoint adapter directory topology differs during config canonicalization"
            )
        initial_names = _checkpoint_directory_names(
            adapter_descriptor,
            label="checkpoint adapter before config canonicalization",
        )
        if "adapter_config.json" not in initial_names:
            fail("generated PEFT adapter config is absent")

        read_flags = (
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            config_descriptor = os.open(
                "adapter_config.json", read_flags, dir_fd=adapter_descriptor
            )
        except OSError as error:
            raise QuotientTrainingError(
                "generated PEFT adapter config topology differs"
            ) from error
        try:
            before = os.fstat(config_descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.getuid()
                or before.st_nlink != 1
                or before.st_size <= 0
            ):
                fail("generated PEFT adapter config topology differs")
            first = bytearray()
            while True:
                block = os.read(config_descriptor, 1024 * 1024)
                if not block:
                    break
                first.extend(block)
            middle = os.fstat(config_descriptor)
            os.lseek(config_descriptor, 0, os.SEEK_SET)
            second = bytearray()
            while True:
                block = os.read(config_descriptor, 1024 * 1024)
                if not block:
                    break
                second.extend(block)
            after = os.fstat(config_descriptor)
            try:
                named = os.stat(
                    "adapter_config.json",
                    dir_fd=adapter_descriptor,
                    follow_symlinks=False,
                )
            except OSError as error:
                raise QuotientTrainingError(
                    "generated PEFT adapter config pathname differs"
                ) from error
            if not (
                _checkpoint_identity(before)
                == _checkpoint_identity(middle)
                == _checkpoint_identity(after)
                == _checkpoint_identity(named)
            ):
                fail("generated PEFT adapter config changed during stable capture")
            raw = bytes(first)
            if raw != bytes(second) or len(raw) != before.st_size:
                fail("generated PEFT adapter config bytes changed during stable capture")
            config_authority_descriptor = os.dup(config_descriptor)
            os.set_inheritable(config_authority_descriptor, False)
            if _checkpoint_identity(os.fstat(config_authority_descriptor)) != (
                _checkpoint_identity(after)
            ):
                fail("generated PEFT adapter config authority duplication differs")
        finally:
            os.close(config_descriptor)

        try:
            config = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_unique_checkpoint_json_pairs,
                parse_constant=_reject_checkpoint_json_constant,
            )
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
            QuotientTrainingError,
        ) as error:
            raise QuotientTrainingError(
                "generated PEFT adapter config is not strict JSON"
            ) from error
        if not isinstance(config, dict):
            fail("generated PEFT adapter config root differs")
        serialized_targets = config.get("target_modules")
        if (
            not isinstance(serialized_targets, list)
            or any(type(item) is not str for item in serialized_targets)
            or len(serialized_targets) != len(expected_targets)
            or len(set(serialized_targets)) != len(serialized_targets)
            or set(serialized_targets) != set(expected_targets)
        ):
            fail("generated PEFT adapter serialized target scope differs")
        config["target_modules"] = list(expected_targets)
        try:
            canonical_raw = canonical(config) + b"\n"
        except (TypeError, ValueError) as error:
            raise QuotientTrainingError(
                "generated PEFT adapter config is not canonicalizable"
            ) from error

        directory_after_capture = os.fstat(adapter_descriptor)
        try:
            directory_named_after_capture = adapter.lstat()
            named_before_replace = os.stat(
                "adapter_config.json",
                dir_fd=adapter_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise QuotientTrainingError(
                "generated PEFT adapter config pathname differs before replacement"
            ) from error
        if (
            _checkpoint_identity(directory_after_capture)
            != _checkpoint_identity(directory_before)
            or _checkpoint_identity(directory_named_after_capture)
            != _checkpoint_identity(directory_before)
            or _checkpoint_identity(named_before_replace)
            != _checkpoint_identity(after)
            or _checkpoint_directory_names(
                adapter_descriptor,
                label="checkpoint adapter before canonical replacement",
            )
            != initial_names
        ):
            fail("generated PEFT adapter config changed before replacement")

        replacement_name = f".adapter_config.json.canonical-{os.getpid()}"
        replacement_flags = (
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        replacement_descriptor: int | None = None
        try:
            replacement_descriptor = os.open(
                replacement_name,
                replacement_flags,
                0o600,
                dir_fd=adapter_descriptor,
            )
        except OSError as error:
            raise QuotientTrainingError(
                "generated PEFT adapter config replacement is not fresh"
            ) from error
        try:
            view = memoryview(canonical_raw)
            while view:
                written = os.write(replacement_descriptor, view)
                if written <= 0:
                    fail("short checkpoint adapter config replacement write")
                view = view[written:]
            os.fchmod(replacement_descriptor, 0o600)
            os.fsync(replacement_descriptor)
            replacement_before = os.fstat(replacement_descriptor)
            if (
                not stat.S_ISREG(replacement_before.st_mode)
                or replacement_before.st_uid != os.getuid()
                or replacement_before.st_nlink != 1
                or stat.S_IMODE(replacement_before.st_mode) != 0o600
                or replacement_before.st_size != len(canonical_raw)
            ):
                fail("checkpoint adapter config replacement topology differs")
            os.lseek(replacement_descriptor, 0, os.SEEK_SET)
            replacement_first = bytearray()
            while True:
                block = os.read(replacement_descriptor, 1024 * 1024)
                if not block:
                    break
                replacement_first.extend(block)
            replacement_middle = os.fstat(replacement_descriptor)
            os.lseek(replacement_descriptor, 0, os.SEEK_SET)
            replacement_second = bytearray()
            while True:
                block = os.read(replacement_descriptor, 1024 * 1024)
                if not block:
                    break
                replacement_second.extend(block)
            replacement_after_capture = os.fstat(replacement_descriptor)
            replacement_named = os.stat(
                replacement_name,
                dir_fd=adapter_descriptor,
                follow_symlinks=False,
            )
            if not (
                _checkpoint_identity(replacement_before)
                == _checkpoint_identity(replacement_middle)
                == _checkpoint_identity(replacement_after_capture)
                == _checkpoint_identity(replacement_named)
            ):
                fail("checkpoint adapter config replacement changed during capture")
            if (
                bytes(replacement_first) != canonical_raw
                or bytes(replacement_second) != canonical_raw
            ):
                fail("checkpoint adapter config replacement bytes differ")

            if config_authority_descriptor is None:
                fail("generated PEFT adapter config authority is absent")
            quarantine_name = f".adapter_config.json.original-{os.getpid()}"
            if _checkpoint_directory_names(
                adapter_descriptor,
                label="checkpoint adapter after replacement creation",
            ) != initial_names | {replacement_name}:
                fail("checkpoint adapter closure differs after replacement creation")
            try:
                named_before_quarantine = os.stat(
                    "adapter_config.json",
                    dir_fd=adapter_descriptor,
                    follow_symlinks=False,
                )
                directory_before_quarantine = os.fstat(adapter_descriptor)
                directory_named_before_quarantine = adapter.lstat()
            except OSError as error:
                raise QuotientTrainingError(
                    "generated PEFT adapter config pathname differs before quarantine"
                ) from error
            if (
                _checkpoint_identity(named_before_quarantine)
                != _checkpoint_identity(after)
                or _checkpoint_directory_anchor(directory_before_quarantine)
                != _checkpoint_directory_anchor(directory_before)
                or _checkpoint_identity(directory_before_quarantine)
                != _checkpoint_identity(directory_named_before_quarantine)
            ):
                fail("generated PEFT adapter config changed before quarantine")
            try:
                os.link(
                    "adapter_config.json",
                    quarantine_name,
                    src_dir_fd=adapter_descriptor,
                    dst_dir_fd=adapter_descriptor,
                    follow_symlinks=False,
                )
                os.fsync(adapter_descriptor)
            except OSError as error:
                raise QuotientTrainingError(
                    "generated PEFT adapter config quarantine is not fresh"
                ) from error

            original_linked = os.fstat(config_authority_descriptor)
            try:
                original_named_linked = os.stat(
                    "adapter_config.json",
                    dir_fd=adapter_descriptor,
                    follow_symlinks=False,
                )
                quarantine_named_linked = os.stat(
                    quarantine_name,
                    dir_fd=adapter_descriptor,
                    follow_symlinks=False,
                )
            except OSError as error:
                raise QuotientTrainingError(
                    "generated PEFT adapter config quarantine pathname differs"
                ) from error
            if (
                original_linked.st_nlink != 2
                or _checkpoint_identity(original_linked)
                != _checkpoint_identity(original_named_linked)
                or _checkpoint_identity(original_linked)
                != _checkpoint_identity(quarantine_named_linked)
                or _checkpoint_directory_names(
                    adapter_descriptor,
                    label="checkpoint adapter after config quarantine link",
                )
                != initial_names | {replacement_name, quarantine_name}
            ):
                fail("generated PEFT adapter config quarantine identity differs")
            os.close(config_authority_descriptor)
            config_authority_descriptor = None
            try:
                os.unlink("adapter_config.json", dir_fd=adapter_descriptor)
                os.fsync(adapter_descriptor)
            except OSError as error:
                raise QuotientTrainingError(
                    "generated PEFT adapter config quarantine unlink failed"
                ) from error

            _capture_checkpoint_member_before_unlink(
                adapter_descriptor,
                quarantine_name,
                label="generated PEFT adapter config quarantine",
                expected_raw=raw,
                expected_content_identity=_checkpoint_content_identity(after),
                allowed_nlinks=frozenset({1, 2}),
            )
            if _checkpoint_directory_names(
                adapter_descriptor,
                label="checkpoint adapter after formal config quarantine",
            ) != (initial_names - {"adapter_config.json"}) | {
                replacement_name,
                quarantine_name,
            }:
                fail("generated PEFT adapter config quarantine identity differs")

            published_flags = (
                os.O_RDWR
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
            )
            try:
                published_descriptor = os.open(
                    "adapter_config.json",
                    published_flags,
                    0o600,
                    dir_fd=adapter_descriptor,
                )
            except OSError as error:
                raise QuotientTrainingError(
                    "canonical PEFT adapter config target is not fresh"
                ) from error
            try:
                view = memoryview(canonical_raw)
                while view:
                    written = os.write(published_descriptor, view)
                    if written <= 0:
                        fail("short canonical PEFT adapter config publication write")
                    view = view[written:]
                os.fchmod(published_descriptor, 0o600)
                os.fsync(published_descriptor)
                os.fsync(adapter_descriptor)
                published_before = os.fstat(published_descriptor)
                if (
                    not stat.S_ISREG(published_before.st_mode)
                    or published_before.st_uid != os.getuid()
                    or published_before.st_nlink != 1
                    or stat.S_IMODE(published_before.st_mode) != 0o600
                    or published_before.st_size != len(canonical_raw)
                ):
                    fail("canonical PEFT adapter config topology differs")
                os.lseek(published_descriptor, 0, os.SEEK_SET)
                published_first = bytearray()
                while True:
                    block = os.read(published_descriptor, 1024 * 1024)
                    if not block:
                        break
                    published_first.extend(block)
                published_middle = os.fstat(published_descriptor)
                os.lseek(published_descriptor, 0, os.SEEK_SET)
                published_second = bytearray()
                while True:
                    block = os.read(published_descriptor, 1024 * 1024)
                    if not block:
                        break
                    published_second.extend(block)
                published_after = os.fstat(published_descriptor)
                try:
                    published_named = os.stat(
                        "adapter_config.json",
                        dir_fd=adapter_descriptor,
                        follow_symlinks=False,
                    )
                except OSError as error:
                    raise QuotientTrainingError(
                        "canonical PEFT adapter config publication pathname differs"
                    ) from error
                if (
                    _checkpoint_identity(published_before)
                    != _checkpoint_identity(published_middle)
                    or _checkpoint_identity(published_before)
                    != _checkpoint_identity(published_after)
                    or _checkpoint_identity(published_before)
                    != _checkpoint_identity(published_named)
                    or bytes(published_first) != canonical_raw
                    or bytes(published_second) != canonical_raw
                    or _checkpoint_directory_names(
                        adapter_descriptor,
                        label="checkpoint adapter after canonical config creation",
                    )
                    != initial_names | {replacement_name, quarantine_name}
                ):
                    fail("canonical PEFT adapter config publication identity differs")

                replacement_before_unlink = os.fstat(replacement_descriptor)
                try:
                    replacement_named_before_unlink = os.stat(
                        replacement_name,
                        dir_fd=adapter_descriptor,
                        follow_symlinks=False,
                    )
                except OSError as error:
                    raise QuotientTrainingError(
                        "canonical PEFT adapter config temporary pathname differs"
                    ) from error
                if _checkpoint_identity(replacement_before_unlink) != (
                    _checkpoint_identity(replacement_named_before_unlink)
                ):
                    fail("canonical PEFT adapter config temporary identity differs")
                os.close(replacement_descriptor)
                replacement_descriptor = None
                try:
                    os.unlink(replacement_name, dir_fd=adapter_descriptor)
                    os.fsync(adapter_descriptor)
                except OSError as error:
                    raise QuotientTrainingError(
                        "canonical PEFT adapter config temporary unlink failed"
                    ) from error
                if _checkpoint_directory_names(
                    adapter_descriptor,
                    label="checkpoint adapter after canonical temporary unlink",
                ) != initial_names | {quarantine_name}:
                    fail("checkpoint adapter closure differs after temporary unlink")

                _capture_checkpoint_member_before_unlink(
                    adapter_descriptor,
                    quarantine_name,
                    label="generated PEFT adapter config quarantine",
                    expected_raw=raw,
                    expected_content_identity=_checkpoint_content_identity(after),
                    allowed_nlinks=frozenset({1, 2}),
                )
                try:
                    os.unlink(quarantine_name, dir_fd=adapter_descriptor)
                    os.fsync(adapter_descriptor)
                except OSError as error:
                    raise QuotientTrainingError(
                        "generated PEFT adapter config quarantine removal failed"
                    ) from error

                directory_after_publish = os.fstat(adapter_descriptor)
                published_final = os.fstat(published_descriptor)
                try:
                    directory_named_after_publish = adapter.lstat()
                    published_named_final = os.stat(
                        "adapter_config.json",
                        dir_fd=adapter_descriptor,
                        follow_symlinks=False,
                    )
                except OSError as error:
                    raise QuotientTrainingError(
                        "checkpoint adapter changed after config publication"
                    ) from error
                if (
                    published_final.st_nlink != 1
                    or _checkpoint_identity(published_final)
                    != _checkpoint_identity(published_named_final)
                    or _checkpoint_directory_anchor(directory_after_publish)
                    != _checkpoint_directory_anchor(directory_before)
                    or _checkpoint_identity(directory_after_publish)
                    != _checkpoint_identity(directory_named_after_publish)
                    or _checkpoint_directory_names(
                        adapter_descriptor,
                        label="checkpoint adapter after config publication",
                    )
                    != initial_names
                ):
                    fail("canonical PEFT adapter config publication identity differs")
            finally:
                os.close(published_descriptor)
        finally:
            if replacement_descriptor is not None:
                os.close(replacement_descriptor)
    finally:
        if config_authority_descriptor is not None:
            os.close(config_authority_descriptor)
        os.close(adapter_descriptor)


def _validate_and_remove_peft_readme(adapter: Path) -> None:
    """Remove PEFT's generated documentation without widening checkpoint authority.

    ``README.md`` is not consumed by ``PeftModel.from_pretrained`` and is not a
    model-reload artifact.  A pinned PEFT release nevertheless creates it by
    default.  Accept only that one auxiliary name, capture it through the
    already-open adapter directory and a same-file descriptor double read, and
    unlink it before the formal checkpoint closure is sealed.  No unknown entry
    is silently discarded.
    """

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        adapter_descriptor = os.open(adapter, directory_flags)
    except OSError as error:
        raise QuotientTrainingError(
            "checkpoint adapter directory topology differs"
        ) from error
    try:
        directory_before = os.fstat(adapter_descriptor)
        try:
            directory_named = adapter.lstat()
        except OSError as error:
            raise QuotientTrainingError(
                "checkpoint adapter directory pathname differs"
            ) from error
        if (
            not stat.S_ISDIR(directory_before.st_mode)
            or stat.S_ISLNK(directory_named.st_mode)
            or directory_before.st_uid != os.getuid()
            or _checkpoint_identity(directory_before)
            != _checkpoint_identity(directory_named)
        ):
            fail("checkpoint adapter directory topology differs")

        try:
            names_before = frozenset(os.listdir(adapter_descriptor))
        except OSError as error:
            raise QuotientTrainingError(
                "checkpoint adapter entry capture failed"
            ) from error
        allowed_with_readme = (
            CHECKPOINT_ADAPTER_ENTRY_NAMES
            | PEFT_REMOVABLE_ADAPTER_ENTRY_NAMES
        )
        if names_before not in (
            CHECKPOINT_ADAPTER_ENTRY_NAMES,
            allowed_with_readme,
        ):
            fail("checkpoint adapter entry closure differs before README removal")
        if "README.md" not in names_before:
            return

        read_flags = (
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            readme_descriptor = os.open(
                "README.md", read_flags, dir_fd=adapter_descriptor
            )
        except OSError as error:
            raise QuotientTrainingError(
                "generated PEFT README topology differs"
            ) from error
        try:
            before = os.fstat(readme_descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.getuid()
                or before.st_nlink != 1
                or before.st_size <= 0
            ):
                fail("generated PEFT README topology differs")
            first = bytearray()
            while True:
                block = os.read(readme_descriptor, 1024 * 1024)
                if not block:
                    break
                first.extend(block)
            middle = os.fstat(readme_descriptor)
            os.lseek(readme_descriptor, 0, os.SEEK_SET)
            second = bytearray()
            while True:
                block = os.read(readme_descriptor, 1024 * 1024)
                if not block:
                    break
                second.extend(block)
            after = os.fstat(readme_descriptor)
            try:
                named = os.stat(
                    "README.md",
                    dir_fd=adapter_descriptor,
                    follow_symlinks=False,
                )
            except OSError as error:
                raise QuotientTrainingError(
                    "generated PEFT README pathname differs"
                ) from error
            if not (
                _checkpoint_identity(before)
                == _checkpoint_identity(middle)
                == _checkpoint_identity(after)
                == _checkpoint_identity(named)
            ):
                fail("generated PEFT README changed during stable capture")
            if bytes(first) != bytes(second) or len(first) != before.st_size:
                fail("generated PEFT README bytes changed during stable capture")
        finally:
            os.close(readme_descriptor)

        directory_after_capture = os.fstat(adapter_descriptor)
        if _checkpoint_identity(directory_before) != _checkpoint_identity(
            directory_after_capture
        ):
            fail("checkpoint adapter directory changed during README capture")
        try:
            named_before_unlink = os.stat(
                "README.md",
                dir_fd=adapter_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise QuotientTrainingError(
                "generated PEFT README pathname differs before removal"
            ) from error
        if _checkpoint_identity(named_before_unlink) != _checkpoint_identity(after):
            fail("generated PEFT README pathname changed before removal")
        try:
            os.unlink("README.md", dir_fd=adapter_descriptor)
            os.fsync(adapter_descriptor)
        except OSError as error:
            raise QuotientTrainingError(
                "generated PEFT README removal failed"
            ) from error

        directory_after_unlink = os.fstat(adapter_descriptor)
        try:
            directory_named_after_unlink = adapter.lstat()
            names_after = frozenset(os.listdir(adapter_descriptor))
        except OSError as error:
            raise QuotientTrainingError(
                "checkpoint adapter closure capture failed after README removal"
            ) from error
        if (
            _checkpoint_directory_anchor(directory_after_unlink)
            != _checkpoint_directory_anchor(directory_before)
            or _checkpoint_identity(directory_after_unlink)
            != _checkpoint_identity(directory_named_after_unlink)
            or names_after != CHECKPOINT_ADAPTER_ENTRY_NAMES
        ):
            fail("checkpoint adapter closure differs after README removal")
    finally:
        os.close(adapter_descriptor)


def write_create_only(path: Path, raw: bytes, *, mode: int) -> None:
    if path.exists() or path.is_symlink():
        fail(f"refusing to overwrite output: {path}")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                fail(f"short output write: {path}")
            view = view[written:]
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
        details = os.fstat(descriptor)
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_uid != os.getuid()
            or details.st_nlink != 1
            or details.st_size != len(raw)
        ):
            fail(f"published output topology differs: {path}")
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def require_replication_seed(seed: int) -> None:
    if type(seed) is not int or seed != REPLICATION_SEED:
        fail(f"initialization seed must be {REPLICATION_SEED}")


def objective_family(args: argparse.Namespace) -> str:
    value = getattr(args, "objective_family", "legacy_v1")
    if value not in OBJECTIVE_FAMILIES:
        fail(f"objective family differs: {value}")
    return value


def require_objective_contract(args: argparse.Namespace) -> str:
    family = objective_family(args)
    if family == "legacy_v1":
        require_replication_seed(args.seed)
        if args.arm not in quotient.ARM_NAMES:
            fail("legacy objective arm differs")
        if args.slots <= 0 or args.max_steps <= 0:
            fail("slots/max_steps must be positive")
        return family
    if type(args.seed) is not int or args.seed != V2_CANARY_SEED:
        fail(f"v2 canary initialization/cache seed must be {V2_CANARY_SEED}")
    if args.arm not in preservation_v2.ARM_NAMES:
        fail("v2 objective arm differs")
    if args.slots != len(V2_SIGMA_BINS):
        fail("v2 canary requires exactly five stratified sigma slots")
    if args.max_steps != 20:
        fail("v2 canary requires exactly 20 optimizer updates")
    if args.limit_cells != 0:
        fail("v2 formal cache forbids a partial cell limit")
    return family


def require_sha256(value: str | None, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        fail(f"{label} must be a lowercase SHA-256")
    return value


def validate_file_sha(path: Path, expected_sha256: str, label: str) -> str:
    observed = file_sha(path)
    if observed != expected_sha256:
        fail(f"{label} SHA-256 differs")
    return observed


def validate_teacher_cache_seed(cache: Mapping[str, Any], expected_seed: int) -> int:
    declared = cache.get("teacher_cache_seed", cache.get("seed"))
    if declared == LEGACY_CACHE_SEED:
        fail(f"legacy teacher cache seed {LEGACY_CACHE_SEED} is forbidden")
    if "teacher_cache_seed" not in cache or type(cache["teacher_cache_seed"]) is not int:
        fail("teacher cache lacks an explicit numeric teacher_cache_seed")
    teacher_seed = cache["teacher_cache_seed"]
    if teacher_seed != expected_seed:
        fail("teacher cache seed differs from initialization seed")
    if cache.get("initialization_seed") != expected_seed or cache.get("seed") != expected_seed:
        fail("teacher cache initialization seed metadata differs")
    return teacher_seed


def validate_teacher_cache_cells(
    cache: Mapping[str, Any], *, slots: int, expected_seed: int,
) -> tuple[list[Mapping[str, Any]], dict[tuple[int, int], Mapping[str, Any]]]:
    cells = cache.get("cells")
    if cache.get("slots") != slots or not isinstance(cells, list) or len(cells) != 4 * slots:
        fail("formal training requires the complete row x slot cache")
    expected_keys = {(row_index, slot) for row_index in range(4) for slot in range(slots)}
    by_key: dict[tuple[int, int], Mapping[str, Any]] = {}
    for cell in cells:
        if not isinstance(cell, Mapping):
            fail("teacher cache cell differs")
        row_index, slot, seed = cell.get("row_index"), cell.get("slot"), cell.get("seed")
        if type(row_index) is not int or type(slot) is not int or type(seed) is not int:
            fail("teacher cache cell indices/seed must be numeric")
        key = (row_index, slot)
        if key not in expected_keys or key in by_key:
            fail("teacher cache row x slot keys differ")
        if seed != legacy.step_seed(expected_seed, slot, row_index):
            fail("teacher cache cell seed differs")
        by_key[key] = cell
    if set(by_key) != expected_keys:
        fail("teacher cache row x slot keys differ")
    return cells, by_key


def sigma_bin_index(value: float) -> int:
    sigma = float(value)
    if not math.isfinite(sigma):
        fail("sigma is non-finite")
    for index, (lower, upper) in enumerate(V2_SIGMA_BINS):
        if lower <= sigma < upper or (index == len(V2_SIGMA_BINS) - 1 and sigma <= upper):
            return index
    fail(f"sigma lies outside the v2 stratification: {sigma}")


def validate_teacher_cache_cells_v2(
    cache: Mapping[str, Any], *, expected_seed: int,
) -> tuple[list[Mapping[str, Any]], dict[tuple[int, int], Mapping[str, Any]]]:
    import torch

    cells = cache.get("cells")
    expected_count = 4 * len(V2_SIGMA_BINS)
    if (
        cache.get("slots") != len(V2_SIGMA_BINS)
        or not isinstance(cells, list)
        or len(cells) != expected_count
        or cache.get("sigma_bins") != [list(item) for item in V2_SIGMA_BINS]
    ):
        fail("v2 cache requires the complete row x sigma-bin grid")
    expected_keys = {
        (row_index, slot)
        for row_index in range(4)
        for slot in range(len(V2_SIGMA_BINS))
    }
    by_key: dict[tuple[int, int], Mapping[str, Any]] = {}
    observed_order: list[tuple[int, int]] = []
    trials_by_row: dict[int, set[int]] = {row_index: set() for row_index in range(4)}
    required = {
        "iid", "row_index", "slot", "seed", "seed_trial", "sigma_bin",
        "sigma", "source_state_digest", "source_noop_raw", "teacher_unit",
        "camera_unit", "appearance_unit", "amplitude_floor",
        "source_amplitude", "teacher_amplitude",
        "frozen_source_action_velocity",
        "frozen_source_action_velocity_sha256",
    }
    for cell in cells:
        if not isinstance(cell, Mapping) or set(cell) != required:
            fail("v2 teacher cache cell field closure differs")
        row_index, slot = cell["row_index"], cell["slot"]
        trial, seed = cell["seed_trial"], cell["seed"]
        if any(type(item) is not int for item in (row_index, slot, trial, seed)):
            fail("v2 cache row/slot/trial/seed must be numeric")
        key = (row_index, slot)
        if key not in expected_keys or key in by_key or trial < 0:
            fail("v2 cache row x sigma-bin keys differ")
        if trial in trials_by_row[row_index]:
            fail("v2 cache reuses a sigma-selection trial within one IID")
        trials_by_row[row_index].add(trial)
        if seed != legacy.step_seed(expected_seed, trial, row_index):
            fail("v2 cache trial seed differs")
        if cell["sigma_bin"] != slot or sigma_bin_index(float(cell["sigma"])) != slot:
            fail("v2 cache sigma bin differs")
        if not isinstance(cell["iid"], str) or not cell["iid"]:
            fail("v2 cache IID differs")
        if not isinstance(cell["source_state_digest"], str) or ":" not in cell["source_state_digest"]:
            fail("v2 cache source state digest differs")
        for label in ("source_amplitude", "teacher_amplitude"):
            value = cell[label]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                fail(f"v2 cache {label} differs")
        for label in ("source_noop_raw", "teacher_unit", "camera_unit", "appearance_unit"):
            value = cell[label]
            if (
                not isinstance(value, torch.Tensor)
                or value.requires_grad
                or value.device.type != "cpu"
                or value.dtype != torch.float32
                or not value.is_contiguous()
                or tuple(int(item) for item in value.shape) != (1, 21, 32)
                or not bool(torch.isfinite(value.float()).all().item())
            ):
                fail(f"v2 cache {label} authority differs")
            if bool(torch.count_nonzero(value[:, :1]).item()):
                fail(f"v2 cache {label} phase-zero authority differs")
        teacher_flat = cell["teacher_unit"].reshape(1, -1)
        camera_flat = cell["camera_unit"].reshape(1, -1)
        appearance_flat = cell["appearance_unit"].reshape(1, -1)
        ones = torch.ones(1, dtype=torch.float32)
        if (
            not torch.allclose(torch.linalg.vector_norm(teacher_flat, dim=1), ones, rtol=1e-5, atol=1e-6)
            or not torch.allclose(torch.linalg.vector_norm(camera_flat, dim=1), ones, rtol=1e-5, atol=1e-6)
            or not torch.allclose(torch.linalg.vector_norm(appearance_flat, dim=1), ones, rtol=1e-5, atol=1e-6)
            or not torch.allclose((camera_flat * appearance_flat).sum(dim=1), torch.zeros(1), rtol=0.0, atol=1e-5)
        ):
            fail("v2 cache teacher/nuisance unit geometry differs")
        amplitude_floor = cell["amplitude_floor"]
        if (
            not isinstance(amplitude_floor, torch.Tensor)
            or amplitude_floor.requires_grad
            or amplitude_floor.device.type != "cpu"
            or tuple(int(item) for item in amplitude_floor.shape) != (1,)
            or not bool(torch.isfinite(amplitude_floor.float()).all().item())
            or not bool((amplitude_floor.float() > 0.0).all().item())
        ):
            fail("v2 cache amplitude floor authority differs")
        velocity = cell["frozen_source_action_velocity"]
        if (
            not isinstance(velocity, torch.Tensor)
            or velocity.requires_grad
            or velocity.device.type != "cpu"
            or velocity.dtype != torch.float32
            or not velocity.is_contiguous()
            or velocity.ndim != 5
            or tuple(int(item) for item in velocity.shape[:3]) != (1, 16, 21)
            or not bool(torch.isfinite(velocity.float()).all().item())
            or tensor_sha(velocity) != cell["frozen_source_action_velocity_sha256"]
        ):
            fail("v2 frozen source action velocity authority differs")
        require_sha256(
            cell["frozen_source_action_velocity_sha256"],
            "v2 frozen source action velocity SHA-256",
        )
        packet = action_core.NuisancePacket(
            camera_unit=cell["camera_unit"],
            appearance_unit=cell["appearance_unit"],
            camera_norm=ones,
            appearance_norm=ones,
            appearance_residual_ratio=ones,
        )
        source_raw = action_core.psiout_raw_v1(velocity) - cell["source_noop_raw"]
        source_projected = action_core.project_nuisances_v1(source_raw, packet)
        observed_source_amplitude = float(
            torch.linalg.vector_norm(source_projected.reshape(1, -1), dim=1).item()
        )
        if not math.isclose(
            float(cell["source_amplitude"]),
            observed_source_amplitude,
            rel_tol=1.0e-6,
            abs_tol=1.0e-6,
        ):
            fail("v2 cache source amplitude differs from frozen physical authority")
        expected_floor = max(
            observed_source_amplitude,
            0.10 * float(cell["teacher_amplitude"]),
            1.0e-4,
        )
        if not math.isclose(
            float(amplitude_floor.item()),
            expected_floor,
            rel_tol=1.0e-6,
            abs_tol=1.0e-6,
        ):
            fail("v2 cache amplitude floor formula differs")
        by_key[key] = cell
        observed_order.append(key)
    if set(by_key) != expected_keys:
        fail("v2 cache row x sigma-bin grid is incomplete")
    if observed_order != sorted(expected_keys):
        fail("v2 cache row x sigma-bin order differs")
    return cells, by_key


def tensor_sha(value: Any) -> str:
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(canonical(list(tensor.shape)))
    digest.update(tensor.view(__import__("torch").uint8).numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        fail(f"refusing to overwrite JSON: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_create_only(path, canonical(value) + b"\n", mode=0o600)


def load_manifest(
    path: Path, expected_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = path.resolve(strict=True)
    raw = stable_plain_file_bytes(
        path, expected_sha256=expected_sha256, label="source manifest"
    )
    value = json.loads(raw)
    stored = value.pop("manifest_digest", None)
    if value.get("schema_version") != SOURCE_MANIFEST_SCHEMA or object_sha(value) != stored:
        fail("source-only manifest identity differs")
    if value.get("training_process_can_reach_historical_selected_target") is not False:
        fail("historical selected target is reachable")
    if value.get("self_generated_anchor_is_rv2v_supervision_target") is not False:
        fail("self-generated anchor role differs")
    declared_rows = value.get("rows")
    if not isinstance(declared_rows, list) or len(declared_rows) != 4:
        fail("source-only manifest must contain four exploratory rows")
    runtime_rows: list[dict[str, Any]] = []
    for row in declared_rows:
        if not isinstance(row, dict):
            fail("source-only manifest row differs")
        source = Path(row["source_posterior"]["path"]).resolve(strict=True)
        anchor = Path(row["action_anchor"]["latent_path"]).resolve(strict=True)
        if str(source) != row["source_posterior"]["path"]:
            fail(f"source posterior path is not canonical: {row.get('iid')}")
        if str(anchor) != row["action_anchor"]["latent_path"]:
            fail(f"anchor latent path is not canonical: {row.get('iid')}")
        source_blob = stable_plain_file_bytes(
            source,
            expected_sha256=row["source_posterior"]["sha256"],
            label=f"source posterior {row.get('iid')}",
        )
        anchor_blob = stable_plain_file_bytes(
            anchor,
            expected_sha256=row["action_anchor"]["latent_sha256"],
            label=f"action anchor {row.get('iid')}",
        )
        if set(row.get("teacher_captions", {})) != {
            "action", "noop", "camera_only", "appearance_only"
        }:
            fail("teacher caption roles differ")
        runtime_row = dict(row)
        runtime_row["_source_posterior_blob"] = source_blob
        runtime_row["_action_anchor_blob"] = anchor_blob
        runtime_rows.append(runtime_row)
    return {**value, "manifest_digest": stored}, runtime_rows


def blob_tensor(blob: bytes) -> Any:
    import torch

    buffer = io.BytesIO(blob)
    try:
        return torch.load(buffer, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise QuotientTrainingError(
            "source posterior loading requires torch.load(weights_only=True)"
        ) from error


def tensor_blob(tensor: Any) -> bytes:
    import torch

    buffer = io.BytesIO()
    torch.save(tensor.detach().cpu().contiguous(), buffer)
    return buffer.getvalue()


def source_clean_from_posterior(blob: bytes, mean: Any, std: Any) -> Any:
    import torch

    posterior = blob_tensor(blob).float().contiguous()
    if posterior.ndim != 5 or tuple(posterior.shape[:3]) != (1, 32, 21):
        fail(f"source posterior geometry differs: {tuple(posterior.shape)}")
    normalized = (posterior[:, :16] - mean.unsqueeze(0)) / std.unsqueeze(0)
    if not bool(torch.isfinite(normalized).all().item()):
        fail("source normalized clean latent is non-finite")
    return normalized.contiguous()


def anchor_posterior_blob(blob: bytes, mean: Any, std: Any) -> tuple[bytes, Any]:
    import torch
    from safetensors.torch import load

    tensors = load(blob)
    if set(tensors) != {"normalized_clean_latent"}:
        fail("anchor safetensors key closure differs")
    clean = tensors["normalized_clean_latent"].float().contiguous()
    if clean.ndim != 5 or tuple(clean.shape[:3]) != (1, 16, 21):
        fail("anchor normalized latent geometry differs")
    raw_mean = clean * std.unsqueeze(0) + mean.unsqueeze(0)
    posterior = torch.cat((raw_mean, torch.full_like(raw_mean, -30.0)), dim=1).contiguous()
    return tensor_blob(posterior), clean


def messages(instruction: str, *, source: bool) -> str:
    rows = []
    if source:
        rows.append({"type": "video", "has_loss": 0})
    rows.extend((
        {"type": "text", "text": instruction, "has_loss": 0},
        {"type": "video_gen", "has_loss": 1},
    ))
    return canonical(rows).decode("utf-8")


def make_sample(*, instruction: str, target_blob: bytes, source_blob: bytes | None) -> dict[str, Any]:
    values = [target_blob] if source_blob is None else [source_blob, target_blob]
    return {
        "inputs": messages(instruction, source=source_blob is not None),
        "video_vae_latents": values,
        # Force one noise schedule for paired source/T2V states; the teacher is
        # still structurally source-free because it has no source-video message.
        "source_name": legacy.TASK_SOURCE_NAME,
    }


def collate(transformed: Any, device: Any) -> dict[str, Any]:
    batch = legacy.collate_single_renderer_sample(transformed)
    return legacy._move_batch(batch, device)


def target_patches(batch: Mapping[str, Any]) -> Any:
    selector = batch["vae_latents_mask"].squeeze(0).bool()
    return batch["input_vae_latents"][selector].contiguous()


def patches_to_spatial(patches: Any, *, spatial_shape: Sequence[int]) -> Any:
    packed = patches.permute(0, 2, 3, 4, 1).reshape(1, int(patches.shape[0]), 64)
    return unpack_wan_target_velocity(packed, spatial_shape=spatial_shape)


def unpack_wan_target_velocity(packed: Any, *, spatial_shape: Sequence[int]) -> Any:
    """Graph-preserving inverse of Wan's native ``(1,2,2)`` patch order."""

    import torch

    batch, channels, phases, height, width = map(int, spatial_shape)
    if (batch, channels, phases) != (1, 16, 21) or height % 2 or width % 2:
        fail("Wan spatial shape differs")
    patch_h, patch_w = height // 2, width // 2
    tokens = phases * patch_h * patch_w
    if not isinstance(packed, torch.Tensor) or tuple(packed.shape) != (1, tokens, 64):
        fail("Wan packed target shape differs")
    patches = packed.reshape(batch, phases, patch_h, patch_w, 1, 2, 2, channels)
    return (
        patches.permute(0, 7, 1, 4, 2, 5, 3, 6)
        .reshape(batch, channels, phases, height, width)
        .contiguous()
    )


def predicted_target_velocity(renderer: Any, batch: Mapping[str, Any], *, spatial_shape: Sequence[int]) -> Any:
    import torch

    text_lens, text_embs = renderer.get_t5_text_embeddings(
        batch["input_ids"], batch["attention_mask"], batch["t5_input_lens"]
    )
    lengths_tensor = batch["vae_seqlen"].squeeze(0)
    lengths = [int(item) for item in lengths_tensor[lengths_tensor > 0].tolist()]
    if len(lengths) != 1:
        fail("one packed VAE sequence is required")
    decoder = renderer.diff_dec
    transformer = decoder.transformer
    if transformer is None or decoder.transformer_2 is not None:
        fail("exactly one Wan expert is required")
    latent = batch["input_vae_latents"]
    embedded = transformer.patch_embedding(latent).flatten(1).unsqueeze(0)
    rope = batch["input_vae_rope"].permute(1, 0, 2).unsqueeze(0)
    timesteps = batch["timesteps"].squeeze(0)[:1]
    full = decoder.shared_step(
        model_id="transformer_1",
        noisy_latents=embedded,
        timesteps=timesteps,
        cond_embeds=text_embs,
        rotary_embs=rope,
        batch_vae_seqlen=lengths,
        batch_text_seqlen=text_lens,
    )
    selector = batch["vae_latents_mask"].squeeze(0).bool()
    selected = full[:, selector, :]
    result = unpack_wan_target_velocity(selected, spatial_shape=spatial_shape)
    if not bool(torch.isfinite(result).all().item()):
        fail("post-head velocity is non-finite")
    return result.float().contiguous()


def paired_state(batch: Mapping[str, Any], source_clean: Any) -> tuple[Any, Any, float, str]:
    import torch

    shape = tuple(int(x) for x in source_clean.shape)
    x_sigma = patches_to_spatial(target_patches(batch), spatial_shape=shape).float()
    true_velocity = patches_to_spatial(batch["target_velocity"], spatial_shape=shape).float()
    delta = x_sigma - source_clean.to(x_sigma.device)
    denominator = true_velocity.double().square().sum()
    sigma = float((delta.double() * true_velocity.double()).sum().div(denominator).item())
    if not math.isfinite(sigma) or not -1e-4 <= sigma <= 1.0001:
        fail(f"recovered sigma is invalid: {sigma}")
    timestep = float(batch["timesteps"].float().reshape(-1)[0].item())
    return x_sigma, true_velocity, sigma, tensor_sha(target_patches(batch)) + f":{timestep:.9g}"


def build_transform(
    *,
    tokenizer: Any,
    rope: Any,
    mean: Any,
    std: Any,
    scheduler: Any,
    device: Any,
    source_name: str = legacy.TASK_SOURCE_NAME,
):
    from bernini.training.data import process_renderer_sample

    if not isinstance(source_name, str) or not source_name.strip():
        fail("renderer training source_name must be non-empty")

    def transform(sample: Mapping[str, Any], seed: int) -> dict[str, Any]:
        legacy.seed_same_sample(seed)
        value = process_renderer_sample(
            sample,
            tokenizer=tokenizer,
            vae_rope_func=rope,
            vae_latent_mean=mean,
            vae_latent_std=std,
            noise_scheduler=scheduler,
            text_dropout_rate=0.0,
            img_dropout_rate=0.0,
            video_dropout_rate=0.0,
            max_vae_frames=21,
            source_name=source_name,
        )
        return collate(value, device)

    return transform


def cache_cells(
    *, rows: Sequence[Mapping[str, Any]], model: Any, transform: Any,
    mean: Any, std: Any, slots: int, base_seed: int, limit_cells: int,
) -> list[dict[str, Any]]:
    import torch

    cells = []
    model.eval()
    with torch.inference_mode(), model.disable_adapter(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        for row_index, row in enumerate(rows):
            source_blob = row["_source_posterior_blob"]
            source_clean = source_clean_from_posterior(source_blob, mean, std)
            anchor_blob, anchor_clean = anchor_posterior_blob(
                row["_action_anchor_blob"], mean, std
            )
            if tuple(source_clean.shape) != tuple(anchor_clean.shape):
                fail(f"source/anchor geometry differs: {row['iid']}")
            shape = tuple(int(x) for x in source_clean.shape)
            for slot in range(slots):
                if limit_cells and len(cells) >= limit_cells:
                    return cells
                seed = legacy.step_seed(base_seed, slot, row_index)
                source_batches = {}
                for role, instruction in (("action", row["instruction"]), ("noop", NOOP_INSTRUCTION)):
                    source_batches[role] = transform(
                        make_sample(instruction=instruction, source_blob=source_blob, target_blob=source_blob), seed
                    )
                _, _, sigma, state_digest = paired_state(source_batches["action"], source_clean)
                if paired_state(source_batches["noop"], source_clean)[3] != state_digest:
                    fail("source action/noop did not share exact noisy state")
                source_velocity = {
                    role: predicted_target_velocity(model, batch, spatial_shape=shape)
                    for role, batch in source_batches.items()
                }
                teacher_velocity = {}
                teacher_state = None
                for role in ("action", "noop", "camera_only", "appearance_only"):
                    batch = transform(
                        make_sample(
                            instruction=row["teacher_captions"][role],
                            source_blob=None,
                            target_blob=anchor_blob,
                        ),
                        seed,
                    )
                    current_state = tensor_sha(target_patches(batch)) + ":" + str(
                        float(batch["timesteps"].float().reshape(-1)[0].item())
                    )
                    if teacher_state is None:
                        teacher_state = current_state
                    elif current_state != teacher_state:
                        fail("teacher captions did not share exact anchor/noise/timestep")
                    teacher_velocity[role] = predicted_target_velocity(model, batch, spatial_shape=shape)
                teacher_noop = teacher_velocity["noop"]
                action_raw = action_core.psiout_raw_v1(teacher_velocity["action"] - teacher_noop)
                camera_raw = action_core.psiout_raw_v1(teacher_velocity["camera_only"] - teacher_noop)
                appearance_raw = action_core.psiout_raw_v1(
                    teacher_velocity["appearance_only"] - teacher_noop
                )
                packet = action_core.build_nuisance_packet_v1(camera_raw, appearance_raw)
                teacher_projected = action_core.project_nuisances_v1(action_raw, packet)
                teacher_unit = action_core.teacher_unit_v1(teacher_projected)
                source_noop_raw = action_core.psiout_raw_v1(source_velocity["noop"])
                source_delta_raw = action_core.psiout_raw_v1(
                    source_velocity["action"] - source_velocity["noop"]
                )
                source_projected = action_core.project_nuisances_v1(source_delta_raw, packet)
                source_amp = torch.linalg.vector_norm(source_projected.reshape(1, -1), dim=1)
                teacher_amp = torch.linalg.vector_norm(teacher_projected.reshape(1, -1), dim=1)
                floor = torch.maximum(source_amp, 0.10 * teacher_amp).clamp_min(1e-4)
                cells.append(
                    {
                        "iid": row["iid"], "row_index": row_index, "slot": slot,
                        "seed": seed, "sigma": sigma, "source_state_digest": state_digest,
                        "source_noop_raw": source_noop_raw.cpu(),
                        "teacher_unit": teacher_unit.cpu(),
                        "camera_unit": packet.camera_unit.cpu(),
                        "appearance_unit": packet.appearance_unit.cpu(),
                        "amplitude_floor": floor.cpu(),
                        "source_amplitude": float(source_amp.item()),
                        "teacher_amplitude": float(teacher_amp.item()),
                    }
                )
                if int(os.environ.get("RANK", "0")) == 0:
                    print(json.dumps({"cache_cell": len(cells), "iid": row["iid"], "slot": slot,
                                      "sigma": sigma, "source_amp": float(source_amp.item()),
                                      "teacher_amp": float(teacher_amp.item())}), flush=True)
    return cells


def cache_cells_v2(
    *, rows: Sequence[Mapping[str, Any]], model: Any, transform: Any,
    mean: Any, std: Any, base_seed: int,
) -> list[dict[str, Any]]:
    """Build one detached authority cell per IID and sigma stratum.

    Trials are deterministically rejection-selected before any model forward.
    This fixes the legacy cache's accidental lack of low-sigma examples while
    retaining the exact same source/noise state for every source and teacher
    role within a selected cell.
    """

    import torch

    cells: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode(), model.disable_adapter(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        for row_index, row in enumerate(rows):
            source_blob = row["_source_posterior_blob"]
            source_clean = source_clean_from_posterior(source_blob, mean, std)
            anchor_blob, anchor_clean = anchor_posterior_blob(
                row["_action_anchor_blob"], mean, std
            )
            if tuple(source_clean.shape) != tuple(anchor_clean.shape):
                fail(f"source/anchor geometry differs: {row['iid']}")
            shape = tuple(int(item) for item in source_clean.shape)
            selected_bins: set[int] = set()
            for trial in range(V2_MAX_SIGMA_TRIALS):
                seed = legacy.step_seed(base_seed, trial, row_index)
                action_batch = transform(
                    make_sample(
                        instruction=row["instruction"],
                        source_blob=source_blob,
                        target_blob=source_blob,
                    ),
                    seed,
                )
                _, _, sigma, state_digest = paired_state(action_batch, source_clean)
                slot = sigma_bin_index(sigma)
                if slot in selected_bins:
                    continue
                noop_batch = transform(
                    make_sample(
                        instruction=NOOP_INSTRUCTION,
                        source_blob=source_blob,
                        target_blob=source_blob,
                    ),
                    seed,
                )
                if paired_state(noop_batch, source_clean)[3] != state_digest:
                    fail("v2 source action/noop did not share exact noisy state")
                source_velocity = {
                    "action": predicted_target_velocity(
                        model, action_batch, spatial_shape=shape
                    ),
                    "noop": predicted_target_velocity(
                        model, noop_batch, spatial_shape=shape
                    ),
                }
                teacher_velocity: dict[str, Any] = {}
                teacher_state: str | None = None
                for role in ("action", "noop", "camera_only", "appearance_only"):
                    batch = transform(
                        make_sample(
                            instruction=row["teacher_captions"][role],
                            source_blob=None,
                            target_blob=anchor_blob,
                        ),
                        seed,
                    )
                    current_state = tensor_sha(target_patches(batch)) + ":" + str(
                        float(batch["timesteps"].float().reshape(-1)[0].item())
                    )
                    if teacher_state is None:
                        teacher_state = current_state
                    elif current_state != teacher_state:
                        fail("v2 teacher captions did not share exact anchor/noise/timestep")
                    teacher_velocity[role] = predicted_target_velocity(
                        model, batch, spatial_shape=shape
                    )
                teacher_noop = teacher_velocity["noop"]
                action_raw = action_core.psiout_raw_v1(
                    teacher_velocity["action"] - teacher_noop
                )
                camera_raw = action_core.psiout_raw_v1(
                    teacher_velocity["camera_only"] - teacher_noop
                )
                appearance_raw = action_core.psiout_raw_v1(
                    teacher_velocity["appearance_only"] - teacher_noop
                )
                packet = action_core.build_nuisance_packet_v1(
                    camera_raw, appearance_raw
                )
                teacher_projected = action_core.project_nuisances_v1(
                    action_raw, packet
                )
                teacher_unit = action_core.teacher_unit_v1(teacher_projected)
                source_noop_raw = action_core.psiout_raw_v1(source_velocity["noop"])
                source_delta_raw = action_core.psiout_raw_v1(
                    source_velocity["action"] - source_velocity["noop"]
                )
                source_projected = action_core.project_nuisances_v1(
                    source_delta_raw, packet
                )
                source_amp = torch.linalg.vector_norm(
                    source_projected.reshape(1, -1), dim=1
                )
                teacher_amp = torch.linalg.vector_norm(
                    teacher_projected.reshape(1, -1), dim=1
                )
                floor = torch.maximum(source_amp, 0.10 * teacher_amp).clamp_min(1e-4)
                frozen_action = (
                    source_velocity["action"]
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                    .contiguous()
                )
                cells.append(
                    {
                        "iid": row["iid"],
                        "row_index": row_index,
                        "slot": slot,
                        "seed": seed,
                        "seed_trial": trial,
                        "sigma_bin": slot,
                        "sigma": sigma,
                        "source_state_digest": state_digest,
                        "source_noop_raw": source_noop_raw.detach().cpu(),
                        "teacher_unit": teacher_unit.detach().cpu(),
                        "camera_unit": packet.camera_unit.detach().cpu(),
                        "appearance_unit": packet.appearance_unit.detach().cpu(),
                        "amplitude_floor": floor.detach().cpu(),
                        "source_amplitude": float(source_amp.item()),
                        "teacher_amplitude": float(teacher_amp.item()),
                        "frozen_source_action_velocity": frozen_action,
                        "frozen_source_action_velocity_sha256": tensor_sha(
                            frozen_action
                        ),
                    }
                )
                selected_bins.add(slot)
                if int(os.environ.get("RANK", "0")) == 0:
                    print(
                        json.dumps(
                            {
                                "cache_cell": len(cells),
                                "iid": row["iid"],
                                "slot": slot,
                                "seed_trial": trial,
                                "sigma": sigma,
                                "source_amp": float(source_amp.item()),
                                "teacher_amp": float(teacher_amp.item()),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                if len(selected_bins) == len(V2_SIGMA_BINS):
                    break
            if selected_bins != set(range(len(V2_SIGMA_BINS))):
                fail(
                    f"v2 sigma stratification incomplete for {row['iid']}: "
                    f"{sorted(selected_bins)}"
                )
    cells.sort(key=lambda item: (int(item["row_index"]), int(item["slot"])))
    return cells


def teacher_cache_payload(
    *, args: argparse.Namespace, manifest: Mapping[str, Any], cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if objective_family(args) == "preservation_v2":
        return {
            "schema_version": CACHE_SCHEMA_V2,
            "objective_family": "preservation_v2",
            "manifest_digest": manifest["manifest_digest"],
            "source_manifest_sha256": args.source_manifest_sha256,
            "method_source_revision": args.method_source_revision,
            "method_source_archive_sha256": args.method_source_archive_sha256,
            "slots": len(V2_SIGMA_BINS),
            "sigma_bins": [list(item) for item in V2_SIGMA_BINS],
            "seed": args.seed,
            "initialization_seed": args.seed,
            "teacher_cache_seed": args.seed,
            "cells": list(cells),
            "teacher_graph": "detached",
            "frozen_source_action_velocity": "cpu_float32_sha256_bound",
            "anchor_role": "action_phase_representation_only",
            "decoded_identity_background_camera_claim_authorized": False,
        }
    return {
        "schema_version": CACHE_SCHEMA,
        "manifest_digest": manifest["manifest_digest"],
        "source_manifest_sha256": args.source_manifest_sha256,
        "method_source_revision": args.method_source_revision,
        "method_source_archive_sha256": args.method_source_archive_sha256,
        "slots": args.slots,
        "seed": args.seed,
        "initialization_seed": args.seed,
        "teacher_cache_seed": args.seed,
        "cells": list(cells),
        "teacher_graph": "detached",
        "anchor_role": "action_phase_representation_only",
    }


def teacher_cache_receipt(
    *, args: argparse.Namespace, manifest: Mapping[str, Any], cell_count: int,
    cache_sha256: str,
) -> dict[str, Any]:
    family = objective_family(args)
    receipt = {
        "schema_version": CACHE_SCHEMA_V2 if family == "preservation_v2" else CACHE_SCHEMA,
        "cell_count": cell_count,
        "manifest_digest": manifest["manifest_digest"],
        "cache_sha256": cache_sha256,
        "source_manifest_sha256": args.source_manifest_sha256,
        "seed": args.seed,
        "initialization_seed": args.seed,
        "teacher_cache_seed": args.seed,
        "method_source_revision": args.method_source_revision,
        "method_source_archive_sha256": args.method_source_archive_sha256,
        "historical_selected_target_reachable": False,
    }
    if family == "preservation_v2":
        receipt.update(
            {
                "objective_family": family,
                "sigma_bins": [list(item) for item in V2_SIGMA_BINS],
                "frozen_source_action_velocity": "cpu_float32_sha256_bound",
                "decoded_identity_background_camera_claim_authorized": False,
            }
        )
    receipt["receipt_digest"] = object_sha(receipt)
    return receipt


def checkpoint_receipt(
    *, args: argparse.Namespace, manifest: Mapping[str, Any], step: int, loss: float,
    grad_norm: float, target_modules: Sequence[str], trainable_count: int,
    bernini_revision: str, veomni_revision: str, transformers_version: str,
    initial_digest: str, teacher_cache_seed: int, teacher_cache_sha256: str,
    loss_components: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    family = objective_family(args)
    if family == "preservation_v2":
        spec_v2 = preservation_v2.arm_spec(args.arm)
        learning_rate = spec_v2.learning_rate
        training_contract = {
            "model": "Bernini-R-1.3B-Diffusers renderer-only",
            "single_expert": "transformer_1",
            "mv2v_flow_shift": 5.0,
            "num_frames": 81,
            "latent_frames": 21,
            "task_source_name": legacy.TASK_SOURCE_NAME,
            "external_spatial_mask": False,
            "external_tracking_or_swept_tube": False,
            "conditioning": ["clean_source_video_vae", "edit_instruction"],
            "target_embedding_or_caption_conditioning": False,
            "lora_rank": 8,
            "lora_alpha": 8,
            "tokenizer_fix_mistral_regex": True,
            "transformers_version": transformers_version,
            "objective": METHOD_V2,
            "objective_family": family,
            "arm": args.arm,
            "weights": {
                "noop": spec_v2.noop_weight,
                "onset": spec_v2.onset_weight,
                "nuisance": spec_v2.nuisance_weight,
                "functional": spec_v2.functional_weight,
            },
            "onset_latent_phase_weights": list(preservation_v2.ONSET_WEIGHTS),
            "functional_components": [
                "teacher_direction_exempt_post_head_orthogonal_drift",
                "post_onset_temporal_dc_clean_latent_drift",
            ],
            "lora_route_scope": spec_v2.route_scope,
            "lora_route_scope_semantics": (
                "observable Wan attention topology only; no temporal-only or "
                "source-only route claim"
            ),
            "sigma_bins": [list(item) for item in V2_SIGMA_BINS],
            "checkpoint_updates": list(V2_SAVE_STEPS),
            "rv2v_supervision_target": "source_video_only",
            "self_generated_anchor_role": "detached_post_head_action_phase_code_only",
            "historical_selected_target_reachable": False,
            "decoded_identity_background_camera_claim_authorized": False,
            "post_decode_gate_schema": "bernini-action-preservation-decision-v1",
            "blind_full_video_review_required_for_promotion": True,
        }
        expected_components = {
            "action",
            "onset",
            "nuisance",
            "noop",
            "functional_code",
            "functional_temporal_dc",
            "functional_total",
        }
        if loss_components is None or set(loss_components) != expected_components:
            fail("v2 checkpoint loss component closure differs")
        normalized_components = {
            key: float(value) for key, value in loss_components.items()
        }
        if not all(math.isfinite(value) for value in normalized_components.values()):
            fail("v2 checkpoint loss component is non-finite")
    else:
        spec = quotient.arm_spec(args.arm)
        learning_rate = spec.learning_rate
        training_contract = {
            "model": "Bernini-R-1.3B-Diffusers renderer-only",
            "single_expert": "transformer_1", "mv2v_flow_shift": 5.0,
            "num_frames": 81, "latent_frames": 21,
            "task_source_name": legacy.TASK_SOURCE_NAME,
            "external_spatial_mask": False, "external_tracking_or_swept_tube": False,
            "conditioning": ["clean_source_video_vae", "edit_instruction"],
            "target_embedding_or_caption_conditioning": False,
            "lora_rank": 8, "lora_alpha": 8,
            "tokenizer_fix_mistral_regex": True,
            "transformers_version": transformers_version,
            "objective": METHOD,
            "arm": args.arm,
            "weights": {
                "noop": spec.noop_weight, "start": spec.start_weight,
                "nuisance": spec.nuisance_weight, "border": spec.border_weight,
            },
            "rv2v_supervision_target": "source_video_only",
            "self_generated_anchor_role": "detached_post_head_action_phase_code_only",
            "historical_selected_target_reachable": False,
        }
        normalized_components = None
    receipt = {
        "schema_version": RECEIPT_SCHEMA_V2 if family == "preservation_v2" else RECEIPT_SCHEMA,
        "global_step": step,
        "max_steps": args.max_steps,
        "last_loss": loss,
        "last_preclip_gradient_norm": grad_norm,
        "bernini_commit": bernini_revision,
        "bernini_training_files_index_sha256": legacy.object_sha256(legacy.BERNINI_PINNED_FILE_HASHES),
        "veomni_commit": veomni_revision,
        "method_source_revision": args.method_source_revision,
        "method_source_archive_sha256": args.method_source_archive_sha256,
        "initialization_seed": args.seed,
        "teacher_cache_seed": teacher_cache_seed,
        "checkpoint_tree_sha256": legacy.CHECKPOINT_TREE_SHA256,
        "training_contract": training_contract,
        "source_manifest_digest": manifest["manifest_digest"],
        "source_manifest_sha256": args.source_manifest_sha256,
        "teacher_cache_sha256": teacher_cache_sha256,
        "optimizer": {
            "type": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": V2_WEIGHT_DECAY if family == "preservation_v2" else 0.0,
        },
        "distributed": {
            "world_size": 4, "ulysses_size": 4, "backend": "nccl/rccl",
            "same_sample_all_ranks": True, "same_seed_all_ranks": True,
            "explicit_lora_gradient_all_reduce": True,
            "lora_initialization_digest": initial_digest,
        },
        "target_module_count": len(target_modules),
        "target_modules_sha256": legacy.object_sha256(list(target_modules)),
        "trainable_parameter_count": trainable_count,
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
        "experimental_training": True,
    }
    if family == "preservation_v2":
        receipt.update(
            {
                "objective_family": family,
                "last_loss_components": normalized_components,
                "target_modules": list(target_modules),
                "decoded_preservation_evidence_present": False,
                "automatic_scientific_promotion_authorized": False,
            }
        )
    receipt["receipt_digest"] = legacy.object_sha256(receipt)
    return receipt


def save_checkpoint(*, model: Any, optimizer: Any, output: Path, step: int, receipt: Mapping[str, Any], rank: int) -> None:
    import torch
    import torch.distributed as dist

    final = output / f"checkpoint-{step:08d}"
    if rank == 0:
        if final.exists():
            fail(f"checkpoint exists: {final}")
        temporary = output / f".{final.name}.tmp-{os.getpid()}"
        temporary.mkdir(parents=True)
        model.save_pretrained(temporary / "adapter", safe_serialization=True)
        adapter = temporary / "adapter"
        route_scope: str | None = None
        receipt_schema = receipt.get("schema_version")
        receipt_family = receipt.get("objective_family")
        if (
            receipt_schema == RECEIPT_SCHEMA_V2
            or receipt_family == "preservation_v2"
        ):
            if (
                receipt_schema != RECEIPT_SCHEMA_V2
                or receipt_family != "preservation_v2"
            ):
                fail("v2 checkpoint receipt identity differs during publication")
            training_contract = receipt.get("training_contract")
            if not isinstance(training_contract, Mapping):
                fail("v2 checkpoint training contract is absent during publication")
            candidate_scope = training_contract.get("lora_route_scope")
            if type(candidate_scope) is not str:
                fail("v2 checkpoint PEFT route scope is absent during publication")
            route_scope = candidate_scope
        if route_scope is not None:
            _canonicalize_peft_adapter_config(adapter, route_scope)
        _validate_and_remove_peft_readme(adapter)
        torch.save({"global_step": step, "optimizer": optimizer.state_dict()}, temporary / "optimizer.pt")
        atomic_json(temporary / "receipt.json", receipt)
        if (
            {path.name for path in temporary.iterdir()} != CHECKPOINT_ENTRY_NAMES
            or {path.name for path in adapter.iterdir()}
            != CHECKPOINT_ADAPTER_ENTRY_NAMES
        ):
            fail("checkpoint artifact closure differs before publication")
        for artifact in (
            temporary / "optimizer.pt",
            temporary / "receipt.json",
            adapter / "adapter_config.json",
            adapter / "adapter_model.safetensors",
        ):
            descriptor = os.open(
                artifact, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                os.fchmod(descriptor, 0o444)
                os.fsync(descriptor)
                details = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(details.st_mode)
                    or details.st_uid != os.getuid()
                    or details.st_nlink != 1
                    or stat.S_IMODE(details.st_mode) != 0o444
                    or details.st_size <= 0
                ):
                    fail(f"checkpoint artifact topology differs: {artifact}")
            finally:
                os.close(descriptor)
        for directory_path in (adapter,):
            descriptor = os.open(
                directory_path,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                os.fchmod(descriptor, 0o555)
                os.fsync(descriptor)
                details = os.fstat(descriptor)
                if (
                    not stat.S_ISDIR(details.st_mode)
                    or details.st_uid != os.getuid()
                    or stat.S_IMODE(details.st_mode) != 0o555
                ):
                    fail(f"checkpoint directory topology differs: {directory_path}")
            finally:
                os.close(descriptor)
        temporary_descriptor = os.open(
            temporary,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(temporary_descriptor)
        finally:
            os.close(temporary_descriptor)
        os.replace(temporary, final)
        final_descriptor = os.open(
            final,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fchmod(final_descriptor, 0o555)
            os.fsync(final_descriptor)
            details = os.fstat(final_descriptor)
            if (
                not stat.S_ISDIR(details.st_mode)
                or details.st_uid != os.getuid()
                or stat.S_IMODE(details.st_mode) != 0o555
            ):
                fail(f"checkpoint directory topology differs: {final}")
        finally:
            os.close(final_descriptor)
        descriptor = os.open(
            output, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    if dist.is_initialized():
        dist.barrier()


def require_finite_scalar_all_ranks(value: Any, *, label: str) -> None:
    """Collectively reject a bad scalar before backward/optimizer mutation."""

    import torch
    import torch.distributed as dist

    local_ok = bool(
        isinstance(value, torch.Tensor)
        and value.ndim == 0
        and torch.isfinite(value.detach()).item()
    )
    status = torch.tensor(
        1 if local_ok else 0,
        device=value.device if isinstance(value, torch.Tensor) else "cpu",
        dtype=torch.int32,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(status, op=dist.ReduceOp.MIN)
    if int(status.item()) != 1:
        fail(f"{label} is non-finite or non-scalar on at least one rank")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--mode", choices=("cache", "train"), required=True)
    value.add_argument("--bernini-root", required=True)
    value.add_argument("--veomni-root", required=True)
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--source-manifest", required=True)
    value.add_argument("--source-manifest-sha256", required=True)
    value.add_argument("--cache", required=True)
    value.add_argument("--expected-cache-sha256")
    value.add_argument("--output", required=True)
    value.add_argument("--objective-family", choices=OBJECTIVE_FAMILIES, default="legacy_v1")
    value.add_argument(
        "--arm",
        choices=tuple(quotient.ARM_NAMES) + tuple(preservation_v2.ARM_NAMES),
        default="action_only",
    )
    value.add_argument("--slots", type=int, default=4)
    value.add_argument("--limit-cells", type=int, default=0)
    value.add_argument("--max-steps", type=int, default=160)
    value.add_argument("--seed", type=int, required=True)
    value.add_argument("--method-source-revision", required=True)
    value.add_argument("--method-source-archive-sha256", required=True)
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    family = require_objective_contract(args)
    require_sha256(args.source_manifest_sha256, "source manifest SHA-256")
    cache_path: Path | None = None
    cache_raw: bytes | None = None
    teacher_cache_sha256: str | None = None
    if args.mode == "train":
        require_sha256(args.expected_cache_sha256, "expected teacher cache SHA-256")
        cache_path = Path(args.cache).resolve(strict=True)
        cache_raw = stable_plain_file_bytes(
            cache_path,
            expected_sha256=args.expected_cache_sha256,
            label="teacher cache",
        )
        teacher_cache_sha256 = args.expected_cache_sha256
    bernini_root, veomni_root, bernini_revision, veomni_revision = legacy.validate_source_trees(
        args.bernini_root, args.veomni_root,
        expected_bernini_commit=legacy.BERNINI_OFFICIAL_COMMIT,
        expected_veomni_commit=legacy.VEOMNI_TESTED_COMMIT,
    )
    checkpoint, _ = legacy.validate_checkpoint(args.checkpoint)
    legacy.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, __version__ as transformers_version
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.models.transformer_wan import WanRotaryPosEmbed
    from bernini.training.data import NoiseScheduler
    from bernini.parallel import init_parallel_state

    contract = legacy.distributed_contract()
    if contract.world_size != 4:
        fail("cache/train requires SP4")
    device, _ = legacy.initialise_distributed(contract)
    init_parallel_state(ulysses_size=4)
    legacy.seed_same_sample(args.seed)
    source_manifest_path = Path(args.source_manifest).resolve(strict=True)
    manifest, rows = load_manifest(
        source_manifest_path, args.source_manifest_sha256
    )

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True, **legacy.renderer_config_overrides(checkpoint)
    )
    config.dtype = torch.bfloat16
    with serialized_model_load():
        base = BerniniRendererModel(config)
        base.requires_grad_(False)
        base.t5_text_encoder.eval()
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        all_targets = legacy.select_attention_projection_names(base)
        if family == "preservation_v2":
            targets = preservation_v2.select_projection_scope(
                all_targets,
                scope=preservation_v2.arm_spec(args.arm).route_scope,
            )
        else:
            targets = all_targets
        model = get_peft_model(base, LoraConfig(
            r=8, lora_alpha=8, lora_dropout=0.0, bias="none", target_modules=targets
        ))
        model.to(device)
        gc.collect()
        torch.cuda.empty_cache()
    named = legacy.trainable_lora_parameters(model)
    initial_digest = legacy.synchronize_trainable_parameters(named, source_rank=0)
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), subfolder="tokenizer", padding_side="right", trust_remote_code=True,
        local_files_only=True, fix_mistral_regex=True,
    )
    rope = WanRotaryPosEmbed(128, (1, 2, 2), 1024, use_src_id_rotary_emb=True)
    mean, std, _ = legacy._vae_statistics(checkpoint)
    scheduler = NoiseScheduler(**legacy.noise_scheduler_kwargs())
    transform = build_transform(
        tokenizer=tokenizer, rope=rope, mean=mean, std=std,
        scheduler=scheduler, device=device,
    )

    if args.mode == "cache":
        if family == "preservation_v2":
            cells = cache_cells_v2(
                rows=rows,
                model=model,
                transform=transform,
                mean=mean,
                std=std,
                base_seed=args.seed,
            )
        else:
            cells = cache_cells(
                rows=rows, model=model, transform=transform, mean=mean, std=std,
                slots=args.slots, base_seed=args.seed, limit_cells=args.limit_cells,
            )
        payload = teacher_cache_payload(args=args, manifest=manifest, cells=cells)
        if contract.rank == 0:
            output = Path(args.output).resolve()
            if output.exists() or output.is_symlink():
                fail(f"cache output exists: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            buffer = io.BytesIO()
            torch.save(payload, buffer)
            output_raw = buffer.getvalue()
            output_sha256 = hashlib.sha256(output_raw).hexdigest()
            write_create_only(output, output_raw, mode=0o444)
            cache_receipt = teacher_cache_receipt(
                args=args,
                manifest=manifest,
                cell_count=len(cells),
                cache_sha256=output_sha256,
            )
            write_create_only(
                output.with_suffix(output.suffix + ".receipt.json"),
                canonical(cache_receipt) + b"\n",
                mode=0o444,
            )
        dist.barrier()
        dist.destroy_process_group()
        return 0

    if cache_path is None or cache_raw is None or teacher_cache_sha256 is None:
        fail("teacher cache contract was not established")
    cache = torch.load(
        io.BytesIO(cache_raw), map_location="cpu", weights_only=True
    )
    expected_schema = CACHE_SCHEMA_V2 if family == "preservation_v2" else CACHE_SCHEMA
    if (
        cache.get("schema_version") != expected_schema
        or cache.get("manifest_digest") != manifest["manifest_digest"]
    ):
        fail("teacher cache identity differs")
    if family == "preservation_v2" and cache.get("objective_family") != family:
        fail("v2 teacher cache objective family differs")
    teacher_cache_seed = validate_teacher_cache_seed(cache, args.seed)
    if (
        cache.get("source_manifest_sha256") != args.source_manifest_sha256
        or cache.get("method_source_revision") != args.method_source_revision
        or cache.get("method_source_archive_sha256") != args.method_source_archive_sha256
    ):
        fail("teacher cache source authority differs")
    if family == "preservation_v2":
        cells, by_key = validate_teacher_cache_cells_v2(
            cache, expected_seed=args.seed
        )
        for (row_index, _), cell in by_key.items():
            if cell["iid"] != rows[row_index]["iid"]:
                fail("v2 teacher cache IID binding differs")
    else:
        cells, by_key = validate_teacher_cache_cells(
            cache, slots=args.slots, expected_seed=args.seed
        )
    output = Path(args.output).resolve()
    if contract.rank == 0:
        if output.exists() or output.is_symlink():
            fail(f"training output exists: {output}")
        output.mkdir(parents=True)
    dist.barrier()

    if family == "preservation_v2":
        spec = preservation_v2.arm_spec(args.arm)
    else:
        spec = quotient.arm_spec(args.arm)
    # The v2 receipt freezes weight_decay=0.0.  Pass it explicitly: AdamW's
    # library default is 0.01, so relying on the default would make the
    # serialized optimizer disagree with the signed training contract.
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in named],
        lr=spec.learning_rate,
        weight_decay=V2_WEIGHT_DECAY if family == "preservation_v2" else 0.01,
    )
    model.train()
    model.get_base_model().t5_text_encoder.eval()
    last_loss = last_grad = 0.0
    last_components: dict[str, float] | None = None
    trainable_count = sum(int(parameter.numel()) for _, parameter in named)
    if family == "preservation_v2":
        last_components = {
            "action": 0.0,
            "onset": 0.0,
            "nuisance": 0.0,
            "noop": 0.0,
            "functional_code": 0.0,
            "functional_temporal_dc": 0.0,
            "functional_total": 0.0,
        }
        initial_receipt = checkpoint_receipt(
            args=args,
            manifest=manifest,
            step=0,
            loss=0.0,
            grad_norm=0.0,
            target_modules=targets,
            trainable_count=trainable_count,
            bernini_revision=bernini_revision,
            veomni_revision=veomni_revision,
            transformers_version=transformers_version,
            initial_digest=initial_digest,
            teacher_cache_seed=teacher_cache_seed,
            teacher_cache_sha256=teacher_cache_sha256,
            loss_components=last_components,
        )
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            output=output,
            step=0,
            receipt=initial_receipt,
            rank=contract.rank,
        )
    for global_step in range(args.max_steps):
        row_index = global_step % 4
        slot = (global_step // 4) % args.slots
        row = rows[row_index]
        cell = by_key[(row_index, slot)]
        source_blob = row["_source_posterior_blob"]
        source_clean_cpu = source_clean_from_posterior(source_blob, mean, std)
        shape = tuple(int(x) for x in source_clean_cpu.shape)
        action_batch = transform(
            make_sample(instruction=row["instruction"], source_blob=source_blob, target_blob=source_blob),
            int(cell["seed"]),
        )
        x_sigma, _, sigma, state_digest = paired_state(action_batch, source_clean_cpu)
        if state_digest != cell["source_state_digest"] or abs(sigma - float(cell["sigma"])) > 1e-6:
            fail("trainable source state differs from detached cache")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            velocity = predicted_target_velocity(model, action_batch, spatial_shape=shape)
            source_noop_raw = cell["source_noop_raw"].to(device)
            raw = action_core.psiout_raw_v1(velocity) - source_noop_raw
            packet = action_core.NuisancePacket(
                camera_unit=cell["camera_unit"].to(device),
                appearance_unit=cell["appearance_unit"].to(device),
                camera_norm=torch.ones(1, device=device),
                appearance_norm=torch.ones(1, device=device),
                appearance_residual_ratio=torch.ones(1, device=device),
            )
            projected = action_core.project_nuisances_v1(raw, packet)
            action_loss = action_core.paired_action_loss_v1(
                projected, cell["teacher_unit"].to(device), cell["amplitude_floor"].to(device)
            ).total
            predicted_clean = x_sigma - float(sigma) * velocity
            nuisance_loss = quotient.nuisance_coefficient_loss(
                raw, packet.camera_unit, packet.appearance_unit
            )
            noop_loss = torch.zeros((), device=device, dtype=torch.float32)
            if spec.noop_weight > 0:
                noop_batch = transform(
                    make_sample(instruction=NOOP_INSTRUCTION, source_blob=source_blob, target_blob=source_blob),
                    int(cell["seed"]),
                )
                noop_loss = model(**noop_batch, use_cache=False).diff_loss.float().mean()
            if family == "preservation_v2":
                source_clean_device = source_clean_cpu.to(device)
                onset_loss = preservation_v2.onset_envelope_loss(
                    predicted_clean=predicted_clean,
                    source_clean=source_clean_device,
                )
                frozen_action_velocity = (
                    cell["frozen_source_action_velocity"]
                    .to(device=device)
                    .float()
                    .detach()
                )
                frozen_action_code = (
                    action_core.psiout_raw_v1(frozen_action_velocity)
                    - source_noop_raw.detach().float()
                ).detach()
                functional_code_loss = preservation_v2.functional_non_regression_loss(
                    student_action_code=raw.float(),
                    frozen_action_code=frozen_action_code,
                    teacher_action_unit=cell["teacher_unit"].to(device).float().detach(),
                )
                functional_temporal_dc_loss = (
                    preservation_v2.temporal_dc_non_regression_loss(
                        student_velocity=velocity.float(),
                        frozen_velocity=frozen_action_velocity,
                        sigma=sigma,
                    )
                )
                functional_loss = functional_code_loss + functional_temporal_dc_loss
                total = preservation_v2.weighted_total(
                    spec=spec,
                    action=action_loss.float(),
                    onset=onset_loss.float(),
                    nuisance=nuisance_loss.float(),
                    noop=noop_loss.float(),
                    functional=functional_loss.float(),
                )
                preserve = None
            else:
                preserve = quotient.preservation_losses(
                    predicted_clean=predicted_clean,
                    source_clean=source_clean_cpu.to(device), border_width=4,
                )
                zero_proxy = torch.zeros((), device=device, dtype=torch.float32)
                onset_loss = zero_proxy
                functional_code_loss = zero_proxy
                functional_temporal_dc_loss = zero_proxy
                functional_loss = zero_proxy
                total = quotient.weighted_total(
                    spec=spec, action=action_loss.float(), noop=noop_loss.float(),
                    start=preserve["start"].float(), nuisance=nuisance_loss.float(),
                    border=preserve["border"].float(),
                )
        require_finite_scalar_all_ranks(total, label="training objective")
        total.backward()
        last_grad = legacy.all_reduce_lora_gradients(named)
        torch.nn.utils.clip_grad_norm_([parameter for _, parameter in named], 1.0)
        optimizer.step()
        step = global_step + 1
        last_loss = float(total.detach().item())
        direction = float(
            action_core.paired_action_loss_v1(
                projected.detach(),
                cell["teacher_unit"].to(device),
                cell["amplitude_floor"].to(device),
            ).direction_mean.item()
        )
        if family == "preservation_v2":
            last_components = {
                "action": float(action_loss.detach().item()),
                "onset": float(onset_loss.detach().item()),
                "nuisance": float(nuisance_loss.detach().item()),
                "noop": float(noop_loss.detach().item()),
                "functional_code": float(functional_code_loss.detach().item()),
                "functional_temporal_dc": float(
                    functional_temporal_dc_loss.detach().item()
                ),
                "functional_total": float(functional_loss.detach().item()),
            }
        if contract.rank == 0:
            log_row = {
                "step": step, "arm": args.arm, "iid": row["iid"], "slot": slot,
                "total": last_loss, "action": float(action_loss.detach().item()),
                "direction": direction,
                "noop": float(noop_loss.detach().item()),
                "nuisance": float(nuisance_loss.detach().item()),
                "sigma": sigma, "preclip_grad_norm": last_grad,
                "objective_family": family,
            }
            if family == "preservation_v2":
                log_row.update(last_components or {})
                log_row["sigma_bin"] = int(cell["sigma_bin"])
                log_row["seed_trial"] = int(cell["seed_trial"])
                log_row["lora_route_scope"] = spec.route_scope
            else:
                if preserve is None:
                    fail("legacy preservation losses are missing")
                log_row.update(
                    {
                        "start": float(preserve["start"].detach().item()),
                        "border": float(preserve["border"].detach().item()),
                    }
                )
            print(json.dumps(log_row, sort_keys=True), flush=True)
        save_steps = V2_SAVE_STEPS if family == "preservation_v2" else SAVE_STEPS
        if step in save_steps or (family == "legacy_v1" and step == args.max_steps):
            receipt = checkpoint_receipt(
                args=args, manifest=manifest, step=step, loss=last_loss, grad_norm=last_grad,
                target_modules=targets, trainable_count=trainable_count,
                bernini_revision=bernini_revision, veomni_revision=veomni_revision,
                transformers_version=transformers_version, initial_digest=initial_digest,
                teacher_cache_seed=teacher_cache_seed,
                teacher_cache_sha256=teacher_cache_sha256,
                loss_components=last_components,
            )
            save_checkpoint(model=model, optimizer=optimizer, output=output, step=step,
                            receipt=receipt, rank=contract.rank)
    if family == "preservation_v2" and contract.rank == 0:
        descriptor = os.open(
            output,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fchmod(descriptor, 0o555)
            os.fsync(descriptor)
            details = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(details.st_mode)
                or details.st_uid != os.getuid()
                or stat.S_IMODE(details.st_mode) != 0o555
            ):
                fail("completed v2 arm directory topology differs")
        finally:
            os.close(descriptor)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
