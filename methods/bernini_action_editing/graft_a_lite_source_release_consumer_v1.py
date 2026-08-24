#!/usr/bin/env python3
"""Fail-closed, source-owning consumer for the GRAFT A-lite canary4 release.

This module does not train a model and does not grant scientific or training
authority.  It turns the five externally pinned pieces of release evidence
into four immutable, owned source rows which a later short-training runner can
route mechanically.  In particular, the two confirmation rows can never be
promoted to update rows by this consumer.

The producer's execution receipt is intentionally insufficient on its own: it
is written before Slurm records the terminal state.  A caller must therefore
provide the bytes of an independently materialized, externally SHA-pinned
``sacct`` admission receipt.  A submission receipt (whose ``job_success`` is
``null``) is never accepted as terminal evidence.

All release artifacts are read through retained ``O_NOFOLLOW`` descriptors
and must be absolute, exact-realpath, regular, link-count-one, mode-0444
files.  Each source is opened once through a retained parent descriptor,
hashed, freshly probed as exact81/25 through the frozen portable ffprobe, and
hashed again on the same descriptor.  The returned source payload is a new
``bytes`` object, so pathname replacement after admission cannot change what
the caller consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Callable, Mapping, Sequence


ROW_SCHEMA = "bernini-graft-a-lite-source-noop-row-v1"
PRODUCER_RECEIPT_SCHEMA = "bernini-graft-a-lite-source-noop-receipt-v1"
EXECUTION_RECEIPT_SCHEMA = (
    "bernini-graft-a-lite-source-noop-execution-receipt-v2"
)
SUBMISSION_RECEIPT_SCHEMA = (
    "bernini-graft-a-lite-source-submission-receipt-v1"
)
TERMINAL_ADMISSION_SCHEMA = (
    "bernini-graft-a-lite-source-independent-sacct-admission-v1"
)
TERMINAL_MATERIALIZER_SCHEMA = (
    "bernini-graft-independent-sacct-admission-materializer-v1"
)

EXPECTED_JOB_ID = "132549"
MANIFEST_SUFFIX = ".manifest.jsonl"
PRODUCER_SUFFIX = ".receipt.json"
EXECUTION_SUFFIX = ".execution.receipt.json"
SUBMISSION_SUFFIX = ".submission.receipt.json"
FRAME_COUNT = 81
FPS_NUMERATOR = 25
FPS_DENOMINATOR = 1
SHORT_SIDE = 704
NOOP_INSTRUCTION = (
    "Keep every subject, action, timing, camera motion, framing, appearance, "
    "and background unchanged."
)

CANARY4 = (
    ("7b88a1ca1f804f41", "optimizer_train", True, False),
    ("a35b590961d24694", "optimizer_train", True, False),
    ("841b5e0080a1441d", "optimizer_confirmation", False, True),
    ("a66e6818e4144928", "optimizer_confirmation", False, True),
)
V16_COHORT = "goku_fullmotion_v16_exact128"
V17_COHORT = "goku_fullmotion_v17_next1000"
V16_ROWS = 128
V17_ROWS = 1000
V16_MANIFEST_SHA256 = (
    "834e5a70e7c87683730ac644ce233b9343e4fc98eb3b3a45f55f93c8da94688d"
)
V17_MANIFEST_SHA256 = (
    "24021e6a4c5d1758340f9e61df1a987383e1ad39063071526726e9658ccd1c10"
)

PORTABLE_FFPROBE_PROBE_KIND = (
    "frozen_shared_portable_compute_verified_ffprobe_v2"
)
PORTABLE_FFPROBE_PIN_LABEL = "shared_portable_compute_verified_auh_ffprobe_v1"
PORTABLE_FFPROBE_PATH = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_graft_v1_20260810/runtime/"
    "ffprobe_conda_cf_ffmpeg9_hdabad70_r2/runtime/ffprobe"
)
PORTABLE_FFPROBE_SHA256 = (
    "356754aa8e327b139dd54dda6846af6425673b73572ab6c1c182ed970f1107f5"
)
PORTABLE_FFPROBE_VERSION_STDOUT_SHA256 = (
    "2271b81138bdaf07532b801ac7abd5b48d9e84dd66a6287a82fb44bc04c84f6b"
)
PORTABLE_FFPROBE_VERSION_FIRST_LINE = (
    "ffprobe version 9.0 Copyright (c) 2007-2026 the FFmpeg developers"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_HEX40_RE = re.compile(r"[0-9a-f]{40}\Z")
_IID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")

_AUTHORITY_FIELDS = (
    "action_authority",
    "identity_authority",
    "cross_clip_identity_authority",
    "quality_authority",
    "training_authority",
    "production_authority",
    "data_governance_authority",
    "data_license_authority",
    "scientific_success_claimed",
)


class GraftALiteConsumerError(RuntimeError):
    """Raised before any source row is admitted."""


@dataclass(frozen=True)
class ReleaseArtifactPins:
    """Out-of-band SHA-256 pins; none may be learned from the artifacts."""

    manifest_sha256: str
    producer_receipt_sha256: str
    execution_receipt_sha256: str
    submission_receipt_sha256: str
    terminal_admission_sha256: str
    terminal_materializer_implementation_sha256: str
    terminal_materializer_runtime_sha256: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _require_sha256(getattr(self, name), label=name)
        if len({getattr(self, name) for name in self.__dataclass_fields__}) != 7:
            raise GraftALiteConsumerError("release artifact pins must be pairwise distinct")


@dataclass(frozen=True)
class OwnedSourceMedia:
    frame_count: int
    fps_numerator: int
    fps_denominator: int
    width: int
    height: int
    rgb_or_codec_content_not_interpreted: bool = True


@dataclass(frozen=True)
class OwnedValidatedSourceRow:
    """An immutable copy of exactly one admitted source endpoint."""

    row_index: int
    iid: str
    split: str
    optimizer_update_allowed: bool
    optimizer_confirmation_only: bool
    source_path_observed: str
    source_sha256: str
    source_size_bytes: int
    source_bytes: bytes
    media: OwnedSourceMedia
    noop_instruction: str
    producer_row_digest: str
    source_cohort: str
    upstream_manifest_sha256: str
    action_authority: bool = False
    identity_authority: bool = False
    cross_clip_identity_authority: bool = False
    quality_authority: bool = False
    training_authority: bool = False
    production_authority: bool = False


@dataclass(frozen=True)
class ArtifactProvenance:
    path: str
    sha256: str
    size_bytes: int
    mode: str = "0444"
    regular_file: bool = True
    link_count_one: bool = True
    opened_o_nofollow: bool = True
    retained_through_validation: bool = True


@dataclass(frozen=True)
class ReleaseProvenance:
    job_id: str
    scheduler_state: str
    scheduler_exit_code: str
    terminal_admission_sha256: str
    terminal_admission_digest: str
    terminal_materializer_implementation_sha256: str
    terminal_materializer_runtime_sha256: str
    producer_receipt_digest: str
    execution_receipt_digest: str
    submission_receipt_digest: str
    manifest: ArtifactProvenance
    producer_receipt: ArtifactProvenance
    execution_receipt: ArtifactProvenance
    submission_receipt: ArtifactProvenance
    portable_ffprobe_path: str
    portable_ffprobe_sha256: str
    portable_ffprobe_version_stdout_sha256: str
    portable_ffprobe_probe_kind: str
    consumer_fresh_portable_ffprobe_verified: bool
    sources_hash_and_probe_same_fd: bool = True
    sources_rehashed_after_probe: bool = True
    sources_returned_as_owned_bytes: bool = True


@dataclass(frozen=True)
class AuthorityBoundary:
    action_authority: bool = False
    identity_authority: bool = False
    cross_clip_identity_authority: bool = False
    quality_authority: bool = False
    training_authority: bool = False
    production_authority: bool = False
    data_governance_authority: bool = False
    data_license_authority: bool = False
    scientific_success_claimed: bool = False


class SealedALiteSourceRelease:
    """Opaque production mint; rows are not themselves a trainer routing."""

    __slots__ = (
        "_rows",
        "_provenance",
        "_authority",
        "_pins",
        "_pinset_digest",
        "_result_digest",
        "_mint",
        "_locked",
    )

    def __init__(self, *_: Any, **__: Any) -> None:
        raise GraftALiteConsumerError(
            "SealedALiteSourceRelease may only be minted by the production consumer"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("SealedALiteSourceRelease is immutable")

    def __reduce__(self) -> Any:
        raise TypeError("opaque source release mints are not pickleable")

    @property
    def rows(self) -> tuple[OwnedValidatedSourceRow, ...]:
        """Diagnostic rows; training must call ``validate_for_training``."""

        return self._rows

    @property
    def provenance(self) -> ReleaseProvenance:
        return self._provenance

    @property
    def authority(self) -> AuthorityBoundary:
        return self._authority

    @property
    def pinset_digest(self) -> str:
        return self._pinset_digest

    @property
    def result_digest(self) -> str:
        return self._result_digest


@dataclass(frozen=True)
class TestOnlySourceReleaseObservation:
    """Fake-probe diagnostic which is never accepted for trainer routing."""

    schema_version: str
    rows: tuple[OwnedValidatedSourceRow, ...]
    provenance: ReleaseProvenance
    authority: AuthorityBoundary
    test_only: bool = True
    production_release_minted: bool = False
    eligible_for_training_validation: bool = False


class TrainerOwnedSourceRow:
    """Opaque path-free trainer row backed only by immutable owned bytes."""

    __slots__ = (
        "_iid",
        "_split",
        "_optimizer_update_allowed",
        "_optimizer_confirmation_only",
        "_source_sha256",
        "_source_bytes",
        "_media",
        "_noop_instruction",
        "_mint",
    )

    def __init__(self, *_: Any, **__: Any) -> None:
        raise GraftALiteConsumerError(
            "trainer rows may only be minted by validate_for_training"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("TrainerOwnedSourceRow is immutable")

    @property
    def iid(self) -> str:
        return self._iid

    @property
    def split(self) -> str:
        return self._split

    @property
    def optimizer_update_allowed(self) -> bool:
        return self._optimizer_update_allowed

    @property
    def optimizer_confirmation_only(self) -> bool:
        return self._optimizer_confirmation_only

    @property
    def source_sha256(self) -> str:
        return self._source_sha256

    @property
    def source_bytes(self) -> bytes:
        return self._source_bytes

    @property
    def media(self) -> OwnedSourceMedia:
        return self._media

    @property
    def noop_instruction(self) -> str:
        return self._noop_instruction


class TrainerRouting:
    """Opaque, path-free split routing minted only after full revalidation."""

    __slots__ = (
        "_update_rows",
        "_confirmation_rows",
        "_source_release_result_digest",
        "_pinset_digest",
        "_routing_digest",
        "_authority",
        "_mint",
    )

    def __init__(self, *_: Any, **__: Any) -> None:
        raise GraftALiteConsumerError(
            "TrainerRouting may only be minted by validate_for_training"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("TrainerRouting is immutable")

    def __reduce__(self) -> Any:
        raise TypeError("opaque trainer routings are not pickleable")

    @property
    def update_rows(self) -> tuple[TrainerOwnedSourceRow, ...]:
        return self._update_rows

    @property
    def confirmation_rows(self) -> tuple[TrainerOwnedSourceRow, ...]:
        return self._confirmation_rows

    @property
    def source_release_result_digest(self) -> str:
        return self._source_release_result_digest

    @property
    def pinset_digest(self) -> str:
        return self._pinset_digest

    @property
    def routing_digest(self) -> str:
        return self._routing_digest

    @property
    def authority(self) -> AuthorityBoundary:
        return self._authority


@dataclass
class _OpenedFile:
    path: Path
    fd: int
    identity: tuple[int, int, int, int, int, int, int]
    raw: bytes
    sha256: str


@dataclass
class _OpenedSource:
    path: Path
    parent: Path
    fd: int
    parent_fd: int
    identity: tuple[int, int, int, int, int, int, int]
    parent_identity: tuple[int, int, int, int]


@dataclass
class _OpenedFFprobe:
    path: Path
    fd: int
    identity: tuple[int, int, int, int, int, int, int]
    execution_path: str
    transport: str


@dataclass(frozen=True)
class _CoreConsumedEvidence:
    """Probe-agnostic validation output; never accepted by the trainer gate."""

    rows: tuple[OwnedValidatedSourceRow, ...]
    provenance_without_production_probe_attestation: ReleaseProvenance


@dataclass(frozen=True)
class _TrainerRoutingBlueprint:
    """Validated data only; it is not accepted as trainer routing."""

    rows: tuple[OwnedValidatedSourceRow, ...]
    source_release_result_digest: str
    pinset_digest: str
    routing_digest: str
    authority: AuthorityBoundary


MediaProbe = Callable[[_OpenedSource], Mapping[str, Any]]


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise GraftALiteConsumerError(
            f"value is not canonical finite ASCII JSON: {error}"
        ) from error


def bytes_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def object_sha256(value: Any) -> str:
    return bytes_sha256(canonical_json_bytes(value))


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise GraftALiteConsumerError(f"{label} must be lowercase SHA-256")
    return value


def _require_hex40(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _HEX40_RE.fullmatch(value) is None:
        raise GraftALiteConsumerError(f"{label} must be lowercase hex-40")
    return value


def _require_positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GraftALiteConsumerError(f"{label} must be a positive integer")
    return value


def _require_nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GraftALiteConsumerError(f"{label} must be a nonnegative integer")
    return value


def _require_iid(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _IID_RE.fullmatch(value) is None:
        raise GraftALiteConsumerError(f"{label} must be a safe IID")
    return value


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GraftALiteConsumerError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise GraftALiteConsumerError(f"non-finite JSON constant: {value}")


def _parse_canonical_object(raw: bytes, *, label: str) -> dict[str, Any]:
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise GraftALiteConsumerError(
            f"{label} must be one newline-terminated JSON object"
        )
    try:
        value = json.loads(
            raw[:-1].decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except GraftALiteConsumerError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GraftALiteConsumerError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) + b"\n" != raw:
        raise GraftALiteConsumerError(f"{label} is not canonical JSON")
    return value


def _verify_self_digest(value: Mapping[str, Any], *, label: str) -> str:
    unsigned = dict(value)
    declared = unsigned.pop("receipt_digest", None)
    declared = _require_sha256(declared, label=f"{label} receipt_digest")
    if declared != object_sha256(unsigned):
        raise GraftALiteConsumerError(f"{label} self digest differs")
    return declared


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        stat.S_IMODE(metadata.st_mode),
        metadata.st_nlink,
    )


def _directory_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_fd_all(fd: int, *, label: str) -> bytes:
    before = os.fstat(fd)
    chunks: list[bytes] = []
    try:
        if hasattr(os, "pread"):
            offset = 0
            while True:
                block = os.pread(fd, 1024 * 1024, offset)
                if not block:
                    break
                chunks.append(block)
                offset += len(block)
        else:  # pragma: no cover - supported production/test hosts have pread
            original = os.lseek(fd, 0, os.SEEK_CUR)
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                for block in iter(lambda: os.read(fd, 1024 * 1024), b""):
                    chunks.append(block)
            finally:
                os.lseek(fd, original, os.SEEK_SET)
    except OSError as error:
        raise GraftALiteConsumerError(f"cannot read {label}: {error}") from error
    after = os.fstat(fd)
    if _identity(before) != _identity(after):
        raise GraftALiteConsumerError(f"{label} changed while reading")
    return b"".join(chunks)


def _exact_absolute_path(path_value: str | Path, *, label: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        raise GraftALiteConsumerError(f"{label} must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise GraftALiteConsumerError(f"cannot resolve {label}: {error}") from error
    if resolved != path:
        raise GraftALiteConsumerError(f"{label} must be its exact realpath")
    return path


def _open_sealed_file(
    path_value: str | Path, *, expected_sha256: str, label: str
) -> _OpenedFile:
    expected_sha256 = _require_sha256(expected_sha256, label=f"{label} pin")
    path = _exact_absolute_path(path_value, label=label)
    try:
        before = path.lstat()
    except OSError as error:
        raise GraftALiteConsumerError(f"cannot stat {label}: {error}") from error
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o444
        or before.st_nlink != 1
    ):
        raise GraftALiteConsumerError(
            f"{label} must be regular mode-0444 link-count-one"
        )
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise GraftALiteConsumerError(f"cannot open {label}: {error}") from error
    try:
        opened = os.fstat(fd)
        if _identity(opened) != _identity(before):
            raise GraftALiteConsumerError(f"{label} changed while opening")
        raw = _read_fd_all(fd, label=label)
        observed = bytes_sha256(raw)
        if observed != expected_sha256:
            raise GraftALiteConsumerError(f"{label} SHA-256 differs from pin")
        return _OpenedFile(
            path=path,
            fd=fd,
            identity=_identity(opened),
            raw=raw,
            sha256=observed,
        )
    except Exception:
        os.close(fd)
        raise


def _revalidate_opened_file(opened: _OpenedFile, *, label: str) -> None:
    try:
        current_fd = os.fstat(opened.fd)
        current_path = opened.path.lstat()
    except OSError as error:
        raise GraftALiteConsumerError(f"cannot revalidate {label}: {error}") from error
    if (
        _identity(current_fd) != opened.identity
        or _identity(current_path) != opened.identity
        or not stat.S_ISREG(current_path.st_mode)
        or stat.S_ISLNK(current_path.st_mode)
        or stat.S_IMODE(current_path.st_mode) != 0o444
        or current_path.st_nlink != 1
    ):
        raise GraftALiteConsumerError(f"{label} path or identity changed")
    raw = _read_fd_all(opened.fd, label=f"retained {label}")
    if raw != opened.raw or bytes_sha256(raw) != opened.sha256:
        raise GraftALiteConsumerError(f"{label} bytes changed")


def _open_source(path_value: str, *, label: str) -> _OpenedSource:
    path = _exact_absolute_path(path_value, label=label)
    try:
        before = path.lstat()
        parent = path.parent.resolve(strict=True)
        parent_before = parent.lstat()
    except OSError as error:
        raise GraftALiteConsumerError(f"cannot stat {label}: {error}") from error
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) & 0o022
    ):
        raise GraftALiteConsumerError(
            f"{label} must be regular, link-count-one, and not group/world writable"
        )
    if stat.S_ISLNK(parent_before.st_mode) or not stat.S_ISDIR(parent_before.st_mode):
        raise GraftALiteConsumerError(f"{label} parent must be a plain directory")
    parent_fd = os.open(
        parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    fd = -1
    try:
        opened_parent = os.fstat(parent_fd)
        if _directory_identity(opened_parent) != _directory_identity(parent_before):
            raise GraftALiteConsumerError(f"{label} parent changed while opening")
        leaf = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        fd = os.open(
            path.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        opened = os.fstat(fd)
        if _identity(leaf) != _identity(before) or _identity(opened) != _identity(before):
            raise GraftALiteConsumerError(f"{label} changed while opening")
        return _OpenedSource(
            path=path,
            parent=parent,
            fd=fd,
            parent_fd=parent_fd,
            identity=_identity(opened),
            parent_identity=_directory_identity(opened_parent),
        )
    except Exception:
        if fd >= 0:
            os.close(fd)
        os.close(parent_fd)
        raise


def _revalidate_opened_source(source: _OpenedSource, *, label: str) -> None:
    try:
        fd_stat = os.fstat(source.fd)
        leaf = os.stat(
            source.path.name, dir_fd=source.parent_fd, follow_symlinks=False
        )
        path_stat = source.path.lstat()
        parent_fd_stat = os.fstat(source.parent_fd)
        parent_path_stat = source.parent.lstat()
    except OSError as error:
        raise GraftALiteConsumerError(f"cannot revalidate {label}: {error}") from error
    if (
        _identity(fd_stat) != source.identity
        or _identity(leaf) != source.identity
        or _identity(path_stat) != source.identity
    ):
        raise GraftALiteConsumerError(f"{label} source path or identity changed")
    if (
        _directory_identity(parent_fd_stat) != source.parent_identity
        or _directory_identity(parent_path_stat) != source.parent_identity
    ):
        raise GraftALiteConsumerError(f"{label} source parent changed")


def _all_authority_false(
    value: Any,
    *,
    label: str,
    required_fields: Sequence[str] = _AUTHORITY_FIELDS,
) -> None:
    if not isinstance(value, Mapping):
        raise GraftALiteConsumerError(f"{label} must be an object")
    if set(value) != set(required_fields):
        raise GraftALiteConsumerError(f"{label} fields differ")
    for field in required_fields:
        if value.get(field) is not False:
            raise GraftALiteConsumerError(f"{label}.{field} must be false")


def _manifest_rows(raw: bytes) -> list[dict[str, Any]]:
    if not raw.endswith(b"\n"):
        raise GraftALiteConsumerError("manifest is not newline terminated")
    lines = raw.splitlines(keepends=True)
    if len(lines) != len(CANARY4) or b"".join(lines) != raw:
        raise GraftALiteConsumerError("manifest must contain exactly four framed rows")
    return [
        _parse_canonical_object(line, label=f"manifest row {index}")
        for index, line in enumerate(lines)
    ]


def _validate_row(
    row: Mapping[str, Any], *, index: int, expected: tuple[str, str, bool, bool]
) -> None:
    iid, split, update_allowed, confirmation_only = expected
    unsigned = dict(row)
    declared = unsigned.pop("row_digest", None)
    declared = _require_sha256(declared, label=f"row {index} digest")
    if declared != object_sha256(unsigned):
        raise GraftALiteConsumerError(f"manifest row {index} self digest differs")
    forbidden_release_fields = {
        "tgt_video",
        "target_video",
        "target_video_path",
        "edited_caption",
        "prompt",
        "resolved_anchor_image",
        "anchor_image",
        "latent_path",
    }
    if forbidden_release_fields & set(row):
        raise GraftALiteConsumerError(
            f"manifest row {index} contains a forbidden non-source field"
        )
    if (
        row.get("schema_version") != ROW_SCHEMA
        or row.get("release_mode") != "canary4"
        or _require_nonnegative_int(
            row.get("row_index"), label=f"row {index} row_index"
        )
        != index
        or _require_iid(row.get("iid"), label=f"row {index} iid") != iid
        or row.get("split") != split
        or row.get("optimizer_update_authorized") is not update_allowed
        or row.get("optimizer_confirmation_only") is not confirmation_only
        or row.get("prior_research_exposure") is not True
        or row.get("global_holdout") is not False
        or row.get("stable_identity_disjoint_split_claimed") is not False
        or row.get("same_clip_noop_only") is not True
        or row.get("source_video_is_clean_noop_endpoint") is not True
        or row.get("publication_eligible") is not True
        or row.get("noop_instruction") != NOOP_INSTRUCTION
    ):
        raise GraftALiteConsumerError(f"manifest row {index} routing differs")
    for field in (
        "cross_clip_identity_authority",
        "action_authority",
        "quality_authority",
        "production_authority",
    ):
        if row.get(field) is not False:
            raise GraftALiteConsumerError(f"manifest row {index} authority differs")
    for field in (
        "source_hash_and_probe_same_open_fd",
        "source_sha256_recomputed_before_and_after_probe",
        "source_pre_post_probe_sha256_matched",
        "source_identity_includes_ctime_ns",
        "source_path_inode_binding_revalidated",
    ):
        if row.get(field) is not True:
            raise GraftALiteConsumerError(
                f"manifest row {index} source evidence differs"
            )
    source_path = row.get("source_video_path")
    if not isinstance(source_path, str) or not Path(source_path).is_absolute():
        raise GraftALiteConsumerError(f"manifest row {index} source path differs")
    _require_sha256(
        row.get("source_video_sha256"), label=f"row {index} source SHA-256"
    )
    _require_positive_int(
        row.get("source_file_size_bytes"), label=f"row {index} source size"
    )
    _require_positive_int(row.get("source_mtime_ns"), label=f"row {index} mtime")
    _require_positive_int(
        row.get("source_ctime_ns_observed"), label=f"row {index} ctime"
    )
    media = row.get("source_media")
    if not isinstance(media, Mapping):
        raise GraftALiteConsumerError(f"manifest row {index} media differs")
    width = _require_positive_int(media.get("width"), label=f"row {index} width")
    height = _require_positive_int(media.get("height"), label=f"row {index} height")
    if (
        media.get("frame_count") != FRAME_COUNT
        or media.get("fps") != 25.0
        or media.get("fps_fraction") != "25/1"
        or media.get("reported_fps_fraction") != "25/1"
        or media.get("resolution_hw") != [height, width]
        or media.get("short_side") != SHORT_SIDE
        or min(width, height) != SHORT_SIDE
        or media.get("probe_contract_matched") is not True
        or media.get("fresh_ffprobe_verified") is not True
        or media.get("test_only_probe_contract_matched") is not False
        or media.get("source_fd_transport")
        not in {"linux_proc_self_fd", "portable_dev_fd"}
    ):
        raise GraftALiteConsumerError(f"manifest row {index} exact81/25 differs")


def _validate_producer(
    producer: Mapping[str, Any], *, manifest_raw: bytes, rows: Sequence[Mapping[str, Any]]
) -> str:
    digest = _verify_self_digest(producer, label="producer receipt")
    if (
        producer.get("schema_version") != PRODUCER_RECEIPT_SCHEMA
        or producer.get("status") != "complete"
        or producer.get("release_mode") != "canary4"
    ):
        raise GraftALiteConsumerError("producer receipt identity differs")
    artifact = producer.get("artifact")
    row_digests = [row["row_digest"] for row in rows]
    if not isinstance(artifact, Mapping) or (
        artifact.get("manifest_rows") != 4
        or artifact.get("manifest_bytes") != len(manifest_raw)
        or artifact.get("manifest_sha256") != bytes_sha256(manifest_raw)
        or artifact.get("row_digest_sequence_sha256") != object_sha256(row_digests)
        or artifact.get("manifest_suffix") != MANIFEST_SUFFIX
        or artifact.get("receipt_suffix") != PRODUCER_SUFFIX
    ):
        raise GraftALiteConsumerError("producer receipt does not bind manifest")
    semantics = producer.get("semantics")
    if not isinstance(semantics, Mapping) or (
        semantics.get("source_only") is not True
        or semantics.get("same_clip_noop_only") is not True
        or semantics.get("source_video_is_clean_noop_endpoint") is not True
        or semantics.get("cross_clip_identity_authority") is not False
        or semantics.get("action_authority") is not False
        or semantics.get("quality_authority") is not False
        or semantics.get("production_authority") is not False
        or semantics.get("scientific_success_claimed") is not False
        or semantics.get("canonical_noop_instruction") != NOOP_INSTRUCTION
    ):
        raise GraftALiteConsumerError("producer semantics differ")
    input_policy = producer.get("input_policy")
    if not isinstance(input_policy, Mapping) or (
        input_policy.get("external_target_artifacts_opened") is not False
        or input_policy.get("wan_preview_opened") is not False
        or input_policy.get("generated_target_opened") is not False
        or input_policy.get("legacy_latent_or_receipt_opened") is not False
        or input_policy.get("anchor_image_opened") is not False
        or input_policy.get("code_frozen_v16_v17_manifest_pins_required") is not True
        or input_policy.get("code_frozen_v16_v17_manifest_pins_matched") is not True
        or input_policy.get("custom_manifest_test_path") is not False
        or input_policy.get("custom_manifest_path_publication_eligible") is not False
    ):
        raise GraftALiteConsumerError("producer input policy differs")
    producer_inputs = producer.get("inputs")
    if not isinstance(producer_inputs, list) or len(producer_inputs) != 2:
        raise GraftALiteConsumerError("producer input descriptors differ")
    if producer.get("input_binding_digest") != object_sha256(producer_inputs):
        raise GraftALiteConsumerError("producer input binding digest differs")
    inputs_by_cohort: dict[str, Mapping[str, Any]] = {}
    for descriptor in producer_inputs:
        if not isinstance(descriptor, Mapping):
            raise GraftALiteConsumerError("producer input descriptor differs")
        cohort = descriptor.get("cohort")
        if cohort not in {V16_COHORT, V17_COHORT} or cohort in inputs_by_cohort:
            raise GraftALiteConsumerError("producer input cohort differs")
        if not isinstance(descriptor.get("path"), str) or not Path(
            descriptor["path"]
        ).is_absolute():
            raise GraftALiteConsumerError("producer input path differs")
        _require_sha256(
            descriptor.get("row_binding_digest"),
            label=f"producer {cohort} row binding",
        )
        inputs_by_cohort[str(cohort)] = descriptor
    if (
        inputs_by_cohort[V16_COHORT].get("rows") != V16_ROWS
        or inputs_by_cohort[V16_COHORT].get("file_sha256")
        != V16_MANIFEST_SHA256
        or inputs_by_cohort[V17_COHORT].get("rows") != V17_ROWS
        or inputs_by_cohort[V17_COHORT].get("file_sha256")
        != V17_MANIFEST_SHA256
    ):
        raise GraftALiteConsumerError("producer frozen input pins differ")
    v17 = inputs_by_cohort[V17_COHORT]
    for index, row in enumerate(rows):
        upstream = row.get("upstream_candidate")
        if not isinstance(upstream, Mapping) or (
            row.get("source_cohort") != V17_COHORT
            or upstream.get("cohort") != V17_COHORT
            or upstream.get("manifest_path") != v17.get("path")
            or upstream.get("manifest_sha256") != V17_MANIFEST_SHA256
            or upstream.get("row_schema_version")
            != "motive-goku-action-anchor-prefilter-v1"
        ):
            raise GraftALiteConsumerError(
                f"producer row {index} upstream binding differs"
            )
        _require_positive_int(
            upstream.get("line_number"), label=f"producer row {index} line"
        )
        _require_sha256(
            upstream.get("row_bytes_sha256"),
            label=f"producer row {index} upstream bytes",
        )
        _require_sha256(
            upstream.get("row_canonical_sha256"),
            label=f"producer row {index} upstream canonical",
        )
    selection = producer.get("selection")
    if not isinstance(selection, Mapping) or (
        selection.get("selected_rows") != 4
        or selection.get("selected_iid_digest")
        != object_sha256([expected[0] for expected in CANARY4])
        or selection.get("order") != "preregistered_core4_fit_then_confirmation"
        or selection.get("v16_v17_iid_unique") is not True
        or selection.get("v16_v17_source_path_unique") is not True
        or selection.get("v16_v17_source_sha256_unique") is not True
    ):
        raise GraftALiteConsumerError("producer selection differs")
    split = producer.get("split")
    train_iids = [expected[0] for expected in CANARY4[:2]]
    confirmation_iids = [expected[0] for expected in CANARY4[2:]]
    if not isinstance(split, Mapping) or (
        split.get("optimizer_train_rows") != 2
        or split.get("optimizer_confirmation_rows") != 2
        or split.get("optimizer_train_iid_digest") != object_sha256(train_iids)
        or split.get("optimizer_confirmation_iid_digest")
        != object_sha256(confirmation_iids)
        or split.get("iid_sets_disjoint") is not True
        or split.get("optimizer_confirmation_update_intended") is not False
        or split.get("optimizer_confirmation_update_authorized") is not False
        or split.get("optimizer_confirmation_actual_use_claimed") is not False
        or split.get("global_holdout") is not False
    ):
        raise GraftALiteConsumerError("producer split contract differs")
    media = producer.get("media_contract")
    if not isinstance(media, Mapping) or (
        media.get("probe_kind") != PORTABLE_FFPROBE_PROBE_KIND
        or media.get("fresh_ffprobe") is not True
        or media.get("fresh_ffprobe_verified_rows") != 4
        or media.get("source_sha256_verified_rows") != 4
        or media.get("source_open_once_rows") != 4
        or media.get("hash_and_probe_same_open_fd") is not True
        or media.get("source_sha256_recomputed_before_and_after_probe") is not True
        or media.get("source_pre_post_probe_sha256_matched_rows") != 4
        or media.get("frame_count") != FRAME_COUNT
        or media.get("fps_fraction") != "25/1"
        or media.get("short_side") != SHORT_SIDE
        or media.get("temporal_padding_allowed") is not False
        or media.get("temporal_truncation_allowed") is not False
        or media.get("retiming_allowed") is not False
    ):
        raise GraftALiteConsumerError("producer media contract differs")
    requirements = producer.get("training_consumer_requirements")
    required_true = (
        "must_revalidate_source_video_sha256",
        "must_fresh_probe_frame_count_fps_and_resolution",
        "must_hash_and_probe_same_open_source_fd",
        "must_recompute_source_sha256_before_and_after_probe",
        "must_bind_source_identity_including_ctime_ns",
        "must_revalidate_source_path_inode_and_parent_binding",
        "must_verify_preregistered_ffprobe_pin_in_sealed_runtime",
        "must_supply_independent_training_execution_receipt",
        "must_record_revalidation_in_training_execution_receipt",
        "must_reject_optimizer_confirmation_rows_for_updates",
        "actual_split_use_must_be_recorded_in_training_execution_receipt",
    )
    if not isinstance(requirements, Mapping) or any(
        requirements.get(field) is not True for field in required_true
    ):
        raise GraftALiteConsumerError("producer consumer requirements differ")
    implementation = producer.get("implementation")
    if not isinstance(implementation, Mapping) or (
        implementation.get("media_probe_kind") != PORTABLE_FFPROBE_PROBE_KIND
        or implementation.get("formal_runtime_authority_claimed") is not False
        or implementation.get("independent_execution_receipt_verified_by_this_receipt")
        is not False
    ):
        raise GraftALiteConsumerError("producer implementation boundary differs")
    probe = implementation.get("ffprobe_executable_observation")
    if not isinstance(probe, Mapping) or (
        probe.get("pin_label") != PORTABLE_FFPROBE_PIN_LABEL
        or probe.get("configured_path") != PORTABLE_FFPROBE_PATH
        or probe.get("resolved_path") != PORTABLE_FFPROBE_PATH
        or probe.get("exact_realpath_matched") is not True
        or probe.get("path_lookup_used") is not False
        or probe.get("file_sha256_expected") != PORTABLE_FFPROBE_SHA256
        or probe.get("file_sha256_observed") != PORTABLE_FFPROBE_SHA256
        or probe.get("file_sha256_matched") is not True
        or probe.get("version_stdout_sha256_expected")
        != PORTABLE_FFPROBE_VERSION_STDOUT_SHA256
        or probe.get("version_stdout_sha256_observed")
        != PORTABLE_FFPROBE_VERSION_STDOUT_SHA256
        or probe.get("version_stdout_sha256_matched") is not True
        or probe.get("version_first_line_expected")
        != PORTABLE_FFPROBE_VERSION_FIRST_LINE
        or probe.get("version_first_line_observed")
        != PORTABLE_FFPROBE_VERSION_FIRST_LINE
        or probe.get("version_first_line_matched") is not True
        or probe.get("pre_and_post_version_identity_and_file_sha_revalidated")
        is not True
        or probe.get("caller_process_observation_only") is not True
        or probe.get("trusted_or_official_authority_claimed") is not False
    ):
        raise GraftALiteConsumerError("producer portable ffprobe provenance differs")
    return digest


def _artifact_record_matches(
    record: Any,
    *,
    opened: _OpenedFile,
    expected_access: str,
    expected_leaf_name: str,
    label: str,
) -> None:
    if not isinstance(record, Mapping):
        raise GraftALiteConsumerError(f"{label} artifact binding differs")

    # ``st_dev`` is allocated by the observing mount namespace.  The same
    # Lustre inode can therefore have a different device number on the
    # producer node and on a later consumer node.  Keep the producer value as
    # authenticated provenance, but do not mistake it for a cross-node file
    # identity.  The current-process identity remains fail-closed through the
    # retained O_NOFOLLOW descriptor/path checks, while the externally pinned
    # receipt is bound to the stable inode, bytes, size, mode and leaf name.
    _require_positive_int(
        record.get("device"), label=f"{label} producer-namespace device"
    )
    if (
        record.get("leaf_name") != expected_leaf_name
        or record.get("sha256") != opened.sha256
        or record.get("size_bytes") != len(opened.raw)
        or record.get("mode") != "0444"
        or record.get("access") != expected_access
        or record.get("inode") != opened.identity[1]
    ):
        raise GraftALiteConsumerError(f"{label} artifact binding differs")


def _validate_execution(
    execution: Mapping[str, Any],
    *,
    stem: Path,
    manifest_file: _OpenedFile,
    producer_file: _OpenedFile,
    producer_digest: str,
    producer: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> str:
    digest = _verify_self_digest(execution, label="execution receipt")
    if (
        execution.get("schema_version") != EXECUTION_RECEIPT_SCHEMA
        or execution.get("status") != "complete"
        or execution.get("successful_return") is not True
        or execution.get("builder_successful_return") is not True
        or execution.get("builder_publication_reopened_and_verified") is not True
        or execution.get("formal_runtime_authority") is not False
    ):
        raise GraftALiteConsumerError("execution receipt success boundary differs")
    slurm = execution.get("slurm")
    if not isinstance(slurm, Mapping) or (
        slurm.get("job_id") != EXPECTED_JOB_ID
        or slurm.get("cpu_only_workload") is not True
        or slurm.get("gpu_resource_requested_by_launcher") is not True
        or slurm.get("gpu_computation_used") is not False
        or slurm.get("effective_submission_request_verified") is not False
    ):
        raise GraftALiteConsumerError("execution receipt Slurm observation differs")
    outputs = execution.get("outputs")
    if not isinstance(outputs, Mapping) or (
        outputs.get("logical_output_stem") != str(stem)
        or outputs.get("manifest_rows") != 4
        or outputs.get("release_mode") != "canary4"
        or outputs.get("producer_receipt_digest") != producer_digest
        or outputs.get("canonical_json_and_digests_verified") is not True
    ):
        raise GraftALiteConsumerError("execution output binding differs")
    _artifact_record_matches(
        outputs.get("manifest"),
        opened=manifest_file,
        expected_access="retained_output_parent_fd_openat",
        expected_leaf_name=manifest_file.path.name,
        label="execution manifest",
    )
    _artifact_record_matches(
        outputs.get("producer_receipt"),
        opened=producer_file,
        expected_access="retained_output_parent_fd_openat",
        expected_leaf_name=producer_file.path.name,
        label="execution producer receipt",
    )
    inputs = execution.get("inputs")
    selected = inputs.get("selected_source_videos") if isinstance(inputs, Mapping) else None
    if not isinstance(inputs, Mapping) or not isinstance(selected, list) or len(selected) != 4:
        raise GraftALiteConsumerError("execution selected source bindings differ")
    for index, (record, row) in enumerate(zip(selected, rows)):
        if not isinstance(record, Mapping) or (
            record.get("iid") != row.get("iid")
            or record.get("path") != row.get("source_video_path")
            or record.get("resolved_path") != row.get("source_video_path")
            or record.get("sha256") != row.get("source_video_sha256")
            or record.get("size_bytes") != row.get("source_file_size_bytes")
        ):
            raise GraftALiteConsumerError(
                f"execution source binding {index} differs"
            )
    if (
        inputs.get("all_selected_source_sha256_recomputed_after_publication") is not True
        or inputs.get("target_video_opened") is not False
        or inputs.get("wan_preview_opened") is not False
        or inputs.get("anchor_image_opened") is not False
        or inputs.get("legacy_latent_or_receipt_opened") is not False
    ):
        raise GraftALiteConsumerError("execution input boundary differs")
    producer_inputs = producer.get("inputs")
    if not isinstance(producer_inputs, list):
        raise GraftALiteConsumerError("producer input descriptors differ")
    producer_by_cohort = {
        item.get("cohort"): item
        for item in producer_inputs
        if isinstance(item, Mapping)
    }
    for key, cohort in (
        ("v16_candidates", V16_COHORT),
        ("v17_candidates", V17_COHORT),
    ):
        record = inputs.get(key)
        descriptor = producer_by_cohort.get(cohort)
        if not isinstance(record, Mapping) or not isinstance(descriptor, Mapping) or (
            record.get("path") != descriptor.get("path")
            or record.get("resolved_path") != descriptor.get("path")
            or record.get("sha256") != descriptor.get("file_sha256")
        ):
            raise GraftALiteConsumerError(
                f"execution {cohort} candidate binding differs"
            )
    split = execution.get("split_execution")
    if not isinstance(split, Mapping) or (
        split.get("optimizer_train_rows") != 2
        or split.get("optimizer_confirmation_rows") != 2
        or split.get("optimizer_update_performed") is not False
        or split.get("optimizer_confirmation_update_performed") is not False
        or split.get("optimizer_confirmation_update_authorized") is not False
        or split.get("global_holdout_claimed") is not False
    ):
        raise GraftALiteConsumerError("execution split evidence differs")
    expected_probe = execution.get("runtime_observations", {}).get(
        "builder_ffprobe_expected_contract"
    ) if isinstance(execution.get("runtime_observations"), Mapping) else None
    if not isinstance(expected_probe, Mapping) or (
        expected_probe.get("media_probe_kind") != PORTABLE_FFPROBE_PROBE_KIND
        or expected_probe.get("pin_label") != PORTABLE_FFPROBE_PIN_LABEL
        or expected_probe.get("configured_and_resolved_path") != PORTABLE_FFPROBE_PATH
        or expected_probe.get("file_sha256") != PORTABLE_FFPROBE_SHA256
        or expected_probe.get("version_stdout_sha256")
        != PORTABLE_FFPROBE_VERSION_STDOUT_SHA256
        or expected_probe.get("version_first_line")
        != PORTABLE_FFPROBE_VERSION_FIRST_LINE
        or expected_probe.get(
            "shared_portable_compute_verified_label_is_provenance_not_runtime_authority"
        ) is not True
    ):
        raise GraftALiteConsumerError("execution ffprobe contract differs")
    runtime = execution.get("runtime_observations")
    builder_runtime = runtime.get("builder") if isinstance(runtime, Mapping) else None
    producer_implementation = producer.get("implementation")
    if not isinstance(builder_runtime, Mapping) or not isinstance(
        producer_implementation, Mapping
    ) or (
        builder_runtime.get("sha256_observed_and_matched")
        != producer_implementation.get("sha256")
        or builder_runtime.get("archive_member")
        != "build_graft_a_lite_source_release_v1.py"
        or builder_runtime.get("archive_exactly_one_plain_member") is not True
        or builder_runtime.get("compiled_from_exact_in_memory_archive_member_bytes")
        is not True
        or builder_runtime.get("executed_or_imported_from_builder_path") is not False
        or builder_runtime.get("live_repository_imported") is not False
    ):
        raise GraftALiteConsumerError("execution builder provenance differs")
    _require_hex40(
        builder_runtime.get("git_commit_observed_pin"),
        label="execution builder Git commit",
    )
    _require_hex40(
        builder_runtime.get("git_blob_sha1_observed_and_matched"),
        label="execution builder Git blob",
    )
    _all_authority_false(execution.get("authority"), label="execution authority")
    failure = execution.get("failure_semantics")
    if not isinstance(failure, Mapping) or (
        failure.get("consumer_must_also_require_slurm_completed_exit_zero") is not True
        or failure.get("consumer_must_require_this_valid_execution_receipt") is not True
        or failure.get("receipt_alone_proves_successful_process_return") is not False
    ):
        raise GraftALiteConsumerError("execution failure semantics differ")
    return digest


def _validate_submission(
    submission: Mapping[str, Any], *, stem: Path
) -> str:
    digest = _verify_self_digest(submission, label="submission receipt")
    if (
        submission.get("schema_version") != SUBMISSION_RECEIPT_SCHEMA
        or submission.get("status") != "submitted"
        or submission.get("submission_success") is not True
        or submission.get("job_success") is not None
        or submission.get("job_terminal_state_observed") is not False
        or submission.get("effective_submission_request_verified") is not False
    ):
        raise GraftALiteConsumerError(
            "submission receipt must remain non-terminal and non-authoritative"
        )
    job = submission.get("submitted_job")
    if not isinstance(job, Mapping) or job.get("job_id") != EXPECTED_JOB_ID:
        raise GraftALiteConsumerError("submission job ID differs")
    outputs = submission.get("outputs")
    if not isinstance(outputs, Mapping) or (
        outputs.get("logical_output_stem") != str(stem)
        or outputs.get("submission_receipt_path")
        != str(stem.with_name(f"{stem.name}{SUBMISSION_SUFFIX}"))
        or outputs.get("submission_receipt_create_only") is not True
        or outputs.get("submission_receipt_mode") != "0444"
    ):
        raise GraftALiteConsumerError("submission output binding differs")
    observations = submission.get("export_contract", {}).get(
        "exported_value_observations"
    ) if isinstance(submission.get("export_contract"), Mapping) else None
    if not isinstance(observations, list):
        raise GraftALiteConsumerError("submission export observations differ")
    output_observations = [
        row
        for row in observations
        if isinstance(row, Mapping)
        and row.get("name") == "GRAFT_A_LITE_OUTPUT_STEM"
    ]
    encoded_stem = str(stem).encode("utf-8")
    if len(output_observations) != 1 or (
        output_observations[0].get("value_sha256") != bytes_sha256(encoded_stem)
        or output_observations[0].get("value_size_bytes") != len(encoded_stem)
    ):
        raise GraftALiteConsumerError("submission output-stem export differs")
    _all_authority_false(
        submission.get("authority"),
        label="submission authority",
        required_fields=tuple(
            field
            for field in _AUTHORITY_FIELDS
            if field != "cross_clip_identity_authority"
        ),
    )
    failure = submission.get("failure_semantics")
    if not isinstance(failure, Mapping) or (
        failure.get("submission_success_is_not_job_success") is not True
        or failure.get("job_success_requires_terminal_scheduler_and_execution_receipts")
        is not True
    ):
        raise GraftALiteConsumerError("submission failure semantics differ")
    return digest


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise GraftALiteConsumerError(f"{label} keys differ")


def _validate_terminal_admission(
    raw: bytes,
    *,
    expected_sha256: str,
    manifest_file_sha256: str,
    producer_file_sha256: str,
    producer_digest: str,
    execution_file_sha256: str,
    execution_digest: str,
    submission_file_sha256: str,
    submission_digest: str,
    expected_materializer_implementation_sha256: str,
    expected_materializer_runtime_sha256: str,
) -> tuple[str, str, str, str, str]:
    if not isinstance(raw, bytes):
        raise GraftALiteConsumerError("terminal admission evidence must be bytes")
    expected_sha256 = _require_sha256(
        expected_sha256, label="terminal admission external pin"
    )
    if bytes_sha256(raw) != expected_sha256:
        raise GraftALiteConsumerError("terminal admission SHA-256 differs from pin")
    admission = _parse_canonical_object(raw, label="terminal admission")
    _require_exact_keys(
        admission,
        {
            "schema_version",
            "status",
            "materializer",
            "sacct_admission",
            "artifact_bindings",
            "authority",
            "receipt_digest",
        },
        label="terminal admission",
    )
    receipt_digest = _verify_self_digest(admission, label="terminal admission")
    if (
        admission.get("schema_version") != TERMINAL_ADMISSION_SCHEMA
        or admission.get("status") != "admitted"
    ):
        raise GraftALiteConsumerError("terminal admission identity differs")
    materializer = admission.get("materializer")
    if not isinstance(materializer, Mapping):
        raise GraftALiteConsumerError("terminal admission materializer differs")
    _require_exact_keys(
        materializer,
        {
            "schema_version",
            "implementation_sha256",
            "runtime_sha256",
            "independent_of_submitted_job_process",
            "job_process_wrote_this_receipt",
            "observed_after_job_became_terminal",
        },
        label="terminal materializer",
    )
    implementation_sha = _require_sha256(
        materializer.get("implementation_sha256"),
        label="terminal materializer implementation",
    )
    runtime_sha = _require_sha256(
        materializer.get("runtime_sha256"),
        label="terminal materializer runtime",
    )
    if (
        materializer.get("schema_version") != TERMINAL_MATERIALIZER_SCHEMA
        or implementation_sha != expected_materializer_implementation_sha256
        or runtime_sha != expected_materializer_runtime_sha256
        or materializer.get("independent_of_submitted_job_process") is not True
        or materializer.get("job_process_wrote_this_receipt") is not False
        or materializer.get("observed_after_job_became_terminal") is not True
    ):
        raise GraftALiteConsumerError("terminal admission is not independently materialized")
    sacct = admission.get("sacct_admission")
    if not isinstance(sacct, Mapping):
        raise GraftALiteConsumerError("terminal sacct admission differs")
    _require_exact_keys(
        sacct,
        {
            "source",
            "queried_fields",
            "job_id",
            "state",
            "exit_code",
            "terminal_state_observed",
            "job_success",
            "raw_stdout_sha256",
            "raw_stdout_size_bytes",
            "selected_record_sha256",
        },
        label="terminal sacct admission",
    )
    if (
        sacct.get("source") != "sacct"
        or sacct.get("queried_fields") != ["JobIDRaw", "State", "ExitCode"]
        or sacct.get("job_id") != EXPECTED_JOB_ID
        or sacct.get("state") != "COMPLETED"
        or sacct.get("exit_code") != "0:0"
        or sacct.get("terminal_state_observed") is not True
        or sacct.get("job_success") is not True
    ):
        raise GraftALiteConsumerError("terminal status is not COMPLETED 0:0")
    _require_sha256(
        sacct.get("raw_stdout_sha256"), label="sacct raw stdout SHA-256"
    )
    _require_positive_int(
        sacct.get("raw_stdout_size_bytes"), label="sacct raw stdout size"
    )
    _require_sha256(
        sacct.get("selected_record_sha256"), label="sacct selected record SHA-256"
    )
    bindings = admission.get("artifact_bindings")
    if not isinstance(bindings, Mapping):
        raise GraftALiteConsumerError("terminal artifact bindings differ")
    expected_bindings = {
        "manifest_file_sha256": manifest_file_sha256,
        "producer_receipt_file_sha256": producer_file_sha256,
        "producer_receipt_digest": producer_digest,
        "execution_receipt_file_sha256": execution_file_sha256,
        "execution_receipt_digest": execution_digest,
        "submission_receipt_file_sha256": submission_file_sha256,
        "submission_receipt_digest": submission_digest,
    }
    _require_exact_keys(bindings, set(expected_bindings), label="terminal bindings")
    if dict(bindings) != expected_bindings:
        raise GraftALiteConsumerError("terminal admission does not bind all artifacts")
    _all_authority_false(admission.get("authority"), label="terminal authority")
    return receipt_digest, "COMPLETED", "0:0", implementation_sha, runtime_sha


def _derive_stem(
    manifest: Path, producer: Path, execution: Path, submission: Path
) -> Path:
    manifest_text = str(manifest)
    if not manifest_text.endswith(MANIFEST_SUFFIX):
        raise GraftALiteConsumerError("manifest path suffix differs")
    stem = Path(manifest_text[: -len(MANIFEST_SUFFIX)])
    expected = (
        stem.with_name(f"{stem.name}{PRODUCER_SUFFIX}"),
        stem.with_name(f"{stem.name}{EXECUTION_SUFFIX}"),
        stem.with_name(f"{stem.name}{SUBMISSION_SUFFIX}"),
    )
    if (producer, execution, submission) != expected:
        raise GraftALiteConsumerError("release artifacts are not one sibling stem")
    return stem


def _subprocess_environment() -> dict[str, str]:
    return {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"}


def _fd_transport(fd: int, *, label: str) -> tuple[str, str]:
    for root, transport in (
        (Path("/proc/self/fd"), "linux_proc_self_fd"),
        (Path("/dev/fd"), "portable_dev_fd"),
    ):
        candidate = root / str(fd)
        try:
            observed = candidate.stat()
            opened = os.fstat(fd)
        except OSError:
            continue
        if (observed.st_dev, observed.st_ino) == (opened.st_dev, opened.st_ino):
            return str(candidate), transport
    raise GraftALiteConsumerError(f"{label} has no verified inherited-fd transport")


def _open_frozen_ffprobe() -> _OpenedFFprobe:
    path = _exact_absolute_path(PORTABLE_FFPROBE_PATH, label="portable ffprobe")
    before = path.lstat()
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o555
        or before.st_nlink != 1
    ):
        raise GraftALiteConsumerError(
            "portable ffprobe must be regular mode-0555 link-count-one"
        )
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        identity = _identity(os.fstat(fd))
        if identity != _identity(before):
            raise GraftALiteConsumerError("portable ffprobe changed while opening")
        raw = _read_fd_all(fd, label="portable ffprobe")
        if bytes_sha256(raw) != PORTABLE_FFPROBE_SHA256:
            raise GraftALiteConsumerError("portable ffprobe SHA-256 differs")
        execution_path, transport = _fd_transport(fd, label="portable ffprobe")
        completed = subprocess.run(
            [execution_path, "-version"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=60,
            pass_fds=(fd,),
            env=_subprocess_environment(),
        )
        if completed.returncode != 0:
            raise GraftALiteConsumerError("portable ffprobe version command failed")
        try:
            first_line = completed.stdout.decode("utf-8").splitlines()[0]
        except (UnicodeDecodeError, IndexError) as error:
            raise GraftALiteConsumerError("portable ffprobe version differs") from error
        if (
            bytes_sha256(completed.stdout)
            != PORTABLE_FFPROBE_VERSION_STDOUT_SHA256
            or first_line != PORTABLE_FFPROBE_VERSION_FIRST_LINE
            or _identity(os.fstat(fd)) != identity
            or _read_fd_all(fd, label="portable ffprobe post-version") != raw
            or _identity(path.lstat()) != identity
        ):
            raise GraftALiteConsumerError("portable ffprobe pin changed across version")
        return _OpenedFFprobe(
            path=path,
            fd=fd,
            identity=identity,
            execution_path=execution_path,
            transport=transport,
        )
    except Exception:
        os.close(fd)
        raise


def _probe_with_frozen_ffprobe(
    source: _OpenedSource, executable: _OpenedFFprobe
) -> Mapping[str, Any]:
    _revalidate_opened_source(source, label="source before ffprobe")
    source_path, source_transport = _fd_transport(source.fd, label="source")
    command = [
        executable.execution_path,
        "-v",
        "error",
        "-select_streams",
        "v",
        "-count_frames",
        "-show_entries",
        "stream=index,width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames",
        "-of",
        "json",
        source_path,
    ]
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=180,
        pass_fds=(source.fd, executable.fd),
        env=_subprocess_environment(),
    )
    if completed.returncode != 0:
        raise GraftALiteConsumerError(
            f"portable ffprobe rejected {source.path} with exit {completed.returncode}"
        )
    try:
        value = json.loads(
            completed.stdout.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GraftALiteConsumerError("portable ffprobe JSON differs") from error
    streams = value.get("streams") if isinstance(value, Mapping) else None
    if not isinstance(streams, list) or len(streams) != 1 or not isinstance(streams[0], Mapping):
        raise GraftALiteConsumerError("source must have exactly one video stream")
    stream = streams[0]
    frame_text = stream.get("nb_read_frames")
    if not isinstance(frame_text, str) or not frame_text.isdigit():
        raise GraftALiteConsumerError("portable ffprobe frame count differs")

    def fraction(value: Any, label: str) -> Fraction:
        if not isinstance(value, str):
            raise GraftALiteConsumerError(f"{label} differs")
        try:
            result = Fraction(value)
        except (ValueError, ZeroDivisionError) as error:
            raise GraftALiteConsumerError(f"{label} differs") from error
        if result <= 0:
            raise GraftALiteConsumerError(f"{label} differs")
        return result

    average = fraction(stream.get("avg_frame_rate"), "average fps")
    reported = fraction(stream.get("r_frame_rate"), "reported fps")
    _revalidate_opened_source(source, label="source after ffprobe")
    if (
        _identity(os.fstat(executable.fd)) != executable.identity
        or _identity(executable.path.lstat()) != executable.identity
        or bytes_sha256(_read_fd_all(executable.fd, label="portable ffprobe post-probe"))
        != PORTABLE_FFPROBE_SHA256
    ):
        raise GraftALiteConsumerError("portable ffprobe changed during source probe")
    return {
        "frame_count": int(frame_text),
        "fps_numerator": average.numerator,
        "fps_denominator": average.denominator,
        "reported_fps_numerator": reported.numerator,
        "reported_fps_denominator": reported.denominator,
        "width": stream.get("width"),
        "height": stream.get("height"),
        "source_fd_transport": source_transport,
        "executable_fd_transport": executable.transport,
    }


def _normalize_probe(value: Mapping[str, Any], *, label: str) -> OwnedSourceMedia:
    if not isinstance(value, Mapping):
        raise GraftALiteConsumerError(f"{label} media probe differs")
    frame_count = _require_positive_int(value.get("frame_count"), label=f"{label} frames")
    fps_num = _require_positive_int(value.get("fps_numerator"), label=f"{label} fps numerator")
    fps_den = _require_positive_int(value.get("fps_denominator"), label=f"{label} fps denominator")
    reported_num = _require_positive_int(
        value.get("reported_fps_numerator"), label=f"{label} reported fps numerator"
    )
    reported_den = _require_positive_int(
        value.get("reported_fps_denominator"), label=f"{label} reported fps denominator"
    )
    width = _require_positive_int(value.get("width"), label=f"{label} width")
    height = _require_positive_int(value.get("height"), label=f"{label} height")
    if (
        frame_count != FRAME_COUNT
        or Fraction(fps_num, fps_den) != Fraction(FPS_NUMERATOR, FPS_DENOMINATOR)
        or Fraction(reported_num, reported_den)
        != Fraction(FPS_NUMERATOR, FPS_DENOMINATOR)
        or min(width, height) != SHORT_SIDE
    ):
        raise GraftALiteConsumerError(f"{label} is not fresh exact81/25/short704")
    return OwnedSourceMedia(
        frame_count=frame_count,
        fps_numerator=FPS_NUMERATOR,
        fps_denominator=FPS_DENOMINATOR,
        width=width,
        height=height,
    )


def _pinset_record(pins: ReleaseArtifactPins) -> dict[str, str]:
    return {
        name: getattr(pins, name)
        for name in pins.__dataclass_fields__
    }


def _authority_record(authority: AuthorityBoundary) -> dict[str, bool]:
    return {field: getattr(authority, field) for field in _AUTHORITY_FIELDS}


def _media_record(media: OwnedSourceMedia) -> dict[str, Any]:
    return {
        "frame_count": media.frame_count,
        "fps_numerator": media.fps_numerator,
        "fps_denominator": media.fps_denominator,
        "width": media.width,
        "height": media.height,
        "rgb_or_codec_content_not_interpreted": (
            media.rgb_or_codec_content_not_interpreted
        ),
    }


def _owned_row_record(row: OwnedValidatedSourceRow) -> dict[str, Any]:
    return {
        "row_index": row.row_index,
        "iid": row.iid,
        "split": row.split,
        "optimizer_update_allowed": row.optimizer_update_allowed,
        "optimizer_confirmation_only": row.optimizer_confirmation_only,
        "source_path_observed": row.source_path_observed,
        "source_sha256": row.source_sha256,
        "source_size_bytes": row.source_size_bytes,
        "owned_source_bytes_sha256": bytes_sha256(row.source_bytes),
        "owned_source_bytes_size": len(row.source_bytes),
        "media": _media_record(row.media),
        "noop_instruction": row.noop_instruction,
        "producer_row_digest": row.producer_row_digest,
        "source_cohort": row.source_cohort,
        "upstream_manifest_sha256": row.upstream_manifest_sha256,
        "authority": {
            "action_authority": row.action_authority,
            "identity_authority": row.identity_authority,
            "cross_clip_identity_authority": row.cross_clip_identity_authority,
            "quality_authority": row.quality_authority,
            "training_authority": row.training_authority,
            "production_authority": row.production_authority,
        },
    }


def _artifact_record(artifact: ArtifactProvenance) -> dict[str, Any]:
    return {
        "path": artifact.path,
        "sha256": artifact.sha256,
        "size_bytes": artifact.size_bytes,
        "mode": artifact.mode,
        "regular_file": artifact.regular_file,
        "link_count_one": artifact.link_count_one,
        "opened_o_nofollow": artifact.opened_o_nofollow,
        "retained_through_validation": artifact.retained_through_validation,
    }


def _provenance_record(provenance: ReleaseProvenance) -> dict[str, Any]:
    return {
        "job_id": provenance.job_id,
        "scheduler_state": provenance.scheduler_state,
        "scheduler_exit_code": provenance.scheduler_exit_code,
        "terminal_admission_sha256": provenance.terminal_admission_sha256,
        "terminal_admission_digest": provenance.terminal_admission_digest,
        "terminal_materializer_implementation_sha256": (
            provenance.terminal_materializer_implementation_sha256
        ),
        "terminal_materializer_runtime_sha256": (
            provenance.terminal_materializer_runtime_sha256
        ),
        "producer_receipt_digest": provenance.producer_receipt_digest,
        "execution_receipt_digest": provenance.execution_receipt_digest,
        "submission_receipt_digest": provenance.submission_receipt_digest,
        "manifest": _artifact_record(provenance.manifest),
        "producer_receipt": _artifact_record(provenance.producer_receipt),
        "execution_receipt": _artifact_record(provenance.execution_receipt),
        "submission_receipt": _artifact_record(provenance.submission_receipt),
        "portable_ffprobe_path": provenance.portable_ffprobe_path,
        "portable_ffprobe_sha256": provenance.portable_ffprobe_sha256,
        "portable_ffprobe_version_stdout_sha256": (
            provenance.portable_ffprobe_version_stdout_sha256
        ),
        "portable_ffprobe_probe_kind": provenance.portable_ffprobe_probe_kind,
        "consumer_fresh_portable_ffprobe_verified": (
            provenance.consumer_fresh_portable_ffprobe_verified
        ),
        "sources_hash_and_probe_same_fd": provenance.sources_hash_and_probe_same_fd,
        "sources_rehashed_after_probe": provenance.sources_rehashed_after_probe,
        "sources_returned_as_owned_bytes": provenance.sources_returned_as_owned_bytes,
    }


def _release_result_record(
    *,
    rows: tuple[OwnedValidatedSourceRow, ...],
    provenance: ReleaseProvenance,
    authority: AuthorityBoundary,
    pins: ReleaseArtifactPins,
) -> dict[str, Any]:
    return {
        "schema_version": "bernini-graft-a-lite-source-consumer-result-v2",
        "rows": [_owned_row_record(row) for row in rows],
        "provenance": _provenance_record(provenance),
        "authority": _authority_record(authority),
        "pinset": _pinset_record(pins),
        "pinset_digest": object_sha256(_pinset_record(pins)),
        "trainer_routing_included": False,
        "validate_for_training_required": True,
    }


def _consume_probe_neutral(
    *,
    manifest_path: str | Path,
    producer_receipt_path: str | Path,
    execution_receipt_path: str | Path,
    submission_receipt_path: str | Path,
    terminal_admission_bytes: bytes,
    pins: ReleaseArtifactPins,
    media_probe: MediaProbe,
) -> _CoreConsumedEvidence:
    """Shared validation core; its injectable probe result is never mintable."""
    if not isinstance(pins, ReleaseArtifactPins):
        raise GraftALiteConsumerError("pins must be ReleaseArtifactPins")
    opened_files: list[tuple[_OpenedFile, str]] = []
    opened_sources: list[_OpenedSource] = []
    try:
        manifest_file = _open_sealed_file(
            manifest_path,
            expected_sha256=pins.manifest_sha256,
            label="manifest",
        )
        opened_files.append((manifest_file, "manifest"))
        producer_file = _open_sealed_file(
            producer_receipt_path,
            expected_sha256=pins.producer_receipt_sha256,
            label="producer receipt",
        )
        opened_files.append((producer_file, "producer receipt"))
        execution_file = _open_sealed_file(
            execution_receipt_path,
            expected_sha256=pins.execution_receipt_sha256,
            label="execution receipt",
        )
        opened_files.append((execution_file, "execution receipt"))
        submission_file = _open_sealed_file(
            submission_receipt_path,
            expected_sha256=pins.submission_receipt_sha256,
            label="submission receipt",
        )
        opened_files.append((submission_file, "submission receipt"))

        stem = _derive_stem(
            manifest_file.path,
            producer_file.path,
            execution_file.path,
            submission_file.path,
        )
        rows = _manifest_rows(manifest_file.raw)
        for index, (row, expected) in enumerate(zip(rows, CANARY4)):
            _validate_row(row, index=index, expected=expected)
        paths = [row["source_video_path"] for row in rows]
        source_hashes = [row["source_video_sha256"] for row in rows]
        if len(set(paths)) != 4 or len(set(source_hashes)) != 4:
            raise GraftALiteConsumerError("canary4 sources must be path/hash unique")

        producer = _parse_canonical_object(
            producer_file.raw, label="producer receipt"
        )
        producer_digest = _validate_producer(
            producer, manifest_raw=manifest_file.raw, rows=rows
        )
        execution = _parse_canonical_object(
            execution_file.raw, label="execution receipt"
        )
        execution_digest = _validate_execution(
            execution,
            stem=stem,
            manifest_file=manifest_file,
            producer_file=producer_file,
            producer_digest=producer_digest,
            producer=producer,
            rows=rows,
        )
        submission = _parse_canonical_object(
            submission_file.raw, label="submission receipt"
        )
        submission_digest = _validate_submission(submission, stem=stem)
        (
            terminal_digest,
            terminal_state,
            terminal_exit,
            terminal_materializer_implementation_sha,
            terminal_materializer_runtime_sha,
        ) = _validate_terminal_admission(
            terminal_admission_bytes,
            expected_sha256=pins.terminal_admission_sha256,
            manifest_file_sha256=manifest_file.sha256,
            producer_file_sha256=producer_file.sha256,
            producer_digest=producer_digest,
            execution_file_sha256=execution_file.sha256,
            execution_digest=execution_digest,
            submission_file_sha256=submission_file.sha256,
            submission_digest=submission_digest,
            expected_materializer_implementation_sha256=(
                pins.terminal_materializer_implementation_sha256
            ),
            expected_materializer_runtime_sha256=(
                pins.terminal_materializer_runtime_sha256
            ),
        )

        owned_rows: list[OwnedValidatedSourceRow] = []
        execution_source_records = execution["inputs"]["selected_source_videos"]
        for index, row in enumerate(rows):
            source = _open_source(
                row["source_video_path"], label=f"source row {index}"
            )
            opened_sources.append(source)
            first_raw = _read_fd_all(source.fd, label=f"source row {index} pre-probe")
            first_digest = bytes_sha256(first_raw)
            execution_source = execution_source_records[index]
            if not isinstance(execution_source, Mapping):
                raise GraftALiteConsumerError(
                    f"source row {index} content/provenance differs"
                )
            _require_positive_int(
                execution_source.get("device"),
                label=f"source row {index} producer-namespace device",
            )
            if (
                first_digest != row["source_video_sha256"]
                or len(first_raw) != row["source_file_size_bytes"]
                or source.identity[2] != row["source_file_size_bytes"]
                or source.identity[3] != row["source_mtime_ns"]
                or source.identity[4] != row["source_ctime_ns_observed"]
                or execution_source.get("inode") != source.identity[1]
                or execution_source.get("mode") != format(source.identity[5], "04o")
            ):
                raise GraftALiteConsumerError(
                    f"source row {index} content/provenance differs"
                )
            observed_media = _normalize_probe(
                media_probe(source), label=f"source row {index}"
            )
            producer_media = row["source_media"]
            if (
                observed_media.width != producer_media["width"]
                or observed_media.height != producer_media["height"]
            ):
                raise GraftALiteConsumerError(
                    f"source row {index} fresh resolution differs"
                )
            second_raw = _read_fd_all(
                source.fd, label=f"source row {index} post-probe"
            )
            _revalidate_opened_source(source, label=f"source row {index}")
            if second_raw != first_raw or bytes_sha256(second_raw) != first_digest:
                raise GraftALiteConsumerError(
                    f"source row {index} changed across probe"
                )
            upstream = row.get("upstream_candidate")
            if not isinstance(upstream, Mapping):
                raise GraftALiteConsumerError(
                    f"source row {index} upstream provenance differs"
                )
            upstream_sha = _require_sha256(
                upstream.get("manifest_sha256"),
                label=f"source row {index} upstream manifest",
            )
            owned_rows.append(
                OwnedValidatedSourceRow(
                    row_index=index,
                    iid=row["iid"],
                    split=row["split"],
                    optimizer_update_allowed=row["optimizer_update_authorized"],
                    optimizer_confirmation_only=row["optimizer_confirmation_only"],
                    source_path_observed=row["source_video_path"],
                    source_sha256=first_digest,
                    source_size_bytes=len(first_raw),
                    source_bytes=bytes(first_raw),
                    media=observed_media,
                    noop_instruction=NOOP_INSTRUCTION,
                    producer_row_digest=row["row_digest"],
                    source_cohort=str(row.get("source_cohort")),
                    upstream_manifest_sha256=upstream_sha,
                )
            )

        # Keep every release/source descriptor alive until all cross-evidence
        # and source observations have completed, then perform one last pass.
        for source_index, source in enumerate(opened_sources):
            _revalidate_opened_source(source, label=f"source row {source_index} terminal")
        for opened, label in opened_files:
            _revalidate_opened_file(opened, label=label)

        immutable_rows = tuple(owned_rows)
        provenance = ReleaseProvenance(
                job_id=EXPECTED_JOB_ID,
                scheduler_state=terminal_state,
                scheduler_exit_code=terminal_exit,
                terminal_admission_sha256=pins.terminal_admission_sha256,
                terminal_admission_digest=terminal_digest,
                terminal_materializer_implementation_sha256=(
                    terminal_materializer_implementation_sha
                ),
                terminal_materializer_runtime_sha256=(
                    terminal_materializer_runtime_sha
                ),
                producer_receipt_digest=producer_digest,
                execution_receipt_digest=execution_digest,
                submission_receipt_digest=submission_digest,
                manifest=ArtifactProvenance(
                    path=str(manifest_file.path),
                    sha256=manifest_file.sha256,
                    size_bytes=len(manifest_file.raw),
                ),
                producer_receipt=ArtifactProvenance(
                    path=str(producer_file.path),
                    sha256=producer_file.sha256,
                    size_bytes=len(producer_file.raw),
                ),
                execution_receipt=ArtifactProvenance(
                    path=str(execution_file.path),
                    sha256=execution_file.sha256,
                    size_bytes=len(execution_file.raw),
                ),
                submission_receipt=ArtifactProvenance(
                    path=str(submission_file.path),
                    sha256=submission_file.sha256,
                    size_bytes=len(submission_file.raw),
                ),
                portable_ffprobe_path=PORTABLE_FFPROBE_PATH,
                portable_ffprobe_sha256=PORTABLE_FFPROBE_SHA256,
                portable_ffprobe_version_stdout_sha256=(
                    PORTABLE_FFPROBE_VERSION_STDOUT_SHA256
                ),
                portable_ffprobe_probe_kind=PORTABLE_FFPROBE_PROBE_KIND,
                consumer_fresh_portable_ffprobe_verified=False,
            )
        train_rows = tuple(
            row for row in immutable_rows if row.optimizer_update_allowed
        )
        confirmation_rows = tuple(
            row for row in immutable_rows if row.optimizer_confirmation_only
        )
        if len(train_rows) != 2 or len(confirmation_rows) != 2:
            raise GraftALiteConsumerError("internal immutable split differs")
        if any(row.optimizer_update_allowed for row in confirmation_rows):
            raise GraftALiteConsumerError("confirmation row was promoted")
        return _CoreConsumedEvidence(
            rows=immutable_rows,
            provenance_without_production_probe_attestation=provenance,
        )
    finally:
        for source in reversed(opened_sources):
            for fd in (source.fd, source.parent_fd):
                try:
                    os.close(fd)
                except OSError:
                    pass
        for opened, _ in reversed(opened_files):
            try:
                os.close(opened.fd)
            except OSError:
                pass


def _revalidate_frozen_ffprobe(executable: _OpenedFFprobe) -> None:
    if type(executable) is not _OpenedFFprobe:
        raise GraftALiteConsumerError("production portable ffprobe type differs")
    if (
        executable.path != Path(PORTABLE_FFPROBE_PATH)
        or executable.transport not in {"linux_proc_self_fd", "portable_dev_fd"}
        or _identity(os.fstat(executable.fd)) != executable.identity
        or _identity(executable.path.lstat()) != executable.identity
        or bytes_sha256(
            _read_fd_all(executable.fd, label="portable ffprobe final mint rehash")
        )
        != PORTABLE_FFPROBE_SHA256
    ):
        raise GraftALiteConsumerError("production portable ffprobe changed")


def _install_production_consumer_boundary() -> tuple[Callable[..., Any], Callable[[Any], bool]]:
    """Keep the production release capability out of module-visible data."""

    production_release_mint = object()

    def production_core(
        *,
        manifest_path: str | Path,
        producer_receipt_path: str | Path,
        execution_receipt_path: str | Path,
        submission_receipt_path: str | Path,
        terminal_admission_bytes: bytes,
        pins: ReleaseArtifactPins,
    ) -> SealedALiteSourceRelease:
        # This boundary owns both opening and using the frozen executable.  It
        # accepts no callback, neutral evidence, or caller-set trust flag.
        executable = _open_frozen_ffprobe()
        try:
            evidence = _consume_probe_neutral(
                manifest_path=manifest_path,
                producer_receipt_path=producer_receipt_path,
                execution_receipt_path=execution_receipt_path,
                submission_receipt_path=submission_receipt_path,
                terminal_admission_bytes=terminal_admission_bytes,
                pins=pins,
                media_probe=lambda source: _probe_with_frozen_ffprobe(
                    source, executable
                ),
            )
            if type(evidence) is not _CoreConsumedEvidence:
                raise GraftALiteConsumerError("production core evidence differs")
            _revalidate_frozen_ffprobe(executable)
            base = evidence.provenance_without_production_probe_attestation
            if base.consumer_fresh_portable_ffprobe_verified is not False:
                raise GraftALiteConsumerError(
                    "neutral core already claims production probe"
                )
            provenance = replace(
                base, consumer_fresh_portable_ffprobe_verified=True
            )
            authority = AuthorityBoundary()
            pinset_digest = object_sha256(_pinset_record(pins))
            result_digest = object_sha256(
                _release_result_record(
                    rows=evidence.rows,
                    provenance=provenance,
                    authority=authority,
                    pins=pins,
                )
            )
            release = object.__new__(SealedALiteSourceRelease)
            object.__setattr__(release, "_rows", evidence.rows)
            object.__setattr__(release, "_provenance", provenance)
            object.__setattr__(release, "_authority", authority)
            object.__setattr__(release, "_pins", pins)
            object.__setattr__(release, "_pinset_digest", pinset_digest)
            object.__setattr__(release, "_result_digest", result_digest)
            object.__setattr__(release, "_mint", production_release_mint)
            object.__setattr__(release, "_locked", True)
            return release
        finally:
            os.close(executable.fd)

    def has_valid_mint(value: Any) -> bool:
        return (
            type(value) is SealedALiteSourceRelease
            and value._mint is production_release_mint
        )

    return production_core, has_valid_mint


_consume_core, _has_valid_production_release_mint = (
    _install_production_consumer_boundary()
)
del _install_production_consumer_boundary


def consume_graft_a_lite_source_release(
    *,
    manifest_path: str | Path,
    producer_receipt_path: str | Path,
    execution_receipt_path: str | Path,
    submission_receipt_path: str | Path,
    terminal_admission_bytes: bytes,
    pins: ReleaseArtifactPins,
) -> SealedALiteSourceRelease:
    """Consume the pinned job-132549 release through the frozen ffprobe only."""

    return _consume_core(
        manifest_path=manifest_path,
        producer_receipt_path=producer_receipt_path,
        execution_receipt_path=execution_receipt_path,
        submission_receipt_path=submission_receipt_path,
        terminal_admission_bytes=terminal_admission_bytes,
        pins=pins,
    )


def _consume_with_test_probe(
    *,
    manifest_path: str | Path,
    producer_receipt_path: str | Path,
    execution_receipt_path: str | Path,
    submission_receipt_path: str | Path,
    terminal_admission_bytes: bytes,
    pins: ReleaseArtifactPins,
    media_probe: MediaProbe,
) -> TestOnlySourceReleaseObservation:
    """Private unit-test seam; it always records that portable probe is absent."""

    evidence = _consume_probe_neutral(
        manifest_path=manifest_path,
        producer_receipt_path=producer_receipt_path,
        execution_receipt_path=execution_receipt_path,
        submission_receipt_path=submission_receipt_path,
        terminal_admission_bytes=terminal_admission_bytes,
        pins=pins,
        media_probe=media_probe,
    )
    if type(evidence) is not _CoreConsumedEvidence:
        raise RuntimeError("internal test-only core evidence differs")
    return TestOnlySourceReleaseObservation(
        schema_version="bernini-graft-a-lite-source-consumer-test-only-observation-v1",
        rows=evidence.rows,
        provenance=evidence.provenance_without_production_probe_attestation,
        authority=AuthorityBoundary(),
    )


def _validate_false_authority_boundary(authority: AuthorityBoundary) -> None:
    if type(authority) is not AuthorityBoundary or set(vars(authority)) != set(
        _AUTHORITY_FIELDS
    ) or any(getattr(authority, field) is not False for field in _AUTHORITY_FIELDS):
        raise GraftALiteConsumerError("production result authority boundary differs")


def _validated_trainer_blueprint(
    release: SealedALiteSourceRelease,
) -> _TrainerRoutingBlueprint:
    """Revalidate the opaque result and return non-routing validated data.

    This function never opens a path.  Trainer rows contain only owned source
    bytes and immutable routing metadata; the observed source/artifact paths
    remain confined to diagnostic release provenance.
    """

    if type(release) is not SealedALiteSourceRelease:
        raise GraftALiteConsumerError(
            "validate_for_training requires an opaque production release mint"
        )
    if not _has_valid_production_release_mint(release):
        raise GraftALiteConsumerError("production release mint differs")
    pins = release._pins
    if type(pins) is not ReleaseArtifactPins:
        raise GraftALiteConsumerError("production release pinset type differs")
    pinset_digest = object_sha256(_pinset_record(pins))
    if pinset_digest != release._pinset_digest:
        raise GraftALiteConsumerError("production release pinset digest differs")
    _validate_false_authority_boundary(release._authority)
    provenance = release._provenance
    if type(provenance) is not ReleaseProvenance or (
        provenance.job_id != EXPECTED_JOB_ID
        or provenance.scheduler_state != "COMPLETED"
        or provenance.scheduler_exit_code != "0:0"
        or provenance.terminal_admission_sha256
        != pins.terminal_admission_sha256
        or provenance.terminal_materializer_implementation_sha256
        != pins.terminal_materializer_implementation_sha256
        or provenance.terminal_materializer_runtime_sha256
        != pins.terminal_materializer_runtime_sha256
        or provenance.portable_ffprobe_path != PORTABLE_FFPROBE_PATH
        or provenance.portable_ffprobe_sha256 != PORTABLE_FFPROBE_SHA256
        or provenance.portable_ffprobe_version_stdout_sha256
        != PORTABLE_FFPROBE_VERSION_STDOUT_SHA256
        or provenance.portable_ffprobe_probe_kind
        != PORTABLE_FFPROBE_PROBE_KIND
        or provenance.consumer_fresh_portable_ffprobe_verified is not True
        or provenance.sources_hash_and_probe_same_fd is not True
        or provenance.sources_rehashed_after_probe is not True
        or provenance.sources_returned_as_owned_bytes is not True
    ):
        raise GraftALiteConsumerError("production release provenance differs")
    for artifact, expected_sha, label in (
        (provenance.manifest, pins.manifest_sha256, "manifest"),
        (
            provenance.producer_receipt,
            pins.producer_receipt_sha256,
            "producer receipt",
        ),
        (
            provenance.execution_receipt,
            pins.execution_receipt_sha256,
            "execution receipt",
        ),
        (
            provenance.submission_receipt,
            pins.submission_receipt_sha256,
            "submission receipt",
        ),
    ):
        if type(artifact) is not ArtifactProvenance or (
            artifact.sha256 != expected_sha
            or not isinstance(artifact.path, str)
            or not Path(artifact.path).is_absolute()
            or isinstance(artifact.size_bytes, bool)
            or artifact.size_bytes <= 0
            or artifact.mode != "0444"
            or artifact.regular_file is not True
            or artifact.link_count_one is not True
            or artifact.opened_o_nofollow is not True
            or artifact.retained_through_validation is not True
        ):
            raise GraftALiteConsumerError(
                f"production release {label} provenance differs"
            )
    rows = release._rows
    if not isinstance(rows, tuple) or len(rows) != 4:
        raise GraftALiteConsumerError("production release must contain four rows")
    source_hashes: list[str] = []
    for index, (row, expected) in enumerate(zip(rows, CANARY4)):
        iid, split, update_allowed, confirmation_only = expected
        if type(row) is not OwnedValidatedSourceRow or (
            row.row_index != index
            or row.iid != iid
            or row.split != split
            or row.optimizer_update_allowed is not update_allowed
            or row.optimizer_confirmation_only is not confirmation_only
            or not isinstance(row.source_path_observed, str)
            or not Path(row.source_path_observed).is_absolute()
            or type(row.source_bytes) is not bytes
            or row.source_size_bytes != len(row.source_bytes)
            or row.source_sha256 != bytes_sha256(row.source_bytes)
            or row.noop_instruction != NOOP_INSTRUCTION
            or row.source_cohort != V17_COHORT
            or row.upstream_manifest_sha256 != V17_MANIFEST_SHA256
            or row.action_authority is not False
            or row.identity_authority is not False
            or row.cross_clip_identity_authority is not False
            or row.quality_authority is not False
            or row.training_authority is not False
            or row.production_authority is not False
        ):
            raise GraftALiteConsumerError(
                f"production release owned row {index} differs"
            )
        _require_sha256(row.producer_row_digest, label=f"owned row {index} digest")
        media = row.media
        if type(media) is not OwnedSourceMedia or (
            media.frame_count != FRAME_COUNT
            or media.fps_numerator != FPS_NUMERATOR
            or media.fps_denominator != FPS_DENOMINATOR
            or isinstance(media.width, bool)
            or isinstance(media.height, bool)
            or not isinstance(media.width, int)
            or not isinstance(media.height, int)
            or min(media.width, media.height) != SHORT_SIDE
            or media.rgb_or_codec_content_not_interpreted is not True
        ):
            raise GraftALiteConsumerError(
                f"production release owned row {index} media differs"
            )
        source_hashes.append(row.source_sha256)
    if len(set(source_hashes)) != 4:
        raise GraftALiteConsumerError("production release source hashes alias")
    recomputed_result_digest = object_sha256(
        _release_result_record(
            rows=rows,
            provenance=provenance,
            authority=release._authority,
            pins=pins,
        )
    )
    if recomputed_result_digest != release._result_digest:
        raise GraftALiteConsumerError("production release canonical result digest differs")

    update_rows = rows[:2]
    confirmation_rows = rows[2:]
    if any(not row.optimizer_update_allowed for row in update_rows) or any(
        row.optimizer_update_allowed or not row.optimizer_confirmation_only
        for row in confirmation_rows
    ):
        raise GraftALiteConsumerError("trainer split routing differs")
    authority = AuthorityBoundary()
    routing_record = {
        "schema_version": "bernini-graft-a-lite-owned-trainer-routing-v1",
        "source_release_result_digest": release._result_digest,
        "pinset_digest": pinset_digest,
        "rows": [
            {
                "iid": row.iid,
                "split": row.split,
                "optimizer_update_allowed": row.optimizer_update_allowed,
                "optimizer_confirmation_only": row.optimizer_confirmation_only,
                "source_sha256": row.source_sha256,
                "owned_source_bytes_sha256": bytes_sha256(row.source_bytes),
                "owned_source_bytes_size": len(row.source_bytes),
                "media": _media_record(row.media),
                "noop_instruction": row.noop_instruction,
            }
            for row in rows
        ],
        "path_reopen_allowed": False,
        "owned_bytes_only": True,
        "authority": _authority_record(authority),
    }
    routing_digest = object_sha256(routing_record)
    return _TrainerRoutingBlueprint(
        rows=rows,
        source_release_result_digest=release._result_digest,
        pinset_digest=pinset_digest,
        routing_digest=routing_digest,
        authority=authority,
    )


def _install_trainer_routing_boundary() -> Callable[[Any], TrainerRouting]:
    """Hide the only trainer-row/routing construction capability in a closure."""

    trainer_routing_mint = object()

    def validate_for_training(release: SealedALiteSourceRelease) -> TrainerRouting:
        """Mint path-free routing after full immutable release revalidation."""

        blueprint = _validated_trainer_blueprint(release)
        if type(blueprint) is not _TrainerRoutingBlueprint:
            raise GraftALiteConsumerError("trainer routing blueprint differs")
        routed_rows: list[TrainerOwnedSourceRow] = []
        for row in blueprint.rows:
            routed = object.__new__(TrainerOwnedSourceRow)
            object.__setattr__(routed, "_iid", row.iid)
            object.__setattr__(routed, "_split", row.split)
            object.__setattr__(
                routed, "_optimizer_update_allowed", row.optimizer_update_allowed
            )
            object.__setattr__(
                routed,
                "_optimizer_confirmation_only",
                row.optimizer_confirmation_only,
            )
            object.__setattr__(routed, "_source_sha256", row.source_sha256)
            object.__setattr__(routed, "_source_bytes", bytes(row.source_bytes))
            object.__setattr__(routed, "_media", row.media)
            object.__setattr__(routed, "_noop_instruction", row.noop_instruction)
            object.__setattr__(routed, "_mint", trainer_routing_mint)
            routed_rows.append(routed)
        update_rows = tuple(routed_rows[:2])
        confirmation_rows = tuple(routed_rows[2:])
        routing = object.__new__(TrainerRouting)
        object.__setattr__(routing, "_update_rows", update_rows)
        object.__setattr__(routing, "_confirmation_rows", confirmation_rows)
        object.__setattr__(
            routing,
            "_source_release_result_digest",
            blueprint.source_release_result_digest,
        )
        object.__setattr__(routing, "_pinset_digest", blueprint.pinset_digest)
        object.__setattr__(routing, "_routing_digest", blueprint.routing_digest)
        object.__setattr__(routing, "_authority", blueprint.authority)
        object.__setattr__(routing, "_mint", trainer_routing_mint)
        return routing

    return validate_for_training


validate_for_training = _install_trainer_routing_boundary()
del _install_trainer_routing_boundary


__all__ = [
    "AuthorityBoundary",
    "CANARY4",
    "EXPECTED_JOB_ID",
    "GraftALiteConsumerError",
    "OwnedSourceMedia",
    "OwnedValidatedSourceRow",
    "PORTABLE_FFPROBE_PATH",
    "PORTABLE_FFPROBE_PROBE_KIND",
    "PORTABLE_FFPROBE_SHA256",
    "PORTABLE_FFPROBE_VERSION_STDOUT_SHA256",
    "ReleaseArtifactPins",
    "ReleaseProvenance",
    "SealedALiteSourceRelease",
    "TestOnlySourceReleaseObservation",
    "TERMINAL_ADMISSION_SCHEMA",
    "TERMINAL_MATERIALIZER_SCHEMA",
    "TrainerOwnedSourceRow",
    "TrainerRouting",
    "bytes_sha256",
    "canonical_json_bytes",
    "consume_graft_a_lite_source_release",
    "object_sha256",
    "validate_for_training",
]
