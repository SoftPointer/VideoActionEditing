#!/usr/bin/env python3
"""Build a source-only experimental release for GRAFT-Edit Stage A-lite.

The only admissible inputs are the frozen v16/v17 raw candidate JSONL files
and the source MP4 named by each selected row.  Candidate target fields are
not projected, and this builder has no argument through which a Wan preview,
generated target, or an older latent receipt can be supplied.

Stage A-lite is deliberately narrow: the source video is the clean endpoint
of a same-clip no-op reconstruction task.  The release does not assert a
stable identity across clips and carries no action, quality, production, or
data-governance authority.  The current research-thread request is recorded
only as permission to prepare and evaluate this non-production experiment.

The programmatic builder does not publish output, but it does open, hash, and
probe its inputs and the pinned executable.  Publication writes two flat,
create-only siblings under a pinned parent fd: ``<stem>.manifest.jsonl`` then
``<stem>.receipt.json``.  The complete canonical receipt is the last commit
marker; readers must also require successful producer return.  The CLI writes
only when the caller supplies ``--publish``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Callable, Mapping, Optional, Sequence


ROW_SCHEMA = "bernini-graft-a-lite-source-noop-row-v1"
RECEIPT_SCHEMA = "bernini-graft-a-lite-source-noop-receipt-v1"
RESEARCH_SCOPE_SCHEMA = "bernini-graft-user-research-scope-record-v1"
UPSTREAM_ROW_SCHEMA = "motive-goku-action-anchor-prefilter-v1"

MANIFEST_SUFFIX = ".manifest.jsonl"
RECEIPT_SUFFIX = ".receipt.json"

FRAME_COUNT = 81
FPS_NUMERATOR = 25
FPS_DENOMINATOR = 1
FPS = 25.0
SHORT_SIDE = 704
NOOP_INSTRUCTION = (
    "Keep every subject, action, timing, camera motion, framing, appearance, "
    "and background unchanged."
)
SPLIT_NAMESPACE = "bernini-graft-a-lite-source-split-v1"

V16_COHORT = "goku_fullmotion_v16_exact128"
V17_COHORT = "goku_fullmotion_v17_next1000"
EXPECTED_V16_ROWS = 128
EXPECTED_V17_ROWS = 1000
EXPECTED_FULL_ROWS = EXPECTED_V16_ROWS + EXPECTED_V17_ROWS
CURRENT_V16_MANIFEST_SHA256 = (
    "834e5a70e7c87683730ac644ce233b9343e4fc98eb3b3a45f55f93c8da94688d"
)
CURRENT_V17_MANIFEST_SHA256 = (
    "24021e6a4c5d1758340f9e61df1a987383e1ad39063071526726e9658ccd1c10"
)

# This is the preregistered, shared portable AUH executable observation after
# successful compute-node verification.  It is deliberately a content/path/
# version pin rather than a PATH lookup.  dev/ino are recorded at execution
# time but are not frozen because they are host/filesystem-local.  This pin
# label is provenance only; it does not claim a sealed Python/runtime closure.
FROZEN_FFPROBE_PIN_LABEL = "shared_portable_compute_verified_auh_ffprobe_v1"
FROZEN_FFPROBE_REALPATH = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_graft_v1_20260810/runtime/"
    "ffprobe_conda_cf_ffmpeg9_hdabad70_r2/runtime/ffprobe"
)
FROZEN_FFPROBE_SHA256 = (
    "356754aa8e327b139dd54dda6846af6425673b73572ab6c1c182ed970f1107f5"
)
FROZEN_FFPROBE_VERSION_STDOUT_SHA256 = (
    "2271b81138bdaf07532b801ac7abd5b48d9e84dd66a6287a82fb44bc04c84f6b"
)
FROZEN_FFPROBE_VERSION_FIRST_LINE = (
    "ffprobe version 9.0 Copyright (c) 2007-2026 the FFmpeg developers"
)

# Ordering is part of the canary contract: the two optimizer rows come first,
# followed by confirmation-only rows that must never be consumed by updates.
CANARY4 = (
    ("7b88a1ca1f804f41", "optimizer_train", "dog_fit"),
    ("a35b590961d24694", "optimizer_train", "human_fit"),
    ("841b5e0080a1441d", "optimizer_confirmation", "dog_confirmation"),
    ("a66e6818e4144928", "optimizer_confirmation", "human_confirmation"),
)
CANARY4_BY_IID = {
    iid: (split, role) for iid, split, role in CANARY4
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_IID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class GraftALiteReleaseError(RuntimeError):
    """Raised before an incomplete or misleading release can be emitted."""


@dataclass(frozen=True)
class Candidate:
    cohort: str
    manifest_path: Path
    manifest_sha256: str
    line_number: int
    raw_line: bytes
    value: Mapping[str, Any]
    iid: str
    source_path_text: str
    source_sha256: str
    width: int
    height: int
    source_size_bytes: int
    source_mtime_ns: int


@dataclass(frozen=True)
class InputManifest:
    cohort: str
    path: Path
    sha256: str
    raw: bytes
    stat_identity: tuple[int, int, int, int, int]
    rows: tuple[Candidate, ...]


@dataclass(frozen=True)
class SourceEvidence:
    row: Mapping[str, Any]
    path: Path
    stat_identity: tuple[int, int, int, int, int]
    parent_path: Path
    parent_binding_identity: tuple[int, int, int, int]


@dataclass(frozen=True)
class OpenedSource:
    """Read-only source descriptor retained across hash and media probe."""

    path: Path
    fd: int


@dataclass(frozen=True)
class FFprobePin:
    """Exact executable observation required by one probe session."""

    label: str
    realpath: str
    file_sha256: str
    version_stdout_sha256: str
    version_first_line: str


@dataclass(frozen=True)
class OpenedFFprobe:
    """Read-only executable fd plus verified, receipt-safe provenance."""

    path: Path
    fd: int
    fd_transport: str
    execution_path: str
    execution_transport: str
    stat_identity: tuple[int, int, int, int, int]
    expected_file_sha256: str
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class PublishedRelease:
    logical_output_stem: Path
    manifest_path: Path
    receipt_path: Path


@dataclass(frozen=True)
class ReleasePayload:
    rows: tuple[Mapping[str, Any], ...]
    manifest_bytes: bytes
    receipt: Mapping[str, Any]
    receipt_bytes: bytes
    publication_eligible: bool
    probe_kind: str


MediaProbe = Callable[[OpenedSource], Mapping[str, Any]]
ProbeFinalize = Callable[[], None]
_PRODUCTION_PROBE_KIND = "frozen_shared_portable_compute_verified_ffprobe_v2"
_TEST_PROBE_KIND = "test_only_untrusted_media_probe_v1"
_FROZEN_FFPROBE_PIN = FFprobePin(
    label=FROZEN_FFPROBE_PIN_LABEL,
    realpath=FROZEN_FFPROBE_REALPATH,
    file_sha256=FROZEN_FFPROBE_SHA256,
    version_stdout_sha256=FROZEN_FFPROBE_VERSION_STDOUT_SHA256,
    version_first_line=FROZEN_FFPROBE_VERSION_FIRST_LINE,
)


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
        raise GraftALiteReleaseError(
            f"value is not canonical finite ASCII JSON: {error}"
        ) from error


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def object_sha256(value: Any) -> str:
    return bytes_sha256(canonical_json_bytes(value))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise GraftALiteReleaseError(f"{label} must be lowercase SHA-256")
    return value


def _safe_iid(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _IID_RE.fullmatch(value) is None:
        raise GraftALiteReleaseError(f"{label} is not a safe IID")
    return value


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GraftALiteReleaseError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise GraftALiteReleaseError(f"non-finite JSON constant: {value}")


def _decode_object(payload: bytes, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except GraftALiteReleaseError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GraftALiteReleaseError(f"invalid {context}: {error}") from error
    if not isinstance(value, dict):
        raise GraftALiteReleaseError(f"{context} must be one JSON object")
    return value


def _stat_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _inode_identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _directory_binding_identity(
    metadata: os.stat_result,
) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _plain_absolute_file(path_value: str | Path, *, label: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        raise GraftALiteReleaseError(f"{label} must be absolute")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise GraftALiteReleaseError(f"cannot stat {label}: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise GraftALiteReleaseError(f"{label} must be a plain non-symlink file")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:  # pragma: no cover - lstat already catches normal absence
        raise GraftALiteReleaseError(f"cannot resolve {label}: {error}") from error
    return resolved


def _stable_read(
    path: Path, *, label: str
) -> tuple[bytes, tuple[int, int, int, int, int]]:
    before = path.stat()
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            raw = handle.read()
    except OSError as error:
        raise GraftALiteReleaseError(f"cannot read {label}: {error}") from error
    after = path.stat()
    identities = {_stat_identity(before), _stat_identity(opened), _stat_identity(after)}
    if len(identities) != 1:
        raise GraftALiteReleaseError(f"{label} changed while being read")
    return raw, _stat_identity(after)


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GraftALiteReleaseError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise GraftALiteReleaseError(f"{label} must be finite")
    return result


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GraftALiteReleaseError(f"{label} must be a positive integer")
    return value


def _validate_candidate(
    value: Mapping[str, Any],
    *,
    cohort: str,
    manifest_path: Path,
    manifest_sha256: str,
    line_number: int,
    raw_line: bytes,
) -> Candidate:
    iid = _safe_iid(value.get("iid"), label=f"{cohort} line {line_number} IID")
    if value.get("schema_version") != UPSTREAM_ROW_SCHEMA:
        raise GraftALiteReleaseError(f"iid={iid} upstream row schema differs")
    if value.get("eligible") is not True or value.get("selected") is not True:
        raise GraftALiteReleaseError(f"iid={iid} is not selected and eligible")
    path_text = value.get("resolved_src_video")
    if (
        not isinstance(path_text, str)
        or not path_text
        or path_text != path_text.strip()
        or "\x00" in path_text
        or not Path(path_text).is_absolute()
    ):
        raise GraftALiteReleaseError(f"iid={iid} resolved source path is invalid")
    if Path(path_text).suffix.lower() != ".mp4":
        raise GraftALiteReleaseError(f"iid={iid} source must be an MP4")
    source_sha = _sha256(
        value.get("source_video_sha256"), label=f"iid={iid} source SHA-256"
    )
    media = value.get("media")
    if not isinstance(media, Mapping):
        raise GraftALiteReleaseError(f"iid={iid} media must be an object")
    if media.get("frame_count") != FRAME_COUNT or isinstance(
        media.get("frame_count"), bool
    ):
        raise GraftALiteReleaseError(f"iid={iid} frame_count must equal 81")
    fps = _finite_number(media.get("fps"), label=f"iid={iid} media.fps")
    if Fraction(str(fps)) != Fraction(FPS_NUMERATOR, FPS_DENOMINATOR):
        raise GraftALiteReleaseError(f"iid={iid} fps must equal 25")
    width = _positive_int(media.get("width"), label=f"iid={iid} media.width")
    height = _positive_int(media.get("height"), label=f"iid={iid} media.height")
    short_side = _positive_int(
        media.get("short_side"), label=f"iid={iid} media.short_side"
    )
    if short_side != min(width, height) or short_side != SHORT_SIDE:
        raise GraftALiteReleaseError(
            f"iid={iid} resolution is not the frozen 704-short-side source"
        )
    source_size = _positive_int(
        media.get("file_size_bytes"), label=f"iid={iid} media.file_size_bytes"
    )
    source_mtime = _positive_int(
        media.get("mtime_ns_at_analysis"),
        label=f"iid={iid} media.mtime_ns_at_analysis",
    )
    return Candidate(
        cohort=cohort,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        line_number=line_number,
        raw_line=raw_line,
        value=value,
        iid=iid,
        source_path_text=path_text,
        source_sha256=source_sha,
        width=width,
        height=height,
        source_size_bytes=source_size,
        source_mtime_ns=source_mtime,
    )


def _read_input_manifest(
    path_value: str | Path,
    *,
    cohort: str,
    expected_sha256: str,
    expected_rows: int,
) -> InputManifest:
    expected = _sha256(expected_sha256, label=f"{cohort} expected manifest SHA-256")
    path = _plain_absolute_file(path_value, label=f"{cohort} candidate manifest")
    raw, identity = _stable_read(path, label=f"{cohort} candidate manifest")
    actual = bytes_sha256(raw)
    if actual != expected:
        raise GraftALiteReleaseError(
            f"{cohort} candidate manifest SHA-256 differs: {actual} != {expected}"
        )
    if not raw or not raw.endswith(b"\n"):
        raise GraftALiteReleaseError(
            f"{cohort} candidate manifest must be non-empty newline-terminated JSONL"
        )
    lines = raw.splitlines(keepends=True)
    if len(lines) != expected_rows:
        raise GraftALiteReleaseError(
            f"{cohort} row count differs: {len(lines)}/{expected_rows}"
        )
    rows: list[Candidate] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.endswith(b"\n") or not line[:-1].strip():
            raise GraftALiteReleaseError(
                f"{cohort} line {line_number} is not one JSONL record"
            )
        value = _decode_object(line, context=f"{cohort} line {line_number}")
        candidate = _validate_candidate(
            value,
            cohort=cohort,
            manifest_path=path,
            manifest_sha256=actual,
            line_number=line_number,
            raw_line=line,
        )
        if candidate.iid in seen:
            raise GraftALiteReleaseError(
                f"duplicate IID inside {cohort}: {candidate.iid}"
            )
        seen.add(candidate.iid)
        rows.append(candidate)
    return InputManifest(
        cohort=cohort,
        path=path,
        sha256=actual,
        raw=raw,
        stat_identity=identity,
        rows=tuple(rows),
    )


def _parse_fraction(value: Any, *, label: str) -> Fraction:
    if not isinstance(value, str) or not value or value == "N/A":
        raise GraftALiteReleaseError(f"{label} is not an ffprobe fraction")
    try:
        fraction = Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise GraftALiteReleaseError(f"{label} is invalid: {value!r}") from error
    if fraction <= 0:
        raise GraftALiteReleaseError(f"{label} must be positive")
    return fraction


def _inherited_fd_path(fd: int, *, label: str) -> tuple[str, str]:
    if sys.platform.startswith("linux") and Path("/proc/self/fd").is_dir():
        return f"/proc/self/fd/{fd}", "linux_proc_self_fd"
    if Path("/dev/fd").is_dir():
        return f"/dev/fd/{fd}", "portable_dev_fd"
    raise GraftALiteReleaseError(
        f"no inherited-fd path is available for fail-closed {label}"
    )


def _subprocess_environment() -> dict[str, str]:
    # PATH is intentionally irrelevant to executable selection; it is present
    # only as a conservative runtime default for the already-open binary.
    return {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"}


def _run_opened_executable(
    executable: OpenedFFprobe,
    arguments: Sequence[str],
    *,
    timeout: int,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            [executable.execution_path, *arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
            pass_fds=(executable.fd,),
            env=_subprocess_environment(),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise GraftALiteReleaseError(
            f"verified ffprobe execution failed: {error}"
        ) from error


def _single_lstat_matches(
    path: Path,
    identity: tuple[int, int, int, int, int],
    *,
    regular: bool,
) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    wanted_mode = stat.S_ISREG(metadata.st_mode) if regular else stat.S_ISDIR(metadata.st_mode)
    return wanted_mode and _stat_identity(metadata) == identity


def _validate_opened_ffprobe(executable: OpenedFFprobe) -> None:
    before = os.fstat(executable.fd)
    digest, hashed_identity = _hash_open_fd(
        executable.fd, label="opened ffprobe executable"
    )
    after = os.fstat(executable.fd)
    if (
        _stat_identity(before) != executable.stat_identity
        or hashed_identity != executable.stat_identity
        or _stat_identity(after) != executable.stat_identity
        or digest != executable.expected_file_sha256
        or not _single_lstat_matches(
            executable.path, executable.stat_identity, regular=True
        )
    ):
        raise GraftALiteReleaseError(
            "frozen ffprobe executable changed after verification"
        )


def _open_ffprobe_pin(pin: FFprobePin) -> OpenedFFprobe:
    """Open and verify one caller-process executable observation.

    This private primitive also supports hostile tests with a local executable.
    Only ``_open_frozen_ffprobe`` may feed structurally publication-eligible
    output, and that status is not scientific or formal runtime authority.
    """

    if (
        not isinstance(pin.label, str)
        or not pin.label
        or not pin.label.isascii()
        or len(pin.label) > 128
    ):
        raise GraftALiteReleaseError("ffprobe pin label must be nonempty ASCII")
    expected_file_sha = _sha256(pin.file_sha256, label="ffprobe file SHA-256")
    expected_version_sha = _sha256(
        pin.version_stdout_sha256, label="ffprobe version stdout SHA-256"
    )
    if not isinstance(pin.realpath, str) or not Path(pin.realpath).is_absolute():
        raise GraftALiteReleaseError("ffprobe pin path must be absolute")
    path = Path(pin.realpath)
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise GraftALiteReleaseError(f"cannot stat frozen ffprobe: {error}") from error
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or str(resolved) != pin.realpath
    ):
        raise GraftALiteReleaseError(
            "ffprobe pin must be its exact plain-file absolute realpath"
        )
    if stat.S_IMODE(metadata.st_mode) & 0o111 == 0:
        raise GraftALiteReleaseError("ffprobe pin is not executable")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise GraftALiteReleaseError(f"cannot open frozen ffprobe: {error}") from error
    try:
        opened_metadata = os.fstat(descriptor)
        identity = _stat_identity(opened_metadata)
        if (
            not stat.S_ISREG(opened_metadata.st_mode)
            or _stat_identity(metadata) != identity
        ):
            raise GraftALiteReleaseError("ffprobe changed while opening")
        executable_fd_path, transport = _inherited_fd_path(
            descriptor, label="ffprobe executable"
        )
        digest, hashed_identity = _hash_open_fd(
            descriptor, label="opened ffprobe executable"
        )
        if digest != expected_file_sha or hashed_identity != identity:
            raise GraftALiteReleaseError("ffprobe file SHA-256 differs")
        executable = OpenedFFprobe(
            path=path,
            fd=descriptor,
            fd_transport=executable_fd_path.rsplit("/", 1)[0],
            execution_path=executable_fd_path,
            execution_transport=transport,
            stat_identity=identity,
            expected_file_sha256=expected_file_sha,
            provenance={},
        )
        try:
            completed = _run_opened_executable(
                executable, ("-version",), timeout=30
            )
        except GraftALiteReleaseError:
            if sys.platform.startswith("linux"):
                raise
            # Some non-Linux /dev/fd mounts are not executable.  The explicit
            # fallback is still PATH-free but is weaker: it relies on exact
            # realpath plus pre/post inode/content revalidation, and says so.
            executable = OpenedFFprobe(
                path=path,
                fd=descriptor,
                fd_transport=executable.fd_transport,
                execution_path=str(path),
                execution_transport="absolute_realpath_pre_post_inode_sha_fallback",
                stat_identity=identity,
                expected_file_sha256=expected_file_sha,
                provenance={},
            )
            _validate_opened_ffprobe(executable)
            completed = _run_opened_executable(
                executable, ("-version",), timeout=30
            )
        if completed.returncode != 0:
            diagnostic = completed.stderr.decode("utf-8", errors="replace")[-1000:]
            raise GraftALiteReleaseError(
                f"verified ffprobe -version failed ({completed.returncode}): {diagnostic}"
            )
        observed_version_sha = bytes_sha256(completed.stdout)
        try:
            version_lines = completed.stdout.decode("utf-8", errors="strict").splitlines()
        except UnicodeDecodeError as error:
            raise GraftALiteReleaseError(
                "ffprobe version stdout is not UTF-8"
            ) from error
        if not version_lines:
            raise GraftALiteReleaseError("ffprobe version stdout is empty")
        first_line = version_lines[0]
        if (
            observed_version_sha != expected_version_sha
            or first_line != pin.version_first_line
        ):
            raise GraftALiteReleaseError("ffprobe version observation differs")
        provenance = {
            "pin_label": pin.label,
            "configured_path": pin.realpath,
            "resolved_path": str(resolved),
            "exact_realpath_matched": True,
            "path_lookup_used": False,
            "file_sha256_expected": expected_file_sha,
            "file_sha256_observed": digest,
            "file_sha256_matched": True,
            "version_command": "opened_executable_fd -version",
            "version_stdout_sha256_expected": expected_version_sha,
            "version_stdout_sha256_observed": observed_version_sha,
            "version_stdout_sha256_matched": True,
            "version_first_line_expected": pin.version_first_line,
            "version_first_line_observed": first_line,
            "version_first_line_matched": True,
            "executable_transport": executable.execution_transport,
            "executable_fixed_inode_execution": executable.execution_transport
            in {"linux_proc_self_fd", "portable_dev_fd"},
            "absolute_path_fallback_pre_post_inode_sha": (
                executable.execution_transport
                == "absolute_realpath_pre_post_inode_sha_fallback"
            ),
            "executable_opened_o_nofollow": bool(getattr(os, "O_NOFOLLOW", 0)),
            "executable_identity": {
                "device": identity[0],
                "inode": identity[1],
                "size_bytes": identity[2],
                "mtime_ns": identity[3],
                "ctime_ns": identity[4],
                "mode": format(stat.S_IMODE(opened_metadata.st_mode), "04o"),
                "uid": opened_metadata.st_uid,
                "gid": opened_metadata.st_gid,
            },
            "device_inode_are_observed_not_cross_host_pins": True,
            "pre_and_post_version_identity_and_file_sha_revalidated": True,
            "caller_process_observation_only": True,
            "trusted_or_official_authority_claimed": False,
        }
        executable = OpenedFFprobe(
            path=path,
            fd=descriptor,
            fd_transport=executable.fd_transport,
            execution_path=executable.execution_path,
            execution_transport=executable.execution_transport,
            stat_identity=identity,
            expected_file_sha256=expected_file_sha,
            provenance=provenance,
        )
        _validate_opened_ffprobe(executable)
        return executable
    except Exception:
        os.close(descriptor)
        raise


def _open_frozen_ffprobe() -> OpenedFFprobe:
    return _open_ffprobe_pin(_FROZEN_FFPROBE_PIN)


def probe_source_media(
    source: OpenedSource, executable: OpenedFFprobe
) -> Mapping[str, Any]:
    """Probe one already-open source fd with one already-verified executable fd."""

    if not isinstance(source, OpenedSource) or source.fd < 0:
        raise GraftALiteReleaseError("probe_source_media requires an open source fd")
    if not isinstance(executable, OpenedFFprobe) or executable.fd < 0:
        raise GraftALiteReleaseError("probe_source_media requires an open ffprobe fd")
    _validate_opened_ffprobe(executable)
    fd_path, fd_transport = _inherited_fd_path(source.fd, label="source video")
    command = [
        "-v",
        "error",
        "-select_streams",
        "v",
        "-count_frames",
        "-show_entries",
        "stream=index,width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames",
        "-of",
        "json",
        fd_path,
    ]
    # Both fds are inherited.  The executable itself is selected through its
    # fd, never through PATH or a mutable pathname lookup in the child.
    try:
        completed = subprocess.run(
            [executable.execution_path, *command],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=180,
            pass_fds=(source.fd, executable.fd),
            env=_subprocess_environment(),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise GraftALiteReleaseError(
            f"ffprobe failed for pinned source {source.path}: {error}"
        ) from error
    if completed.returncode != 0:
        diagnostic = completed.stderr.decode("utf-8", errors="replace")[-1000:]
        raise GraftALiteReleaseError(
            f"ffprobe rejected {source.path} (exit {completed.returncode}): {diagnostic}"
        )
    value = _decode_object(
        completed.stdout, context=f"ffprobe output for {source.path}"
    )
    streams = value.get("streams")
    if not isinstance(streams, list) or len(streams) != 1:
        raise GraftALiteReleaseError(
            f"source must contain exactly one video stream: {source.path}"
        )
    stream = streams[0]
    if not isinstance(stream, Mapping):
        raise GraftALiteReleaseError(f"ffprobe stream is invalid: {source.path}")
    frames_text = stream.get("nb_read_frames")
    if not isinstance(frames_text, str) or not frames_text.isdigit():
        raise GraftALiteReleaseError(
            f"ffprobe did not count frames for {source.path}"
        )
    frame_count = int(frames_text)
    width = _positive_int(
        stream.get("width"), label=f"{source.path} ffprobe width"
    )
    height = _positive_int(
        stream.get("height"), label=f"{source.path} ffprobe height"
    )
    average = _parse_fraction(
        stream.get("avg_frame_rate"),
        label=f"{source.path} ffprobe avg_frame_rate",
    )
    reported = _parse_fraction(
        stream.get("r_frame_rate"),
        label=f"{source.path} ffprobe r_frame_rate",
    )
    _validate_opened_ffprobe(executable)
    return {
        "frame_count": frame_count,
        "fps_numerator": average.numerator,
        "fps_denominator": average.denominator,
        "reported_fps_numerator": reported.numerator,
        "reported_fps_denominator": reported.denominator,
        "width": width,
        "height": height,
        "fd_transport": fd_transport,
        "executable_transport": executable.provenance["executable_transport"],
        "executable_pre_post_verified": True,
        "path_lookup_used": False,
    }


def _validate_probe(probe: Mapping[str, Any], *, candidate: Candidate) -> dict[str, Any]:
    iid = candidate.iid
    if not isinstance(probe, Mapping):
        raise GraftALiteReleaseError(f"iid={iid} media probe did not return an object")
    frame_count = probe.get("frame_count")
    width = probe.get("width")
    height = probe.get("height")
    fps_numerator = probe.get("fps_numerator")
    fps_denominator = probe.get("fps_denominator")
    reported_numerator = probe.get("reported_fps_numerator")
    reported_denominator = probe.get("reported_fps_denominator")
    for label, value in (
        ("frame_count", frame_count),
        ("width", width),
        ("height", height),
        ("fps_numerator", fps_numerator),
        ("fps_denominator", fps_denominator),
        ("reported_fps_numerator", reported_numerator),
        ("reported_fps_denominator", reported_denominator),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise GraftALiteReleaseError(f"iid={iid} probed {label} is invalid")
    average = Fraction(fps_numerator, fps_denominator)
    reported = Fraction(reported_numerator, reported_denominator)
    expected_fps = Fraction(FPS_NUMERATOR, FPS_DENOMINATOR)
    if frame_count != FRAME_COUNT:
        raise GraftALiteReleaseError(f"iid={iid} probed frame_count is not exact81")
    if average != expected_fps or reported != expected_fps:
        raise GraftALiteReleaseError(f"iid={iid} probed fps is not exact25")
    if width != candidate.width or height != candidate.height:
        raise GraftALiteReleaseError(
            f"iid={iid} probed resolution differs from upstream metadata"
        )
    return {
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "fps_fraction": f"{FPS_NUMERATOR}/{FPS_DENOMINATOR}",
        "reported_fps_fraction": f"{reported.numerator}/{reported.denominator}",
        "width": width,
        "height": height,
        "resolution_hw": [height, width],
        "short_side": min(width, height),
        "probe_contract_matched": True,
    }


def _hash_open_fd(
    fd: int, *, label: str
) -> tuple[str, tuple[int, int, int, int, int]]:
    before = os.fstat(fd)
    digest = hashlib.sha256()
    try:
        if hasattr(os, "pread"):
            offset = 0
            while True:
                block = os.pread(fd, 1024 * 1024, offset)
                if not block:
                    break
                digest.update(block)
                offset += len(block)
        else:  # pragma: no cover - AUH/Linux and macOS both provide pread
            original_offset = os.lseek(fd, 0, os.SEEK_CUR)
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                for block in iter(lambda: os.read(fd, 1024 * 1024), b""):
                    digest.update(block)
            finally:
                os.lseek(fd, original_offset, os.SEEK_SET)
    except OSError as error:
        raise GraftALiteReleaseError(f"cannot hash {label}: {error}") from error
    after = os.fstat(fd)
    if _stat_identity(before) != _stat_identity(after):
        raise GraftALiteReleaseError(f"{label} changed while hashing")
    return digest.hexdigest(), _stat_identity(after)


def _hash_open_source_fd(
    fd: int,
) -> tuple[str, tuple[int, int, int, int, int]]:
    return _hash_open_fd(fd, label="open source fd")


def _open_pinned_source(
    path_value: str, *, label: str
) -> tuple[OpenedSource, int, tuple[int, int, int, int]]:
    raw_path = Path(path_value).expanduser()
    if not raw_path.is_absolute():
        raise GraftALiteReleaseError(f"{label} must be absolute")
    try:
        raw_metadata = raw_path.lstat()
    except OSError as error:
        raise GraftALiteReleaseError(f"cannot stat {label}: {error}") from error
    if stat.S_ISLNK(raw_metadata.st_mode) or not stat.S_ISREG(raw_metadata.st_mode):
        raise GraftALiteReleaseError(f"{label} must be a plain non-symlink file")
    try:
        path = raw_path.resolve(strict=True)
        parent = path.parent.resolve(strict=True)
    except OSError as error:
        raise GraftALiteReleaseError(f"cannot resolve {label}: {error}") from error
    parent_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        parent_fd = os.open(parent, parent_flags)
    except OSError as error:
        raise GraftALiteReleaseError(f"cannot pin {label} parent: {error}") from error
    source_fd: Optional[int] = None
    try:
        parent_metadata = os.fstat(parent_fd)
        parent_path_metadata = parent.lstat()
        if (
            not stat.S_ISDIR(parent_metadata.st_mode)
            or _inode_identity(parent_metadata) != _inode_identity(parent_path_metadata)
        ):
            raise GraftALiteReleaseError(f"{label} parent identity differs")
        leaf_metadata = os.stat(
            path.name, dir_fd=parent_fd, follow_symlinks=False
        )
        if not stat.S_ISREG(leaf_metadata.st_mode):
            raise GraftALiteReleaseError(f"{label} leaf is not a regular file")
        source_fd = os.open(
            path.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        opened_metadata = os.fstat(source_fd)
        if _inode_identity(opened_metadata) != _inode_identity(leaf_metadata):
            raise GraftALiteReleaseError(f"{label} changed while opening")
        return (
            OpenedSource(path=path, fd=source_fd),
            parent_fd,
            _directory_binding_identity(parent_metadata),
        )
    except Exception:
        if source_fd is not None:
            os.close(source_fd)
        os.close(parent_fd)
        raise


def _split_assignment(iid: str) -> tuple[str, str]:
    preregistered = CANARY4_BY_IID.get(iid)
    if preregistered is not None:
        split, role = preregistered
        return split, f"preregistered_core4:{role}"
    digest = hashlib.sha256(f"{SPLIT_NAMESPACE}\0{iid}".encode("ascii")).digest()
    split = (
        "optimizer_confirmation"
        if int.from_bytes(digest[:8], "big") % 10 == 0
        else "optimizer_train"
    )
    return split, "fixed_iid_hash_mod10"


def _materialize_source_row(
    item: tuple[int, Candidate],
    *,
    mode: str,
    media_probe: MediaProbe,
    production_probe_observed: bool,
    publication_eligible: bool,
) -> SourceEvidence:
    row_index, candidate = item
    source, parent_fd, parent_binding = _open_pinned_source(
        candidate.source_path_text,
        label=f"iid={candidate.iid} source video",
    )
    try:
        initial_file_identity = _stat_identity(os.fstat(source.fd))
        digest, identity = _hash_open_source_fd(source.fd)
        if identity != initial_file_identity:
            raise GraftALiteReleaseError(
                f"iid={candidate.iid} source identity changed during initial hash"
            )
        if digest != candidate.source_sha256:
            raise GraftALiteReleaseError(
                f"iid={candidate.iid} source SHA-256 differs"
            )
        if identity[2] != candidate.source_size_bytes:
            raise GraftALiteReleaseError(
                f"iid={candidate.iid} source file size differs"
            )
        if identity[3] != candidate.source_mtime_ns:
            raise GraftALiteReleaseError(
                f"iid={candidate.iid} source mtime differs from upstream analysis"
            )
        before_probe_identity = _stat_identity(os.fstat(source.fd))
        if before_probe_identity != initial_file_identity:
            raise GraftALiteReleaseError(
                f"iid={candidate.iid} source identity changed before probe"
            )
        probe_value = media_probe(source)
        after_probe_identity = _stat_identity(os.fstat(source.fd))
        if after_probe_identity != initial_file_identity:
            raise GraftALiteReleaseError(
                f"iid={candidate.iid} source identity changed during probe"
            )
        if production_probe_observed:
            transport = probe_value.get("fd_transport")
            if transport not in {"linux_proc_self_fd", "portable_dev_fd"}:
                raise GraftALiteReleaseError(
                    f"iid={candidate.iid} production probe did not bind an inherited fd"
                )
            if (
                probe_value.get("executable_transport")
                not in {"linux_proc_self_fd", "portable_dev_fd"}
                or probe_value.get("executable_pre_post_verified") is not True
                or probe_value.get("path_lookup_used") is not False
            ):
                raise GraftALiteReleaseError(
                    f"iid={candidate.iid} production ffprobe executable was not fd-bound"
                )
        else:
            transport = "test_only_untrusted_probe"
        media = _validate_probe(probe_value, candidate=candidate)
        media["source_fd_transport"] = transport

        second_digest, second_hash_identity = _hash_open_source_fd(source.fd)
        if (
            second_digest != digest
            or second_hash_identity != initial_file_identity
            or _stat_identity(os.fstat(source.fd)) != initial_file_identity
        ):
            raise GraftALiteReleaseError(
                f"iid={candidate.iid} source content or identity changed across probe"
            )

        after_file = os.fstat(source.fd)
        after_leaf = os.stat(
            source.path.name, dir_fd=parent_fd, follow_symlinks=False
        )
        after_full_path = source.path.lstat()
        after_parent = os.fstat(parent_fd)
        after_parent_path = source.path.parent.lstat()
        if (
            _stat_identity(after_file) != initial_file_identity
            or _stat_identity(after_leaf) != initial_file_identity
            or _stat_identity(after_full_path) != initial_file_identity
        ):
            raise GraftALiteReleaseError(
                f"iid={candidate.iid} source path/inode binding changed during probe"
            )
        if (
            _directory_binding_identity(after_parent) != parent_binding
            or _directory_binding_identity(after_parent_path) != parent_binding
        ):
            raise GraftALiteReleaseError(
                f"iid={candidate.iid} source parent changed during probe"
            )
    finally:
        os.close(source.fd)
        os.close(parent_fd)
    media["fresh_ffprobe_verified"] = production_probe_observed
    media["test_only_probe_contract_matched"] = not production_probe_observed
    split, assignment = _split_assignment(candidate.iid)
    optimizer_confirmation = split == "optimizer_confirmation"
    prior_exposure: Optional[bool] = (
        True if candidate.iid in CANARY4_BY_IID else None
    )
    core = {
        "schema_version": ROW_SCHEMA,
        "release_mode": mode,
        "row_index": row_index,
        "iid": candidate.iid,
        "split": split,
        "split_assignment": assignment,
        "optimizer_update_authorized": not optimizer_confirmation,
        "optimizer_confirmation_only": optimizer_confirmation,
        "prior_research_exposure": prior_exposure,
        "global_holdout": False,
        "stable_identity_disjoint_split_claimed": False,
        "source_cohort": candidate.cohort,
        "source_video_path": str(source.path),
        "source_video_sha256": digest,
        "source_file_size_bytes": identity[2],
        "source_mtime_ns": identity[3],
        "source_ctime_ns_observed": identity[4],
        "source_hash_and_probe_same_open_fd": True,
        "source_sha256_recomputed_before_and_after_probe": True,
        "source_pre_post_probe_sha256_matched": True,
        "source_identity_includes_ctime_ns": True,
        "source_path_inode_binding_revalidated": True,
        "source_media": media,
        "noop_instruction": NOOP_INSTRUCTION,
        "same_clip_noop_only": True,
        "source_video_is_clean_noop_endpoint": True,
        "cross_clip_identity_authority": False,
        "action_authority": False,
        "quality_authority": False,
        "production_authority": False,
        "publication_eligible": publication_eligible,
        "upstream_candidate": {
            "cohort": candidate.cohort,
            "manifest_path": str(candidate.manifest_path),
            "manifest_sha256": candidate.manifest_sha256,
            "line_number": candidate.line_number,
            "row_bytes_sha256": bytes_sha256(candidate.raw_line),
            "row_canonical_sha256": object_sha256(candidate.value),
            "row_schema_version": candidate.value["schema_version"],
        },
    }
    return SourceEvidence(
        row={**core, "row_digest": object_sha256(core)},
        path=source.path,
        stat_identity=identity,
        parent_path=source.path.parent,
        parent_binding_identity=parent_binding,
    )


def _select_candidates(
    candidates: Sequence[Candidate], *, mode: str
) -> tuple[Candidate, ...]:
    by_iid = {candidate.iid: candidate for candidate in candidates}
    if len(by_iid) != len(candidates):
        raise GraftALiteReleaseError("v16/v17 candidate IIDs are not disjoint")
    paths = [candidate.source_path_text for candidate in candidates]
    digests = [candidate.source_sha256 for candidate in candidates]
    if len(set(paths)) != len(paths):
        raise GraftALiteReleaseError("v16/v17 source paths are not unique")
    if len(set(digests)) != len(digests):
        raise GraftALiteReleaseError("v16/v17 source SHA-256 values are not unique")
    if len(candidates) != EXPECTED_FULL_ROWS:
        raise GraftALiteReleaseError(
            f"combined candidate count differs: {len(candidates)}/{EXPECTED_FULL_ROWS}"
        )
    if mode == "full1128":
        return tuple(candidates)
    if mode != "canary4":
        raise GraftALiteReleaseError("mode must be canary4 or full1128")
    missing = [iid for iid, _, _ in CANARY4 if iid not in by_iid]
    if missing:
        raise GraftALiteReleaseError(f"canary4 IIDs are absent: {missing}")
    return tuple(by_iid[iid] for iid, _, _ in CANARY4)


def _manifest_input_descriptor(manifest: InputManifest) -> dict[str, Any]:
    return {
        "cohort": manifest.cohort,
        "path": str(manifest.path),
        "rows": len(manifest.rows),
        "file_sha256": manifest.sha256,
        "row_binding_digest": object_sha256(
            [
                {
                    "iid": row.iid,
                    "line_number": row.line_number,
                    "row_bytes_sha256": bytes_sha256(row.raw_line),
                    "row_canonical_sha256": object_sha256(row.value),
                }
                for row in manifest.rows
            ]
        ),
    }


def _implementation_sha256() -> str:
    return file_sha256(Path(__file__).resolve(strict=True))


def _build_payload(
    *,
    v16_candidates: str | Path,
    v17_candidates: str | Path,
    expected_v16_manifest_sha256: str,
    expected_v17_manifest_sha256: str,
    mode: str = "canary4",
    workers: int = 8,
    media_probe: MediaProbe,
    probe_kind: str,
    probe_implementation: Mapping[str, Any],
    probe_finalize: ProbeFinalize,
    frozen_manifest_contract: bool,
) -> ReleasePayload:
    """Private core shared by the real builder and unpublishable test fixtures."""

    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise GraftALiteReleaseError("workers must be a positive integer")
    if not callable(media_probe):
        raise GraftALiteReleaseError("media_probe must be callable")
    if not callable(probe_finalize):
        raise GraftALiteReleaseError("probe_finalize must be callable")
    if probe_kind not in {_PRODUCTION_PROBE_KIND, _TEST_PROBE_KIND}:
        raise GraftALiteReleaseError("unknown media probe kind")
    production_probe_observed = probe_kind == _PRODUCTION_PROBE_KIND
    if frozen_manifest_contract:
        if (
            expected_v16_manifest_sha256 != CURRENT_V16_MANIFEST_SHA256
            or expected_v17_manifest_sha256 != CURRENT_V17_MANIFEST_SHA256
        ):
            raise GraftALiteReleaseError(
                "public manifest contract must use the code-frozen v16/v17 SHA-256 pins"
            )
    elif production_probe_observed:
        raise GraftALiteReleaseError(
            "custom manifests cannot use the production probe publication path"
        )
    publication_eligible = production_probe_observed and frozen_manifest_contract
    implementation_sha256 = _implementation_sha256()
    v16 = _read_input_manifest(
        v16_candidates,
        cohort=V16_COHORT,
        expected_sha256=expected_v16_manifest_sha256,
        expected_rows=EXPECTED_V16_ROWS,
    )
    v17 = _read_input_manifest(
        v17_candidates,
        cohort=V17_COHORT,
        expected_sha256=expected_v17_manifest_sha256,
        expected_rows=EXPECTED_V17_ROWS,
    )
    all_candidates = (*v16.rows, *v17.rows)
    selected = _select_candidates(all_candidates, mode=mode)
    indexed = list(enumerate(selected))
    if workers == 1:
        evidence = [
            _materialize_source_row(
                item,
                mode=mode,
                media_probe=media_probe,
                production_probe_observed=production_probe_observed,
                publication_eligible=publication_eligible,
            )
            for item in indexed
        ]
    else:
        def materialize(item: tuple[int, Candidate]) -> SourceEvidence:
            return _materialize_source_row(
                item,
                mode=mode,
                media_probe=media_probe,
                production_probe_observed=production_probe_observed,
                publication_eligible=publication_eligible,
            )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            evidence = list(executor.map(materialize, indexed))
    rows = tuple(item.row for item in evidence)
    if [row["row_index"] for row in rows] != list(range(len(rows))):
        raise RuntimeError("internal source verification order differs")
    if len({str(row["iid"]) for row in rows}) != len(rows):
        raise RuntimeError("internal selected IID set differs")

    # Catch replacement or mutation of any selected source after its hash and
    # media probe, without paying for a second full content hash.
    for item in evidence:
        if (
            _stat_identity(item.path.lstat()) != item.stat_identity
            or _directory_binding_identity(item.parent_path.lstat())
            != item.parent_binding_identity
        ):
            raise GraftALiteReleaseError(
                f"source path or parent changed after verification: {item.path}"
            )
    for manifest in (v16, v17):
        final_raw, final_identity = _stable_read(
            manifest.path, label=f"{manifest.cohort} final stability check"
        )
        if final_raw != manifest.raw or final_identity != manifest.stat_identity:
            raise GraftALiteReleaseError(
                f"{manifest.cohort} candidate manifest changed during build"
            )
    probe_finalize()
    if _implementation_sha256() != implementation_sha256:
        raise GraftALiteReleaseError("builder implementation changed during build")

    manifest_raw = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    manifest_sha256 = bytes_sha256(manifest_raw)
    train_iids = [
        str(row["iid"]) for row in rows if row["split"] == "optimizer_train"
    ]
    confirmation_iids = [
        str(row["iid"])
        for row in rows
        if row["split"] == "optimizer_confirmation"
    ]
    input_descriptors = [
        _manifest_input_descriptor(v16),
        _manifest_input_descriptor(v17),
    ]
    receipt_core: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "complete",
        "release_id": f"graft-a-lite-{mode}-{manifest_sha256[:16]}",
        "release_mode": mode,
        "semantics": {
            "source_only": True,
            "same_clip_noop_only": True,
            "source_video_is_clean_noop_endpoint": True,
            "cross_clip_identity_authority": False,
            "action_authority": False,
            "quality_authority": False,
            "production_authority": False,
            "scientific_success_claimed": False,
            "canonical_noop_instruction": NOOP_INSTRUCTION,
        },
        "research_authorization_record": {
            "schema_version": RESEARCH_SCOPE_SCHEMA,
            "recorded_date": "2026-08-09",
            "source": "user_instruction_in_current_research_thread",
            "scope": "graft_a_lite_source_only_noop_nonproduction_experiment",
            "source_only_release_construction_requested": True,
            "nonproduction_experimental_training_in_scope": True,
            "data_governance_authority_claimed": False,
            "source_license_authority_claimed": False,
            "supersedes_upstream_release_or_receipt": False,
            "production_use_authorized": False,
        },
        "input_policy": {
            "allowed_external_artifacts": [
                "v16_raw_candidates_jsonl",
                "v17_raw_candidates_jsonl",
                "selected_source_mp4",
            ],
            "candidate_non_source_fields": {
                "values_not_copied": True,
                "whole_upstream_row_bytes_and_digest_committed": True,
            },
            "external_target_artifacts_opened": False,
            "wan_preview_opened": False,
            "generated_target_opened": False,
            "legacy_latent_or_receipt_opened": False,
            "anchor_image_opened": False,
            "code_frozen_v16_v17_manifest_pins_required": frozen_manifest_contract,
            "code_frozen_v16_v17_manifest_pins_matched": frozen_manifest_contract,
            "custom_manifest_test_path": not frozen_manifest_contract,
            "custom_manifest_path_publication_eligible": False,
        },
        "inputs": input_descriptors,
        "input_binding_digest": object_sha256(input_descriptors),
        "selection": {
            "combined_candidate_rows": len(all_candidates),
            "selected_rows": len(rows),
            "selected_iid_digest": object_sha256([row["iid"] for row in rows]),
            "v16_v17_iid_unique": True,
            "v16_v17_source_path_unique": True,
            "v16_v17_source_sha256_unique": True,
            "order": (
                "preregistered_core4_fit_then_confirmation"
                if mode == "canary4"
                else "v16_input_order_then_v17_input_order"
            ),
        },
        "split": {
            "policy": "preregistered_core4_override_then_fixed_iid_hash_mod10",
            "namespace": SPLIT_NAMESPACE,
            "optimizer_train_rows": len(train_iids),
            "optimizer_confirmation_rows": len(confirmation_iids),
            "optimizer_train_iid_digest": object_sha256(train_iids),
            "optimizer_confirmation_iid_digest": object_sha256(confirmation_iids),
            "iid_sets_disjoint": not bool(
                set(train_iids) & set(confirmation_iids)
            ),
            "stable_identity_labels_available": False,
            "stable_identity_disjoint_claimed": False,
            "optimizer_confirmation_update_intended": False,
            "optimizer_confirmation_update_authorized": False,
            "optimizer_confirmation_actual_use_claimed": False,
            "actual_optimizer_use_requires_future_training_execution_receipt": True,
            "core4_optimizer_confirmation_prior_research_exposure": True,
            "all_optimizer_confirmation_exposure_audited": mode == "canary4",
            "global_holdout": False,
        },
        "media_contract": {
            "source_sha256_verified_rows": len(rows),
            "source_stat_verified_rows": len(rows),
            "source_open_once_rows": len(rows),
            "hash_and_probe_same_open_fd": True,
            "source_sha256_recomputed_before_and_after_probe": True,
            "source_pre_post_probe_sha256_matched_rows": len(rows),
            "source_identity_includes_ctime_ns": True,
            "source_fstat_checked_at_each_probe_boundary": True,
            "source_path_and_parent_binding_revalidated": True,
            "probe_kind": probe_kind,
            "fresh_ffprobe": production_probe_observed,
            "fresh_ffprobe_verified_rows": (
                len(rows) if production_probe_observed else 0
            ),
            "test_only_probe_contract_matched_rows": (
                0 if production_probe_observed else len(rows)
            ),
            "publication_eligible": publication_eligible,
            "publication_eligible_is_structural_not_scientific_authority": True,
            "frame_count": FRAME_COUNT,
            "fps": FPS,
            "fps_fraction": f"{FPS_NUMERATOR}/{FPS_DENOMINATOR}",
            "short_side": SHORT_SIDE,
            "resolution_bound_per_row": True,
            "temporal_padding_allowed": False,
            "temporal_truncation_allowed": False,
            "retiming_allowed": False,
        },
        "training_consumer_requirements": {
            "must_revalidate_source_video_sha256": True,
            "must_fresh_probe_frame_count_fps_and_resolution": True,
            "must_hash_and_probe_same_open_source_fd": True,
            "must_recompute_source_sha256_before_and_after_probe": True,
            "must_bind_source_identity_including_ctime_ns": True,
            "must_revalidate_source_path_inode_and_parent_binding": True,
            "must_verify_preregistered_ffprobe_pin_in_sealed_runtime": True,
            "must_supply_independent_training_execution_receipt": True,
            "must_record_revalidation_in_training_execution_receipt": True,
            "must_reject_optimizer_confirmation_rows_for_updates": True,
            "actual_split_use_must_be_recorded_in_training_execution_receipt": True,
        },
        "implementation": {
            "path": str(Path(__file__).resolve(strict=True)),
            "sha256": implementation_sha256,
            "canonical_json": "ascii_sorted_keys_compact_finite_v1",
            "media_probe_kind": probe_kind,
            "ffprobe_executable_observation": dict(probe_implementation),
            "python_closure_or_token_immutability_claimed": False,
            "arbitrary_same_process_python_mutation_prevented": False,
            "arbitrary_same_process_python_mutation_in_provable_boundary": False,
            "sealed_runtime_archive_verified_by_this_receipt": False,
            "independent_execution_receipt_verified_by_this_receipt": False,
            "formal_runtime_authority_claimed": False,
            "formal_runtime_authority_requires_sealed_archive": True,
            "formal_runtime_authority_requires_independent_execution_receipt": True,
        },
        "publication": {
            "publication_eligible": publication_eligible,
            "publication_eligibility_scope": "structural_artifact_emission_only",
            "scientific_or_training_authority_implied": False,
            "create_only": True,
            "directory_atomic": False,
            "automatic_cleanup": False,
            "layout": "flat_sibling_artifacts_v1",
            "logical_output_stem_created": False,
            "manifest_suffix": MANIFEST_SUFFIX,
            "receipt_suffix": RECEIPT_SUFFIX,
            "parent_directory_pinned_openat": True,
            "parent_path_identity_revalidated": True,
            "receipt_last_commit": True,
            "commit_marker": "logical_output_stem_plus_receipt_suffix",
            "manifest_published_before_receipt": True,
            "receipt_complete_canonical_json_commit_marker": True,
            "receipt_filename_exists_during_write": True,
            "reader_must_parse_newline_terminated_canonical_json": True,
            "reader_must_verify_receipt_digest": True,
            "reader_must_require_producer_success": True,
            "receipt_alone_proves_producer_success": False,
            "producer_success_evidence": (
                "successful_publisher_return_or_orchestrator_execution_receipt"
            ),
            "post_receipt_failure_rolls_back": False,
            "published_file_mode": "0444",
            "reopen_verification_required": True,
        },
        "artifact": {
            "manifest_suffix": MANIFEST_SUFFIX,
            "receipt_suffix": RECEIPT_SUFFIX,
            "manifest_rows": len(rows),
            "manifest_bytes": len(manifest_raw),
            "manifest_sha256": manifest_sha256,
            "row_digest_sequence_sha256": object_sha256(
                [row["row_digest"] for row in rows]
            ),
        },
    }
    receipt = {**receipt_core, "receipt_digest": object_sha256(receipt_core)}
    receipt_raw = canonical_json_bytes(receipt) + b"\n"
    return ReleasePayload(
        rows=rows,
        manifest_bytes=manifest_raw,
        receipt=receipt,
        receipt_bytes=receipt_raw,
        publication_eligible=publication_eligible,
        probe_kind=probe_kind,
    )


def build_payload(
    *,
    v16_candidates: str | Path,
    v17_candidates: str | Path,
    mode: str = "canary4",
    workers: int = 8,
) -> ReleasePayload:
    """Build structurally publishable evidence against code-frozen AUH pins.

    This process does not and cannot prove immunity to arbitrary same-process
    Python mutation.  Formal authority is supplied separately by a sealed
    runtime archive and an independent execution receipt.
    """

    executable = _open_frozen_ffprobe()
    try:
        def production_probe(source: OpenedSource) -> Mapping[str, Any]:
            return probe_source_media(source, executable)

        return _build_payload(
            v16_candidates=v16_candidates,
            v17_candidates=v17_candidates,
            expected_v16_manifest_sha256=CURRENT_V16_MANIFEST_SHA256,
            expected_v17_manifest_sha256=CURRENT_V17_MANIFEST_SHA256,
            mode=mode,
            workers=workers,
            media_probe=production_probe,
            probe_kind=_PRODUCTION_PROBE_KIND,
            probe_implementation=executable.provenance,
            probe_finalize=lambda: _validate_opened_ffprobe(executable),
            frozen_manifest_contract=True,
        )
    finally:
        os.close(executable.fd)


def _build_test_payload(
    *,
    v16_candidates: str | Path,
    v17_candidates: str | Path,
    expected_v16_manifest_sha256: str,
    expected_v17_manifest_sha256: str,
    media_probe: MediaProbe,
    mode: str = "canary4",
    workers: int = 1,
) -> ReleasePayload:
    """Build an explicitly unpublishable payload with a test media double."""

    return _build_payload(
        v16_candidates=v16_candidates,
        v17_candidates=v17_candidates,
        expected_v16_manifest_sha256=expected_v16_manifest_sha256,
        expected_v17_manifest_sha256=expected_v17_manifest_sha256,
        mode=mode,
        workers=workers,
        media_probe=media_probe,
        probe_kind=_TEST_PROBE_KIND,
        probe_implementation={
            "test_only_injected_callback": True,
            "executable_path": None,
            "executable_file_sha256": None,
            "executable_version_stdout_sha256": None,
            "caller_process_observation_only": True,
            "trusted_or_official_authority_claimed": False,
        },
        probe_finalize=lambda: None,
        frozen_manifest_contract=False,
    )


def _parent_path_matches(parent: Path, identity: tuple[int, int]) -> bool:
    try:
        metadata = parent.lstat()
    except OSError:
        return False
    return stat.S_ISDIR(metadata.st_mode) and _inode_identity(metadata) == identity


def _pin_output_parent(
    logical_output_stem: Path,
) -> tuple[Path, str, int, tuple[int, int]]:
    raw = logical_output_stem.expanduser()
    if not raw.is_absolute() or not raw.name or raw.name in {".", ".."}:
        raise GraftALiteReleaseError("logical output stem must be an absolute leaf")
    if raw.name.endswith(MANIFEST_SUFFIX) or raw.name.endswith(RECEIPT_SUFFIX):
        raise GraftALiteReleaseError("logical output stem must not include artifact suffix")
    try:
        parent = raw.parent.resolve(strict=True)
    except OSError as error:
        raise GraftALiteReleaseError("output parent must already exist") from error
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(parent, flags)
    except OSError as error:
        raise GraftALiteReleaseError(f"cannot pin output parent: {error}") from error
    opened = os.fstat(descriptor)
    identity = _inode_identity(opened)
    if not stat.S_ISDIR(opened.st_mode) or not _parent_path_matches(parent, identity):
        os.close(descriptor)
        raise GraftALiteReleaseError("output parent path identity differs")
    return parent, raw.name, descriptor, identity


def _write_create_only_file(
    *, target_fd: int, name: str, payload: bytes
) -> None:
    """Create, fully write, chmod and fsync one artifact without rollback."""

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(
            name,
            flags,
            0o600,
            dir_fd=target_fd,
        )
    except FileExistsError as error:
        raise GraftALiteReleaseError(
            f"create-only artifact already exists: {name}"
        ) from error
    except OSError as error:
        raise GraftALiteReleaseError(f"cannot create {name}: {error}") from error
    try:
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("zero-byte write")
            offset += written
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    except OSError as error:
        raise GraftALiteReleaseError(f"cannot finish {name}: {error}") from error
    finally:
        os.close(descriptor)


def _read_published_file(*, target_fd: int, name: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=target_fd)
    except OSError as error:
        raise GraftALiteReleaseError(f"cannot reopen {name}: {error}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise GraftALiteReleaseError(f"published {name} is not a regular file")
        if stat.S_IMODE(metadata.st_mode) != 0o444:
            raise GraftALiteReleaseError(f"published {name} mode is not 0444")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _verify_published_release(
    parent: Path,
    *,
    expected_parent_identity: tuple[int, int],
    logical_stem_name: str,
    manifest_name: str,
    receipt_name: str,
    payload: ReleasePayload,
) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(parent, flags)
    except OSError as error:
        raise GraftALiteReleaseError(f"cannot reopen output parent: {error}") from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or _inode_identity(metadata) != expected_parent_identity
            or not _parent_path_matches(parent, expected_parent_identity)
        ):
            raise GraftALiteReleaseError("output parent identity differs")
        try:
            os.stat(logical_stem_name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        except OSError as error:
            raise GraftALiteReleaseError(
                f"cannot verify logical output stem absence: {error}"
            ) from error
        else:
            raise GraftALiteReleaseError("logical output stem unexpectedly exists")
        manifest_raw = _read_published_file(
            target_fd=descriptor, name=manifest_name
        )
        receipt_raw = _read_published_file(
            target_fd=descriptor, name=receipt_name
        )
    finally:
        os.close(descriptor)
    if manifest_raw != payload.manifest_bytes:
        raise GraftALiteReleaseError("reopened manifest bytes differ")
    if receipt_raw != payload.receipt_bytes or not receipt_raw.endswith(b"\n"):
        raise GraftALiteReleaseError("reopened receipt bytes differ or are incomplete")
    reopened = _decode_object(receipt_raw[:-1], context="published receipt")
    if canonical_json_bytes(reopened) + b"\n" != receipt_raw:
        raise GraftALiteReleaseError("published receipt is not canonical JSON newline")
    unsigned = dict(reopened)
    declared = unsigned.pop("receipt_digest", None)
    if declared != object_sha256(unsigned):
        raise GraftALiteReleaseError("published receipt digest differs")
    if reopened.get("artifact", {}).get("manifest_sha256") != bytes_sha256(
        manifest_raw
    ):
        raise GraftALiteReleaseError("published receipt does not bind manifest")


def publish_payload(
    logical_output_stem: str | Path, payload: ReleasePayload
) -> PublishedRelease:
    """Publish two create-only sibling files under one pinned parent fd."""

    if not isinstance(payload, ReleasePayload):
        raise GraftALiteReleaseError("payload must be a ReleasePayload")
    if (
        payload.publication_eligible is not True
        or payload.probe_kind != _PRODUCTION_PROBE_KIND
        or payload.receipt.get("media_contract", {}).get("fresh_ffprobe") is not True
        or payload.receipt.get("publication", {}).get("publication_eligible") is not True
        or payload.receipt.get("input_policy", {}).get(
            "code_frozen_v16_v17_manifest_pins_matched"
        ) is not True
    ):
        raise GraftALiteReleaseError(
            "payload is not publication-eligible fresh-ffprobe evidence"
        )
    if payload.manifest_bytes != b"".join(
        canonical_json_bytes(row) + b"\n" for row in payload.rows
    ):
        raise GraftALiteReleaseError("manifest payload is not canonical")
    receipt = dict(payload.receipt)
    declared_receipt_digest = receipt.pop("receipt_digest", None)
    if declared_receipt_digest != object_sha256(receipt):
        raise GraftALiteReleaseError("receipt digest differs")
    if payload.receipt_bytes != canonical_json_bytes(payload.receipt) + b"\n":
        raise GraftALiteReleaseError("receipt payload is not canonical")
    if payload.receipt["artifact"]["manifest_sha256"] != bytes_sha256(
        payload.manifest_bytes
    ):
        raise GraftALiteReleaseError("receipt does not bind manifest bytes")
    expected_publication = {
        "publication_eligible": True,
        "publication_eligibility_scope": "structural_artifact_emission_only",
        "scientific_or_training_authority_implied": False,
        "create_only": True,
        "directory_atomic": False,
        "automatic_cleanup": False,
        "layout": "flat_sibling_artifacts_v1",
        "logical_output_stem_created": False,
        "manifest_suffix": MANIFEST_SUFFIX,
        "receipt_suffix": RECEIPT_SUFFIX,
        "parent_directory_pinned_openat": True,
        "parent_path_identity_revalidated": True,
        "receipt_last_commit": True,
        "commit_marker": "logical_output_stem_plus_receipt_suffix",
        "manifest_published_before_receipt": True,
        "receipt_complete_canonical_json_commit_marker": True,
        "receipt_filename_exists_during_write": True,
        "reader_must_parse_newline_terminated_canonical_json": True,
        "reader_must_verify_receipt_digest": True,
        "reader_must_require_producer_success": True,
        "receipt_alone_proves_producer_success": False,
        "producer_success_evidence": (
            "successful_publisher_return_or_orchestrator_execution_receipt"
        ),
        "post_receipt_failure_rolls_back": False,
        "published_file_mode": "0444",
        "reopen_verification_required": True,
    }
    if payload.receipt.get("publication") != expected_publication:
        raise GraftALiteReleaseError("receipt publication contract differs")

    parent, stem_name, parent_fd, parent_identity = _pin_output_parent(
        Path(logical_output_stem)
    )
    manifest_name = f"{stem_name}{MANIFEST_SUFFIX}"
    receipt_name = f"{stem_name}{RECEIPT_SUFFIX}"
    for name in (stem_name, manifest_name, receipt_name):
        try:
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        except OSError as error:
            os.close(parent_fd)
            raise GraftALiteReleaseError(
                f"cannot preflight create-only output {name}: {error}"
            ) from error
        os.close(parent_fd)
        raise GraftALiteReleaseError(f"create-only output exists: {parent / name}")
    try:
        _write_create_only_file(
            target_fd=parent_fd,
            name=manifest_name,
            payload=payload.manifest_bytes,
        )
        os.fsync(parent_fd)
        if not _parent_path_matches(parent, parent_identity):
            raise GraftALiteReleaseError("output parent changed before receipt")

        # O_EXCL creates the receipt name before its bytes are complete.  It is
        # a commit marker only after strict canonical parse, digest validation,
        # newline validation, and a successful producer return.
        _write_create_only_file(
            target_fd=parent_fd,
            name=receipt_name,
            payload=payload.receipt_bytes,
        )
        os.fsync(parent_fd)
        if not _parent_path_matches(parent, parent_identity):
            raise GraftALiteReleaseError("output parent changed after receipt")
    finally:
        try:
            os.close(parent_fd)
        except OSError:
            pass
    # Verification deliberately happens after the receipt write.  Failure here
    # is reported and never removes either sibling artifact.
    _verify_published_release(
        parent,
        expected_parent_identity=parent_identity,
        logical_stem_name=stem_name,
        manifest_name=manifest_name,
        receipt_name=receipt_name,
        payload=payload,
    )
    stem = parent / stem_name
    return PublishedRelease(
        logical_output_stem=stem,
        manifest_path=parent / manifest_name,
        receipt_path=parent / receipt_name,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v16-candidates", type=Path, required=True)
    parser.add_argument("--v17-candidates", type=Path, required=True)
    parser.add_argument("--mode", choices=("canary4", "full1128"), default="canary4")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output-stem",
        type=Path,
        required=True,
        help=(
            "absolute logical stem; publishes <stem>.manifest.jsonl then "
            "<stem>.receipt.json"
        ),
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="create the two sibling artifacts; without this flag nothing is written",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_payload(
        v16_candidates=args.v16_candidates,
        v17_candidates=args.v17_candidates,
        mode=args.mode,
        workers=args.workers,
    )
    output: Optional[PublishedRelease] = None
    if args.publish:
        output = publish_payload(args.output_stem, payload)
    result = {
        "status": "published" if output is not None else "validated_not_published",
        "logical_output_stem": (
            str(args.output_stem) if output is None else str(output.logical_output_stem)
        ),
        "manifest_path": None if output is None else str(output.manifest_path),
        "receipt_path": None if output is None else str(output.receipt_path),
        "release_id": payload.receipt["release_id"],
        "release_mode": payload.receipt["release_mode"],
        "manifest_rows": payload.receipt["artifact"]["manifest_rows"],
        "manifest_sha256": payload.receipt["artifact"]["manifest_sha256"],
        "receipt_digest": payload.receipt["receipt_digest"],
        "same_clip_noop_only": True,
        "cross_clip_identity_authority": False,
    }
    print(canonical_json_bytes(result).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANARY4",
    "CURRENT_V16_MANIFEST_SHA256",
    "CURRENT_V17_MANIFEST_SHA256",
    "EXPECTED_FULL_ROWS",
    "GraftALiteReleaseError",
    "MANIFEST_SUFFIX",
    "NOOP_INSTRUCTION",
    "RECEIPT_SUFFIX",
    "ROW_SCHEMA",
    "RECEIPT_SCHEMA",
    "ReleasePayload",
    "PublishedRelease",
    "build_parser",
    "build_payload",
    "bytes_sha256",
    "canonical_json_bytes",
    "main",
    "object_sha256",
    "probe_source_media",
    "publish_payload",
]
