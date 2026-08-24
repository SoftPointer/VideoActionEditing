#!/usr/bin/env python3
"""Materialize independent Slurm terminal admission for GRAFT A-lite job 132549.

The submitted job cannot write evidence about its own terminal scheduler
state.  This CPU-only observer therefore runs later, invokes only the absolute
``/usr/bin/sacct`` executable, and admits only the top-level job row with
``State=COMPLETED`` and ``ExitCode=0:0``.  The fixed query also requires
``Start``, ``End``, ``Elapsed``, and ``NodeList``.  To remain byte-compatible
with the already frozen consumer schema, ``queried_fields`` names the three
fields that decide admission; the SHA-256 of the full canonical seven-field
selected record commits the validated timing and node fields without exposing
them as cleartext in the receipt.

Four externally SHA-pinned, read-only release artifacts are retained open
through the scheduler observation.  The receipt binds all four file digests
and the three receipt self-digests.  It grants no scientific, training,
identity, action, production, governance, or license authority.

``materializer.runtime_sha256`` is the canonical digest of the observed
Python executable SHA, absolute sacct executable SHA/path, fixed argv, fixed
environment, and stdout framing contract.  Both that digest and the
materializer source SHA are required as external inputs and compared before
publication; the downstream consumer independently pins them again.

The private fake-sacct seam emits a different, explicitly test-only schema and
cannot be passed to the publisher.  The public API has no state, exit-code,
stdout, executable, or command override.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Mapping, Optional, Sequence


JOB_ID = "132549"
SACCT_PATH = "/usr/bin/sacct"
SACCT_QUERY_FIELDS = (
    "JobIDRaw",
    "State",
    "ExitCode",
    "Start",
    "End",
    "Elapsed",
    "NodeList",
)
ADMISSION_DECISION_FIELDS = ("JobIDRaw", "State", "ExitCode")
SACCT_ARGUMENTS = (
    "-j",
    JOB_ID,
    "-X",
    "-n",
    "-P",
    "-o",
    ",".join(SACCT_QUERY_FIELDS),
)

ROW_SCHEMA = "bernini-graft-a-lite-source-noop-row-v1"
PRODUCER_SCHEMA = "bernini-graft-a-lite-source-noop-receipt-v1"
EXECUTION_SCHEMA = "bernini-graft-a-lite-source-noop-execution-receipt-v2"
SUBMISSION_SCHEMA = "bernini-graft-a-lite-source-submission-receipt-v1"
TERMINAL_SCHEMA = "bernini-graft-a-lite-source-independent-sacct-admission-v1"
MATERIALIZER_SCHEMA = "bernini-graft-independent-sacct-admission-materializer-v1"
TEST_TERMINAL_SCHEMA = (
    "bernini-graft-a-lite-source-independent-sacct-admission-test-only-v1"
)
TEST_MATERIALIZER_SCHEMA = (
    "bernini-graft-independent-sacct-admission-materializer-test-only-v1"
)

MANIFEST_SUFFIX = ".manifest.jsonl"
PRODUCER_SUFFIX = ".receipt.json"
EXECUTION_SUFFIX = ".execution.receipt.json"
SUBMISSION_SUFFIX = ".submission.receipt.json"
TERMINAL_SUFFIX = ".terminal.admission.receipt.json"

CANARY4 = (
    ("7b88a1ca1f804f41", "optimizer_train", True, False),
    ("a35b590961d24694", "optimizer_train", True, False),
    ("841b5e0080a1441d", "optimizer_confirmation", False, True),
    ("a66e6818e4144928", "optimizer_confirmation", False, True),
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_TIME_RE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\Z"
)
_ELAPSED_RE = re.compile(r"(?:(?P<days>[0-9]+)-)?(?P<h>[0-9]{2}):(?P<m>[0-9]{2}):(?P<s>[0-9]{2})\Z")
_NODE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._,\-\[\]]*\Z")

_AUTHORITY = {
    "action_authority": False,
    "identity_authority": False,
    "cross_clip_identity_authority": False,
    "quality_authority": False,
    "training_authority": False,
    "production_authority": False,
    "data_governance_authority": False,
    "data_license_authority": False,
    "scientific_success_claimed": False,
}


class TerminalAdmissionError(RuntimeError):
    """Raised without publishing a terminal admission receipt."""


@dataclass(frozen=True)
class TerminalArtifactPins:
    manifest_sha256: str
    producer_receipt_sha256: str
    execution_receipt_sha256: str
    submission_receipt_sha256: str
    materializer_implementation_sha256: str
    materializer_runtime_sha256: str

    def __post_init__(self) -> None:
        values: list[str] = []
        for name in self.__dataclass_fields__:
            values.append(_require_sha256(getattr(self, name), label=name))
        if len(set(values[:4])) != 4:
            raise TerminalAdmissionError("artifact SHA-256 pins must be pairwise distinct")


class _ProductionAdmissionPayload:
    """Internal bytes that are never accepted as a publication input."""

    __slots__ = ("receipt", "receipt_bytes")

    def __init__(self, *, receipt: Mapping[str, Any], receipt_bytes: bytes) -> None:
        object.__setattr__(self, "receipt", receipt)
        object.__setattr__(self, "receipt_bytes", receipt_bytes)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("production admission payload is immutable")


@dataclass(frozen=True)
class TestOnlyAdmissionObservation:
    """Untrusted fake result; deliberately not accepted by publication."""

    schema_version: str
    receipt_bytes: bytes
    test_only: bool = True
    publication_eligible: bool = False
    production_sacct_observed: bool = False


@dataclass(frozen=True)
class PublishedAdmission:
    path: Path
    sha256: str
    size_bytes: int


@dataclass
class _OpenedFile:
    path: Path
    fd: int
    identity: tuple[int, int, int, int, int, int, int]
    raw: bytes
    sha256: str


@dataclass(frozen=True)
class _ArtifactBundle:
    stem: Path
    manifest: _OpenedFile
    producer: _OpenedFile
    execution: _OpenedFile
    submission: _OpenedFile
    producer_digest: str
    execution_digest: str
    submission_digest: str


@dataclass(frozen=True)
class _ParsedSacctObservation:
    raw_stdout: bytes
    selected_record_sha256: str
    materializer_runtime_sha256: str


@dataclass
class _OutputParent:
    path: Path
    leaf_name: str
    fd: int
    identity: tuple[int, int]


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
        raise TerminalAdmissionError(f"non-canonical JSON value: {error}") from error


def bytes_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def object_sha256(value: Any) -> str:
    return bytes_sha256(canonical_json_bytes(value))


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise TerminalAdmissionError(f"{label} must be lowercase SHA-256")
    return value


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


def _directory_inode_identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _read_fd(fd: int, *, label: str) -> bytes:
    before = os.fstat(fd)
    chunks: list[bytes] = []
    if hasattr(os, "pread"):
        offset = 0
        while True:
            block = os.pread(fd, 1024 * 1024, offset)
            if not block:
                break
            chunks.append(block)
            offset += len(block)
    else:  # pragma: no cover
        original = os.lseek(fd, 0, os.SEEK_CUR)
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            for block in iter(lambda: os.read(fd, 1024 * 1024), b""):
                chunks.append(block)
        finally:
            os.lseek(fd, original, os.SEEK_SET)
    after = os.fstat(fd)
    if _identity(before) != _identity(after):
        raise TerminalAdmissionError(f"{label} changed while reading")
    return b"".join(chunks)


def _exact_path(path_value: str | Path, *, label: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        raise TerminalAdmissionError(f"{label} must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise TerminalAdmissionError(f"cannot resolve {label}: {error}") from error
    if resolved != path:
        raise TerminalAdmissionError(f"{label} must be its exact realpath")
    return path


def _open_artifact(
    path_value: str | Path, *, expected_sha256: str, label: str
) -> _OpenedFile:
    path = _exact_path(path_value, label=label)
    expected = _require_sha256(expected_sha256, label=f"{label} expected SHA-256")
    before = path.lstat()
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o444
        or before.st_nlink != 1
    ):
        raise TerminalAdmissionError(
            f"{label} must be regular mode-0444 link-count-one"
        )
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(fd)
        if _identity(opened) != _identity(before):
            raise TerminalAdmissionError(f"{label} changed while opening")
        raw = _read_fd(fd, label=label)
        observed = bytes_sha256(raw)
        if observed != expected:
            raise TerminalAdmissionError(f"{label} SHA-256 differs from external pin")
        return _OpenedFile(path, fd, _identity(opened), raw, observed)
    except Exception:
        os.close(fd)
        raise


def _revalidate(opened: _OpenedFile, *, label: str) -> None:
    try:
        fd_metadata = os.fstat(opened.fd)
        path_metadata = opened.path.lstat()
    except OSError as error:
        raise TerminalAdmissionError(f"cannot revalidate {label}: {error}") from error
    if _identity(fd_metadata) != opened.identity or _identity(path_metadata) != opened.identity:
        raise TerminalAdmissionError(f"{label} path or identity changed")
    if _read_fd(opened.fd, label=f"retained {label}") != opened.raw:
        raise TerminalAdmissionError(f"{label} bytes changed")


def _reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TerminalAdmissionError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise TerminalAdmissionError(f"non-finite JSON constant: {value}")


def _parse_object(raw: bytes, *, label: str) -> dict[str, Any]:
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise TerminalAdmissionError(f"{label} must be one newline-terminated object")
    try:
        value = json.loads(
            raw[:-1].decode("ascii"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_nonfinite,
        )
    except TerminalAdmissionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TerminalAdmissionError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) + b"\n" != raw:
        raise TerminalAdmissionError(f"{label} is not canonical JSON")
    return value


def _self_digest(value: Mapping[str, Any], *, label: str) -> str:
    unsigned = dict(value)
    declared = _require_sha256(
        unsigned.pop("receipt_digest", None), label=f"{label} receipt_digest"
    )
    if declared != object_sha256(unsigned):
        raise TerminalAdmissionError(f"{label} self digest differs")
    return declared


def _derive_stem(paths: Sequence[Path]) -> Path:
    manifest, producer, execution, submission = paths
    text = str(manifest)
    if not text.endswith(MANIFEST_SUFFIX):
        raise TerminalAdmissionError("manifest suffix differs")
    stem = Path(text[: -len(MANIFEST_SUFFIX)])
    if (
        producer != stem.with_name(f"{stem.name}{PRODUCER_SUFFIX}")
        or execution != stem.with_name(f"{stem.name}{EXECUTION_SUFFIX}")
        or submission != stem.with_name(f"{stem.name}{SUBMISSION_SUFFIX}")
    ):
        raise TerminalAdmissionError("four artifacts are not one sibling release")
    return stem


def _validate_manifest(raw: bytes) -> None:
    if not raw.endswith(b"\n"):
        raise TerminalAdmissionError("manifest is not newline terminated")
    lines = raw.splitlines(keepends=True)
    if len(lines) != 4 or b"".join(lines) != raw:
        raise TerminalAdmissionError("manifest is not fixed canary4")
    for index, (line, expected) in enumerate(zip(lines, CANARY4)):
        row = _parse_object(line, label=f"manifest row {index}")
        unsigned = dict(row)
        declared = _require_sha256(
            unsigned.pop("row_digest", None), label=f"manifest row {index} digest"
        )
        if declared != object_sha256(unsigned):
            raise TerminalAdmissionError(f"manifest row {index} self digest differs")
        iid, split, update, confirmation = expected
        if (
            row.get("schema_version") != ROW_SCHEMA
            or row.get("row_index") != index
            or row.get("iid") != iid
            or row.get("split") != split
            or row.get("optimizer_update_authorized") is not update
            or row.get("optimizer_confirmation_only") is not confirmation
        ):
            raise TerminalAdmissionError(f"manifest row {index} routing differs")


def _open_bundle(
    *,
    manifest_path: str | Path,
    producer_receipt_path: str | Path,
    execution_receipt_path: str | Path,
    submission_receipt_path: str | Path,
    pins: TerminalArtifactPins,
) -> _ArtifactBundle:
    if not isinstance(pins, TerminalArtifactPins):
        raise TerminalAdmissionError("pins must be TerminalArtifactPins")
    opened: list[_OpenedFile] = []
    try:
        for path, wanted, label in (
            (manifest_path, pins.manifest_sha256, "manifest"),
            (producer_receipt_path, pins.producer_receipt_sha256, "producer receipt"),
            (execution_receipt_path, pins.execution_receipt_sha256, "execution receipt"),
            (submission_receipt_path, pins.submission_receipt_sha256, "submission receipt"),
        ):
            opened.append(_open_artifact(path, expected_sha256=wanted, label=label))
        manifest, producer_file, execution_file, submission_file = opened
        stem = _derive_stem([item.path for item in opened])
        _validate_manifest(manifest.raw)
        producer = _parse_object(producer_file.raw, label="producer receipt")
        producer_digest = _self_digest(producer, label="producer receipt")
        artifact = producer.get("artifact")
        if (
            producer.get("schema_version") != PRODUCER_SCHEMA
            or producer.get("status") != "complete"
            or producer.get("release_mode") != "canary4"
            or not isinstance(artifact, Mapping)
            or artifact.get("manifest_rows") != 4
            or artifact.get("manifest_sha256") != manifest.sha256
        ):
            raise TerminalAdmissionError("producer does not bind complete canary4")
        execution = _parse_object(execution_file.raw, label="execution receipt")
        execution_digest = _self_digest(execution, label="execution receipt")
        slurm = execution.get("slurm")
        outputs = execution.get("outputs")
        if (
            execution.get("schema_version") != EXECUTION_SCHEMA
            or execution.get("status") != "complete"
            or execution.get("successful_return") is not True
            or execution.get("builder_successful_return") is not True
            or not isinstance(slurm, Mapping)
            or slurm.get("job_id") != JOB_ID
            or not isinstance(outputs, Mapping)
            or outputs.get("logical_output_stem") != str(stem)
            or outputs.get("manifest_rows") != 4
            or outputs.get("producer_receipt_digest") != producer_digest
            or outputs.get("manifest", {}).get("sha256") != manifest.sha256
            or outputs.get("producer_receipt", {}).get("sha256")
            != producer_file.sha256
        ):
            raise TerminalAdmissionError("execution receipt binding differs")
        failure = execution.get("failure_semantics")
        if not isinstance(failure, Mapping) or (
            failure.get("consumer_must_also_require_slurm_completed_exit_zero")
            is not True
            or failure.get("receipt_alone_proves_successful_process_return")
            is not False
        ):
            raise TerminalAdmissionError("execution is not terminal evidence")
        submission = _parse_object(submission_file.raw, label="submission receipt")
        submission_digest = _self_digest(submission, label="submission receipt")
        submitted_job = submission.get("submitted_job")
        submission_outputs = submission.get("outputs")
        if (
            submission.get("schema_version") != SUBMISSION_SCHEMA
            or submission.get("status") != "submitted"
            or submission.get("submission_success") is not True
            or submission.get("job_success") is not None
            or submission.get("job_terminal_state_observed") is not False
            or not isinstance(submitted_job, Mapping)
            or submitted_job.get("job_id") != JOB_ID
            or not isinstance(submission_outputs, Mapping)
            or submission_outputs.get("logical_output_stem") != str(stem)
            or submission_outputs.get("submission_receipt_path")
            != str(submission_file.path)
        ):
            raise TerminalAdmissionError(
                "submission must remain non-terminal and bind job132549"
            )
        return _ArtifactBundle(
            stem=stem,
            manifest=manifest,
            producer=producer_file,
            execution=execution_file,
            submission=submission_file,
            producer_digest=producer_digest,
            execution_digest=execution_digest,
            submission_digest=submission_digest,
        )
    except Exception:
        for item in reversed(opened):
            os.close(item.fd)
        raise


def _close_bundle(bundle: _ArtifactBundle) -> None:
    for item in (bundle.submission, bundle.execution, bundle.producer, bundle.manifest):
        try:
            os.close(item.fd)
        except OSError:
            pass


def _validate_time(value: str, *, label: str) -> datetime:
    if not isinstance(value, str) or _TIME_RE.fullmatch(value) is None:
        raise TerminalAdmissionError(f"sacct {label} time differs")
    try:
        return datetime.fromisoformat(value)
    except ValueError as error:  # pragma: no cover - regex handles normal invalidity
        raise TerminalAdmissionError(f"sacct {label} time differs") from error


def _parse_sacct_stdout(raw: bytes, *, runtime_sha: str) -> _ParsedSacctObservation:
    """Parse and validate bytes only; this function never grants publication trust."""

    if not isinstance(raw, bytes) or not raw.endswith(b"\n") or not raw:
        raise TerminalAdmissionError("sacct stdout framing differs")
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise TerminalAdmissionError("sacct stdout is not ASCII") from error
    records: list[dict[str, str]] = []
    for line in lines:
        if not line:
            raise TerminalAdmissionError("sacct stdout contains an empty row")
        fields = line.split("|")
        if fields and fields[-1] == "":
            fields.pop()
        if len(fields) != len(SACCT_QUERY_FIELDS):
            raise TerminalAdmissionError("sacct row field count differs")
        records.append(dict(zip(SACCT_QUERY_FIELDS, fields)))
    selected = [record for record in records if record["JobIDRaw"] == JOB_ID]
    if len(selected) != 1 or len(records) != 1:
        raise TerminalAdmissionError("sacct must return exactly one top-level job row")
    record = selected[0]
    if record["State"] != "COMPLETED" or record["ExitCode"] != "0:0":
        raise TerminalAdmissionError("job132549 is not COMPLETED ExitCode 0:0")
    start = _validate_time(record["Start"], label="Start")
    end = _validate_time(record["End"], label="End")
    if end < start:
        raise TerminalAdmissionError("sacct End precedes Start")
    elapsed = _ELAPSED_RE.fullmatch(record["Elapsed"])
    if elapsed is None or int(elapsed.group("m")) >= 60 or int(elapsed.group("s")) >= 60:
        raise TerminalAdmissionError("sacct Elapsed differs")
    elapsed_seconds = (
        int(elapsed.group("days") or "0") * 86400
        + int(elapsed.group("h")) * 3600
        + int(elapsed.group("m")) * 60
        + int(elapsed.group("s"))
    )
    if int((end - start).total_seconds()) != elapsed_seconds:
        raise TerminalAdmissionError("sacct Start/End/Elapsed are inconsistent")
    node = record["NodeList"]
    if (
        not node
        or node.lower() in {"none", "unknown", "n/a", "(null)"}
        or _NODE_RE.fullmatch(node) is None
    ):
        raise TerminalAdmissionError("sacct NodeList differs")
    return _ParsedSacctObservation(
        raw_stdout=raw,
        selected_record_sha256=object_sha256(record),
        materializer_runtime_sha256=_require_sha256(
            runtime_sha, label="materializer runtime observation"
        ),
    )


def _open_plain_observation(path_value: str | Path, *, label: str) -> _OpenedFile:
    path = _exact_path(path_value, label=label)
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise TerminalAdmissionError(f"{label} must be a regular file")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(fd)
        if _identity(opened) != _identity(before):
            raise TerminalAdmissionError(f"{label} changed while opening")
        raw = _read_fd(fd, label=label)
        return _OpenedFile(path, fd, _identity(opened), raw, bytes_sha256(raw))
    except Exception:
        os.close(fd)
        raise


def _running_python_observation() -> _OpenedFile:
    if sys.platform.startswith("linux") and Path("/proc/self/exe").exists():
        source = Path("/proc/self/exe")
        fd = os.open(source, os.O_RDONLY)
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode):
                raise TerminalAdmissionError("running Python executable is not regular")
            raw = _read_fd(fd, label="running Python executable")
            return _OpenedFile(
                Path(sys.executable).resolve(strict=True),
                fd,
                _identity(metadata),
                raw,
                bytes_sha256(raw),
            )
        except Exception:
            os.close(fd)
            raise
    return _open_plain_observation(
        Path(sys.executable).resolve(strict=True), label="running Python executable"
    )


def _observe_implementation() -> _OpenedFile:
    return _open_plain_observation(Path(__file__).resolve(strict=True), label="materializer code")


def _observe_production_sacct(
    *, expected_runtime_sha256: str
) -> _ParsedSacctObservation:
    """Run the code-frozen command and require its externally pinned runtime."""

    code = _observe_implementation()
    python = _running_python_observation()
    sacct = _open_plain_observation(SACCT_PATH, label="absolute /usr/bin/sacct")
    try:
        if not sacct.identity[5] & 0o111:
            raise TerminalAdmissionError("absolute /usr/bin/sacct is not executable")
        environment = {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"}
        command = [SACCT_PATH, *SACCT_ARGUMENTS]
        runtime_observation = {
            "python_executable_sha256": python.sha256,
            "sacct_absolute_path": SACCT_PATH,
            "sacct_executable_sha256": sacct.sha256,
            "sacct_arguments": list(SACCT_ARGUMENTS),
            "environment": environment,
            "stdout_contract": "pipe_delimited_no_header_exact_one_top_level_row_v1",
        }
        runtime_sha256 = object_sha256(runtime_observation)
        if runtime_sha256 != _require_sha256(
            expected_runtime_sha256, label="expected materializer runtime"
        ):
            raise TerminalAdmissionError(
                "materializer runtime observation differs from external pin"
            )
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=60,
            env=environment,
        )
        if completed.returncode != 0:
            raise TerminalAdmissionError(
                f"/usr/bin/sacct failed with exit {completed.returncode}; "
                f"stderr_sha256={bytes_sha256(completed.stderr)}"
            )
        for opened, label in (
            (code, "materializer code"),
            (python, "running Python executable"),
            (sacct, "absolute /usr/bin/sacct"),
        ):
            # /proc/self/exe has no stable pathname binding; its retained fd is
            # still re-read.  The other two also retain exact path identity.
            if opened is python and sys.platform.startswith("linux"):
                if _read_fd(opened.fd, label=label) != opened.raw:
                    raise TerminalAdmissionError(f"{label} bytes changed")
            else:
                _revalidate(opened, label=label)
        return _parse_sacct_stdout(completed.stdout, runtime_sha=runtime_sha256)
    finally:
        for opened in (sacct, python, code):
            os.close(opened.fd)


def _receipt_core(
    bundle: _ArtifactBundle,
    observation: _ParsedSacctObservation,
    *,
    implementation_sha256: str,
    schema_version: str,
    status: str,
    materializer_schema: str,
    source: str,
    independently_observed: bool,
) -> tuple[Mapping[str, Any], bytes]:
    core: dict[str, Any] = {
        "schema_version": schema_version,
        "status": status,
        "materializer": {
            "schema_version": materializer_schema,
            "implementation_sha256": implementation_sha256,
            "runtime_sha256": observation.materializer_runtime_sha256,
            "independent_of_submitted_job_process": independently_observed,
            "job_process_wrote_this_receipt": False,
            "observed_after_job_became_terminal": independently_observed,
        },
        "sacct_admission": {
            "source": source,
            # These are admission-decision fields.  The selected-record hash
            # commits all seven SACCT_QUERY_FIELDS, including time and node.
            "queried_fields": list(ADMISSION_DECISION_FIELDS),
            "job_id": JOB_ID,
            "state": "COMPLETED",
            "exit_code": "0:0",
            "terminal_state_observed": independently_observed,
            "job_success": independently_observed,
            "raw_stdout_sha256": bytes_sha256(observation.raw_stdout),
            "raw_stdout_size_bytes": len(observation.raw_stdout),
            "selected_record_sha256": observation.selected_record_sha256,
        },
        "artifact_bindings": {
            "manifest_file_sha256": bundle.manifest.sha256,
            "producer_receipt_file_sha256": bundle.producer.sha256,
            "producer_receipt_digest": bundle.producer_digest,
            "execution_receipt_file_sha256": bundle.execution.sha256,
            "execution_receipt_digest": bundle.execution_digest,
            "submission_receipt_file_sha256": bundle.submission.sha256,
            "submission_receipt_digest": bundle.submission_digest,
        },
        "authority": dict(_AUTHORITY),
    }
    receipt = {**core, "receipt_digest": object_sha256(core)}
    raw = canonical_json_bytes(receipt) + b"\n"
    return receipt, raw


def _test_receipt_from_observation(
    bundle: _ArtifactBundle,
    observation: _ParsedSacctObservation,
    *,
    implementation_sha256: str,
) -> TestOnlyAdmissionObservation:
    if type(observation) is not _ParsedSacctObservation:
        raise TerminalAdmissionError("test sacct parse result differs")
    _, raw = _receipt_core(
        bundle,
        observation,
        implementation_sha256=implementation_sha256,
        schema_version=TEST_TERMINAL_SCHEMA,
        status="test_only_untrusted",
        materializer_schema=TEST_MATERIALIZER_SCHEMA,
        source="test_only_fake_sacct",
        independently_observed=False,
    )
    return TestOnlyAdmissionObservation(
        schema_version=TEST_TERMINAL_SCHEMA,
        receipt_bytes=raw,
    )


def _build_production_payload(
    bundle: _ArtifactBundle, *, pins: TerminalArtifactPins
) -> _ProductionAdmissionPayload:
    """The sole neutral-parse-to-production transition; it owns real sacct."""

    if type(pins) is not TerminalArtifactPins:
        raise TerminalAdmissionError("pins must be TerminalArtifactPins")
    code = _observe_implementation()
    try:
        implementation_sha = code.sha256
        if implementation_sha != pins.materializer_implementation_sha256:
            raise TerminalAdmissionError(
                "materializer implementation differs from external pin"
            )
        observation = _observe_production_sacct(
            expected_runtime_sha256=pins.materializer_runtime_sha256
        )
        if (
            type(observation) is not _ParsedSacctObservation
            or observation.materializer_runtime_sha256
            != pins.materializer_runtime_sha256
        ):
            raise TerminalAdmissionError(
                "production sacct observation/runtime binding differs"
            )
        _revalidate(code, label="materializer code after sacct")
        receipt, raw = _receipt_core(
            bundle,
            observation,
            implementation_sha256=implementation_sha,
            schema_version=TERMINAL_SCHEMA,
            status="admitted",
            materializer_schema=MATERIALIZER_SCHEMA,
            source="sacct",
            independently_observed=True,
        )
        return _ProductionAdmissionPayload(
            receipt=receipt,
            receipt_bytes=raw,
        )
    finally:
        os.close(code.fd)


def _build_with_test_sacct(
    *,
    manifest_path: str | Path,
    producer_receipt_path: str | Path,
    execution_receipt_path: str | Path,
    submission_receipt_path: str | Path,
    pins: TerminalArtifactPins,
    fake_sacct_stdout: bytes,
) -> TestOnlyAdmissionObservation:
    """CPU unit-test seam.  Its receipt schema is never publication eligible."""

    bundle = _open_bundle(
        manifest_path=manifest_path,
        producer_receipt_path=producer_receipt_path,
        execution_receipt_path=execution_receipt_path,
        submission_receipt_path=submission_receipt_path,
        pins=pins,
    )
    code = _observe_implementation()
    try:
        observation = _parse_sacct_stdout(
            fake_sacct_stdout,
            runtime_sha=object_sha256(
                {
                    "kind": "test_only_fake_sacct_runtime_v1",
                    "python": sys.version_info[:3],
                }
            ),
        )
        for item, label in (
            (bundle.manifest, "manifest"),
            (bundle.producer, "producer receipt"),
            (bundle.execution, "execution receipt"),
            (bundle.submission, "submission receipt"),
        ):
            _revalidate(item, label=label)
        return _test_receipt_from_observation(
            bundle, observation, implementation_sha256=code.sha256
        )
    finally:
        os.close(code.fd)
        _close_bundle(bundle)


def _pin_output_parent(output_path: str | Path, *, expected_stem: Path) -> _OutputParent:
    path = Path(output_path)
    expected = expected_stem.with_name(f"{expected_stem.name}{TERMINAL_SUFFIX}")
    if not path.is_absolute() or path != expected:
        raise TerminalAdmissionError("terminal output path must be the release sibling suffix")
    parent = path.parent
    try:
        before = parent.lstat()
        resolved = parent.resolve(strict=True)
    except OSError as error:
        raise TerminalAdmissionError(f"cannot resolve output parent: {error}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode) or resolved != parent:
        raise TerminalAdmissionError("output parent must be an exact plain directory")
    fd = os.open(
        parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(fd)
        identity = _directory_inode_identity(opened)
        if identity != _directory_inode_identity(before):
            raise TerminalAdmissionError("output parent changed while opening")
        try:
            os.stat(path.name, dir_fd=fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise TerminalAdmissionError("create-only terminal receipt already exists")
        return _OutputParent(parent, path.name, fd, identity)
    except Exception:
        os.close(fd)
        raise


def _parent_matches(output: _OutputParent) -> bool:
    try:
        path_metadata = output.path.lstat()
        fd_metadata = os.fstat(output.fd)
    except OSError:
        return False
    return (
        stat.S_ISDIR(path_metadata.st_mode)
        and not stat.S_ISLNK(path_metadata.st_mode)
        and _directory_inode_identity(path_metadata) == output.identity
        and _directory_inode_identity(fd_metadata) == output.identity
    )


def _publish_production_at(
    output: _OutputParent,
    bundle: _ArtifactBundle,
    *,
    pins: TerminalArtifactPins,
) -> PublishedAdmission:
    """Build from real sacct and publish; accepts no caller-supplied payload bytes."""

    payload = _build_production_payload(bundle, pins=pins)
    for item, label in (
        (bundle.manifest, "manifest"),
        (bundle.producer, "producer receipt"),
        (bundle.execution, "execution receipt"),
        (bundle.submission, "submission receipt"),
    ):
        _revalidate(item, label=label)
    if type(payload) is not _ProductionAdmissionPayload or (
        payload.receipt.get("schema_version") != TERMINAL_SCHEMA
        or payload.receipt.get("status") != "admitted"
        or payload.receipt_bytes != canonical_json_bytes(payload.receipt) + b"\n"
    ):
        raise TerminalAdmissionError("payload is not production terminal admission")
    unsigned = dict(payload.receipt)
    declared = unsigned.pop("receipt_digest", None)
    if declared != object_sha256(unsigned):
        raise TerminalAdmissionError("terminal payload self digest differs")
    if not _parent_matches(output):
        raise TerminalAdmissionError("output parent changed before publication")
    fd = os.open(
        output.leaf_name,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=output.fd,
    )
    try:
        view = memoryview(payload.receipt_bytes)
        offset = 0
        while offset < len(view):
            written = os.write(fd, view[offset:])
            if written <= 0:
                raise TerminalAdmissionError("zero-byte terminal receipt write")
            offset += written
        os.fchmod(fd, 0o444)
        os.fsync(fd)
        metadata = os.fstat(fd)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o444
            or metadata.st_nlink != 1
        ):
            raise TerminalAdmissionError("terminal receipt seal differs")
        if _read_fd(fd, label="terminal receipt same-fd reopen") != payload.receipt_bytes:
            raise TerminalAdmissionError("terminal receipt same-fd reopen differs")
        leaf = os.stat(output.leaf_name, dir_fd=output.fd, follow_symlinks=False)
        if _identity(leaf) != _identity(metadata) or not _parent_matches(output):
            raise TerminalAdmissionError("terminal receipt path binding differs")
    finally:
        os.close(fd)
    # Last fallible durable action.  No semantic operation follows it.
    os.fsync(output.fd)
    return PublishedAdmission(
        path=output.path / output.leaf_name,
        sha256=bytes_sha256(payload.receipt_bytes),
        size_bytes=len(payload.receipt_bytes),
    )


def materialize_graft_a_lite_terminal_admission(
    *,
    manifest_path: str | Path,
    producer_receipt_path: str | Path,
    execution_receipt_path: str | Path,
    submission_receipt_path: str | Path,
    output_path: str | Path,
    pins: TerminalArtifactPins,
) -> PublishedAdmission:
    """Observe fixed job132549 via fixed /usr/bin/sacct and publish once."""

    bundle = _open_bundle(
        manifest_path=manifest_path,
        producer_receipt_path=producer_receipt_path,
        execution_receipt_path=execution_receipt_path,
        submission_receipt_path=submission_receipt_path,
        pins=pins,
    )
    output = _pin_output_parent(output_path, expected_stem=bundle.stem)
    try:
        return _publish_production_at(output, bundle, pins=pins)
    finally:
        os.close(output.fd)
        _close_bundle(bundle)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--producer-receipt", type=Path, required=True)
    parser.add_argument("--producer-receipt-sha256", required=True)
    parser.add_argument("--execution-receipt", type=Path, required=True)
    parser.add_argument("--execution-receipt-sha256", required=True)
    parser.add_argument("--submission-receipt", type=Path, required=True)
    parser.add_argument("--submission-receipt-sha256", required=True)
    parser.add_argument("--materializer-implementation-sha256", required=True)
    parser.add_argument("--materializer-runtime-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = materialize_graft_a_lite_terminal_admission(
        manifest_path=args.manifest,
        producer_receipt_path=args.producer_receipt,
        execution_receipt_path=args.execution_receipt,
        submission_receipt_path=args.submission_receipt,
        output_path=args.output,
        pins=TerminalArtifactPins(
            manifest_sha256=args.manifest_sha256,
            producer_receipt_sha256=args.producer_receipt_sha256,
            execution_receipt_sha256=args.execution_receipt_sha256,
            submission_receipt_sha256=args.submission_receipt_sha256,
            materializer_implementation_sha256=(
                args.materializer_implementation_sha256
            ),
            materializer_runtime_sha256=args.materializer_runtime_sha256,
        ),
    )
    print(
        canonical_json_bytes(
            {
                "status": "published",
                "job_id": JOB_ID,
                "path": str(result.path),
                "sha256": result.sha256,
                "size_bytes": result.size_bytes,
                "authority": dict(_AUTHORITY),
            }
        ).decode("ascii"),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ADMISSION_DECISION_FIELDS",
    "JOB_ID",
    "MATERIALIZER_SCHEMA",
    "SACCT_ARGUMENTS",
    "SACCT_PATH",
    "SACCT_QUERY_FIELDS",
    "TERMINAL_SCHEMA",
    "TERMINAL_SUFFIX",
    "TerminalAdmissionError",
    "TerminalArtifactPins",
    "PublishedAdmission",
    "canonical_json_bytes",
    "materialize_graft_a_lite_terminal_admission",
    "object_sha256",
]
