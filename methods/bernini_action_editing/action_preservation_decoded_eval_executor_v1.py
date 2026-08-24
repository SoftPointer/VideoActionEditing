#!/usr/bin/env python3
"""Local, create-only executor for one decoded-evaluation holder shard.

The executor has no scheduler, SSH, network, upload, retry, or loss-ranking
code.  A caller must explicitly invoke one local shard.  Every attempted task
is claimed once and retains its request, process logs, staging media, and a
terminal success/failure receipt.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

import action_preservation_decoded_eval_plan_v1 as plan
import action_preservation_decoded_eval_bridge_v1 as bridge
import action_preservation_decoded_eval_decoder_adapter_v1 as decoder_adapter


TASK_INPUT_SCHEMA = "bernini-action-preservation-decode-task-input-v1"
PROCESS_SCHEMA = "bernini-action-preservation-decode-process-v1"
TASK_OUTPUT_SCHEMA = "bernini-action-preservation-decode-task-output-v1"
TASK_FAILURE_SCHEMA = "bernini-action-preservation-decode-task-failure-v1"
SHARD_SUMMARY_SCHEMA = "bernini-action-preservation-decode-shard-summary-v1"

EXECUTION_DIRECTORY = "execution_shards"
INPUT_RECEIPT_FILENAME = "input_receipt.json"
STDOUT_FILENAME = "decoder.stdout"
STDERR_FILENAME = "decoder.stderr"
PROCESS_RECEIPT_FILENAME = "process_receipt.json"
STAGING_VIDEO_FILENAME = "candidate.staging.mp4"
OUTPUT_RECEIPT_FILENAME = "output_receipt.json"
FAILURE_RECEIPT_FILENAME = "failure_receipt.json"
SUMMARY_FILENAME = "shard_summary.json"
DECODER_RUNTIME_CAPTURE_FILENAME = "decoder_verified_runtime_capture.json"
SUBPROCESS_ENV_DENYLIST = (
    "BASH_ENV",
    "ENV",
    "ZDOTDIR",
    "PYTHONSTARTUP",
    "PYTHONINSPECT",
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}")


class DecodedEvaluationExecutorError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return plan.canonical_json_bytes(value)
    except plan.DecodedEvaluationPlanError as error:
        raise DecodedEvaluationExecutorError(str(error)) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: str | Path) -> str:
    return plan.file_sha256(path)


def _closed(value: Any, fields: set[str] | frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise DecodedEvaluationExecutorError(f"{label} field closure differs")
    return value


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DecodedEvaluationExecutorError(f"{label} is not a lowercase SHA-256")
    return value


def _identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise DecodedEvaluationExecutorError(f"{label} is invalid")
    return value


def _verify_digest(value: Mapping[str, Any], *, field: str, label: str) -> str:
    digest = _sha(value.get(field), label=f"{label} digest")
    payload = dict(value)
    payload.pop(field)
    if object_sha256(payload) != digest:
        raise DecodedEvaluationExecutorError(f"{label} digest differs")
    return digest


def _plain_file(path: Path, *, label: str) -> Path:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise DecodedEvaluationExecutorError(f"{label} does not exist") from error
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise DecodedEvaluationExecutorError(f"{label} is not a plain file")
    return path


def _plain_directory(path: Path, *, label: str) -> Path:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise DecodedEvaluationExecutorError(f"{label} does not exist") from error
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise DecodedEvaluationExecutorError(f"{label} is not a plain directory")
    return path


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw, _ = bridge._stable_file(path, label=label)
        value = json.loads(raw.decode("utf-8"))
    except bridge.DecodedEvaluationBridgeError as error:
        raise DecodedEvaluationExecutorError(str(error)) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DecodedEvaluationExecutorError(f"cannot load {label}: {error}") from error
    if not isinstance(value, Mapping):
        raise DecodedEvaluationExecutorError(f"{label} root is not an object")
    return dict(value)


def _write_create_only(path: Path, payload: bytes, *, mode: int = 0o400) -> None:
    _plain_directory(path.parent, label="create-only parent")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, mode)
    except FileExistsError as error:
        raise DecodedEvaluationExecutorError(
            f"refusing to overwrite create-only artifact: {path}"
        ) from error
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise DecodedEvaluationExecutorError(
                    "create-only write made no progress"
                )
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if path.read_bytes() != payload:
        raise DecodedEvaluationExecutorError("create-only reread differs")


def _write_create_only_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_create_only(path, canonical_json_bytes(value) + b"\n")


def _ensure_directory_tree(root: Path, relative_parent: Path) -> Path:
    if relative_parent.is_absolute() or ".." in relative_parent.parts:
        raise DecodedEvaluationExecutorError("relative output parent escapes root")
    current = _plain_directory(root, label="evaluation root")
    for component in relative_parent.parts:
        if component in ("", "."):
            continue
        current = current / component
        try:
            os.mkdir(current, 0o700)
        except FileExistsError:
            pass
        _plain_directory(current, label="output directory")
    return current


def _claim_directory(path: Path, *, label: str) -> Path:
    _plain_directory(path.parent, label=f"{label} parent")
    try:
        os.mkdir(path, 0o700)
    except FileExistsError as error:
        raise DecodedEvaluationExecutorError(
            f"{label} already exists; retries are forbidden: {path}"
        ) from error
    return _plain_directory(path, label=label)


def load_published_bundle(evaluation_root: str | Path) -> dict[str, Any]:
    root = Path(evaluation_root)
    if not root.is_absolute() or str(root) == os.path.sep:
        raise DecodedEvaluationExecutorError("evaluation root must be absolute and non-root")
    _plain_directory(root, label="evaluation root")
    input_spec = _load_json(root / plan.INPUT_FILENAME, label="evaluation input")
    review = _load_json(
        root / plan.REVIEW_CONTRACT_FILENAME, label="review packet contract"
    )
    manifest = _load_json(root / plan.MANIFEST_FILENAME, label="evaluation manifest")
    shards = {
        holder["job_id"]: _load_json(
            root / plan.SHARD_DIRECTORY / f"{holder['job_id']}.json",
            label=f"holder {holder['job_id']} shard",
        )
        for holder in plan.HOLDER_ROWS
    }
    bundle = {
        "input_spec": input_spec,
        "review_contract": review,
        "manifest": manifest,
        "shards": shards,
    }
    receipt = _load_json(root / plan.PUBLICATION_FILENAME, label="publication receipt")
    try:
        validated_receipt = plan.validate_publication_receipt(receipt, bundle=bundle)
    except plan.DecodedEvaluationPlanError as error:
        raise DecodedEvaluationExecutorError(str(error)) from error
    for item in validated_receipt["files"]:
        path = _plain_file(root / item["relpath"], label=item["relpath"])
        if file_sha256(path) != item["sha256"]:
            raise DecodedEvaluationExecutorError(
                f"published bundle file hash differs: {item['relpath']}"
            )
    if manifest["evaluation_root"] != str(root):
        raise DecodedEvaluationExecutorError("manifest evaluation root binding differs")
    return {**bundle, "publication_receipt": validated_receipt}


def _task_id(task: Mapping[str, Any]) -> str:
    if task["task_kind"] == "adapter_candidate":
        return _identifier(task["record"]["candidate_id"], label="candidate id")
    if task["task_kind"] == "frozen_base_control":
        return _identifier(task["record"]["control_id"], label="control id")
    raise DecodedEvaluationExecutorError("task kind differs")


def _tool_identity(value: Any, *, label: str, verify_file: bool) -> dict[str, str]:
    row = dict(_closed(value, {"path", "sha256"}, label=label))
    path = row["path"]
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise DecodedEvaluationExecutorError(f"{label} path must be absolute")
    _sha(row["sha256"], label=f"{label} SHA")
    if verify_file:
        file_path = _plain_file(Path(path), label=label)
        if file_sha256(file_path) != row["sha256"]:
            raise DecodedEvaluationExecutorError(f"{label} file hash differs")
        if not os.access(file_path, os.X_OK):
            raise DecodedEvaluationExecutorError(f"{label} is not executable")
    return row


def _artifact_identity(value: Any, *, label: str, verify_file: bool) -> dict[str, str]:
    row = dict(_closed(value, {"path", "sha256"}, label=label))
    path = row["path"]
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise DecodedEvaluationExecutorError(f"{label} path must be absolute")
    _sha(row["sha256"], label=f"{label} SHA")
    if verify_file:
        file_path = _plain_file(Path(path), label=label)
        if file_sha256(file_path) != row["sha256"]:
            raise DecodedEvaluationExecutorError(f"{label} file hash differs")
    return row


def _capture_evidence(
    value: Any, *, label: str, allow_none: bool = False
) -> dict[str, Any] | None:
    if value is None and allow_none:
        return None
    fields = {
        "receipt_path", "receipt_sha256", "capture_digest", "target",
        "target_arguments_sha256",
    }
    row = dict(_closed(value, fields, label=label))
    if not isinstance(row["receipt_path"], str) or not Path(
        row["receipt_path"]
    ).is_absolute():
        raise DecodedEvaluationExecutorError(f"{label} path differs")
    for field in (
        "receipt_sha256", "capture_digest", "target_arguments_sha256"
    ):
        _sha(row[field], label=f"{label} {field}")
    if row["target"] not in bridge.verified_release.ALLOWED_PYTHON_TARGETS:
        raise DecodedEvaluationExecutorError(f"{label} target differs")
    return row


def build_task_input_receipt(
    *,
    bundle: Mapping[str, Any],
    shard: Mapping[str, Any],
    task: Mapping[str, Any],
    decoder_identity: Mapping[str, Any],
    ffprobe_identity: Mapping[str, Any],
    physical_bindings_identity: Mapping[str, Any],
    verify_tools: bool,
    executor_verified_release_capture: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    task_id = _task_id(task)
    decoder = _tool_identity(decoder_identity, label="decoder adapter", verify_file=verify_tools)
    ffprobe = _tool_identity(ffprobe_identity, label="ffprobe", verify_file=verify_tools)
    physical_bindings = _artifact_identity(
        physical_bindings_identity,
        label="physical bindings",
        verify_file=verify_tools,
    )
    executor_capture = _capture_evidence(
        executor_verified_release_capture,
        label="executor verified release capture",
        allow_none=not verify_tools,
    )
    if verify_tools and (
        executor_capture is None
        or executor_capture["target"]
        != "action_preservation_decoded_eval_executor_v1.py"
    ):
        raise DecodedEvaluationExecutorError(
            "production task lacks executor verified release capture"
        )
    if not verify_tools and executor_capture is not None:
        raise DecodedEvaluationExecutorError(
            "injected task may not claim an executor verified capture"
        )
    value = {
        "schema_version": TASK_INPUT_SCHEMA,
        "evaluation_id": bundle["manifest"]["evaluation_id"],
        "evaluation_manifest_digest": bundle["manifest"]["manifest_digest"],
        "publication_digest": bundle["publication_receipt"]["publication_digest"],
        "shard_digest": shard["shard_digest"],
        "holder": dict(shard["holder"]),
        "task_id": task_id,
        "task_kind": task["task_kind"],
        "task_record": task["record"],
        "task_record_digest": task["record"]["record_digest"],
        "decoder_adapter": decoder,
        "ffprobe": ffprobe,
        "physical_bindings": physical_bindings,
        "executor_source_sha256": file_sha256(__file__),
        "executor_verified_release_capture": executor_capture,
        "execution_backend": "pinned_local_subprocess"
        if verify_tools
        else "injected_stub",
        "tool_files_verified": verify_tools,
        "attempt_number": 1,
        "retry_allowed": False,
        "training_loss_read_or_used": False,
        "network_allowed": False,
        "remote_launch_performed": False,
        "direct_exec_shell": False,
        "subprocess_environment_denylist": list(SUBPROCESS_ENV_DENYLIST),
    }
    value["input_digest"] = object_sha256(value)
    return validate_task_input_receipt(value, task=task, bundle=bundle, shard=shard)


def validate_task_input_receipt(
    value: Any,
    *,
    task: Mapping[str, Any],
    bundle: Mapping[str, Any],
    shard: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "evaluation_id",
        "evaluation_manifest_digest",
        "publication_digest",
        "shard_digest",
        "holder",
        "task_id",
        "task_kind",
        "task_record",
        "task_record_digest",
        "decoder_adapter",
        "ffprobe",
        "physical_bindings",
        "executor_source_sha256",
        "executor_verified_release_capture",
        "execution_backend",
        "tool_files_verified",
        "attempt_number",
        "retry_allowed",
        "training_loss_read_or_used",
        "network_allowed",
        "remote_launch_performed",
        "direct_exec_shell",
        "subprocess_environment_denylist",
        "input_digest",
    }
    row = dict(_closed(value, fields, label="task input receipt"))
    if row["schema_version"] != TASK_INPUT_SCHEMA:
        raise DecodedEvaluationExecutorError("task input schema differs")
    expected = {
        "evaluation_id": bundle["manifest"]["evaluation_id"],
        "evaluation_manifest_digest": bundle["manifest"]["manifest_digest"],
        "publication_digest": bundle["publication_receipt"]["publication_digest"],
        "shard_digest": shard["shard_digest"],
        "holder": shard["holder"],
        "task_id": _task_id(task),
        "task_kind": task["task_kind"],
        "task_record": task["record"],
        "task_record_digest": task["record"]["record_digest"],
        "physical_bindings": row["physical_bindings"],
        "attempt_number": 1,
        "retry_allowed": False,
        "training_loss_read_or_used": False,
        "network_allowed": False,
        "remote_launch_performed": False,
        "direct_exec_shell": False,
        "subprocess_environment_denylist": list(SUBPROCESS_ENV_DENYLIST),
    }
    for key, expected_value in expected.items():
        if row[key] != expected_value:
            raise DecodedEvaluationExecutorError(f"task input binding differs: {key}")
    _tool_identity(row["decoder_adapter"], label="decoder adapter", verify_file=False)
    _tool_identity(row["ffprobe"], label="ffprobe", verify_file=False)
    _artifact_identity(row["physical_bindings"], label="physical bindings", verify_file=False)
    _sha(row["executor_source_sha256"], label="executor source")
    if row["execution_backend"] not in {"pinned_local_subprocess", "injected_stub"}:
        raise DecodedEvaluationExecutorError("execution backend differs")
    if type(row["tool_files_verified"]) is not bool:
        raise DecodedEvaluationExecutorError("tool verification flag differs")
    if (row["execution_backend"] == "pinned_local_subprocess") is not row[
        "tool_files_verified"
    ]:
        raise DecodedEvaluationExecutorError("execution backend/tool verification closure differs")
    capture = _capture_evidence(
        row["executor_verified_release_capture"],
        label="executor verified release capture",
        allow_none=not row["tool_files_verified"],
    )
    if row["tool_files_verified"] and (
        capture is None
        or capture["target"]
        != "action_preservation_decoded_eval_executor_v1.py"
    ):
        raise DecodedEvaluationExecutorError(
            "task executor verified release capture differs"
        )
    if not row["tool_files_verified"] and capture is not None:
        raise DecodedEvaluationExecutorError(
            "injected task claims a verified release capture"
        )
    _verify_digest(row, field="input_digest", label="task input receipt")
    return row


def _validate_process_observation(value: Any) -> dict[str, Any]:
    row = dict(
        _closed(
            value,
            {"return_code", "stdout", "stderr"},
            label="decoder process observation",
        )
    )
    if type(row["return_code"]) is not int:
        raise DecodedEvaluationExecutorError("decoder return code is not an integer")
    for key in ("stdout", "stderr"):
        if not isinstance(row[key], bytes):
            raise DecodedEvaluationExecutorError(f"decoder {key} is not bytes")
    return row


def build_process_receipt(
    *,
    input_receipt: Mapping[str, Any],
    observation: Mapping[str, Any],
    request_path: Path,
    staging_path: Path,
) -> dict[str, Any]:
    observed = _validate_process_observation(observation)
    value = {
        "schema_version": PROCESS_SCHEMA,
        "task_id": input_receipt["task_id"],
        "input_digest": input_receipt["input_digest"],
        "decoder_adapter_sha256": input_receipt["decoder_adapter"]["sha256"],
        "protocol_target": (
            "action_preservation_decoded_eval_decoder_adapter_v1.py"
        ),
        "protocol_arguments": [
            "--request",
            str(request_path),
            "--output",
            str(staging_path),
        ],
        "verified_runtime_required": input_receipt["tool_files_verified"],
        "root_bootstrap_source_sha256": (
            bytes_sha256(
                bridge.verified_release.ROOT_BOOTSTRAP_SOURCE.encode("utf-8")
            )
            if input_receipt["tool_files_verified"]
            else None
        ),
        "decoder_runtime_capture_receipt_path": (
            str(request_path.parent / DECODER_RUNTIME_CAPTURE_FILENAME)
            if input_receipt["tool_files_verified"]
            else None
        ),
        "release_member_path_executed_directly": False,
        "return_code": observed["return_code"],
        "stdout_sha256": bytes_sha256(observed["stdout"]),
        "stdout_size": len(observed["stdout"]),
        "stderr_sha256": bytes_sha256(observed["stderr"]),
        "stderr_size": len(observed["stderr"]),
        "retry_attempted": False,
        "shell": False,
        "subprocess_environment_denylist": list(SUBPROCESS_ENV_DENYLIST),
    }
    value["process_digest"] = object_sha256(value)
    return value


def _fraction(value: Any, *, label: str) -> Fraction:
    if not isinstance(value, str):
        raise DecodedEvaluationExecutorError(f"{label} is not a fraction string")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise DecodedEvaluationExecutorError(f"{label} is invalid") from error
    if result <= 0:
        raise DecodedEvaluationExecutorError(f"{label} is non-positive")
    return result


def parse_ffprobe_json(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or not isinstance(value.get("streams"), list)
        or not isinstance(value.get("frames"), list)
    ):
        raise DecodedEvaluationExecutorError("ffprobe JSON stream closure differs")
    video_streams = [
        stream
        for stream in value["streams"]
        if isinstance(stream, Mapping) and stream.get("codec_type") == "video"
    ]
    if len(video_streams) != 1:
        raise DecodedEvaluationExecutorError("decoded output must have exactly one video stream")
    stream = video_streams[0]
    frame_value = stream.get("nb_read_frames", stream.get("nb_frames"))
    try:
        frame_count = int(frame_value)
    except (TypeError, ValueError) as error:
        raise DecodedEvaluationExecutorError("decoded frame count is unavailable") from error
    rate = _fraction(
        stream.get("avg_frame_rate", stream.get("r_frame_rate")),
        label="decoded average frame rate",
    )
    format_value = value.get("format", {})
    format_name = format_value.get("format_name") if isinstance(format_value, Mapping) else None
    if not isinstance(format_name, str) or not format_name:
        raise DecodedEvaluationExecutorError("decoded container format is unavailable")
    video_frames = [
        item
        for item in value["frames"]
        if isinstance(item, Mapping) and item.get("media_type") == "video"
    ]
    timestamps: list[str] = []
    for item in video_frames:
        timestamp = item.get("best_effort_timestamp_time", item.get("pts_time"))
        if not isinstance(timestamp, str):
            raise DecodedEvaluationExecutorError(
                "decoded frame timestamp is unavailable"
            )
        try:
            Fraction(timestamp)
        except (ValueError, ZeroDivisionError) as error:
            raise DecodedEvaluationExecutorError(
                "decoded frame timestamp is invalid"
            ) from error
        timestamps.append(timestamp)
    return {
        "video_stream_count": len(video_streams),
        "frame_count": frame_count,
        "fps_num": rate.numerator,
        "fps_den": rate.denominator,
        "format_name": format_name,
        "frame_timestamp_times": timestamps,
    }


def validate_probe_result(value: Any) -> dict[str, Any]:
    row = dict(
        _closed(
            value,
            {
                "video_stream_count", "frame_count", "fps_num", "fps_den",
                "format_name", "frame_timestamp_times",
            },
            label="probe result",
        )
    )
    if (
        row["video_stream_count"] != 1
        or row["frame_count"] != 81
        or row["fps_num"] != 25
        or row["fps_den"] != 1
    ):
        raise DecodedEvaluationExecutorError("decoded output is not exact full81@25fps")
    if not isinstance(row["format_name"], str) or not row["format_name"]:
        raise DecodedEvaluationExecutorError("decoded format name differs")
    if "mp4" not in row["format_name"].lower() and "mov" not in row["format_name"].lower():
        raise DecodedEvaluationExecutorError("decoded output is not an MP4-family container")
    timestamps = row["frame_timestamp_times"]
    if not isinstance(timestamps, list) or len(timestamps) != 81:
        raise DecodedEvaluationExecutorError(
            "decoded output lacks exact 81-frame PTS evidence"
        )
    try:
        parsed = [Fraction(item) for item in timestamps]
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise DecodedEvaluationExecutorError(
            "decoded frame PTS evidence is invalid"
        ) from error
    if any(
        current - previous != Fraction(1, 25)
        for previous, current in zip(parsed, parsed[1:])
    ):
        raise DecodedEvaluationExecutorError(
            "decoded output is variable-rate or not exact 25fps PTS cadence"
        )
    return row


def _staging_observation(path: Path) -> dict[str, Any]:
    if not os.path.lexists(path):
        return {"exists": False, "sha256": None, "size": None}
    try:
        info = path.lstat()
    except OSError:
        return {"exists": True, "sha256": None, "size": None}
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        return {"exists": True, "sha256": None, "size": None}
    return {"exists": True, "sha256": file_sha256(path), "size": info.st_size}


def build_failure_receipt(
    *,
    input_receipt: Mapping[str, Any],
    process_receipt: Mapping[str, Any] | None,
    staging_path: Path,
    final_path: Path,
    failure_kind: str,
    failure_detail: str,
) -> dict[str, Any]:
    _identifier(failure_kind, label="failure kind")
    if not isinstance(failure_detail, str) or not failure_detail:
        raise DecodedEvaluationExecutorError("failure detail is empty")
    value = {
        "schema_version": TASK_FAILURE_SCHEMA,
        "task_id": input_receipt["task_id"],
        "input_digest": input_receipt["input_digest"],
        "process_digest": process_receipt["process_digest"]
        if process_receipt is not None
        else None,
        "failure_kind": failure_kind,
        "failure_detail": failure_detail,
        "staging_artifact": _staging_observation(staging_path),
        "final_artifact": _staging_observation(final_path),
        "attempt_number": 1,
        "retry_attempted": False,
        "retry_allowed": False,
        "failure_artifacts_retained": True,
        "training_loss_read_or_used": False,
    }
    value["failure_digest"] = object_sha256(value)
    return value


def validate_failure_receipt(
    value: Any, *, input_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "task_id",
        "input_digest",
        "process_digest",
        "failure_kind",
        "failure_detail",
        "staging_artifact",
        "final_artifact",
        "attempt_number",
        "retry_attempted",
        "retry_allowed",
        "failure_artifacts_retained",
        "training_loss_read_or_used",
        "failure_digest",
    }
    row = dict(_closed(value, fields, label="task failure receipt"))
    if row["schema_version"] != TASK_FAILURE_SCHEMA:
        raise DecodedEvaluationExecutorError("task failure schema differs")
    if (
        row["task_id"] != input_receipt["task_id"]
        or row["input_digest"] != input_receipt["input_digest"]
    ):
        raise DecodedEvaluationExecutorError("task failure input binding differs")
    if row["process_digest"] is not None:
        _sha(row["process_digest"], label="failure process digest")
    _identifier(row["failure_kind"], label="failure kind")
    if not isinstance(row["failure_detail"], str) or not row["failure_detail"]:
        raise DecodedEvaluationExecutorError("failure detail differs")
    artifact_fields = {"exists", "sha256", "size"}
    for label in ("staging_artifact", "final_artifact"):
        artifact = _closed(row[label], artifact_fields, label=label)
        if type(artifact["exists"]) is not bool:
            raise DecodedEvaluationExecutorError(f"{label} existence differs")
        if artifact["sha256"] is not None:
            _sha(artifact["sha256"], label=f"{label} SHA")
        if artifact["size"] is not None and (
            type(artifact["size"]) is not int or artifact["size"] < 0
        ):
            raise DecodedEvaluationExecutorError(f"{label} size differs")
    expected = {
        "attempt_number": 1,
        "retry_attempted": False,
        "retry_allowed": False,
        "failure_artifacts_retained": True,
        "training_loss_read_or_used": False,
    }
    for key, expected_value in expected.items():
        if row[key] != expected_value:
            raise DecodedEvaluationExecutorError(f"failure policy differs: {key}")
    _verify_digest(row, field="failure_digest", label="task failure receipt")
    return row


def build_output_receipt(
    *,
    input_receipt: Mapping[str, Any],
    process_receipt: Mapping[str, Any],
    output_path: Path,
    probe_result: Mapping[str, Any],
    native_inference_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    probe = validate_probe_result(probe_result)
    info = _plain_file(output_path, label="published decoded output").lstat()
    if info.st_size <= 0:
        raise DecodedEvaluationExecutorError("published decoded output is empty")
    if input_receipt["tool_files_verified"]:
        if not isinstance(native_inference_evidence, Mapping) or set(
            native_inference_evidence
        ) != {
            "receipt_path",
            "receipt_sha256",
            "receipt_digest",
            "decoder_verified_release_capture",
            "inference_verified_release_capture",
        }:
            raise DecodedEvaluationExecutorError(
                "verified execution lacks native inference receipt evidence"
            )
        native = dict(native_inference_evidence)
        if not isinstance(native["receipt_path"], str) or not Path(
            native["receipt_path"]
        ).is_absolute():
            raise DecodedEvaluationExecutorError("native inference receipt path differs")
        _sha(native["receipt_sha256"], label="native inference receipt file")
        _sha(native["receipt_digest"], label="native inference receipt")
        decoder_capture = _capture_evidence(
            native["decoder_verified_release_capture"],
            label="decoder verified release capture",
        )
        inference_capture = _capture_evidence(
            native["inference_verified_release_capture"],
            label="inference verified release capture",
        )
        if (
            decoder_capture is None
            or decoder_capture["target"]
            != "action_preservation_decoded_eval_decoder_adapter_v1.py"
            or inference_capture is None
            or inference_capture["target"] != "infer_lora.py"
        ):
            raise DecodedEvaluationExecutorError(
                "native verified release capture target differs"
            )
    elif native_inference_evidence is not None:
        raise DecodedEvaluationExecutorError(
            "injected execution may not claim native inference evidence"
        )
    else:
        native = None
    value = {
        "schema_version": TASK_OUTPUT_SCHEMA,
        "task_id": input_receipt["task_id"],
        "task_kind": input_receipt["task_kind"],
        "input_digest": input_receipt["input_digest"],
        "process_digest": process_receipt["process_digest"],
        "task_record_digest": input_receipt["task_record_digest"],
        "source_video_sha256": input_receipt["task_record"]["source_video_sha256"],
        "instruction_sha256": input_receipt["task_record"]["instruction_sha256"],
        "seed": input_receipt["task_record"]["seed"],
        "onset_policy": input_receipt["task_record"]["onset_policy"]["name"],
        "execution_backend": input_receipt["execution_backend"],
        "tool_files_verified": input_receipt["tool_files_verified"],
        "output_relpath": input_receipt["task_record"]["output_relpath"],
        "output_video_sha256": file_sha256(output_path),
        "output_byte_size": info.st_size,
        "probe": probe,
        "media_contract_satisfied": True,
        "native_inference_receipt": native,
        "exact_input_binding_satisfied": native is not None,
        "training_loss_read_or_used": False,
        "retry_attempted": False,
        "retry_allowed": False,
        "remote_launch_performed": False,
    }
    value["output_digest"] = object_sha256(value)
    return value


def validate_output_receipt(
    value: Any,
    *,
    input_receipt: Mapping[str, Any],
    process_receipt: Mapping[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "task_id",
        "task_kind",
        "input_digest",
        "process_digest",
        "task_record_digest",
        "source_video_sha256",
        "instruction_sha256",
        "seed",
        "onset_policy",
        "execution_backend",
        "tool_files_verified",
        "output_relpath",
        "output_video_sha256",
        "output_byte_size",
        "probe",
        "media_contract_satisfied",
        "native_inference_receipt",
        "exact_input_binding_satisfied",
        "training_loss_read_or_used",
        "retry_attempted",
        "retry_allowed",
        "remote_launch_performed",
        "output_digest",
    }
    row = dict(_closed(value, fields, label="task output receipt"))
    if row["schema_version"] != TASK_OUTPUT_SCHEMA:
        raise DecodedEvaluationExecutorError("task output schema differs")
    expected = build_output_receipt(
        input_receipt=input_receipt,
        process_receipt=process_receipt,
        output_path=output_path,
        probe_result=row["probe"],
        native_inference_evidence=row["native_inference_receipt"],
    )
    if row != expected:
        raise DecodedEvaluationExecutorError("task output receipt binding differs")
    _verify_digest(row, field="output_digest", label="task output receipt")
    return row


DecoderRunner = Callable[[Path, Path], Mapping[str, Any]]
VideoProber = Callable[[Path], Mapping[str, Any]]


def _verify_native_inference_receipt(
    *, input_receipt_path: Path, staging_path: Path
) -> dict[str, Any]:
    request = _load_json(input_receipt_path, label="sealed decode request")
    try:
        normalized_request, bindings, source, checkpoint = decoder_adapter.resolve_request(
            request, verify_files=True
        )
        receipt_path = staging_path.with_name(staging_path.name + ".receipt.json")
        receipt = _load_json(receipt_path, label="native inference receipt")
        validated = decoder_adapter.validate_inference_receipt(
            receipt,
            request=normalized_request,
            bindings=bindings,
            source=source,
            checkpoint=checkpoint,
            output_path=staging_path,
        )
        decoder_capture = bridge.validate_verified_capture_receipt(
            bindings,
            receipt_path=input_receipt_path.parent
            / DECODER_RUNTIME_CAPTURE_FILENAME,
            target="action_preservation_decoded_eval_decoder_adapter_v1.py",
            expected_arguments=[
                "--request", str(input_receipt_path),
                "--output", str(staging_path),
            ],
            verify_file=True,
        )
        inference_arguments = decoder_adapter.inference_target_arguments(
            request=normalized_request, bindings=bindings, source=source,
            checkpoint=checkpoint, output_path=staging_path,
        )
        inference_capture = bridge.validate_verified_capture_receipt(
            bindings,
            receipt_path=decoder_adapter.inference_runtime_capture_path(
                staging_path
            ),
            target="infer_lora.py",
            expected_arguments=inference_arguments,
            verify_file=True,
        )
    except (
        decoder_adapter.DecodedEvaluationDecoderError,
        bridge.DecodedEvaluationBridgeError,
    ) as error:
        raise DecodedEvaluationExecutorError(str(error)) from error
    return {
        "receipt_path": str(receipt_path),
        "receipt_sha256": file_sha256(receipt_path),
        "receipt_digest": validated["receipt_digest"],
        "decoder_verified_release_capture": decoder_capture,
        "inference_verified_release_capture": inference_capture,
    }


def _publish_failure(
    *,
    task_root: Path,
    input_receipt: Mapping[str, Any],
    process_receipt: Mapping[str, Any] | None,
    staging_path: Path,
    final_path: Path,
    failure_kind: str,
    failure_detail: str,
) -> dict[str, Any]:
    receipt = build_failure_receipt(
        input_receipt=input_receipt,
        process_receipt=process_receipt,
        staging_path=staging_path,
        final_path=final_path,
        failure_kind=failure_kind,
        failure_detail=failure_detail,
    )
    receipt = validate_failure_receipt(receipt, input_receipt=input_receipt)
    _write_create_only_json(task_root / FAILURE_RECEIPT_FILENAME, receipt)
    return {
        "task_id": input_receipt["task_id"],
        "status": "failure",
        "terminal_receipt_digest": receipt["failure_digest"],
        "output_relpath": input_receipt["task_record"]["output_relpath"],
    }


def execute_task(
    *,
    evaluation_root: Path,
    task_parent: Path,
    bundle: Mapping[str, Any],
    shard: Mapping[str, Any],
    task: Mapping[str, Any],
    decoder_identity: Mapping[str, Any],
    ffprobe_identity: Mapping[str, Any],
    physical_bindings_identity: Mapping[str, Any],
    run_decoder: DecoderRunner,
    probe_video: VideoProber,
    verify_tools: bool,
    executor_verified_release_capture: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    task_id = _task_id(task)
    task_root = _claim_directory(task_parent / task_id, label="task attempt")
    record = task["record"]
    relative_output = Path(record["output_relpath"])
    if relative_output.is_absolute() or ".." in relative_output.parts:
        raise DecodedEvaluationExecutorError("task output path escapes evaluation root")
    _ensure_directory_tree(evaluation_root, relative_output.parent)
    final_path = evaluation_root / relative_output
    staging_path = task_root / STAGING_VIDEO_FILENAME
    input_receipt = build_task_input_receipt(
        bundle=bundle,
        shard=shard,
        task=task,
        decoder_identity=decoder_identity,
        ffprobe_identity=ffprobe_identity,
        physical_bindings_identity=physical_bindings_identity,
        verify_tools=verify_tools,
        executor_verified_release_capture=executor_verified_release_capture,
    )
    input_receipt_path = task_root / INPUT_RECEIPT_FILENAME
    expected_input_bytes = canonical_json_bytes(input_receipt) + b"\n"
    _write_create_only_json(input_receipt_path, input_receipt)
    if os.path.lexists(final_path):
        return _publish_failure(
            task_root=task_root,
            input_receipt=input_receipt,
            process_receipt=None,
            staging_path=staging_path,
            final_path=final_path,
            failure_kind="output_already_exists",
            failure_detail="create-only final output path already exists",
        )
    try:
        observation = _validate_process_observation(
            run_decoder(input_receipt_path, staging_path)
        )
    except Exception as error:  # backend failures become retained terminal evidence
        observation = {
            "return_code": 255,
            "stdout": b"",
            "stderr": f"decoder backend exception: {type(error).__name__}: {error}".encode(
                "utf-8", errors="replace"
            ),
        }
    _write_create_only(task_root / STDOUT_FILENAME, observation["stdout"])
    _write_create_only(task_root / STDERR_FILENAME, observation["stderr"])
    process_receipt = build_process_receipt(
        input_receipt=input_receipt,
        observation=observation,
        request_path=input_receipt_path,
        staging_path=staging_path,
    )
    _write_create_only_json(task_root / PROCESS_RECEIPT_FILENAME, process_receipt)
    try:
        input_receipt_unchanged = input_receipt_path.read_bytes() == expected_input_bytes
    except OSError:
        input_receipt_unchanged = False
    if not input_receipt_unchanged:
        return _publish_failure(
            task_root=task_root,
            input_receipt=input_receipt,
            process_receipt=process_receipt,
            staging_path=staging_path,
            final_path=final_path,
            failure_kind="input_receipt_mutated",
            failure_detail="decoder changed or removed its sealed input receipt",
        )
    if observation["return_code"] != 0:
        return _publish_failure(
            task_root=task_root,
            input_receipt=input_receipt,
            process_receipt=process_receipt,
            staging_path=staging_path,
            final_path=final_path,
            failure_kind="decoder_nonzero",
            failure_detail=f"decoder returned {observation['return_code']}",
        )
    try:
        staging_info = _plain_file(staging_path, label="decoder staging output").lstat()
        if staging_info.st_size <= 0:
            raise DecodedEvaluationExecutorError("decoder staging output is empty")
        if staging_info.st_nlink != 1:
            raise DecodedEvaluationExecutorError(
                "decoder staging output has an unexpected pre-existing hard link"
            )
        probe_result = validate_probe_result(probe_video(staging_path))
        native_inference_evidence = (
            _verify_native_inference_receipt(
                input_receipt_path=input_receipt_path, staging_path=staging_path
            )
            if verify_tools
            else None
        )
    except Exception as error:
        return _publish_failure(
            task_root=task_root,
            input_receipt=input_receipt,
            process_receipt=process_receipt,
            staging_path=staging_path,
            final_path=final_path,
            failure_kind="media_validation_failed",
            failure_detail=f"{type(error).__name__}: {error}",
        )
    try:
        os.link(staging_path, final_path)
        os.chmod(final_path, 0o444)
        staging_published = _plain_file(
            staging_path, label="sealed staging output"
        ).lstat()
        final_published = _plain_file(final_path, label="sealed final output").lstat()
        if (
            (staging_published.st_dev, staging_published.st_ino)
            != (final_published.st_dev, final_published.st_ino)
            or staging_published.st_nlink != 2
            or final_published.st_nlink != 2
            or stat.S_IMODE(final_published.st_mode) != 0o444
        ):
            raise DecodedEvaluationExecutorError(
                "published output hard-link/sealing closure differs"
            )
    except Exception as error:
        return _publish_failure(
            task_root=task_root,
            input_receipt=input_receipt,
            process_receipt=process_receipt,
            staging_path=staging_path,
            final_path=final_path,
            failure_kind="output_publication_failed",
            failure_detail=f"{type(error).__name__}: {error}",
        )
    try:
        output_receipt = build_output_receipt(
            input_receipt=input_receipt,
            process_receipt=process_receipt,
            output_path=final_path,
            probe_result=probe_result,
            native_inference_evidence=native_inference_evidence,
        )
        output_receipt = validate_output_receipt(
            output_receipt,
            input_receipt=input_receipt,
            process_receipt=process_receipt,
            output_path=final_path,
        )
        _write_create_only_json(task_root / OUTPUT_RECEIPT_FILENAME, output_receipt)
    except Exception as error:
        return _publish_failure(
            task_root=task_root,
            input_receipt=input_receipt,
            process_receipt=process_receipt,
            staging_path=staging_path,
            final_path=final_path,
            failure_kind="output_receipt_failed",
            failure_detail=f"{type(error).__name__}: {error}",
        )
    return {
        "task_id": task_id,
        "status": "success",
        "terminal_receipt_digest": output_receipt["output_digest"],
        "output_relpath": record["output_relpath"],
    }


def build_shard_summary(
    *,
    bundle: Mapping[str, Any],
    shard: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    verify_tools: bool,
    holder_execution_authority: Mapping[str, Any] | None = None,
    executor_verified_release_capture: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected_ids = [_task_id(task) for task in shard["tasks"]]
    observed_ids = [row.get("task_id") for row in results]
    if observed_ids != expected_ids:
        raise DecodedEvaluationExecutorError("shard execution result order differs")
    if any(row.get("status") not in {"success", "failure"} for row in results):
        raise DecodedEvaluationExecutorError("shard execution status differs")
    expected_holder = shard["holder"]
    if holder_execution_authority is None:
        if verify_tools:
            raise DecodedEvaluationExecutorError(
                "production shard lacks physical holder execution authority"
            )
        holder_execution = {
            "expected_job_id": expected_holder["job_id"],
            "expected_node": expected_holder["node"],
            "observed_slurm_job_id": None,
            "observed_hostname": None,
            "exact_holder_match": False,
        }
        holder_execution["holder_execution_digest"] = object_sha256(
            holder_execution
        )
    else:
        holder_execution = dict(holder_execution_authority)
        fields = {
            "expected_job_id", "expected_node", "observed_slurm_job_id",
            "observed_hostname", "exact_holder_match",
            "holder_execution_digest",
        }
        if set(holder_execution) != fields:
            raise DecodedEvaluationExecutorError(
                "holder execution authority field closure differs"
            )
        unsigned_holder = dict(holder_execution)
        claimed_holder = unsigned_holder.pop("holder_execution_digest")
        if (
            holder_execution["expected_job_id"] != expected_holder["job_id"]
            or holder_execution["expected_node"] != expected_holder["node"]
            or holder_execution["observed_slurm_job_id"]
            != expected_holder["job_id"]
            or holder_execution["observed_hostname"] != expected_holder["node"]
            or holder_execution["exact_holder_match"] is not True
            or claimed_holder != object_sha256(unsigned_holder)
        ):
            raise DecodedEvaluationExecutorError(
                "holder execution authority differs from planned job/node"
            )
    executor_capture = _capture_evidence(
        executor_verified_release_capture,
        label="shard executor verified release capture",
        allow_none=not verify_tools,
    )
    if verify_tools and (
        executor_capture is None
        or executor_capture["target"]
        != "action_preservation_decoded_eval_executor_v1.py"
    ):
        raise DecodedEvaluationExecutorError(
            "production shard lacks executor verified release capture"
        )
    if not verify_tools and executor_capture is not None:
        raise DecodedEvaluationExecutorError(
            "injected shard may not claim an executor verified capture"
        )
    value = {
        "schema_version": SHARD_SUMMARY_SCHEMA,
        "evaluation_id": bundle["manifest"]["evaluation_id"],
        "evaluation_manifest_digest": bundle["manifest"]["manifest_digest"],
        "publication_digest": bundle["publication_receipt"]["publication_digest"],
        "shard_digest": shard["shard_digest"],
        "holder": shard["holder"],
        "holder_execution_authority": holder_execution,
        "executor_verified_release_capture": executor_capture,
        "planned_task_count": shard["total_task_count"],
        "attempted_task_count": len(results),
        "success_count": sum(row["status"] == "success" for row in results),
        "failure_count": sum(row["status"] == "failure" for row in results),
        "results": [dict(row) for row in results],
        "all_tasks_attempted_exactly_once": len(results) == shard["total_task_count"],
        "automatic_retry_count": 0,
        "retry_allowed": False,
        "failure_artifacts_retained": True,
        "execution_backend": "pinned_local_subprocess"
        if verify_tools
        else "injected_stub",
        "tool_files_verified": verify_tools,
        "training_loss_read_or_used": False,
        "network_used": False,
        "remote_launch_performed": False,
        "scientific_promotion_authorized": False,
    }
    value["summary_digest"] = object_sha256(value)
    return value


def execute_shard(
    *,
    bundle: Mapping[str, Any],
    holder_job_id: str,
    decoder_identity: Mapping[str, Any],
    ffprobe_identity: Mapping[str, Any],
    physical_bindings_identity: Mapping[str, Any],
    run_decoder: DecoderRunner,
    probe_video: VideoProber,
    verify_tools: bool = True,
    holder_execution_authority: Mapping[str, Any] | None = None,
    executor_verified_release_capture: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if holder_job_id not in bundle["shards"]:
        raise DecodedEvaluationExecutorError("holder job is outside the exact four-shard plan")
    shard = bundle["shards"][holder_job_id]
    try:
        shard = plan.validate_shard(
            shard,
            manifest=bundle["manifest"],
            input_spec=bundle["input_spec"],
        )
    except plan.DecodedEvaluationPlanError as error:
        raise DecodedEvaluationExecutorError(str(error)) from error
    _tool_identity(decoder_identity, label="decoder adapter", verify_file=verify_tools)
    _tool_identity(ffprobe_identity, label="ffprobe", verify_file=verify_tools)
    physical_identity = _artifact_identity(
        physical_bindings_identity,
        label="physical bindings",
        verify_file=verify_tools,
    )
    if verify_tools:
        try:
            bindings = bridge.load_physical_bindings(
                physical_identity["path"],
                expected_sha256=physical_identity["sha256"],
                verify_files=True,
            )
        except bridge.DecodedEvaluationBridgeError as error:
            raise DecodedEvaluationExecutorError(str(error)) from error
        try:
            for relative_path, module_path in (
                (
                    "action_preservation_decoded_eval_executor_v1.py",
                    __file__,
                ),
                (
                    "action_preservation_decoded_eval_decoder_adapter_v1.py",
                    decoder_adapter.__file__,
                ),
                (
                    "action_preservation_decoded_eval_bridge_v1.py",
                    bridge.__file__,
                ),
                (
                    "action_preservation_decoded_eval_plan_v1.py",
                    plan.__file__,
                ),
                ("action_preservation_gate_v1.py", plan.gate.__file__),
            ):
                bridge.require_running_eval_release_member(
                    bindings["eval_release"],
                    relative_path=relative_path,
                    running_path=module_path,
                )
        except bridge.DecodedEvaluationBridgeError as error:
            raise DecodedEvaluationExecutorError(str(error)) from error
        if (
            bindings["evaluation_id"] != bundle["manifest"]["evaluation_id"]
            or bindings["input_digest"] != bundle["input_spec"]["input_digest"]
            or bindings["manifest_digest"] != bundle["manifest"]["manifest_digest"]
        ):
            raise DecodedEvaluationExecutorError(
                "physical bindings differ from published evaluation bundle"
            )
        if any(
            bindings["runtime"][runtime_key][field] != identity[field]
            for runtime_key, identity in (
                ("decoder_adapter", decoder_identity),
                ("ffprobe", ffprobe_identity),
            )
            for field in ("path", "sha256")
        ):
            raise DecodedEvaluationExecutorError(
                "decoder/ffprobe tools differ from physical runtime authority"
            )
        pin_file_map = {
            "source_manifest_sha256": "source_manifest",
            "adapter_release_manifest_sha256": "adapter_release_manifest",
            "model_release_manifest_sha256": "model_release_manifest",
            "inference_release_manifest_sha256": "inference_release_manifest",
            "inference_config_sha256": "inference_config",
            "source_preprocessing_sha256": "source_preprocessing",
        }
        if any(
            bindings["pin_files"][file_key]["sha256"]
            != bundle["input_spec"]["pins"][pin_key]
            for pin_key, file_key in pin_file_map.items()
        ) or (
            bindings["runtime"]["infer_lora"]["sha256"]
            != bundle["input_spec"]["pins"]["inference_source_sha256"]
        ) or bindings["calibration_digest"] != bundle["input_spec"]["pins"][
            "calibration_digest"
        ]:
            raise DecodedEvaluationExecutorError(
                "physical pin files differ from evaluation input authority"
            )
        capture = _capture_evidence(
            executor_verified_release_capture,
            label="executor verified release capture",
        )
        if capture is None or capture["target"] != (
            "action_preservation_decoded_eval_executor_v1.py"
        ):
            raise DecodedEvaluationExecutorError(
                "production executor is outside the verified runtime"
            )
        expected_executor_arguments = [
            "--evaluation-root", bundle["manifest"]["evaluation_root"],
            "--holder-job-id", holder_job_id,
            "--decoder-adapter", decoder_identity["path"],
            "--decoder-adapter-sha256", decoder_identity["sha256"],
            "--ffprobe", ffprobe_identity["path"],
            "--ffprobe-sha256", ffprobe_identity["sha256"],
            "--physical-bindings", physical_identity["path"],
            "--physical-bindings-sha256", physical_identity["sha256"],
            "--confirmation",
            f"execute-local-decoded-eval-shard-v1-{holder_job_id}",
        ]
        try:
            replayed_capture = bridge.validate_verified_capture_receipt(
                bindings,
                receipt_path=capture["receipt_path"],
                target="action_preservation_decoded_eval_executor_v1.py",
                expected_arguments=expected_executor_arguments,
                expected_capture_digest=capture["capture_digest"],
                verify_file=True,
            )
        except bridge.DecodedEvaluationBridgeError as error:
            raise DecodedEvaluationExecutorError(str(error)) from error
        if replayed_capture != capture:
            raise DecodedEvaluationExecutorError(
                "executor verified capture replay differs"
            )
        release_executor_sha = next(
            item["sha256"] for item in bindings["eval_release"]["members"]
            if item["relative_path"]
            == "action_preservation_decoded_eval_executor_v1.py"
        )
        if file_sha256(__file__) != release_executor_sha:
            raise DecodedEvaluationExecutorError(
                "executor source differs from exact eval release"
            )
        if getattr(run_decoder, "verified_release_bootstrap", False) is not True:
            raise DecodedEvaluationExecutorError(
                "production decoder backend is not the verified runtime adapter"
            )
    elif executor_verified_release_capture is not None:
        raise DecodedEvaluationExecutorError(
            "injected executor may not claim a verified runtime"
        )
    evaluation_root = Path(bundle["manifest"]["evaluation_root"])
    _plain_directory(evaluation_root, label="evaluation root")
    execution_parent = _ensure_directory_tree(
        evaluation_root, Path(EXECUTION_DIRECTORY)
    )
    shard_root = _claim_directory(
        execution_parent / holder_job_id, label="holder shard execution"
    )
    task_parent = _claim_directory(shard_root / "tasks", label="holder task root")
    results = [
        execute_task(
            evaluation_root=evaluation_root,
            task_parent=task_parent,
            bundle=bundle,
            shard=shard,
            task=task,
            decoder_identity=decoder_identity,
            ffprobe_identity=ffprobe_identity,
            physical_bindings_identity=physical_identity,
            run_decoder=run_decoder,
            probe_video=probe_video,
            verify_tools=verify_tools,
            executor_verified_release_capture=executor_verified_release_capture,
        )
        for task in shard["tasks"]
    ]
    summary = build_shard_summary(
        bundle=bundle,
        shard=shard,
        results=results,
        verify_tools=verify_tools,
        holder_execution_authority=holder_execution_authority,
        executor_verified_release_capture=executor_verified_release_capture,
    )
    _write_create_only_json(shard_root / SUMMARY_FILENAME, summary)
    if {path.name for path in shard_root.iterdir()} != {"tasks", SUMMARY_FILENAME}:
        raise DecodedEvaluationExecutorError("holder execution root closure differs")
    if {path.name for path in task_parent.iterdir()} != set(
        _task_id(task) for task in shard["tasks"]
    ):
        raise DecodedEvaluationExecutorError("holder task root closure differs")
    return summary


def sanitized_subprocess_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for key in SUBPROCESS_ENV_DENYLIST:
        environment.pop(key, None)
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def subprocess_decoder_runner(
    authority: Mapping[str, Any] | str | Path,
) -> DecoderRunner:
    """Return the decoder backend.

    A mapping is the production form and is a validated physical binding.  It
    launches the decoder only through the root bootstrap/captured exact14
    runtime.  The pathname form exists solely for injected unit fixtures and
    must never be paired with ``verify_tools=True``.
    """

    bindings = dict(authority) if isinstance(authority, Mapping) else None
    executable = None if bindings is not None else str(authority)

    def run(request_path: Path, output_path: Path) -> Mapping[str, Any]:
        arguments = [
            "--request", str(request_path), "--output", str(output_path)
        ]
        if bindings is not None:
            try:
                argv = bridge.verified_target_argv(
                    bindings,
                    target=(
                        "action_preservation_decoded_eval_decoder_adapter_v1.py"
                    ),
                    arguments=arguments,
                    capture_receipt_path=(
                        request_path.parent / DECODER_RUNTIME_CAPTURE_FILENAME
                    ),
                )
            except bridge.DecodedEvaluationBridgeError as error:
                raise DecodedEvaluationExecutorError(str(error)) from error
        else:
            assert executable is not None
            argv = [executable, *arguments]
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            shell=False,
            env=sanitized_subprocess_environment(),
        )
        return {
            "return_code": int(completed.returncode),
            "stdout": bytes(completed.stdout),
            "stderr": bytes(completed.stderr),
        }

    setattr(run, "verified_release_bootstrap", bindings is not None)
    return run


def ffprobe_video_prober(ffprobe_path: str | Path) -> VideoProber:
    executable = str(ffprobe_path)

    def probe(video_path: Path) -> Mapping[str, Any]:
        completed = subprocess.run(
            [
                executable,
                "-v",
                "error",
                "-count_frames",
                "-show_streams",
                "-show_format",
                "-show_frames",
                "-of",
                "json",
                str(video_path),
            ],
            check=False,
            capture_output=True,
            shell=False,
            env=sanitized_subprocess_environment(),
        )
        if completed.returncode != 0:
            raise DecodedEvaluationExecutorError(
                f"ffprobe returned {completed.returncode}: "
                + completed.stderr.decode("utf-8", errors="replace")
            )
        try:
            value = json.loads(completed.stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise DecodedEvaluationExecutorError("ffprobe output is invalid JSON") from error
        return parse_ffprobe_json(value)

    return probe


def local_holder_execution_authority(holder_job_id: str) -> dict[str, Any]:
    matches = [
        item for item in plan.HOLDER_ROWS if item["job_id"] == holder_job_id
    ]
    if len(matches) != 1:
        raise DecodedEvaluationExecutorError("holder job is outside the exact plan")
    expected = matches[0]
    observed_job = os.environ.get("SLURM_JOB_ID")
    observed_hostname = socket.gethostname().split(".", 1)[0]
    exact = (
        observed_job == expected["job_id"]
        and observed_hostname == expected["node"]
    )
    if not exact:
        raise DecodedEvaluationExecutorError(
            "local executor is not running on the planned holder job/node"
        )
    value: dict[str, Any] = {
        "expected_job_id": expected["job_id"],
        "expected_node": expected["node"],
        "observed_slurm_job_id": observed_job,
        "observed_hostname": observed_hostname,
        "exact_holder_match": True,
    }
    value["holder_execution_digest"] = object_sha256(value)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--holder-job-id", required=True)
    parser.add_argument("--decoder-adapter", required=True)
    parser.add_argument("--decoder-adapter-sha256", required=True)
    parser.add_argument("--ffprobe", required=True)
    parser.add_argument("--ffprobe-sha256", required=True)
    parser.add_argument("--physical-bindings", required=True)
    parser.add_argument("--physical-bindings-sha256", required=True)
    parser.add_argument("--confirmation", required=True)
    args = parser.parse_args(argv)
    expected_confirmation = (
        f"execute-local-decoded-eval-shard-v1-{args.holder_job_id}"
    )
    if args.confirmation != expected_confirmation:
        raise DecodedEvaluationExecutorError("local execution confirmation differs")
    holder_execution = local_holder_execution_authority(args.holder_job_id)
    bundle = load_published_bundle(args.evaluation_root)
    decoder_identity = {
        "path": str(Path(args.decoder_adapter).resolve(strict=True)),
        "sha256": args.decoder_adapter_sha256,
    }
    ffprobe_identity = {
        "path": str(Path(args.ffprobe).resolve(strict=True)),
        "sha256": args.ffprobe_sha256,
    }
    physical_bindings_identity = {
        "path": str(Path(args.physical_bindings).resolve(strict=True)),
        "sha256": args.physical_bindings_sha256,
    }
    if decoder_identity["sha256"] != file_sha256(decoder_adapter.__file__):
        raise DecodedEvaluationExecutorError(
            "decoder adapter differs from the audited implementation"
        )
    try:
        bindings = bridge.load_physical_bindings(
            physical_bindings_identity["path"],
            expected_sha256=physical_bindings_identity["sha256"],
            verify_files=True,
        )
        executor_capture = bridge.validate_running_verified_capture(
            bindings,
            target="action_preservation_decoded_eval_executor_v1.py",
            expected_arguments=list(sys.argv[1:] if argv is None else argv),
            verify_file=True,
        )
    except bridge.DecodedEvaluationBridgeError as error:
        raise DecodedEvaluationExecutorError(str(error)) from error
    summary = execute_shard(
        bundle=bundle,
        holder_job_id=args.holder_job_id,
        decoder_identity=decoder_identity,
        ffprobe_identity=ffprobe_identity,
        physical_bindings_identity=physical_bindings_identity,
        run_decoder=subprocess_decoder_runner(bindings),
        probe_video=ffprobe_video_prober(ffprobe_identity["path"]),
        verify_tools=True,
        holder_execution_authority=holder_execution,
        executor_verified_release_capture=executor_capture,
    )
    print(canonical_json_bytes(summary).decode("utf-8"))
    return 0 if summary["failure_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
