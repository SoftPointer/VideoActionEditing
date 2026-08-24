"""Fail-closed parallel sharding and aggregation for Wan2.2 I2V batches.

The official Wan2.2 I2V runner uses eight cooperative ranks for one sample.
It therefore cannot safely parallelize independent samples inside one
``output_root``: every invocation would bind a different run contract and
race the same pending prefix.  This module implements the safe topology:

* split the frozen generation manifest into contiguous, disjoint manifests;
* give every eight-GPU job its own output root;
* preserve the original row order across shards; and
* validate every shard receipt before atomically publishing one aggregate
  generated manifest.

Only the Python standard library is imported. ``prepare`` accepts exactly one
source-anchored OpenSSH release covering an eight-row v9 manifest; legacy
approval JSON and authorization booleans cannot create a submission plan.
Validation and finalization utilities remain importable on CPU nodes.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence


GENERATION_MANIFEST_SCHEMA = "motive-goku-action-anchor-generation-v1"
GENERATED_MANIFEST_SCHEMA = "motive-wan22-i2v-generated-target-v1"
RUN_SCHEMA = "motive-wan22-i2v-batch-run-v1"
SAMPLE_SCHEMA = "motive-wan22-i2v-sample-v1"
COMPLETE_SCHEMA = "motive-wan22-i2v-batch-complete-v1"
PLAN_SCHEMA = "motive-wan22-i2v-parallel-plan-v1"
AGGREGATE_SCHEMA = "motive-wan22-i2v-parallel-aggregate-v1"
FIRST_FRAME_POLICY = "wan22-i2v-strict-preencode-frame0-v1"
TEMPORAL_POLICY = "wan22-i2v-source-timebase-preserving-v1"
APPROVAL_SCHEMA = "motive-goku-action-anchor-approval-v1"
APPROVED_MANIFEST_ROLE = "approved_generation"
SIGNED_RELEASE_SCHEMA = "motive-wan22-signed-generation-release-v1"
SIGNED_RELEASE_VERIFIER_AVAILABLE = True
SIGNED_RELEASE_GATE_STATUS = "sshsig_qwen3_vl_32b_smoke_exact_8"
SIGNED_AUTHORIZATION_MODE = "sshsig_qwen3_vl_32b_smoke_release_v1"

PLAN_NAME = "parallel_plan.json"
SHARDS_TSV_NAME = "shards.tsv"
RUN_CONTRACT_NAME = "run_contract.json"
RUN_COMPLETE_NAME = "run_complete.json"
GENERATED_MANIFEST_NAME = "generated_manifest.jsonl"
SAMPLE_RESULT_NAME = "result.json"
AGGREGATE_COMPLETE_NAME = "aggregate_complete.json"

EXPECTED_WORLD_SIZE = 8
EXPECTED_FRAME_NUM = 81
EXPECTED_SAMPLE_STEPS = 40
EXPECTED_SAMPLE_SHIFT = 5.0
EXPECTED_SIZE = "1280*720"
EXPECTED_BASE_SEED = 260730
EXPECTED_SOURCE_FRAME_RATE = "25/1"
EXPECTED_MODEL_SAMPLE_FPS = 16
EXPECTED_DURATION_TOLERANCE_FRAMES = 1

NO_IB_ENVIRONMENT = {
    "NCCL_IB_DISABLE": "1",
    "NCCL_SOCKET_IFNAME": "bond0",
    "NCCL_SOCKET_FAMILY": "AF_INET",
    "GLOO_SOCKET_IFNAME": "bond0",
    "NCCL_ASYNC_ERROR_HANDLING": "1",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "NCCL_DEBUG": "WARN",
    "NCCL_DEBUG_SUBSYS": "INIT,NET",
}
NO_IB_UNSET_ENVIRONMENT = ["NCCL_IB_HCA", "NCCL_IB_GID_INDEX"]

_IID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

_OUTPUT_BINDINGS = (
    (
        "target_preview_mp4",
        "target_preview_mp4_sha256",
        "preview_mp4",
        "preview_mp4_sha256",
    ),
    (
        "conditioning_anchor_original",
        "conditioning_anchor_original_sha256",
        "conditioning_anchor_original",
        "conditioning_anchor_original_sha256",
    ),
    (
        "conditioning_frame0_float32",
        "conditioning_frame0_float32_sha256",
        "conditioning_frame0_float32",
        "conditioning_frame0_float32_sha256",
    ),
    (
        "conditioning_frame0_png",
        "conditioning_frame0_png_sha256",
        "conditioning_frame0_png",
        "conditioning_frame0_png_sha256",
    ),
)


class Wan22ParallelError(RuntimeError):
    """A parallel plan, shard output, or aggregate violates its contract."""


def require_signed_generation_release() -> None:
    """Fail before plan creation when no verified release was supplied."""

    raise Wan22ParallelError(
        "signed generation release gate is unavailable for unsigned inputs; "
        "a verified release is required, and no current "
        "generation manifest, legacy approved_generation record, approval "
        "JSON, boolean authorization field, or re-signed manifest can "
        f"authorize Wan submission (required schema: {SIGNED_RELEASE_SCHEMA})"
    )


def _reject_constant(value: str) -> None:
    raise Wan22ParallelError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Wan22ParallelError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _parse_json(raw: bytes, *, context: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise Wan22ParallelError(f"{context} is not UTF-8") from error
    try:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        if isinstance(error, Wan22ParallelError):
            raise
        raise Wan22ParallelError(f"{context} is not strict JSON: {error}") from error


def _canonical_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise Wan22ParallelError(f"value is not canonical JSON: {error}") from error
    return text.encode("utf-8")


def _pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _object_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_create_bytes(path: Path, payload: bytes) -> None:
    """Publish ``payload`` without replacing an existing file."""

    if path.exists() or path.is_symlink():
        raise Wan22ParallelError(f"refusing to overwrite existing file: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise Wan22ParallelError(
                f"refusing to overwrite existing file: {path}"
            ) from error
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_identical_or_create(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise Wan22ParallelError(f"published path is not a regular file: {path}")
        if path.read_bytes() != payload:
            raise Wan22ParallelError(f"existing published file differs: {path}")
        return
    _atomic_create_bytes(path, payload)


def _string(value: Any, *, context: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise Wan22ParallelError(f"{context} must be a string")
    if value != value.strip() or "\x00" in value:
        raise Wan22ParallelError(f"{context} is not a canonical string")
    if not allow_empty and not value:
        raise Wan22ParallelError(f"{context} must not be empty")
    return value


def _safe_iid(value: Any, *, context: str) -> str:
    iid = _string(value, context=context)
    if _IID_RE.fullmatch(iid) is None or iid in {".", ".."}:
        raise Wan22ParallelError(f"{context} is not a safe IID: {iid!r}")
    return iid


def _sha256_field(value: Any, *, context: str) -> str:
    digest = _string(value, context=context)
    if _SHA256_RE.fullmatch(digest) is None:
        raise Wan22ParallelError(f"{context} must be a lowercase SHA-256")
    return digest


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Wan22ParallelError(f"{context} must be an object")
    return value


def _positive_integer(value: Any, *, context: str) -> int:
    if type(value) is not int or value <= 0:
        raise Wan22ParallelError(f"{context} must be a positive integer")
    return value


def _finite_number(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Wan22ParallelError(f"{context} must be a finite JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise Wan22ParallelError(f"{context} must be finite")
    return result


def _positive_fraction(value: Any, *, context: str) -> Fraction:
    text = _string(value, context=context)
    try:
        result = Fraction(text)
    except (ValueError, ZeroDivisionError) as error:
        raise Wan22ParallelError(
            f"{context} must be a positive rational string"
        ) from error
    if result <= 0:
        raise Wan22ParallelError(f"{context} must be positive")
    return result


def _numbers_close(left: Any, right: Any, *, context: str) -> None:
    actual = _finite_number(left, context=context)
    expected = _finite_number(right, context=f"{context} expected")
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise Wan22ParallelError(
            f"{context} differs: expected={expected!r} actual={actual!r}"
        )


def _exact_object_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    if set(value) != expected:
        raise Wan22ParallelError(
            f"{context} keys differ from closed schema: "
            f"{sorted(set(value) ^ expected)}"
        )


def _validate_contract_temporal_policy(
    value: Any,
    *,
    context: str,
) -> dict[str, Any]:
    policy = _mapping(value, context=context)
    required = {
        "policy_version",
        "source_frame_count",
        "target_frame_count",
        "source_frame_rate",
        "target_container_frame_rate",
        "nominal_duration_seconds",
        "source_duration_range_seconds",
        "duration_match_tolerance_frames",
        "duration_match_tolerance_seconds",
        "model_sample_fps",
        "model_sample_fps_role",
        "output_container_rate_source",
        "source_frame_count_must_be_4n_plus_1",
        "source_target_frame_count_equal",
        "source_target_frame_rate_equal",
        "source_target_duration_within_tolerance",
        "batch_time_grid_uniform",
    }
    _exact_object_keys(policy, required, context=context)
    expected_equal = {
        "policy_version": TEMPORAL_POLICY,
        "source_frame_count": EXPECTED_FRAME_NUM,
        "target_frame_count": EXPECTED_FRAME_NUM,
        "source_frame_rate": EXPECTED_SOURCE_FRAME_RATE,
        "target_container_frame_rate": EXPECTED_SOURCE_FRAME_RATE,
        "duration_match_tolerance_frames": (
            EXPECTED_DURATION_TOLERANCE_FRAMES
        ),
        "model_sample_fps": EXPECTED_MODEL_SAMPLE_FPS,
        "model_sample_fps_role": "diffusion_configuration_only",
        "output_container_rate_source": "source_video",
        "source_frame_count_must_be_4n_plus_1": True,
        "source_target_frame_count_equal": True,
        "source_target_frame_rate_equal": True,
        "source_target_duration_within_tolerance": True,
        "batch_time_grid_uniform": True,
    }
    for field, expected in expected_equal.items():
        if policy.get(field) != expected:
            raise Wan22ParallelError(
                f"{context}.{field} differs: "
                f"expected={expected!r} actual={policy.get(field)!r}"
            )

    rate = _positive_fraction(
        policy["source_frame_rate"],
        context=f"{context}.source_frame_rate",
    )
    nominal = float(Fraction(EXPECTED_FRAME_NUM, 1) / rate)
    tolerance = float(
        Fraction(EXPECTED_DURATION_TOLERANCE_FRAMES, 1) / rate
    )
    _numbers_close(
        policy["nominal_duration_seconds"],
        nominal,
        context=f"{context}.nominal_duration_seconds",
    )
    _numbers_close(
        policy["duration_match_tolerance_seconds"],
        tolerance,
        context=f"{context}.duration_match_tolerance_seconds",
    )
    duration_range = policy["source_duration_range_seconds"]
    if not isinstance(duration_range, list) or len(duration_range) != 2:
        raise Wan22ParallelError(
            f"{context}.source_duration_range_seconds must be a two-item list"
        )
    lower = _finite_number(
        duration_range[0],
        context=f"{context}.source_duration_range_seconds[0]",
    )
    upper = _finite_number(
        duration_range[1],
        context=f"{context}.source_duration_range_seconds[1]",
    )
    if lower <= 0 or upper < lower:
        raise Wan22ParallelError(
            f"{context}.source_duration_range_seconds is invalid"
        )
    if (
        abs(lower - nominal) > tolerance + 1e-9
        or abs(upper - nominal) > tolerance + 1e-9
        or upper - lower > tolerance + 1e-9
    ):
        raise Wan22ParallelError(
            f"{context}.source_duration_range_seconds violates the "
            "one-frame source time-grid tolerance"
        )
    return dict(policy)


def _validate_temporal_probe(
    value: Any,
    *,
    expected: Mapping[str, Any],
    context: str,
) -> None:
    probe = _mapping(value, context=context)
    if probe.get("frames") != expected["frame_count"]:
        raise Wan22ParallelError(f"{context}.frames differs from temporal evidence")
    if probe.get("frame_rate") != expected["frame_rate"]:
        raise Wan22ParallelError(
            f"{context}.frame_rate differs from temporal evidence"
        )
    _numbers_close(
        probe.get("duration_seconds"),
        expected["duration_seconds"],
        context=f"{context}.duration_seconds",
    )


def _validate_pair_temporal_policy(
    value: Any,
    *,
    contract_policy: Mapping[str, Any],
    context: str,
) -> dict[str, Any]:
    policy = _mapping(value, context=context)
    required = {
        "policy_version",
        "model_sample_fps",
        "model_sample_fps_role",
        "output_container_rate_source",
        "source",
        "target",
        "frame_count_equal",
        "frame_rate_equal",
        "duration_delta_seconds",
        "duration_delta_frames",
        "duration_match_tolerance_frames",
        "duration_match_tolerance_seconds",
        "duration_within_tolerance",
    }
    _exact_object_keys(policy, required, context=context)
    expected_equal = {
        "policy_version": TEMPORAL_POLICY,
        "model_sample_fps": EXPECTED_MODEL_SAMPLE_FPS,
        "model_sample_fps_role": "diffusion_configuration_only",
        "output_container_rate_source": "source_video",
        "frame_count_equal": True,
        "frame_rate_equal": True,
        "duration_match_tolerance_frames": (
            EXPECTED_DURATION_TOLERANCE_FRAMES
        ),
        "duration_within_tolerance": True,
    }
    for field, expected in expected_equal.items():
        if policy.get(field) != expected:
            raise Wan22ParallelError(
                f"{context}.{field} differs: "
                f"expected={expected!r} actual={policy.get(field)!r}"
            )

    endpoints: dict[str, Mapping[str, Any]] = {}
    for endpoint in ("source", "target"):
        item = _mapping(
            policy.get(endpoint),
            context=f"{context}.{endpoint}",
        )
        _exact_object_keys(
            item,
            {"frame_count", "frame_rate", "duration_seconds"},
            context=f"{context}.{endpoint}",
        )
        expected_count = contract_policy[
            f"{endpoint}_frame_count"
            if endpoint == "source"
            else "target_frame_count"
        ]
        expected_rate = contract_policy[
            "source_frame_rate"
            if endpoint == "source"
            else "target_container_frame_rate"
        ]
        if item.get("frame_count") != expected_count:
            raise Wan22ParallelError(
                f"{context}.{endpoint}.frame_count differs from run contract"
            )
        if item.get("frame_rate") != expected_rate:
            raise Wan22ParallelError(
                f"{context}.{endpoint}.frame_rate differs from run contract"
            )
        duration = _finite_number(
            item.get("duration_seconds"),
            context=f"{context}.{endpoint}.duration_seconds",
        )
        if duration <= 0:
            raise Wan22ParallelError(
                f"{context}.{endpoint}.duration_seconds must be positive"
            )
        nominal = float(
            Fraction(expected_count, 1)
            / _positive_fraction(
                expected_rate,
                context=f"{context}.{endpoint}.frame_rate",
            )
        )
        tolerance = _finite_number(
            contract_policy["duration_match_tolerance_seconds"],
            context=f"{context} contract duration tolerance",
        )
        if abs(duration - nominal) > tolerance + 1e-9:
            raise Wan22ParallelError(
                f"{context}.{endpoint}.duration_seconds violates its time grid"
            )
        endpoints[endpoint] = item

    source_duration = float(endpoints["source"]["duration_seconds"])
    target_duration = float(endpoints["target"]["duration_seconds"])
    duration_delta = abs(source_duration - target_duration)
    rate = float(
        _positive_fraction(
            endpoints["source"]["frame_rate"],
            context=f"{context}.source.frame_rate",
        )
    )
    duration_delta_frames = duration_delta * rate
    tolerance_seconds = float(
        contract_policy["duration_match_tolerance_seconds"]
    )
    _numbers_close(
        policy["duration_delta_seconds"],
        duration_delta,
        context=f"{context}.duration_delta_seconds",
    )
    _numbers_close(
        policy["duration_delta_frames"],
        duration_delta_frames,
        context=f"{context}.duration_delta_frames",
    )
    _numbers_close(
        policy["duration_match_tolerance_seconds"],
        tolerance_seconds,
        context=f"{context}.duration_match_tolerance_seconds",
    )
    if duration_delta_frames > EXPECTED_DURATION_TOLERANCE_FRAMES + 1e-6:
        raise Wan22ParallelError(
            f"{context} source/target duration differs by more than one frame"
        )
    source_range = contract_policy["source_duration_range_seconds"]
    if not (
        float(source_range[0]) - 1e-9
        <= source_duration
        <= float(source_range[1]) + 1e-9
    ):
        raise Wan22ParallelError(
            f"{context}.source.duration_seconds lies outside contract range"
        )
    return dict(policy)


def _regular_file(path: Path, *, context: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink() or not expanded.is_file():
        raise Wan22ParallelError(
            f"{context} must be a regular non-symlink file: {expanded}"
        )
    if expanded.stat().st_size <= 0:
        raise Wan22ParallelError(f"{context} is empty: {expanded}")
    return expanded.resolve(strict=True)


def _regular_directory(path: Path, *, context: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink() or not expanded.is_dir():
        raise Wan22ParallelError(
            f"{context} must be a non-symlink directory: {expanded}"
        )
    return expanded.resolve(strict=True)


def _strict_json_file(path: Path, *, context: str) -> dict[str, Any]:
    resolved = _regular_file(path, context=context)
    value = _parse_json(resolved.read_bytes(), context=context)
    if not isinstance(value, dict):
        raise Wan22ParallelError(f"{context} must contain one JSON object")
    return value


def _strict_jsonl_bytes(
    raw: bytes,
    *,
    context: str,
) -> tuple[list[bytes], list[dict[str, Any]]]:
    if not raw:
        raise Wan22ParallelError(f"{context} is empty")
    if not raw.endswith(b"\n"):
        raise Wan22ParallelError(f"{context} must end with a newline")
    lines = raw.splitlines(keepends=True)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise Wan22ParallelError(
                f"{context} contains a blank line at {line_number}"
            )
        value = _parse_json(line, context=f"{context} row {line_number}")
        if not isinstance(value, dict):
            raise Wan22ParallelError(
                f"{context} row {line_number} is not an object"
            )
        rows.append(value)
    return lines, rows


def _strict_jsonl_file(
    path: Path,
    *,
    context: str,
) -> tuple[Path, bytes, list[bytes], list[dict[str, Any]]]:
    resolved = _regular_file(path, context=context)
    raw = resolved.read_bytes()
    lines, rows = _strict_jsonl_bytes(raw, context=context)
    return resolved, raw, lines, rows


def _validate_source_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    allow_pending_review: bool,
) -> tuple[list[str], list[str]]:
    if allow_pending_review:
        raise Wan22ParallelError(
            "production parallel generation forbids pending-review override"
        )
    seen_iids: set[str] = set()
    seen_groups: set[str] = set()
    iids: list[str] = []
    groups: list[str] = []
    for index, row in enumerate(rows):
        context = f"source manifest row {index + 1}"
        if row.get("schema_version") != GENERATION_MANIFEST_SCHEMA:
            raise Wan22ParallelError(
                f"{context} schema_version must be "
                f"{GENERATION_MANIFEST_SCHEMA!r}"
            )
        iid = _safe_iid(row.get("iid"), context=f"{context} iid")
        group = _string(row.get("group_id"), context=f"{context} group_id")
        if iid in seen_iids:
            raise Wan22ParallelError(f"duplicate source manifest IID: {iid}")
        if group in seen_groups:
            raise Wan22ParallelError(
                f"duplicate source manifest group_id: {group}"
            )
        seen_iids.add(iid)
        seen_groups.add(group)
        if row.get("action_change_substantive") != "yes":
            raise Wan22ParallelError(
                f"{context} action_change_substantive must be exactly 'yes'"
            )
        _string(
            row.get("absolute_target_prompt"),
            context=f"{context} absolute_target_prompt",
        )
        _string(
            row.get("edit_instruction"),
            context=f"{context} edit_instruction",
        )
        _sha256_field(
            row.get("source_video_sha256"),
            context=f"{context} source_video_sha256",
        )
        _sha256_field(
            row.get("anchor_sha256"),
            context=f"{context} anchor_sha256",
        )
        authorization = row.get("generation_authorized")
        review_status = _string(
            row.get("human_review_status"),
            context=f"{context} human_review_status",
        )
        if type(authorization) is not bool:
            raise Wan22ParallelError(
                f"{context} generation_authorized must be a boolean"
            )
        if (
            authorization is not True
            or review_status != "approved"
            or row.get("manifest_role") != APPROVED_MANIFEST_ROLE
            or row.get("production_eligible") is not True
        ):
            raise Wan22ParallelError(
                f"{context} is not explicitly approved for production"
            )
        approval = _mapping(
            row.get("approval"),
            context=f"{context} approval",
        )
        required_approval = {
            "schema_version",
            "approval_digest",
            "approval_file_sha256",
            "proposal_sha256",
            "reviewer_id",
            "reviewed_at_utc",
            "decision",
            "reason",
        }
        if set(approval) != required_approval:
            raise Wan22ParallelError(
                f"{context} approval is not a closed explicit record"
            )
        if approval.get("schema_version") != APPROVAL_SCHEMA:
            raise Wan22ParallelError(
                f"{context} approval schema differs"
            )
        for field in (
            "approval_digest",
            "approval_file_sha256",
            "proposal_sha256",
        ):
            _sha256_field(
                approval.get(field),
                context=f"{context} approval {field}",
            )
        for field in ("reviewer_id", "reviewed_at_utc", "reason"):
            _string(
                approval.get(field),
                context=f"{context} approval {field}",
            )
        if approval.get("decision") != "approved":
            raise Wan22ParallelError(
                f"{context} approval decision is not approved"
            )
        iids.append(iid)
        groups.append(group)
    return iids, groups


def _balanced_ranges(row_count: int, shard_count: int) -> list[tuple[int, int]]:
    _positive_integer(row_count, context="row_count")
    _positive_integer(shard_count, context="shard_count")
    if shard_count > row_count:
        raise Wan22ParallelError(
            f"shard_count={shard_count} exceeds row_count={row_count}"
        )
    quotient, remainder = divmod(row_count, shard_count)
    ranges: list[tuple[int, int]] = []
    start = 0
    for index in range(shard_count):
        size = quotient + (1 if index < remainder else 0)
        stop = start + size
        ranges.append((start, stop))
        start = stop
    if start != row_count:
        raise AssertionError("internal shard range error")
    return ranges


def sample_seed(base_seed: int, iid: str) -> int:
    if type(base_seed) is not int or base_seed < 0:
        raise Wan22ParallelError("base_seed must be a non-negative integer")
    safe_iid = _safe_iid(iid, context="iid")
    digest = hashlib.sha256(f"{base_seed}\0{safe_iid}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def prepare_parallel_run(
    *,
    manifest_path: str | Path,
    parallel_root: str | Path,
    geometry_job_id: int,
    shard_count: int = 3,
    allow_pending_review: bool,
    expected_row_count: int | None = None,
    signed_release_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create an immutable contiguous-shard plan under a fresh output root."""

    # Keep this before manifest reads and root creation: the shell submitter
    # calls prepare before its first sbatch, so an unavailable release gate
    # guarantees zero generation submissions.
    if signed_release_path is None:
        require_signed_generation_release()
    geometry_job_id = _positive_integer(
        geometry_job_id,
        context="geometry_job_id",
    )
    shard_count = _positive_integer(shard_count, context="shard_count")
    manifest, raw, lines, rows = _strict_jsonl_file(
        Path(manifest_path),
        context="source generation manifest",
    )
    signed_release: Mapping[str, Any] | None = None
    if signed_release_path is not None:
        if allow_pending_review:
            raise Wan22ParallelError(
                "pending-review override is never authorization"
            )
        try:
            from motive.wan22_signed_release import verify_signed_release

            released = verify_signed_release(
                release_path=Path(signed_release_path),
                manifest_path=manifest,
                require_exact_manifest=True,
                verify_media=True,
            )
        except Exception as error:
            raise Wan22ParallelError(
                f"signed release verification failed: {error}"
            ) from error
        if (
            released["manifest_sha256"] != _sha256_bytes(raw)
            or released["manifest_bytes"] != len(raw)
        ):
            raise Wan22ParallelError("signed release manifest bytes differ")
        iids = [str(row["_iid"]) for row in released["selected_rows"]]
        groups = [str(row["group_id"]) for row in released["selected_rows"]]
        signed_release = released["release"]
    else:
        # This branch remains only for non-production contract fixtures; the
        # real call above has already failed closed without a release.
        iids, groups = _validate_source_rows(
            rows,
            allow_pending_review=allow_pending_review,
        )
    if expected_row_count is not None:
        expected_row_count = _positive_integer(
            expected_row_count,
            context="expected_row_count",
        )
        if len(rows) != expected_row_count:
            raise Wan22ParallelError(
                f"source row count differs: expected={expected_row_count} "
                f"actual={len(rows)}"
            )

    requested_root = Path(parallel_root).expanduser()
    if not requested_root.is_absolute() or requested_root == Path("/"):
        raise Wan22ParallelError("parallel_root must be a non-root absolute path")
    parent = _regular_directory(
        requested_root.parent,
        context="parallel-root parent",
    )
    root = parent / requested_root.name
    if root.exists() or root.is_symlink():
        raise Wan22ParallelError(f"parallel root must be fresh: {root}")
    root.mkdir(mode=0o700)
    try:
        for name in ("manifests", "shards", "logs", "final", "submissions"):
            (root / name).mkdir(mode=0o700)

        shard_records: list[dict[str, Any]] = []
        concatenated: list[bytes] = []
        for shard_index, (start, stop) in enumerate(
            _balanced_ranges(len(rows), shard_count)
        ):
            shard_id = f"shard_{shard_index:03d}"
            shard_bytes = b"".join(lines[start:stop])
            shard_manifest = root / "manifests" / f"{shard_id}.jsonl"
            _atomic_create_bytes(shard_manifest, shard_bytes)
            concatenated.append(shard_bytes)
            shard_records.append(
                {
                    "index": shard_index,
                    "shard_id": shard_id,
                    "row_start_zero_based": start,
                    "row_stop_exclusive": stop,
                    "row_count": stop - start,
                    "iids": iids[start:stop],
                    "group_ids": groups[start:stop],
                    "manifest": {
                        "path": str(shard_manifest),
                        "relative_path": str(
                            shard_manifest.relative_to(root)
                        ),
                        "sha256": _sha256_bytes(shard_bytes),
                        "bytes": len(shard_bytes),
                    },
                    "output_root": str(root / "shards" / shard_id),
                    "log_stdout_pattern": str(
                        root / "logs" / f"{shard_id}-%j.out"
                    ),
                    "log_stderr_pattern": str(
                        root / "logs" / f"{shard_id}-%j.err"
                    ),
                    "dependency": f"afterok:{geometry_job_id}",
                }
            )
        if b"".join(concatenated) != raw:
            raise AssertionError("internal shard concatenation differs")

        plan: dict[str, Any] = {
            "schema_version": PLAN_SCHEMA,
            "parallel_root": str(root),
            "geometry_job_id": geometry_job_id,
            "expected_source_row_count": (
                expected_row_count
                if expected_row_count is not None
                else len(rows)
            ),
            "source_manifest": {
                "path": str(manifest),
                "sha256": _sha256_bytes(raw),
                "bytes": len(raw),
                "row_count": len(rows),
                "iids": iids,
                "group_ids": groups,
            },
            "shard_count": shard_count,
            "shards": shard_records,
            "generation_parameters": {
                "size": EXPECTED_SIZE,
                "frame_num": EXPECTED_FRAME_NUM,
                "sample_steps": EXPECTED_SAMPLE_STEPS,
                "sample_shift": EXPECTED_SAMPLE_SHIFT,
                "base_seed": EXPECTED_BASE_SEED,
            },
            "distributed_execution": {
                "world_size": EXPECTED_WORLD_SIZE,
                "nodes_per_shard": 1,
                "gpus_per_node": EXPECTED_WORLD_SIZE,
                "independent_output_root_per_shard": True,
            },
            "authorization": (
                {
                    "mode": SIGNED_AUTHORIZATION_MODE,
                    "allow_pending_review": False,
                    "legacy_approval_records_trusted": False,
                    "release": dict(signed_release),
                }
                if signed_release is not None
                else {
                    "allow_pending_review": False,
                    "requires_explicit_approval": True,
                }
            ),
            "network_environment": {
                "export": dict(NO_IB_ENVIRONMENT),
                "unset": list(NO_IB_UNSET_ENVIRONMENT),
            },
        }
        plan["plan_digest"] = _object_digest(plan)
        _atomic_create_bytes(root / PLAN_NAME, _pretty_json_bytes(plan))

        tsv_lines = [
            (
                "shard_id\trow_start_zero_based\trow_stop_exclusive\t"
                "row_count\tmanifest_sha256\tmanifest\toutput_root\t"
                "log_stdout_pattern\tlog_stderr_pattern\n"
            )
        ]
        for shard in shard_records:
            values = (
                shard["shard_id"],
                shard["row_start_zero_based"],
                shard["row_stop_exclusive"],
                shard["row_count"],
                shard["manifest"]["sha256"],
                shard["manifest"]["path"],
                shard["output_root"],
                shard["log_stdout_pattern"],
                shard["log_stderr_pattern"],
            )
            texts = [str(value) for value in values]
            if any("\t" in value or "\n" in value for value in texts):
                raise Wan22ParallelError("TSV plan value contains tab/newline")
            tsv_lines.append("\t".join(texts) + "\n")
        _atomic_create_bytes(
            root / SHARDS_TSV_NAME,
            "".join(tsv_lines).encode("utf-8"),
        )
        return plan
    except Exception:
        # Preserve a partial root for post-mortem evidence.  Its existence
        # deliberately prevents an accidental duplicate submission.
        raise


def _load_bound_object(
    path: Path,
    *,
    context: str,
    digest_field: str,
) -> dict[str, Any]:
    value = _strict_json_file(path, context=context)
    claimed = _sha256_field(
        value.get(digest_field),
        context=f"{context} {digest_field}",
    )
    bound = dict(value)
    del bound[digest_field]
    actual = _object_digest(bound)
    if claimed != actual:
        raise Wan22ParallelError(
            f"{context} {digest_field} mismatch: "
            f"expected={claimed} actual={actual}"
        )
    return value


def _load_plan(root: Path) -> dict[str, Any]:
    plan = _load_bound_object(
        root / PLAN_NAME,
        context="parallel plan",
        digest_field="plan_digest",
    )
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise Wan22ParallelError(f"parallel plan schema must be {PLAN_SCHEMA!r}")
    if plan.get("parallel_root") != str(root):
        raise Wan22ParallelError("parallel plan root differs from requested root")
    _positive_integer(
        plan.get("geometry_job_id"),
        context="parallel plan geometry_job_id",
    )
    _positive_integer(
        plan.get("expected_source_row_count"),
        context="parallel plan expected_source_row_count",
    )
    if plan.get("generation_parameters") != {
        "size": EXPECTED_SIZE,
        "frame_num": EXPECTED_FRAME_NUM,
        "sample_steps": EXPECTED_SAMPLE_STEPS,
        "sample_shift": EXPECTED_SAMPLE_SHIFT,
        "base_seed": EXPECTED_BASE_SEED,
    }:
        raise Wan22ParallelError("parallel plan generation parameters differ")
    if plan.get("distributed_execution") != {
        "world_size": EXPECTED_WORLD_SIZE,
        "nodes_per_shard": 1,
        "gpus_per_node": EXPECTED_WORLD_SIZE,
        "independent_output_root_per_shard": True,
    }:
        raise Wan22ParallelError("parallel plan distributed execution differs")
    authorization = _mapping(
        plan.get("authorization"),
        context="parallel plan authorization",
    )
    signed_plan = authorization.get("mode") == SIGNED_AUTHORIZATION_MODE
    legacy_authorization = {
        "allow_pending_review": False,
        "requires_explicit_approval": True,
    }
    if signed_plan:
        if (
            authorization.get("allow_pending_review") is not False
            or authorization.get("legacy_approval_records_trusted") is not False
            or not isinstance(authorization.get("release"), Mapping)
        ):
            raise Wan22ParallelError(
                "parallel plan signed authorization policy differs"
            )
    elif authorization != legacy_authorization:
        raise Wan22ParallelError(
            "parallel plan authorization policy differs"
        )
    if plan.get("network_environment") != {
        "export": NO_IB_ENVIRONMENT,
        "unset": NO_IB_UNSET_ENVIRONMENT,
    }:
        raise Wan22ParallelError("parallel plan no-IB environment differs")
    return plan


def _resolve_absolute_regular_file(value: Any, *, context: str) -> Path:
    raw = _string(value, context=context)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise Wan22ParallelError(f"{context} must be absolute: {path}")
    return _regular_file(path, context=context)


def _validate_generation_contract(
    *,
    contract: Mapping[str, Any],
    shard: Mapping[str, Any],
    shard_manifest: Path,
    shard_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if contract.get("schema_version") != RUN_SCHEMA:
        raise Wan22ParallelError(
            f"{shard['shard_id']} run contract schema differs"
        )
    manifest = _mapping(
        contract.get("manifest"),
        context=f"{shard['shard_id']} contract manifest",
    )
    expected_manifest = shard["manifest"]
    expected_manifest_fields = {
        "path": str(shard_manifest),
        "sha256": expected_manifest["sha256"],
        "bytes": expected_manifest["bytes"],
        "row_count": shard["row_count"],
        "selected_row_count": shard["row_count"],
        "max_samples": None,
    }
    for field, expected in expected_manifest_fields.items():
        if manifest.get(field) != expected:
            raise Wan22ParallelError(
                f"{shard['shard_id']} contract manifest.{field} differs"
            )

    distributed = _mapping(
        contract.get("distributed_execution"),
        context=f"{shard['shard_id']} distributed_execution",
    )
    expected_distributed = {
        "world_size": EXPECTED_WORLD_SIZE,
        "cooperative_samples_per_step": 1,
        "independent_model_per_gpu": False,
        "t5_fsdp": True,
        "dit_fsdp": True,
        "ulysses_size": EXPECTED_WORLD_SIZE,
        "max_new_samples_per_allocation": shard["row_count"],
    }
    for field, expected in expected_distributed.items():
        if distributed.get(field) != expected:
            raise Wan22ParallelError(
                f"{shard['shard_id']} distributed_execution.{field} differs"
            )

    parameters = _mapping(
        contract.get("generation_parameters"),
        context=f"{shard['shard_id']} generation_parameters",
    )
    expected_parameters = {
        "size": EXPECTED_SIZE,
        "frame_num": EXPECTED_FRAME_NUM,
        "sample_steps": EXPECTED_SAMPLE_STEPS,
        "sample_shift": EXPECTED_SAMPLE_SHIFT,
        "model_sample_fps": EXPECTED_MODEL_SAMPLE_FPS,
        "output_container_frame_rate": EXPECTED_SOURCE_FRAME_RATE,
        "base_seed": EXPECTED_BASE_SEED,
    }
    for field, expected in expected_parameters.items():
        if parameters.get(field) != expected:
            raise Wan22ParallelError(
                f"{shard['shard_id']} generation_parameters.{field} differs"
            )

    authorization = _mapping(
        contract.get("authorization"),
        context=f"{shard['shard_id']} authorization",
    )
    signed_mode = authorization.get("mode") == SIGNED_AUTHORIZATION_MODE
    if signed_mode:
        if (
            authorization.get("allow_pending_review") is not False
            or authorization.get("legacy_approval_records_trusted") is not False
            or not isinstance(authorization.get("release"), Mapping)
        ):
            raise Wan22ParallelError(
                f"{shard['shard_id']} signed authorization differs"
            )
    else:
        expected_authorization = {
            "allow_pending_review": False,
            "pending_review_override_supported": False,
            "requires_explicit_human_approval": True,
            "approved_manifest_role": APPROVED_MANIFEST_ROLE,
            "approval_schema": APPROVAL_SCHEMA,
        }
        for field, expected in expected_authorization.items():
            if authorization.get(field) != expected:
                raise Wan22ParallelError(
                    f"{shard['shard_id']} authorization.{field} differs"
                )

    temporal_policy = _validate_contract_temporal_policy(
        contract.get("temporal_policy"),
        context=f"{shard['shard_id']} temporal_policy",
    )

    selected = contract.get("selected_inputs")
    if not isinstance(selected, list) or len(selected) != len(shard_rows):
        raise Wan22ParallelError(
            f"{shard['shard_id']} selected_inputs length differs"
        )
    expected_iids = [str(row["iid"]) for row in shard_rows]
    if [item.get("iid") for item in selected if isinstance(item, Mapping)] != (
        expected_iids
    ):
        raise Wan22ParallelError(
            f"{shard['shard_id']} selected_inputs IID order differs"
        )
    for index, (selected_row, manifest_row) in enumerate(
        zip(selected, shard_rows)
    ):
        selected_row = _mapping(
            selected_row,
            context=f"{shard['shard_id']} selected_inputs[{index}]",
        )
        expected = {
            "index": index,
            "iid": manifest_row["iid"],
            "group_id": manifest_row["group_id"],
            "row_digest": _object_digest(manifest_row),
            "seed": sample_seed(EXPECTED_BASE_SEED, str(manifest_row["iid"])),
            "authorization_mode": (
                SIGNED_AUTHORIZATION_MODE
                if signed_mode
                else "bound_human_approval"
            ),
            "manifest_role": (
                "review_proposal" if signed_mode else APPROVED_MANIFEST_ROLE
            ),
            "production_eligible": False if signed_mode else True,
            "approval": manifest_row["approval"],
            "action_change_substantive": "yes",
        }
        if signed_mode:
            release_record = _mapping(
                authorization.get("release"),
                context=f"{shard['shard_id']} signed release",
            )
            expected["signed_release"] = {
                field: release_record[field]
                for field in (
                    "path",
                    "release_id",
                    "payload_sha256",
                    "signer_key_fingerprint",
                )
            }
        for field, expected_value in expected.items():
            if selected_row.get(field) != expected_value:
                raise Wan22ParallelError(
                    f"{shard['shard_id']} selected_inputs[{index}]."
                    f"{field} differs"
                )
        probe = _mapping(
            selected_row.get("source_video_ffprobe"),
            context=(
                f"{shard['shard_id']} selected_inputs[{index}]."
                "source_video_ffprobe"
            ),
        )
        _validate_temporal_probe(
            probe,
            expected={
                "frame_count": temporal_policy["source_frame_count"],
                "frame_rate": temporal_policy["source_frame_rate"],
                "duration_seconds": probe.get("duration_seconds"),
            },
            context=(
                f"{shard['shard_id']} selected_inputs[{index}]."
                "source_video_ffprobe"
            ),
        )
        duration = _finite_number(
            probe.get("duration_seconds"),
            context=(
                f"{shard['shard_id']} selected_inputs[{index}]."
                "source_video_ffprobe.duration_seconds"
            ),
        )
        lower, upper = temporal_policy["source_duration_range_seconds"]
        if not float(lower) - 1e-9 <= duration <= float(upper) + 1e-9:
            raise Wan22ParallelError(
                f"{shard['shard_id']} selected_inputs[{index}] source "
                "duration lies outside contract range"
            )
    return temporal_policy


def _validate_regular_hash(
    path: Path,
    expected: Any,
    *,
    context: str,
    digest_cache: dict[Path, str],
) -> str:
    regular = _regular_file(path, context=context)
    expected_digest = _sha256_field(expected, context=f"{context} SHA-256")
    actual = digest_cache.get(regular)
    if actual is None:
        actual = _sha256_file(regular)
        digest_cache[regular] = actual
    if actual != expected_digest:
        raise Wan22ParallelError(
            f"{context} hash mismatch: expected={expected_digest} actual={actual}"
        )
    return actual


def _validate_generated_sample(
    *,
    generated_row: Mapping[str, Any],
    source_row: Mapping[str, Any],
    sample_index: int,
    output_root: Path,
    contract: Mapping[str, Any],
    completion_result_digest: Any,
    digest_cache: dict[Path, str],
) -> str:
    iid = _safe_iid(
        generated_row.get("iid"),
        context=f"generated row {sample_index} iid",
    )
    if generated_row.get("schema_version") != GENERATED_MANIFEST_SCHEMA:
        raise Wan22ParallelError(f"generated row iid={iid} schema differs")
    authorization = _mapping(
        contract.get("authorization"),
        context=f"generated row iid={iid} contract authorization",
    )
    signed_mode = authorization.get("mode") == SIGNED_AUTHORIZATION_MODE
    expected_source_fields = [
        "iid",
        "group_id",
        "action_category",
        "target_action_verb",
        "edit_instruction",
        "action_change_substantive",
    ]
    if not signed_mode:
        expected_source_fields.append("absolute_target_prompt")
    for field in expected_source_fields:
        if generated_row.get(field) != source_row.get(field):
            raise Wan22ParallelError(
                f"generated row iid={iid} {field} differs from source manifest"
            )
    expected_seed = sample_seed(EXPECTED_BASE_SEED, iid)
    if generated_row.get("seed") != expected_seed:
        raise Wan22ParallelError(f"generated row iid={iid} seed differs")

    sample_dir = output_root / "samples" / iid
    samples_root = output_root / "samples"
    if samples_root.is_symlink() or not samples_root.is_dir():
        raise Wan22ParallelError(
            f"samples root must be a non-symlink directory: {samples_root}"
        )
    if sample_dir.is_symlink() or not sample_dir.is_dir():
        raise Wan22ParallelError(
            f"sample directory must be a non-symlink directory: {sample_dir}"
        )
    result_path = _resolve_absolute_regular_file(
        generated_row.get("result_json"),
        context=f"generated row iid={iid} result_json",
    )
    expected_result_path = (sample_dir / SAMPLE_RESULT_NAME).resolve(strict=True)
    if result_path != expected_result_path:
        raise Wan22ParallelError(
            f"generated row iid={iid} result_json escapes sample directory"
        )
    result = _load_bound_object(
        result_path,
        context=f"sample result iid={iid}",
        digest_field="result_digest",
    )
    result_digest = _sha256_field(
        result.get("result_digest"),
        context=f"sample result iid={iid} result_digest",
    )
    if generated_row.get("result_digest") != result_digest:
        raise Wan22ParallelError(
            f"generated row iid={iid} result_digest differs"
        )
    if completion_result_digest != result_digest:
        raise Wan22ParallelError(
            f"run completion result digest differs for iid={iid}"
        )
    expected_result = {
        "schema_version": SAMPLE_SCHEMA,
        "iid": iid,
        "group_id": source_row["group_id"],
        "sample_index": sample_index,
        "manifest_sha256": contract["manifest"]["sha256"],
        "manifest_row_digest": _object_digest(source_row),
        "contract_digest": contract["contract_digest"],
        "seed": expected_seed,
        "action_change_substantive": "yes",
    }
    for field, expected in expected_result.items():
        if result.get(field) != expected:
            raise Wan22ParallelError(
                f"sample result iid={iid} {field} differs"
            )
    if generated_row.get("authorization_mode") != result.get(
        "authorization_mode"
    ):
        raise Wan22ParallelError(
            f"generated row iid={iid} authorization_mode differs"
        )
    expected_authorization_mode = (
        SIGNED_AUTHORIZATION_MODE if signed_mode else "bound_human_approval"
    )
    if result.get("authorization_mode") != expected_authorization_mode:
        raise Wan22ParallelError(
            f"sample result iid={iid} authorization mode differs"
        )
    expected_role = "review_proposal" if signed_mode else APPROVED_MANIFEST_ROLE
    expected_eligible = False if signed_mode else True
    for value, value_context in (
        (result, f"sample result iid={iid}"),
        (generated_row, f"generated row iid={iid}"),
    ):
        if (
            value.get("manifest_role") != expected_role
            or value.get("production_eligible") is not expected_eligible
            or _canonical_bytes(value.get("approval"))
            != _canonical_bytes(source_row["approval"])
        ):
            raise Wan22ParallelError(
                f"{value_context} authorization provenance differs"
            )
    expected_review = "pending" if signed_mode else "approved"
    expected_manifest_authorization = False if signed_mode else True
    if (
        result.get("human_review_status_at_generation") != expected_review
        or result.get("generation_authorized_in_manifest")
        is not expected_manifest_authorization
    ):
        raise Wan22ParallelError(
            f"sample result iid={iid} authorization evidence differs"
        )
    prompt = _mapping(result.get("prompt"), context=f"result iid={iid} prompt")
    if signed_mode:
        if (
            set(prompt) != {"field", "text", "sha256"}
            or prompt.get("field") != "edit_instruction"
            or prompt.get("text") != source_row["edit_instruction"]
            or prompt.get("sha256")
            != _sha256_bytes(source_row["edit_instruction"].encode("utf-8"))
            or generated_row.get("edit_instruction_sha256")
            != prompt.get("sha256")
            or "absolute_target_prompt" in generated_row
        ):
            raise Wan22ParallelError(
                f"sample result iid={iid} signed instruction differs"
            )
        release_record = _mapping(
            authorization.get("release"),
            context=f"sample result iid={iid} signed release",
        )
        expected_release = {
            field: release_record[field]
            for field in (
                "path",
                "release_id",
                "payload_sha256",
                "signer_key_fingerprint",
            )
        }
        if (
            result.get("signed_release") != expected_release
            or generated_row.get("signed_release") != expected_release
        ):
            raise Wan22ParallelError(
                f"sample result iid={iid} signed release differs"
            )
    elif (
        prompt.get("field") != "absolute_target_prompt"
        or prompt.get("text") != source_row["absolute_target_prompt"]
        or prompt.get("edit_instruction") != source_row["edit_instruction"]
    ):
        raise Wan22ParallelError(f"sample result iid={iid} prompt differs")
    if result.get("generation_parameters") != contract.get(
        "generation_parameters"
    ):
        raise Wan22ParallelError(
            f"sample result iid={iid} generation parameters differ"
        )
    if (
        generated_row.get("first_frame_policy") != FIRST_FRAME_POLICY
        or generated_row.get("mp4_decode_pixel_equality_claimed") is not False
    ):
        raise Wan22ParallelError(
            f"generated row iid={iid} first-frame policy differs"
        )
    policy = _mapping(
        result.get("first_frame_policy"),
        context=f"result iid={iid} first_frame_policy",
    )
    required_policy = {
        "policy_version": FIRST_FRAME_POLICY,
        "tensor_frame0_overridden_before_encoding": True,
        "preencode_frame0_matches_png_pixels": True,
        "mp4_codec_is_lossy": True,
        "mp4_decode_pixel_equality_claimed": False,
    }
    for field, expected in required_policy.items():
        if policy.get(field) != expected:
            raise Wan22ParallelError(
                f"sample result iid={iid} first_frame_policy.{field} differs"
            )

    inputs = _mapping(result.get("inputs"), context=f"result iid={iid} inputs")
    source_path = _resolve_absolute_regular_file(
        generated_row.get("source_video"),
        context=f"generated row iid={iid} source_video",
    )
    _validate_regular_hash(
        source_path,
        generated_row.get("source_video_sha256"),
        context=f"generated source iid={iid}",
        digest_cache=digest_cache,
    )
    if generated_row.get("source_video_sha256") != source_row.get(
        "source_video_sha256"
    ):
        raise Wan22ParallelError(
            f"generated source hash iid={iid} differs from source manifest"
        )
    result_source = _resolve_absolute_regular_file(
        inputs.get("source_video_resolved_path"),
        context=f"result iid={iid} source_video_resolved_path",
    )
    if (
        result_source != source_path
        or inputs.get("source_video_sha256")
        != generated_row.get("source_video_sha256")
    ):
        raise Wan22ParallelError(f"sample result iid={iid} source binding differs")
    if inputs.get("anchor_sha256") != source_row.get("anchor_sha256"):
        raise Wan22ParallelError(f"sample result iid={iid} anchor binding differs")

    outputs = _mapping(
        result.get("outputs"),
        context=f"sample result iid={iid} outputs",
    )
    contract_temporal = _validate_contract_temporal_policy(
        contract.get("temporal_policy"),
        context=f"sample result iid={iid} contract temporal_policy",
    )
    result_temporal = _validate_pair_temporal_policy(
        result.get("temporal_policy"),
        contract_policy=contract_temporal,
        context=f"sample result iid={iid} temporal_policy",
    )
    generated_temporal = _validate_pair_temporal_policy(
        generated_row.get("temporal_policy"),
        contract_policy=contract_temporal,
        context=f"generated row iid={iid} temporal_policy",
    )
    if _canonical_bytes(result_temporal) != _canonical_bytes(
        generated_temporal
    ):
        raise Wan22ParallelError(
            f"generated row iid={iid} temporal policy differs from result"
        )
    _validate_temporal_probe(
        inputs.get("source_video_ffprobe"),
        expected=result_temporal["source"],
        context=f"sample result iid={iid} source_video_ffprobe",
    )
    _validate_temporal_probe(
        outputs.get("preview_mp4_ffprobe"),
        expected=result_temporal["target"],
        context=f"sample result iid={iid} preview_mp4_ffprobe",
    )
    for (
        generated_path_field,
        generated_hash_field,
        result_path_field,
        result_hash_field,
    ) in _OUTPUT_BINDINGS:
        basename = _string(
            outputs.get(result_path_field),
            context=f"result iid={iid} {result_path_field}",
        )
        if Path(basename).name != basename or basename in {".", ".."}:
            raise Wan22ParallelError(
                f"result iid={iid} {result_path_field} is not one basename"
            )
        expected_path = (sample_dir / basename).resolve(strict=True)
        generated_path = _resolve_absolute_regular_file(
            generated_row.get(generated_path_field),
            context=f"generated row iid={iid} {generated_path_field}",
        )
        if generated_path != expected_path:
            raise Wan22ParallelError(
                f"generated row iid={iid} {generated_path_field} "
                "escapes sample directory"
            )
        if generated_row.get(generated_hash_field) != outputs.get(
            result_hash_field
        ):
            raise Wan22ParallelError(
                f"generated row iid={iid} {generated_hash_field} differs"
            )
        _validate_regular_hash(
            generated_path,
            generated_row.get(generated_hash_field),
            context=f"generated iid={iid} {generated_path_field}",
            digest_cache=digest_cache,
        )
    if generated_row.get(
        "conditioning_anchor_original_sha256"
    ) != source_row.get("anchor_sha256"):
        raise Wan22ParallelError(
            f"generated row iid={iid} original anchor hash differs"
        )
    return result_digest


def finalize_parallel_run(
    *,
    parallel_root: str | Path,
) -> dict[str, Any]:
    """Validate every shard and publish a deterministic aggregate receipt."""

    requested = Path(parallel_root).expanduser()
    if not requested.is_absolute() or requested == Path("/"):
        raise Wan22ParallelError("parallel_root must be a non-root absolute path")
    root = _regular_directory(requested, context="parallel root")
    plan = _load_plan(root)

    source_manifest, source_raw, _, source_rows = _strict_jsonl_file(
        Path(plan["source_manifest"]["path"]),
        context="planned source generation manifest",
    )
    plan_authorization = _mapping(
        plan.get("authorization"),
        context="parallel plan authorization",
    )
    signed_plan = plan_authorization.get("mode") == SIGNED_AUTHORIZATION_MODE
    if signed_plan:
        release_record = _mapping(
            plan_authorization.get("release"),
            context="parallel plan signed release",
        )
        try:
            from motive.wan22_signed_release import verify_signed_release

            released = verify_signed_release(
                release_path=Path(str(release_record["path"])),
                manifest_path=source_manifest,
                require_exact_manifest=True,
                verify_media=True,
            )
        except Exception as error:
            raise Wan22ParallelError(
                f"parallel finalizer signed release failed: {error}"
            ) from error
        if released["release"] != release_record:
            raise Wan22ParallelError(
                "parallel plan signed release record differs"
            )
        source_iids = [
            str(row["_iid"]) for row in released["selected_rows"]
        ]
        source_groups = [
            str(row["group_id"]) for row in released["selected_rows"]
        ]
    else:
        source_iids, source_groups = _validate_source_rows(
            source_rows,
            allow_pending_review=False,
        )
    expected_source = plan["source_manifest"]
    source_checks = {
        "path": str(source_manifest),
        "sha256": _sha256_bytes(source_raw),
        "bytes": len(source_raw),
        "row_count": len(source_rows),
        "iids": source_iids,
        "group_ids": source_groups,
    }
    if expected_source != source_checks:
        raise Wan22ParallelError("source manifest differs from parallel plan")
    if len(source_rows) != plan.get("expected_source_row_count"):
        raise Wan22ParallelError(
            "source row count differs from the parallel plan expectation"
        )

    shards = plan.get("shards")
    if (
        not isinstance(shards, list)
        or len(shards) != plan.get("shard_count")
        or not shards
    ):
        raise Wan22ParallelError("parallel plan shard list differs")
    expected_start = 0
    shard_manifest_bytes: list[bytes] = []
    generated_bytes_parts: list[bytes] = []
    generated_rows_all: list[dict[str, Any]] = []
    aggregate_shards: list[dict[str, Any]] = []
    digest_cache: dict[Path, str] = {}
    temporal_common: dict[str, Any] | None = None
    temporal_duration_min: float | None = None
    temporal_duration_max: float | None = None

    for expected_index, shard_value in enumerate(shards):
        shard = _mapping(
            shard_value,
            context=f"parallel plan shard {expected_index}",
        )
        shard_id = f"shard_{expected_index:03d}"
        if shard.get("index") != expected_index or shard.get("shard_id") != shard_id:
            raise Wan22ParallelError(f"parallel plan {shard_id} identity differs")
        start = shard.get("row_start_zero_based")
        stop = shard.get("row_stop_exclusive")
        if (
            type(start) is not int
            or type(stop) is not int
            or start != expected_start
            or stop <= start
            or stop > len(source_rows)
            or shard.get("row_count") != stop - start
        ):
            raise Wan22ParallelError(f"parallel plan {shard_id} range differs")
        expected_start = stop
        shard_source_rows = source_rows[start:stop]
        shard_source_iids = source_iids[start:stop]
        if shard.get("iids") != shard_source_iids:
            raise Wan22ParallelError(f"parallel plan {shard_id} IID order differs")
        if shard.get("group_ids") != source_groups[start:stop]:
            raise Wan22ParallelError(
                f"parallel plan {shard_id} group order differs"
            )
        if shard.get("dependency") != (
            f"afterok:{plan['geometry_job_id']}"
        ):
            raise Wan22ParallelError(
                f"parallel plan {shard_id} dependency differs"
            )

        shard_manifest, shard_raw, _, shard_rows = _strict_jsonl_file(
            Path(shard["manifest"]["path"]),
            context=f"{shard_id} source manifest",
        )
        if shard_manifest != root / "manifests" / f"{shard_id}.jsonl":
            raise Wan22ParallelError(
                f"{shard_id} manifest path differs from reserved layout"
            )
        shard_manifest_bytes.append(shard_raw)
        manifest_checks = {
            "path": str(shard_manifest),
            "relative_path": str(shard_manifest.relative_to(root)),
            "sha256": _sha256_bytes(shard_raw),
            "bytes": len(shard_raw),
        }
        if shard.get("manifest") != manifest_checks:
            raise Wan22ParallelError(f"{shard_id} manifest differs from plan")
        if shard_rows != shard_source_rows:
            raise Wan22ParallelError(
                f"{shard_id} manifest rows differ from source slice"
            )

        output_root = _regular_directory(
            Path(shard["output_root"]),
            context=f"{shard_id} output root",
        )
        if output_root != root / "shards" / shard_id:
            raise Wan22ParallelError(f"{shard_id} output root differs from plan")
        contract_path = output_root / RUN_CONTRACT_NAME
        contract = _load_bound_object(
            contract_path,
            context=f"{shard_id} run contract",
            digest_field="contract_digest",
        )
        contract_temporal = _validate_generation_contract(
            contract=contract,
            shard=shard,
            shard_manifest=shard_manifest,
            shard_rows=shard_rows,
        )
        if signed_plan:
            contract_authorization = _mapping(
                contract.get("authorization"),
                context=f"{shard_id} contract authorization",
            )
            if contract_authorization != plan_authorization:
                raise Wan22ParallelError(
                    f"{shard_id} signed release differs from plan"
                )
        shard_temporal_common = dict(contract_temporal)
        shard_duration_range = shard_temporal_common.pop(
            "source_duration_range_seconds"
        )
        if temporal_common is None:
            temporal_common = shard_temporal_common
        elif _canonical_bytes(temporal_common) != _canonical_bytes(
            shard_temporal_common
        ):
            raise Wan22ParallelError(
                f"{shard_id} temporal policy differs across shards"
            )
        shard_min = float(shard_duration_range[0])
        shard_max = float(shard_duration_range[1])
        temporal_duration_min = (
            shard_min
            if temporal_duration_min is None
            else min(temporal_duration_min, shard_min)
        )
        temporal_duration_max = (
            shard_max
            if temporal_duration_max is None
            else max(temporal_duration_max, shard_max)
        )

        complete_path = output_root / RUN_COMPLETE_NAME
        completion = _load_bound_object(
            complete_path,
            context=f"{shard_id} run completion",
            digest_field="complete_digest",
        )
        if completion.get("schema_version") != COMPLETE_SCHEMA:
            raise Wan22ParallelError(f"{shard_id} completion schema differs")
        expected_completion = {
            "contract_digest": contract["contract_digest"],
            "manifest_sha256": shard["manifest"]["sha256"],
            "selected_sample_count": shard["row_count"],
            "completed_sample_count": shard["row_count"],
            "generated_manifest": GENERATED_MANIFEST_NAME,
            "temporal_policy": contract_temporal,
        }
        for field, expected in expected_completion.items():
            if completion.get(field) != expected:
                raise Wan22ParallelError(
                    f"{shard_id} completion {field} differs"
                )

        generated_path, generated_raw, _, generated_rows = _strict_jsonl_file(
            output_root / GENERATED_MANIFEST_NAME,
            context=f"{shard_id} generated manifest",
        )
        generated_sha = _sha256_bytes(generated_raw)
        if completion.get("generated_manifest_sha256") != generated_sha:
            raise Wan22ParallelError(
                f"{shard_id} generated manifest SHA differs from completion"
            )
        if [row.get("iid") for row in generated_rows] != shard_source_iids:
            raise Wan22ParallelError(
                f"{shard_id} generated manifest IID order differs"
            )
        result_digests = completion.get("sample_result_digests")
        if not isinstance(result_digests, list) or len(result_digests) != len(
            generated_rows
        ):
            raise Wan22ParallelError(
                f"{shard_id} completion result digest count differs"
            )
        validated_result_digests = [
            _validate_generated_sample(
                generated_row=generated_row,
                source_row=source_row,
                sample_index=index,
                output_root=output_root,
                contract=contract,
                completion_result_digest=result_digests[index],
                digest_cache=digest_cache,
            )
            for index, (generated_row, source_row) in enumerate(
                zip(generated_rows, shard_source_rows)
            )
        ]
        if validated_result_digests != result_digests:
            raise Wan22ParallelError(
                f"{shard_id} validated result digest order differs"
            )
        generated_bytes_parts.append(generated_raw)
        generated_rows_all.extend(generated_rows)
        aggregate_shards.append(
            {
                "index": expected_index,
                "shard_id": shard_id,
                "row_start_zero_based": start,
                "row_stop_exclusive": stop,
                "row_count": stop - start,
                "manifest_sha256": shard["manifest"]["sha256"],
                "run_contract": str(contract_path),
                "run_contract_sha256": _sha256_file(contract_path),
                "contract_digest": contract["contract_digest"],
                "run_complete": str(complete_path),
                "run_complete_sha256": _sha256_file(complete_path),
                "complete_digest": completion["complete_digest"],
                "generated_manifest": str(generated_path),
                "generated_manifest_sha256": generated_sha,
                "result_digests": validated_result_digests,
                "temporal_policy": contract_temporal,
            }
        )

    if expected_start != len(source_rows):
        raise Wan22ParallelError("parallel shard ranges do not cover source rows")
    if b"".join(shard_manifest_bytes) != source_raw:
        raise Wan22ParallelError(
            "concatenated shard manifests differ from source manifest bytes"
        )
    generated_iids = [row["iid"] for row in generated_rows_all]
    if generated_iids != source_iids:
        raise Wan22ParallelError(
            "aggregate generated IID order differs from source manifest"
        )
    if len(set(generated_iids)) != len(generated_iids):
        raise Wan22ParallelError("aggregate generated IIDs are not unique")
    if (
        temporal_common is None
        or temporal_duration_min is None
        or temporal_duration_max is None
    ):
        raise Wan22ParallelError("aggregate temporal policy is missing")
    aggregate_temporal = dict(temporal_common)
    aggregate_temporal["source_duration_range_seconds"] = [
        temporal_duration_min,
        temporal_duration_max,
    ]
    aggregate_temporal = _validate_contract_temporal_policy(
        aggregate_temporal,
        context="aggregate temporal_policy",
    )

    merged_bytes = b"".join(generated_bytes_parts)
    _, parsed_merged = _strict_jsonl_bytes(
        merged_bytes,
        context="aggregate generated manifest",
    )
    if [row.get("iid") for row in parsed_merged] != source_iids:
        raise Wan22ParallelError("serialized aggregate IID order differs")
    final_root = _regular_directory(root / "final", context="aggregate final root")
    merged_path = final_root / GENERATED_MANIFEST_NAME
    _publish_identical_or_create(merged_path, merged_bytes)

    aggregate: dict[str, Any] = {
        "schema_version": AGGREGATE_SCHEMA,
        "parallel_plan": str(root / PLAN_NAME),
        "parallel_plan_sha256": _sha256_file(root / PLAN_NAME),
        "plan_digest": plan["plan_digest"],
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": _sha256_bytes(source_raw),
        "source_row_count": len(source_rows),
        "source_iids": source_iids,
        "shard_count": len(shards),
        "shards": aggregate_shards,
        "generated_manifest": str(merged_path),
        "generated_manifest_sha256": _sha256_bytes(merged_bytes),
        "generated_row_count": len(generated_rows_all),
        "generated_iids": generated_iids,
        "temporal_policy": aggregate_temporal,
        "coverage": {
            "contiguous_source_order": True,
            "no_duplicate_iids": True,
            "all_source_rows_generated": True,
        },
        "validated_generation_parameters": {
            "world_size": EXPECTED_WORLD_SIZE,
            "size": EXPECTED_SIZE,
            "frame_num": EXPECTED_FRAME_NUM,
            "sample_steps": EXPECTED_SAMPLE_STEPS,
            "sample_shift": EXPECTED_SAMPLE_SHIFT,
            "model_sample_fps": EXPECTED_MODEL_SAMPLE_FPS,
            "output_container_frame_rate": EXPECTED_SOURCE_FRAME_RATE,
            "base_seed": EXPECTED_BASE_SEED,
        },
        "validated_file_sha256_count": len(digest_cache),
    }
    aggregate["aggregate_digest"] = _object_digest(aggregate)
    aggregate_path = final_root / AGGREGATE_COMPLETE_NAME
    _publish_identical_or_create(
        aggregate_path,
        _pretty_json_bytes(aggregate),
    )
    return aggregate


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare",
        help="split one frozen manifest into fresh contiguous shard manifests",
    )
    prepare.add_argument("--manifest", required=True, type=Path)
    prepare.add_argument("--signed-release", type=Path)
    prepare.add_argument("--parallel-root", required=True, type=Path)
    prepare.add_argument("--geometry-job-id", required=True, type=int)
    prepare.add_argument("--shard-count", default=3, type=int)
    prepare.add_argument("--expected-row-count", type=int)
    prepare.add_argument(
        "--allow-pending-review",
        action="store_true",
        help=(
            "Legacy flag retained only to fail closed; production sharding "
            "never permits pending-review generation."
        ),
    )

    finalize = subparsers.add_parser(
        "finalize",
        help="validate every shard and atomically publish the aggregate",
    )
    finalize.add_argument("--parallel-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_parallel_run(
                manifest_path=args.manifest,
                parallel_root=args.parallel_root,
                geometry_job_id=args.geometry_job_id,
                shard_count=args.shard_count,
                allow_pending_review=args.allow_pending_review,
                expected_row_count=args.expected_row_count,
                signed_release_path=args.signed_release,
            )
        else:
            result = finalize_parallel_run(parallel_root=args.parallel_root)
    except Exception as error:
        print(
            f"[wan22-parallel] fatal {type(error).__name__}: {error}",
            file=os.sys.stderr,
            flush=True,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
