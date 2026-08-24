"""Fail-closed loader for the EPMC K=2 experimental action canary.

The loader binds one caller-pinned config to the preview manifest and VAE
index named by that config.  It then joins exactly two support rows and one
held-out row by IID, revalidates every selected artifact hash, and probes both
source and target videos for exactly 81 frames at 25 FPS.

The upstream rows deliberately remain preview-only and training-forbidden.
Loading them therefore requires an explicit experimental acknowledgement.  A
successful load does not change those authorization fields and cannot be used
as a production or scientific-quality receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "bernini-epmc-exact-micro-program-k2-v1"
PREVIEW_ROW_SCHEMA = "omnivideo2-action-preview-row-v1"
VAE_INDEX_ROW_SCHEMA = "bernini-r-action-vae-index-row-v2"
AUDIT_SCHEMA = "bernini-epmc-exact-micro-program-k2-audit-v1"

# The default binds the checked-in config byte-for-byte.  Tests and future
# deliberately reviewed configs may provide another explicit caller pin.
REFERENCE_CONFIG_SHA256 = (
    "a46d18fce025b0cd3b30a6505514b817cd5d96c43d305d6202405a952eed2446"
)

EXPECTED_RGB_FRAMES = 81
EXPECTED_FPS = 25
EXPECTED_LATENT_PHASES = 21
EXPECTED_BUCKET_HW = (480, 496)
EXPECTED_POSTERIOR_SHAPE = (1, 32, 21, 60, 62)
EXPECTED_PATCH_GRID_THW = (21, 30, 31)
EXPECTED_TARGET_ACTION = "sit_and_turn_head"
EXPECTED_ENTITY_TYPE = "animal"
EXPECTED_SPLIT_SEED = 20260807
EXPECTED_SUPPORT_COUNT = 2
EXPECTED_TOTAL_ROWS = 3

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_IID_RE = re.compile(r"[0-9a-f]{16}")
_DATE_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")

_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "purpose",
        "production_claim_forbidden",
        "scientific_claim_authorized",
        "upstream_authorization",
        "inference_contract",
        "dataset_binding",
        "micro_program",
        "split_seed",
        "rows",
        "manual_contact_sheet_review",
    }
)
_AUTHORIZATION_FIELDS = frozenset(
    {
        "post_video_acceptance",
        "preview_only",
        "training_authorized",
        "training_use_forbidden",
        "user_requested_experimental_training_ack_required",
    }
)
_INFERENCE_FIELDS = frozenset(
    {
        "external_inputs",
        "target_available",
        "support_available",
        "external_mask_flow_pose_track_trajectory",
    }
)
_BINDING_FIELDS = frozenset(
    {"preview_manifest_sha256", "vae_index_sha256"}
)
_MICRO_PROGRAM_FIELDS = frozenset(
    {
        "target_action_signature",
        "entity_type",
        "num_rgb_frames",
        "fps",
        "latent_phases",
        "bucket_hw",
        "posterior_parameters_shape",
        "patch_grid_thw",
    }
)
_CONFIG_COMMON_ROW_FIELDS = frozenset(
    {
        "role",
        "iid",
        "group_id",
        "source_video_sha256",
        "target_video_sha256",
        "edit_instruction_sha256",
        "vae_parquet_sha256",
        "source_action_signature",
    }
)
_MANUAL_REVIEW_FIELDS = frozenset(
    {"date", "frames", "verdict", "observations"}
)
_PREVIEW_FIELDS = frozenset(
    {
        "schema_version",
        "iid",
        "group_id",
        "family",
        "source_video_path",
        "source_video_sha256",
        "target_video_path",
        "target_video_sha256",
        "edit_instruction",
        "edit_instruction_sha256",
        "instruction_source",
        "generation_instruction",
        "generation_instruction_sha256",
        "source_census",
        "target_plan",
        "selection_gates",
        "preview_only",
        "training_authorized",
        "training_use_forbidden",
        "production_eligible",
        "post_video_acceptance",
        "provenance",
        "row_digest",
    }
)
_SELECTION_GATE_FIELDS = frozenset(
    {
        "single_dynamic_actor",
        "source_camera_locked_off",
        "target_camera_locked_off",
        "target_camera_preserve_static",
        "source_census_high_confidence",
        "target_plan_high_confidence",
    }
)
_VAE_INDEX_FIELDS = frozenset(
    {
        "schema_version",
        "iid",
        "parquet_path",
        "parquet_sha256",
        "materialized_row_digest",
        "bucket_hw",
        "posterior_parameters_shape",
        "sample_receipt_path",
        "sample_receipt_sha256",
        "preview_only",
        "production_claim_forbidden",
    }
)


class FewShotEpisodeIOError(RuntimeError):
    """A config, identity, artifact, media, or authorization check failed."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FewShotEpisodeIOError(
            f"value cannot be represented as canonical JSON: {error}"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _required_sha256(value: Any, *, context: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise FewShotEpisodeIOError(f"{context} must be a lowercase SHA-256")
    return value


def _required_iid(value: Any, *, context: str) -> str:
    if type(value) is not str or _IID_RE.fullmatch(value) is None:
        raise FewShotEpisodeIOError(f"{context} must be a 16-character lowercase IID")
    return value


def _required_text(value: Any, *, context: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise FewShotEpisodeIOError(
            f"{context} must be non-empty text without NUL"
        )
    return value


def _required_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FewShotEpisodeIOError(f"{context} must be an object")
    return dict(value)


def _required_list(value: Any, *, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise FewShotEpisodeIOError(f"{context} must be a list")
    return list(value)


def _require_exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], *, context: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise FewShotEpisodeIOError(
            f"{context} fields differ: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FewShotEpisodeIOError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise FewShotEpisodeIOError(f"non-finite JSON number: {value}")


def _decode_json_object(payload: bytes, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except FewShotEpisodeIOError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FewShotEpisodeIOError(f"invalid {context}: {error}") from error
    if not isinstance(value, dict):
        raise FewShotEpisodeIOError(f"{context} must contain one JSON object")
    return value


@dataclass(frozen=True)
class BoundArtifact:
    path: Path
    sha256: str
    size_bytes: int
    device: int
    inode: int
    mtime_ns: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "device": self.device,
            "inode": self.inode,
            "mtime_ns": self.mtime_ns,
        }


def _plain_file(path_value: str | Path, *, context: str) -> Path:
    requested = Path(path_value).expanduser()
    try:
        mode = requested.lstat().st_mode
    except FileNotFoundError as error:
        raise FewShotEpisodeIOError(f"missing {context}: {requested}") from error
    except OSError as error:
        raise FewShotEpisodeIOError(f"cannot inspect {context}: {error}") from error
    if requested.is_symlink() or not stat.S_ISREG(mode):
        raise FewShotEpisodeIOError(f"{context} must be a plain non-symlink file")
    try:
        return requested.resolve(strict=True)
    except OSError as error:
        raise FewShotEpisodeIOError(f"cannot resolve {context}: {error}") from error


def _snapshot_tuple(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _read_stable_bytes(
    path_value: str | Path, *, context: str
) -> tuple[BoundArtifact, bytes]:
    path = _plain_file(path_value, context=context)
    try:
        before = path.stat()
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if _snapshot_tuple(before) != _snapshot_tuple(opened):
                raise FewShotEpisodeIOError(f"{context} changed while opening")
            payload = handle.read()
            after = os.fstat(handle.fileno())
        final = path.stat()
    except FewShotEpisodeIOError:
        raise
    except OSError as error:
        raise FewShotEpisodeIOError(f"cannot read {context}: {error}") from error
    if (
        _snapshot_tuple(opened) != _snapshot_tuple(after)
        or _snapshot_tuple(after) != _snapshot_tuple(final)
    ):
        raise FewShotEpisodeIOError(f"{context} changed while reading")
    digest = bytes_sha256(payload)
    return (
        BoundArtifact(
            path=path,
            sha256=digest,
            size_bytes=after.st_size,
            device=after.st_dev,
            inode=after.st_ino,
            mtime_ns=after.st_mtime_ns,
        ),
        payload,
    )


def _hash_stable_file(
    path_value: str | Path,
    *,
    expected_sha256: str,
    context: str,
) -> BoundArtifact:
    expected = _required_sha256(expected_sha256, context=f"expected {context} hash")
    path = _plain_file(path_value, context=context)
    digest = hashlib.sha256()
    try:
        before = path.stat()
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if _snapshot_tuple(before) != _snapshot_tuple(opened):
                raise FewShotEpisodeIOError(f"{context} changed while opening")
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            after = os.fstat(handle.fileno())
        final = path.stat()
    except FewShotEpisodeIOError:
        raise
    except OSError as error:
        raise FewShotEpisodeIOError(f"cannot hash {context}: {error}") from error
    if (
        _snapshot_tuple(opened) != _snapshot_tuple(after)
        or _snapshot_tuple(after) != _snapshot_tuple(final)
    ):
        raise FewShotEpisodeIOError(f"{context} changed while hashing")
    actual = digest.hexdigest()
    if actual != expected:
        raise FewShotEpisodeIOError(f"{context} SHA-256 differs")
    return BoundArtifact(
        path=path,
        sha256=actual,
        size_bytes=after.st_size,
        device=after.st_dev,
        inode=after.st_ino,
        mtime_ns=after.st_mtime_ns,
    )


def _pinned_json(
    path_value: str | Path, *, expected_sha256: str, context: str
) -> tuple[BoundArtifact, dict[str, Any]]:
    expected = _required_sha256(
        expected_sha256, context=f"caller-pinned {context} hash"
    )
    artifact, payload = _read_stable_bytes(path_value, context=context)
    if artifact.sha256 != expected:
        raise FewShotEpisodeIOError(f"{context} differs from its pinned SHA-256")
    return artifact, _decode_json_object(payload, context=context)


def _pinned_jsonl(
    path_value: str | Path, *, expected_sha256: str, context: str
) -> tuple[BoundArtifact, list[dict[str, Any]]]:
    expected = _required_sha256(expected_sha256, context=f"pinned {context} hash")
    artifact, payload = _read_stable_bytes(path_value, context=context)
    if artifact.sha256 != expected:
        raise FewShotEpisodeIOError(f"{context} differs from its pinned SHA-256")
    if not payload.endswith(b"\n"):
        raise FewShotEpisodeIOError(f"{context} must end with one newline")
    lines = payload.splitlines()
    if not lines or any(not line for line in lines):
        raise FewShotEpisodeIOError(
            f"{context} must contain non-empty JSONL rows"
        )
    return artifact, [
        _decode_json_object(line, context=f"{context} row {number}")
        for number, line in enumerate(lines, 1)
    ]


def _index_rows_by_iid(
    rows: Sequence[Mapping[str, Any]], *, context: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for line_number, row_value in enumerate(rows, 1):
        row = dict(row_value)
        iid = _required_iid(
            row.get("iid"), context=f"{context} IID at row {line_number}"
        )
        if iid in result:
            raise FewShotEpisodeIOError(f"duplicate {context} IID: {iid}")
        result[iid] = row
    if not result:
        raise FewShotEpisodeIOError(f"{context} is empty")
    return result


@dataclass(frozen=True)
class VideoMetadata:
    frame_count: int
    fps_numerator: int
    fps_denominator: int

    @property
    def fps(self) -> Fraction:
        return Fraction(self.fps_numerator, self.fps_denominator)


def _parse_positive_decimal_integer(value: Any, *, context: str) -> int | None:
    if value in (None, "N/A"):
        return None
    if type(value) is not str or not value.isdecimal():
        raise FewShotEpisodeIOError(f"{context} must be a positive decimal integer")
    result = int(value)
    if result <= 0:
        raise FewShotEpisodeIOError(f"{context} must be positive")
    return result


def _probe_video_metadata(path: Path) -> VideoMetadata:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise FewShotEpisodeIOError("ffprobe is required for exact-frame auditing")
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,nb_frames,avg_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise FewShotEpisodeIOError(f"ffprobe failed for {path}: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[-500:].strip()
        raise FewShotEpisodeIOError(
            f"ffprobe rejected {path}: {detail or 'unknown error'}"
        )
    payload = _decode_json_object(completed.stdout, context=f"ffprobe output for {path}")
    streams = _required_list(payload.get("streams"), context="ffprobe streams")
    if len(streams) != 1 or not isinstance(streams[0], Mapping):
        raise FewShotEpisodeIOError(f"{path} must expose exactly one selected video stream")
    stream = dict(streams[0])
    counted = _parse_positive_decimal_integer(
        stream.get("nb_read_frames"), context=f"nb_read_frames for {path}"
    )
    declared = _parse_positive_decimal_integer(
        stream.get("nb_frames"), context=f"nb_frames for {path}"
    )
    if counted is None and declared is None:
        raise FewShotEpisodeIOError(f"ffprobe returned no frame count for {path}")
    if counted is not None and declared is not None and counted != declared:
        raise FewShotEpisodeIOError(f"decoded and declared frame counts differ for {path}")
    frame_count = counted if counted is not None else declared
    assert frame_count is not None
    rate = stream.get("avg_frame_rate")
    if type(rate) is not str:
        raise FewShotEpisodeIOError(f"avg_frame_rate is missing for {path}")
    try:
        fps = Fraction(rate)
    except (ValueError, ZeroDivisionError) as error:
        raise FewShotEpisodeIOError(f"invalid avg_frame_rate for {path}") from error
    if fps <= 0:
        raise FewShotEpisodeIOError(f"avg_frame_rate must be positive for {path}")
    return VideoMetadata(frame_count, fps.numerator, fps.denominator)


def _audit_video(
    path_value: str | Path,
    *,
    expected_sha256: str,
    context: str,
) -> tuple[BoundArtifact, VideoMetadata]:
    artifact = _hash_stable_file(
        path_value, expected_sha256=expected_sha256, context=context
    )
    before = artifact.path.stat()
    metadata = _probe_video_metadata(artifact.path)
    after = artifact.path.stat()
    expected_snapshot = (
        artifact.device,
        artifact.inode,
        artifact.size_bytes,
        artifact.mtime_ns,
    )
    if (
        _snapshot_tuple(before) != expected_snapshot
        or _snapshot_tuple(after) != expected_snapshot
    ):
        raise FewShotEpisodeIOError(f"{context} changed while probing")
    if metadata.frame_count != EXPECTED_RGB_FRAMES:
        raise FewShotEpisodeIOError(
            f"{context} must contain exactly {EXPECTED_RGB_FRAMES} frames"
        )
    if metadata.fps != Fraction(EXPECTED_FPS, 1):
        raise FewShotEpisodeIOError(
            f"{context} must have exact {EXPECTED_FPS} FPS"
        )
    return artifact, metadata


def _validate_micro_program(value: Any) -> dict[str, Any]:
    program = _required_mapping(value, context="micro_program")
    _require_exact_fields(program, _MICRO_PROGRAM_FIELDS, context="micro_program")
    expected = {
        "target_action_signature": EXPECTED_TARGET_ACTION,
        "entity_type": EXPECTED_ENTITY_TYPE,
        "num_rgb_frames": EXPECTED_RGB_FRAMES,
        "fps": EXPECTED_FPS,
        "latent_phases": EXPECTED_LATENT_PHASES,
        "bucket_hw": list(EXPECTED_BUCKET_HW),
        "posterior_parameters_shape": list(EXPECTED_POSTERIOR_SHAPE),
        "patch_grid_thw": list(EXPECTED_PATCH_GRID_THW),
    }
    if program != expected:
        raise FewShotEpisodeIOError(
            "micro_program differs from the exact 81-frame K=2 contract"
        )
    return program


def _validate_authorization(value: Any) -> dict[str, Any]:
    authorization = _required_mapping(value, context="upstream_authorization")
    _require_exact_fields(
        authorization, _AUTHORIZATION_FIELDS, context="upstream_authorization"
    )
    expected = {
        "post_video_acceptance": "pending",
        "preview_only": True,
        "training_authorized": False,
        "training_use_forbidden": True,
        "user_requested_experimental_training_ack_required": True,
    }
    if authorization != expected:
        raise FewShotEpisodeIOError("upstream authorization state differs")
    return authorization


def _validate_config_rows(value: Any) -> tuple[dict[str, Any], ...]:
    rows = _required_list(value, context="config rows")
    if len(rows) != EXPECTED_TOTAL_ROWS:
        raise FewShotEpisodeIOError("config must contain exactly three rows")
    parsed: list[dict[str, Any]] = []
    for position, raw in enumerate(rows, 1):
        row = _required_mapping(raw, context=f"config row {position}")
        role = row.get("role")
        expected_fields = _CONFIG_COMMON_ROW_FIELDS | (
            frozenset({"support_index"}) if role == "support" else frozenset()
        )
        _require_exact_fields(row, expected_fields, context=f"config row {position}")
        if role not in {"support", "heldout"}:
            raise FewShotEpisodeIOError(f"unknown config role at row {position}")
        _required_iid(row.get("iid"), context=f"config IID at row {position}")
        _required_sha256(row.get("group_id"), context=f"group_id at row {position}")
        for field in (
            "source_video_sha256",
            "target_video_sha256",
            "edit_instruction_sha256",
            "vae_parquet_sha256",
        ):
            _required_sha256(row.get(field), context=f"{field} at row {position}")
        _required_text(
            row.get("source_action_signature"),
            context=f"source_action_signature at row {position}",
        )
        if role == "support" and (
            type(row.get("support_index")) is not int
            or row["support_index"] not in {1, 2}
        ):
            raise FewShotEpisodeIOError("support_index must be exactly 1 or 2")
        parsed.append(row)

    supports = sorted(
        (row for row in parsed if row["role"] == "support"),
        key=lambda row: int(row["support_index"]),
    )
    heldout = [row for row in parsed if row["role"] == "heldout"]
    if (
        len(supports) != EXPECTED_SUPPORT_COUNT
        or [row["support_index"] for row in supports] != [1, 2]
        or len(heldout) != 1
    ):
        raise FewShotEpisodeIOError(
            "roles must be support[1], support[2], and one heldout"
        )
    ordered = (*supports, heldout[0])
    for identity_name in ("iid", "source_video_sha256", "group_id"):
        identities = [str(row[identity_name]) for row in ordered]
        if len(set(identities)) != EXPECTED_TOTAL_ROWS:
            raise FewShotEpisodeIOError(
                f"all three {identity_name} identities must be disjoint"
            )
    return ordered


def _validate_config(config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    _require_exact_fields(config, _CONFIG_FIELDS, context="EPMC config")
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise FewShotEpisodeIOError("EPMC config schema differs")
    if (
        config.get("purpose") != "experimental_engineering_canary_only"
        or config.get("production_claim_forbidden") is not True
        or config.get("scientific_claim_authorized") is not False
    ):
        raise FewShotEpisodeIOError("EPMC config claim state differs")
    _validate_authorization(config.get("upstream_authorization"))

    inference = _required_mapping(
        config.get("inference_contract"), context="inference_contract"
    )
    _require_exact_fields(inference, _INFERENCE_FIELDS, context="inference_contract")
    if inference != {
        "external_inputs": ["source_video", "edit_instruction"],
        "target_available": False,
        "support_available": False,
        "external_mask_flow_pose_track_trajectory": False,
    }:
        raise FewShotEpisodeIOError("inference contract differs")

    binding = _required_mapping(
        config.get("dataset_binding"), context="dataset_binding"
    )
    _require_exact_fields(binding, _BINDING_FIELDS, context="dataset_binding")
    for field in _BINDING_FIELDS:
        _required_sha256(binding.get(field), context=f"dataset binding {field}")
    _validate_micro_program(config.get("micro_program"))
    if type(config.get("split_seed")) is not int or config["split_seed"] != EXPECTED_SPLIT_SEED:
        raise FewShotEpisodeIOError("split_seed differs")

    review = _required_mapping(
        config.get("manual_contact_sheet_review"),
        context="manual_contact_sheet_review",
    )
    _require_exact_fields(review, _MANUAL_REVIEW_FIELDS, context="manual review")
    if type(review.get("date")) is not str or _DATE_RE.fullmatch(review["date"]) is None:
        raise FewShotEpisodeIOError("manual review date differs")
    if review.get("frames") != [0, 20, 40, 60, 80]:
        raise FewShotEpisodeIOError("manual review frames differ")
    if review.get("verdict") != "eligible_for_representation_canary_not_dataset_acceptance":
        raise FewShotEpisodeIOError("manual review verdict differs")
    observations = review.get("observations")
    if not isinstance(observations, list) or not observations or any(
        type(item) is not str or not item.strip() for item in observations
    ):
        raise FewShotEpisodeIOError("manual review observations differ")
    return _validate_config_rows(config.get("rows"))


def _validate_preview_row(
    row_value: Mapping[str, Any], config_row: Mapping[str, Any]
) -> dict[str, Any]:
    iid = str(config_row["iid"])
    row = dict(row_value)
    _require_exact_fields(row, _PREVIEW_FIELDS, context=f"preview row {iid}")
    if row.get("schema_version") != PREVIEW_ROW_SCHEMA:
        raise FewShotEpisodeIOError(f"preview row schema differs: {iid}")
    if (
        row.get("preview_only") is not True
        or row.get("training_authorized") is not False
        or row.get("training_use_forbidden") is not True
        or row.get("production_eligible") is not False
        or row.get("post_video_acceptance") != "pending"
    ):
        raise FewShotEpisodeIOError(f"preview safety state differs: {iid}")
    digest = _required_sha256(
        row.get("row_digest"), context=f"preview row digest for {iid}"
    )
    unsigned = dict(row)
    unsigned.pop("row_digest")
    if object_sha256(unsigned) != digest:
        raise FewShotEpisodeIOError(f"preview row digest differs: {iid}")

    for field in (
        "iid",
        "group_id",
        "source_video_sha256",
        "target_video_sha256",
        "edit_instruction_sha256",
    ):
        if row.get(field) != config_row.get(field):
            raise FewShotEpisodeIOError(f"preview/config {field} differs: {iid}")
    instruction = _required_text(
        row.get("edit_instruction"), context=f"edit instruction for {iid}"
    )
    if hashlib.sha256(instruction.encode("utf-8")).hexdigest() != row[
        "edit_instruction_sha256"
    ]:
        raise FewShotEpisodeIOError(f"edit instruction hash differs: {iid}")
    generation = _required_text(
        row.get("generation_instruction"),
        context=f"generation instruction for {iid}",
    )
    if hashlib.sha256(generation.encode("utf-8")).hexdigest() != _required_sha256(
        row.get("generation_instruction_sha256"),
        context=f"generation instruction hash for {iid}",
    ):
        raise FewShotEpisodeIOError(f"generation instruction hash differs: {iid}")

    gates = _required_mapping(
        row.get("selection_gates"), context=f"selection gates for {iid}"
    )
    _require_exact_fields(gates, _SELECTION_GATE_FIELDS, context=f"gates for {iid}")
    if any(value is not True for value in gates.values()):
        raise FewShotEpisodeIOError(f"selection gate is not true: {iid}")

    census = _required_mapping(
        row.get("source_census"), context=f"source census for {iid}"
    )
    plan = _required_mapping(row.get("target_plan"), context=f"target plan for {iid}")
    if census.get("iid") != iid or plan.get("iid") != iid:
        raise FewShotEpisodeIOError(f"nested preview IID differs: {iid}")
    subjects = _required_list(
        census.get("dynamic_subjects"), context=f"dynamic subjects for {iid}"
    )
    targets = _required_list(
        plan.get("dynamic_subject_targets"), context=f"target subjects for {iid}"
    )
    if (
        len(subjects) != 1
        or len(targets) != 1
        or not isinstance(subjects[0], Mapping)
        or not isinstance(targets[0], Mapping)
    ):
        raise FewShotEpisodeIOError(f"preview must contain one matched actor: {iid}")
    subject = dict(subjects[0])
    target = dict(targets[0])
    if (
        subject.get("dynamic") is not True
        or subject.get("entity_type") != EXPECTED_ENTITY_TYPE
        or subject.get("source_action_signature")
        != config_row.get("source_action_signature")
        or target.get("subject_id") != subject.get("subject_id")
        or target.get("substantive_change") is not True
        or target.get("target_action_signature") != EXPECTED_TARGET_ACTION
    ):
        raise FewShotEpisodeIOError(f"preview actor/action identity differs: {iid}")
    camera = _required_mapping(census.get("camera"), context=f"source camera for {iid}")
    camera_target = _required_mapping(
        plan.get("camera_target"), context=f"target camera for {iid}"
    )
    if (
        camera.get("motion_class") != "locked_off"
        or camera_target.get("motion_class") != "locked_off"
        or camera_target.get("relation") != "preserve_static"
        or census.get("confidence") != "high"
        or plan.get("confidence") != "high"
    ):
        raise FewShotEpisodeIOError(f"preview camera/confidence differs: {iid}")
    _required_text(row.get("source_video_path"), context=f"source path for {iid}")
    _required_text(row.get("target_video_path"), context=f"target path for {iid}")
    return row


def _validate_vae_row(
    row_value: Mapping[str, Any], config_row: Mapping[str, Any]
) -> dict[str, Any]:
    iid = str(config_row["iid"])
    row = dict(row_value)
    _require_exact_fields(row, _VAE_INDEX_FIELDS, context=f"VAE index row {iid}")
    if row.get("schema_version") != VAE_INDEX_ROW_SCHEMA:
        raise FewShotEpisodeIOError(f"VAE index row schema differs: {iid}")
    if (
        row.get("preview_only") is not True
        or row.get("production_claim_forbidden") is not True
    ):
        raise FewShotEpisodeIOError(f"VAE index safety state differs: {iid}")
    if row.get("iid") != iid:
        raise FewShotEpisodeIOError(f"VAE/config IID differs: {iid}")
    if row.get("parquet_sha256") != config_row.get("vae_parquet_sha256"):
        raise FewShotEpisodeIOError(f"VAE/config parquet hash differs: {iid}")
    if tuple(row.get("bucket_hw", ())) != EXPECTED_BUCKET_HW:
        raise FewShotEpisodeIOError(f"VAE bucket differs: {iid}")
    if tuple(row.get("posterior_parameters_shape", ())) != EXPECTED_POSTERIOR_SHAPE:
        raise FewShotEpisodeIOError(f"VAE posterior shape differs: {iid}")
    for field in (
        "parquet_sha256",
        "materialized_row_digest",
        "sample_receipt_sha256",
    ):
        _required_sha256(row.get(field), context=f"VAE {field} for {iid}")
    _required_text(row.get("parquet_path"), context=f"VAE parquet path for {iid}")
    _required_text(
        row.get("sample_receipt_path"), context=f"VAE receipt path for {iid}"
    )
    return row


@dataclass(frozen=True)
class AuditedEpisodeRow:
    role: str
    support_index: int | None
    iid: str
    group_id: str
    source_video_sha256: str
    target_video_sha256: str
    edit_instruction: str
    edit_instruction_sha256: str
    source_action_signature: str
    target_action_signature: str
    entity_type: str
    preview_row_digest: str
    vae_index_row_digest: str
    source_video: BoundArtifact
    target_video: BoundArtifact
    source_video_metadata: VideoMetadata
    target_video_metadata: VideoMetadata
    vae_parquet: BoundArtifact
    vae_sample_receipt: BoundArtifact
    bucket_hw: tuple[int, int]
    posterior_parameters_shape: tuple[int, int, int, int, int]

    def receipt(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "support_index": self.support_index,
            "iid": self.iid,
            "group_id": self.group_id,
            "source_video_sha256": self.source_video_sha256,
            "target_video_sha256": self.target_video_sha256,
            "edit_instruction_sha256": self.edit_instruction_sha256,
            "source_action_signature": self.source_action_signature,
            "target_action_signature": self.target_action_signature,
            "entity_type": self.entity_type,
            "preview_row_digest": self.preview_row_digest,
            "vae_index_row_digest": self.vae_index_row_digest,
            "source_video": self.source_video.as_dict(),
            "target_video": self.target_video.as_dict(),
            "source_video_frames": self.source_video_metadata.frame_count,
            "target_video_frames": self.target_video_metadata.frame_count,
            "source_video_fps": str(self.source_video_metadata.fps),
            "target_video_fps": str(self.target_video_metadata.fps),
            "vae_parquet": self.vae_parquet.as_dict(),
            "vae_sample_receipt": self.vae_sample_receipt.as_dict(),
            "bucket_hw": list(self.bucket_hw),
            "posterior_parameters_shape": list(self.posterior_parameters_shape),
        }


@dataclass(frozen=True)
class AuditedFewShotEpisode:
    config: BoundArtifact
    preview_manifest: BoundArtifact
    vae_index: BoundArtifact
    preview_manifest_row_count: int
    vae_index_row_count: int
    supports: tuple[AuditedEpisodeRow, AuditedEpisodeRow]
    heldout: AuditedEpisodeRow
    experimental_training_acknowledged: bool

    @property
    def rows(self) -> tuple[AuditedEpisodeRow, AuditedEpisodeRow, AuditedEpisodeRow]:
        return (*self.supports, self.heldout)

    def audit_receipt(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": AUDIT_SCHEMA,
            "complete": True,
            "experimental_engineering_canary_only": True,
            "experimental_training_acknowledged": self.experimental_training_acknowledged,
            "preview_only": True,
            "training_authorized": False,
            "training_use_forbidden": True,
            "production_claim_forbidden": True,
            "scientific_claim_authorized": False,
            "post_video_acceptance": "pending",
            "config": self.config.as_dict(),
            "preview_manifest": self.preview_manifest.as_dict(),
            "preview_manifest_row_count": self.preview_manifest_row_count,
            "vae_index": self.vae_index.as_dict(),
            "vae_index_row_count": self.vae_index_row_count,
            "role_order": ["support", "support", "heldout"],
            "exact_rgb_frames": EXPECTED_RGB_FRAMES,
            "exact_fps": EXPECTED_FPS,
            "latent_phases": EXPECTED_LATENT_PHASES,
            "iid_disjoint": True,
            "source_video_sha256_disjoint": True,
            "group_id_disjoint": True,
            "selected_artifact_hashes_verified": True,
            "selected_media_probed": True,
            "rows": [row.receipt() for row in self.rows],
        }
        payload["audit_digest"] = object_sha256(payload)
        return payload


def _build_audited_row(
    config_row: Mapping[str, Any],
    preview_row: Mapping[str, Any],
    vae_row: Mapping[str, Any],
) -> AuditedEpisodeRow:
    iid = str(config_row["iid"])
    source_video, source_metadata = _audit_video(
        preview_row["source_video_path"],
        expected_sha256=str(config_row["source_video_sha256"]),
        context=f"source video for {iid}",
    )
    target_video, target_metadata = _audit_video(
        preview_row["target_video_path"],
        expected_sha256=str(config_row["target_video_sha256"]),
        context=f"target video for {iid}",
    )
    parquet = _hash_stable_file(
        vae_row["parquet_path"],
        expected_sha256=str(config_row["vae_parquet_sha256"]),
        context=f"VAE parquet for {iid}",
    )
    receipt = _hash_stable_file(
        vae_row["sample_receipt_path"],
        expected_sha256=str(vae_row["sample_receipt_sha256"]),
        context=f"VAE sample receipt for {iid}",
    )
    support_index = (
        int(config_row["support_index"])
        if config_row["role"] == "support"
        else None
    )
    return AuditedEpisodeRow(
        role=str(config_row["role"]),
        support_index=support_index,
        iid=iid,
        group_id=str(config_row["group_id"]),
        source_video_sha256=str(config_row["source_video_sha256"]),
        target_video_sha256=str(config_row["target_video_sha256"]),
        edit_instruction=str(preview_row["edit_instruction"]),
        edit_instruction_sha256=str(config_row["edit_instruction_sha256"]),
        source_action_signature=str(config_row["source_action_signature"]),
        target_action_signature=EXPECTED_TARGET_ACTION,
        entity_type=EXPECTED_ENTITY_TYPE,
        preview_row_digest=str(preview_row["row_digest"]),
        vae_index_row_digest=object_sha256(vae_row),
        source_video=source_video,
        target_video=target_video,
        source_video_metadata=source_metadata,
        target_video_metadata=target_metadata,
        vae_parquet=parquet,
        vae_sample_receipt=receipt,
        bucket_hw=EXPECTED_BUCKET_HW,
        posterior_parameters_shape=EXPECTED_POSTERIOR_SHAPE,
    )


def load_epmc_k2_canary(
    config_path: str | Path,
    preview_manifest_path: str | Path,
    vae_index_path: str | Path,
    *,
    experimental_training_acknowledged: bool = False,
    expected_config_sha256: str = REFERENCE_CONFIG_SHA256,
) -> AuditedFewShotEpisode:
    """Load and audit the exact two-support/one-heldout EPMC canary.

    The acknowledgement must be the literal boolean ``True``.  It records the
    user's authorization for a non-production engineering canary; it never
    changes the upstream preview-only/training-forbidden state.
    """

    config_artifact, config = _pinned_json(
        config_path,
        expected_sha256=expected_config_sha256,
        context="EPMC K=2 config",
    )
    config_rows = _validate_config(config)
    if experimental_training_acknowledged is not True:
        raise FewShotEpisodeIOError(
            "literal experimental_training_acknowledged=True is required"
        )

    binding = config["dataset_binding"]
    preview_artifact, preview_rows = _pinned_jsonl(
        preview_manifest_path,
        expected_sha256=binding["preview_manifest_sha256"],
        context="preview manifest",
    )
    vae_artifact, vae_rows = _pinned_jsonl(
        vae_index_path,
        expected_sha256=binding["vae_index_sha256"],
        context="VAE index",
    )
    if len({config_artifact.path, preview_artifact.path, vae_artifact.path}) != 3:
        raise FewShotEpisodeIOError("config, preview manifest, and VAE index must differ")
    preview_by_iid = _index_rows_by_iid(preview_rows, context="preview manifest")
    vae_by_iid = _index_rows_by_iid(vae_rows, context="VAE index")

    audited_rows: list[AuditedEpisodeRow] = []
    artifact_paths: set[Path] = {
        config_artifact.path,
        preview_artifact.path,
        vae_artifact.path,
    }
    for config_row in config_rows:
        iid = str(config_row["iid"])
        if iid not in preview_by_iid or iid not in vae_by_iid:
            raise FewShotEpisodeIOError(
                f"exact IID join is incomplete for configured row: {iid}"
            )
        preview_row = _validate_preview_row(preview_by_iid[iid], config_row)
        vae_row = _validate_vae_row(vae_by_iid[iid], config_row)
        audited = _build_audited_row(config_row, preview_row, vae_row)
        selected_paths = {
            audited.source_video.path,
            audited.target_video.path,
            audited.vae_parquet.path,
            audited.vae_sample_receipt.path,
        }
        if len(selected_paths) != 4 or artifact_paths.intersection(selected_paths):
            raise FewShotEpisodeIOError(f"selected artifact path aliases another file: {iid}")
        artifact_paths.update(selected_paths)
        audited_rows.append(audited)

    supports = tuple(row for row in audited_rows if row.role == "support")
    heldout = tuple(row for row in audited_rows if row.role == "heldout")
    if len(supports) != 2 or len(heldout) != 1:
        raise FewShotEpisodeIOError("audited role cardinality changed")
    result = AuditedFewShotEpisode(
        config=config_artifact,
        preview_manifest=preview_artifact,
        vae_index=vae_artifact,
        preview_manifest_row_count=len(preview_rows),
        vae_index_row_count=len(vae_rows),
        supports=(supports[0], supports[1]),
        heldout=heldout[0],
        experimental_training_acknowledged=True,
    )
    # Constructing the receipt here ensures all of its values are canonical
    # JSON before the caller can launch any canary process.
    result.audit_receipt()
    return result


__all__ = [
    "AUDIT_SCHEMA",
    "AuditedEpisodeRow",
    "AuditedFewShotEpisode",
    "BoundArtifact",
    "CONFIG_SCHEMA",
    "EXPECTED_BUCKET_HW",
    "EXPECTED_ENTITY_TYPE",
    "EXPECTED_FPS",
    "EXPECTED_LATENT_PHASES",
    "EXPECTED_PATCH_GRID_THW",
    "EXPECTED_POSTERIOR_SHAPE",
    "EXPECTED_RGB_FRAMES",
    "EXPECTED_SUPPORT_COUNT",
    "EXPECTED_TARGET_ACTION",
    "EXPECTED_TOTAL_ROWS",
    "FewShotEpisodeIOError",
    "PREVIEW_ROW_SCHEMA",
    "REFERENCE_CONFIG_SHA256",
    "VAE_INDEX_ROW_SCHEMA",
    "VideoMetadata",
    "bytes_sha256",
    "canonical_json_bytes",
    "load_epmc_k2_canary",
    "object_sha256",
]
