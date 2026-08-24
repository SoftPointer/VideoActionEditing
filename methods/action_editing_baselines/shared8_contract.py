#!/usr/bin/env python3
"""Closed source-only contract helpers for the action-editing shared-8 run."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable


INPUT_SCHEMA = "action-editing-shared8-input-v1"
RECEIPT_SCHEMA = "action-editing-shared8-model-output-v1"
EXPECTED_ROWS = 8
FRAME_COUNT = 81
FPS = 25.0
INPUT_KEYS = {
    "schema_version",
    "index",
    "iid",
    "split",
    "source_video",
    "instruction",
    "seed",
}
PRIVILEGED_KEY_FRAGMENTS = (
    "target",
    "mask",
    "tube",
    "track",
    "pose",
    "trajectory",
    "reference",
    "shared_i0",
)
_IID_RE = re.compile(r"[0-9a-f]{16}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class Shared8ContractError(RuntimeError):
    """Raised when an input, model invocation, or output fails closed."""


@dataclass(frozen=True)
class InputRow:
    schema_version: str
    index: int
    iid: str
    split: str
    source_video: str
    instruction: str
    seed: int


@dataclass(frozen=True)
class VideoProbe:
    codec: str
    width: int
    height: int
    frame_count: int
    fps: float
    duration_seconds: float


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
        raise Shared8ContractError(f"value is not canonical JSON: {error}") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: str, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise Shared8ContractError(f"{label} must be one lowercase SHA-256 digest")
    return value


def _plain_file(path: Path, *, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise Shared8ContractError(f"cannot resolve {label} {path}: {error}") from error
    if not resolved.is_file():
        raise Shared8ContractError(f"{label} is not a file: {resolved}")
    return resolved


def load_input_manifest(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    require_media: bool = False,
) -> tuple[Path, list[InputRow]]:
    manifest = _plain_file(Path(path), label="input manifest")
    observed_sha256 = file_sha256(manifest)
    if expected_sha256 is not None:
        require_sha256(expected_sha256, label="expected manifest SHA-256")
        if observed_sha256 != expected_sha256:
            raise Shared8ContractError(
                f"input manifest SHA-256 mismatch: {observed_sha256} != {expected_sha256}"
            )

    rows: list[InputRow] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise Shared8ContractError(
                    f"invalid JSON on input-manifest line {line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise Shared8ContractError(f"manifest line {line_number} is not an object")
            keys = set(value)
            if keys != INPUT_KEYS:
                raise Shared8ContractError(
                    f"manifest line {line_number} has non-closed keys: "
                    f"missing={sorted(INPUT_KEYS - keys)} extra={sorted(keys - INPUT_KEYS)}"
                )
            lower_keys = [key.lower() for key in keys]
            if any(fragment in key for key in lower_keys for fragment in PRIVILEGED_KEY_FRAGMENTS):
                raise Shared8ContractError(
                    f"manifest line {line_number} exposes a privileged condition"
                )
            try:
                row = InputRow(**value)
            except TypeError as error:
                raise Shared8ContractError(
                    f"manifest line {line_number} does not match the closed row schema"
                ) from error
            if row.schema_version != INPUT_SCHEMA:
                raise Shared8ContractError(f"unexpected input schema on line {line_number}")
            if type(row.index) is not int or row.index != len(rows):
                raise Shared8ContractError(
                    f"manifest indices must be exact 0..7 order; line {line_number}"
                )
            if not isinstance(row.iid, str) or _IID_RE.fullmatch(row.iid) is None:
                raise Shared8ContractError(f"invalid IID on line {line_number}: {row.iid!r}")
            if row.split not in {"test", "validation"}:
                raise Shared8ContractError(f"invalid split on line {line_number}: {row.split!r}")
            source = Path(row.source_video)
            if not source.is_absolute():
                raise Shared8ContractError(
                    f"source_video must be absolute on line {line_number}: {source}"
                )
            if require_media:
                _plain_file(source, label=f"source video for {row.iid}")
            if (
                not isinstance(row.instruction, str)
                or not row.instruction.strip()
                or "\x00" in row.instruction
            ):
                raise Shared8ContractError(f"invalid instruction on line {line_number}")
            if type(row.seed) is not int or not 0 <= row.seed < 2**63:
                raise Shared8ContractError(f"invalid seed on line {line_number}")
            rows.append(row)

    if len(rows) != EXPECTED_ROWS:
        raise Shared8ContractError(
            f"shared input manifest must have exactly {EXPECTED_ROWS} rows, got {len(rows)}"
        )
    if len({row.iid for row in rows}) != EXPECTED_ROWS:
        raise Shared8ContractError("shared input manifest IIDs are not unique")
    return manifest, rows


def _parse_rate(value: str) -> float:
    try:
        rate = float(Fraction(value))
    except (ValueError, ZeroDivisionError) as error:
        raise Shared8ContractError(f"invalid ffprobe frame rate: {value!r}") from error
    if not math.isfinite(rate) or rate <= 0:
        raise Shared8ContractError(f"non-positive ffprobe frame rate: {value!r}")
    return rate


def probe_video(path: str | Path, *, ffprobe: str = "ffprobe") -> VideoProbe:
    video = _plain_file(Path(path), label="video")
    command = [
        ffprobe,
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,nb_read_frames",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise Shared8ContractError(f"ffprobe failed for {video}: {error}") from error
    try:
        payload = json.loads(completed.stdout)
        streams = payload["streams"]
        stream = streams[0]
        frame_count = int(stream["nb_read_frames"])
        duration = float(payload["format"]["duration"])
        probe = VideoProbe(
            codec=str(stream["codec_name"]),
            width=int(stream["width"]),
            height=int(stream["height"]),
            frame_count=frame_count,
            fps=_parse_rate(str(stream["avg_frame_rate"])),
            duration_seconds=duration,
        )
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise Shared8ContractError(f"malformed ffprobe result for {video}") from error
    if probe.width <= 0 or probe.height <= 0 or not math.isfinite(probe.duration_seconds):
        raise Shared8ContractError(f"invalid video geometry/duration: {video}")
    return probe


def require_81f25(probe: VideoProbe, *, label: str) -> None:
    if probe.frame_count != FRAME_COUNT:
        raise Shared8ContractError(
            f"{label} must contain exactly {FRAME_COUNT} frames, got {probe.frame_count}"
        )
    if abs(probe.fps - FPS) > 1e-3:
        raise Shared8ContractError(f"{label} must report {FPS} FPS, got {probe.fps}")


def source_aspect_bucket(
    *, height: int, width: int, max_pixels: int, stride: int = 16
) -> tuple[int, int]:
    if min(height, width, max_pixels, stride) <= 0:
        raise Shared8ContractError("aspect-bucket inputs must be positive")
    scale = math.sqrt(float(max_pixels) / float(height * width))
    if scale > 1.0:
        scale = 1.0
    bucket_h = max(stride, int(height * scale) // stride * stride)
    bucket_w = max(stride, int(width * scale) // stride * stride)
    if bucket_h * bucket_w > max_pixels:
        raise Shared8ContractError("aspect bucket exceeds its max-pixel contract")
    return bucket_h, bucket_w


def atomic_write_json(path: str | Path, value: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise Shared8ContractError(f"refusing to overwrite receipt: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise Shared8ContractError(f"stale receipt temporary exists: {temporary}")
    payload = dict(value)
    payload["receipt_digest"] = object_sha256(payload)
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def build_output_receipt(
    *,
    model_id: str,
    row: InputRow,
    manifest_path: Path,
    source_probe: VideoProbe,
    output_path: Path,
    output_probe: VideoProbe,
    manifest_sha256: str,
    source_video_sha256: str,
    source_revision: str,
    source_archive_sha256: str,
    model_identity: dict[str, Any],
    sampler: dict[str, Any],
    geometry: dict[str, Any],
    runtime_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    require_sha256(manifest_sha256, label="manifest SHA-256")
    require_sha256(source_video_sha256, label="source-video SHA-256")
    require_sha256(source_archive_sha256, label="source archive SHA-256")
    require_81f25(source_probe, label="source video")
    require_81f25(output_probe, label="output video")
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "model_id": model_id,
        "sample": asdict(row),
        "input_contract": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "accepted_model_conditions": ["source_video", "edit_instruction"],
            "source_video_sha256": source_video_sha256,
            "instruction_utf8_sha256": hashlib.sha256(
                row.instruction.encode("utf-8")
            ).hexdigest(),
            "target_video_argument": False,
            "target_video_accessed": False,
            "external_mask_or_swept_tube": False,
            "external_tracking_pose_or_trajectory": False,
            "reference_media": False,
            "external_shared_i0": False,
        },
        "source_video": asdict(source_probe),
        "output": {
            "path": str(output_path),
            "sha256": file_sha256(output_path),
            **asdict(output_probe),
        },
        "method_source": {
            "revision": source_revision,
            "archive_sha256": source_archive_sha256,
        },
        "model_identity": model_identity,
        "sampler": sampler,
        "geometry": geometry,
    }
    if runtime_evidence is not None:
        receipt["runtime_evidence"] = runtime_evidence
    return receipt


def ensure_empty_directory(path: str | Path) -> Path:
    directory = Path(path).expanduser()
    if not directory.is_absolute():
        raise Shared8ContractError(f"output directory must be absolute: {directory}")
    if directory.exists():
        if not directory.is_dir() or directory.is_symlink():
            raise Shared8ContractError(f"output path is not a plain directory: {directory}")
        if any(directory.iterdir()):
            raise Shared8ContractError(f"refusing non-empty output directory: {directory}")
    else:
        directory.mkdir(parents=True)
    return directory.resolve(strict=True)


def assert_no_privileged_cli(arguments: Iterable[str]) -> None:
    forbidden_flags = {
        "--target",
        "--target-video",
        "--mask",
        "--tube",
        "--track",
        "--pose",
        "--trajectory",
        "--reference",
        "--shared-i0",
    }
    observed = set(arguments)
    overlap = observed & forbidden_flags
    if overlap:
        raise Shared8ContractError(f"model command exposes privileged flags: {sorted(overlap)}")
