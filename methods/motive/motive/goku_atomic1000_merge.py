"""Fail-closed merger for the eight immutable Goku atomic-action epochs.

The four-node launcher publishes one complete, immutable dataset manifest per
allocation.  This module is the only cross-allocation publication step: it
revalidates the eight epoch products, concatenates them in the declared epoch
order, and publishes exactly 1,000 rows.  It deliberately trusts neither an
epoch summary nor paths embedded in a row without replaying the corresponding
hash, media, first-frame, and lineage checks.

Production usage::

    PYTHONPATH=methods/motive python -m motive.goku_atomic1000_merge \
      --epoch /abs/run/epoch_000 128 \
      --epoch /abs/run/epoch_001 128 \
      --epoch /abs/run/epoch_002 128 \
      --epoch /abs/run/epoch_003 128 \
      --epoch /abs/run/epoch_004 128 \
      --epoch /abs/run/epoch_005 128 \
      --epoch /abs/run/epoch_006 128 \
      --epoch /abs/run/epoch_007 104 \
      --output-root /abs/run/exact1000_final \
      --ffprobe /usr/bin/ffprobe

The output directory and all three output files are create-only.  A failed
publication is never silently resumed or replaced; use a new output path after
diagnosing a partial publication.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping, Sequence


EPOCH_MANIFEST_NAME = "atomic1000_dataset_manifest.jsonl"
EPOCH_SUMMARY_NAME = "atomic1000_dataset_summary.json"
OUTPUT_MANIFEST_NAME = EPOCH_MANIFEST_NAME
OUTPUT_SUMMARY_NAME = EPOCH_SUMMARY_NAME
OUTPUT_DONE_NAME = "done.json"

ROW_SCHEMA = "motive-goku-atomic1000-dataset-row-v1"
EPOCH_SUMMARY_SCHEMA = "motive-goku-atomic1000-dataset-summary-v1"
MERGED_SUMMARY_SCHEMA = "motive-goku-atomic1000-merged-summary-v1"
MERGED_DONE_SCHEMA = "motive-goku-atomic1000-merged-done-v1"
PLANNER_PASSED_SCHEMA = "motive-goku-full-motion-qwen-v16-passed-v1"
ATOMIC_RESULT_SCHEMA = "motive-goku-atomic-motion-result-v1"
WAN_RESULT_SCHEMA = "motive-wan22-i2v-sample-v1"
SAMPLE_METADATA_SCHEMA = "motive-goku-atomic-wan-sample-metadata-v1"
FIRST_FRAME_POLICY = "wan22-i2v-strict-preencode-frame0-v1"

PRODUCTION_EPOCH_TARGETS = (128, 128, 128, 128, 128, 128, 128, 104)
PRODUCTION_TOTAL = 1000
FRAME_COUNT = 81
FRAME_RATE = "25/1"

_IID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")

_ROW_KEYS = frozenset(
    {
        "schema_version",
        "iid",
        "lineage",
        "primary_training_label_field",
        "atomic_action_instruction",
        "atomic_action_instruction_sha256",
        "camera_instruction",
        "camera_instruction_sha256",
        "preservation_instruction",
        "preservation_instruction_sha256",
        "full_edit_instruction",
        "full_edit_instruction_sha256",
        "wan_generation_prompt",
        "wan_generation_prompt_sha256",
        "wan_edit_instruction_txt_role",
        "source_video",
        "source_video_sha256",
        "target_video",
        "target_video_sha256",
        "source_temporal_geometry",
        "target_temporal_geometry",
        "strict_target_frame0_float32_npy",
        "strict_target_frame0_float32_npy_sha256",
        "strict_target_frame0_png",
        "strict_target_frame0_png_sha256",
        "strict_source_frame0_anchor_png",
        "strict_source_frame0_anchor_png_sha256",
        "decoded_target_frame0_override_required",
        "target_mp4_decoded_frame0_pixel_equality_claimed",
        "atomic_result",
        "atomic_result_sha256",
        "planner_passed",
        "planner_passed_sha256",
        "wan_result",
        "wan_result_sha256",
        "sample_metadata",
        "sample_metadata_sha256",
    }
)

_EPOCH_SUMMARY_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "minimum_success",
        "total_rows",
        "new_wan_rows",
        "legacy_reused_rows",
        "manifest_sha256",
        "primary_training_label_field",
        "wan_generation_prompt_is_separate",
    }
)

_PLANNER_KEYS = frozenset(
    {
        "schema_version",
        "iid",
        "group_id",
        "family",
        "source_video",
        "resolved_source_video",
        "anchor_image",
        "resolved_anchor_image",
        "source_video_sha256",
        "anchor_sha256",
        "strict_temporal_geometry",
        "edit_instruction",
        "edit_instruction_sha256",
        "source_census",
        "target_plan",
        "compiled_instruction",
        "qwen_record_digest",
        "action_change_substantive",
        "all_dynamic_subjects_covered",
        "camera_covered",
        "human_review_status",
        "generation_authorized",
        "production_eligible",
    }
)

_ATOMIC_RESULT_KEYS = frozenset(
    {
        "schema_version",
        "iid",
        "original_candidate_index",
        "status",
        "input_row_digest",
        "source_passed_path",
        "source_passed_sha256",
        "source_frame_grid_generation_prompt",
        "source_frame_grid_generation_prompt_sha256",
        "target_plan_sha256",
        "backend",
        "plan_audit_attempts",
        "plan_audit",
        "rewrite_attempts",
        "rewrite",
        "semantic_audit",
        "atomic_action_instruction",
        "atomic_action_instruction_sha256",
        "camera_instruction",
        "camera_instruction_sha256",
        "preservation_instruction",
        "preservation_instruction_sha256",
        "full_edit_instruction",
        "full_edit_instruction_sha256",
        "error",
        "record_digest",
    }
)

_METADATA_KEYS = frozenset(
    {
        "schema_version",
        "iid",
        "primary_training_label_field",
        "wan_generation_prompt_field",
        "wan_generation_prompt_is_training_label",
        "edit_instruction_txt_role",
        "wan_generation_prompt_txt_role",
        "atomic_action_instruction",
        "atomic_action_instruction_sha256",
        "camera_instruction",
        "camera_instruction_sha256",
        "preservation_instruction",
        "preservation_instruction_sha256",
        "full_edit_instruction",
        "full_edit_instruction_sha256",
        "wan_generation_prompt",
        "wan_generation_prompt_sha256",
        "source_video_sha256",
        "artifacts",
        "metadata_digest",
    }
)

_ROW_ARTIFACTS = (
    ("source_video", "source_video_sha256"),
    ("target_video", "target_video_sha256"),
    (
        "strict_target_frame0_float32_npy",
        "strict_target_frame0_float32_npy_sha256",
    ),
    ("strict_target_frame0_png", "strict_target_frame0_png_sha256"),
    (
        "strict_source_frame0_anchor_png",
        "strict_source_frame0_anchor_png_sha256",
    ),
    ("atomic_result", "atomic_result_sha256"),
    ("planner_passed", "planner_passed_sha256"),
    ("wan_result", "wan_result_sha256"),
    ("sample_metadata", "sample_metadata_sha256"),
)


class Atomic1000MergeError(RuntimeError):
    """An epoch, artifact, lineage binding, or output contract differs."""


@dataclass(frozen=True)
class EpochSpec:
    run_root: Path
    expected_rows: int


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _pretty(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_text(text: str) -> str:
    return _sha_bytes(text.encode("utf-8"))


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _load_json_bytes(raw: bytes, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise Atomic1000MergeError(f"invalid JSON in {context}: {error}") from error
    if not isinstance(value, dict):
        raise Atomic1000MergeError(f"{context} is not one JSON object")
    return value


def _plain_file(path: Path, *, context: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise Atomic1000MergeError(
            f"{context} must be an absolute, existing plain file: {path}"
        )
    return path


def _plain_dir(path: Path, *, context: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise Atomic1000MergeError(
            f"{context} must be an absolute, existing plain directory: {path}"
        )
    return path


def _sha_field(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise Atomic1000MergeError(f"{context} is not a lowercase SHA-256 digest")
    return value


def _text(value: Any, *, context: str, maximum: int = 100_000) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or len(value) > maximum
    ):
        raise Atomic1000MergeError(f"{context} is not bounded non-empty text")
    return value


def _object_digest(value: Mapping[str, Any], *, omit: str) -> str:
    return _sha_bytes(_canonical({key: item for key, item in value.items() if key != omit}))


def _sha_file(path: Path, cache: dict[Path, str]) -> str:
    observed = cache.get(path)
    if observed is not None:
        return observed
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    observed = digest.hexdigest()
    cache[path] = observed
    return observed


def _bound_artifact(
    row: Mapping[str, Any],
    path_field: str,
    sha_field: str,
    *,
    iid: str,
    cache: dict[Path, str],
) -> Path:
    raw_path = row.get(path_field)
    if not isinstance(raw_path, str) or "\n" in raw_path:
        raise Atomic1000MergeError(f"row {iid} {path_field} path is invalid")
    path = _plain_file(Path(raw_path), context=f"row {iid} {path_field}")
    expected = _sha_field(row.get(sha_field), context=f"row {iid} {sha_field}")
    if _sha_file(path, cache) != expected:
        raise Atomic1000MergeError(f"row {iid} {path_field} SHA-256 differs")
    return path


def _single_jsonl(path: Path, *, context: str) -> dict[str, Any]:
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    if len(lines) != 1 or not lines[0].endswith(b"\n") or not lines[0].strip():
        raise Atomic1000MergeError(f"{context} must contain exactly one JSONL row")
    return _load_json_bytes(lines[0][:-1], context=context)


def _probe_video(path: Path, *, ffprobe: Path) -> dict[str, Any]:
    try:
        process = subprocess.run(
            [
                str(ffprobe),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=avg_frame_rate,r_frame_rate,nb_read_frames",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise Atomic1000MergeError(f"ffprobe failed for {path}: {error}") from error
    value = _load_json_bytes(process.stdout.encode("utf-8"), context=f"ffprobe {path}")
    streams = value.get("streams")
    if not isinstance(streams, list) or len(streams) != 1 or not isinstance(streams[0], dict):
        raise Atomic1000MergeError(f"ffprobe stream closure differs: {path}")
    stream = streams[0]
    try:
        frames = int(stream["nb_read_frames"])
    except (KeyError, TypeError, ValueError) as error:
        raise Atomic1000MergeError(f"ffprobe frame count differs: {path}") from error
    if (
        frames != FRAME_COUNT
        or stream.get("avg_frame_rate") != FRAME_RATE
        or stream.get("r_frame_rate") != FRAME_RATE
    ):
        raise Atomic1000MergeError(f"video is not exactly 81 frames at 25 fps: {path}")
    return {"frame_count": FRAME_COUNT, "frame_rate": FRAME_RATE}


def _verify_strict_sidecars(
    *,
    source_video: Path,
    source_anchor: Path,
    conditioning_npy: Path,
    conditioning_png: Path,
    iid: str,
) -> dict[str, str]:
    """Replay the exact-I0 sidecar checks used by the epoch finalizer.

    Imports are intentionally lazy: the merger's JSON/schema helpers remain
    usable in lightweight environments, while production verification fails
    closed unless OpenCV, Pillow, and NumPy are present.
    """

    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError as error:  # pragma: no cover - production dependency guard
        raise Atomic1000MergeError(
            "strict I0 verification requires OpenCV, Pillow, and NumPy"
        ) from error

    def png_pixels(path: Path) -> Any:
        try:
            with Image.open(path) as image:
                if image.format != "PNG":
                    raise Atomic1000MergeError(
                        f"strict sidecar is not PNG iid={iid}: {path}"
                    )
                return np.asarray(image.convert("RGB"), dtype=np.uint8)
        except Atomic1000MergeError:
            raise
        except Exception as error:
            raise Atomic1000MergeError(
                f"could not decode strict PNG iid={iid}: {path}"
            ) from error

    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise Atomic1000MergeError(f"could not open source video for I0 iid={iid}")
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None or frame.size == 0:
        raise Atomic1000MergeError(f"could not decode source I0 iid={iid}")
    source_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    anchor_rgb = png_pixels(source_anchor)
    conditioning_rgb = png_pixels(conditioning_png)
    if not np.array_equal(anchor_rgb, source_rgb):
        raise Atomic1000MergeError(f"source anchor is not decoded source I0 iid={iid}")
    try:
        array = np.load(conditioning_npy, allow_pickle=False)
    except Exception as error:
        raise Atomic1000MergeError(f"could not load strict NPY iid={iid}") from error
    if (
        array.dtype != np.dtype("<f4")
        or array.ndim != 3
        or array.shape[0] != 3
        or array.shape[1] <= 0
        or array.shape[2] <= 0
        or not np.isfinite(array).all()
        or float(array.min()) < -1.0001
        or float(array.max()) > 1.0001
    ):
        raise Atomic1000MergeError(f"strict float32 NPY contract differs iid={iid}")
    projected = (
        np.rint((array.astype(np.float32) + 1.0) * 127.5)
        .clip(0, 255)
        .astype(np.uint8)
        .transpose(1, 2, 0)
    )
    if not np.array_equal(projected, conditioning_rgb):
        raise Atomic1000MergeError(f"strict NPY projection differs from PNG iid={iid}")
    return {
        "source_anchor_rgb_sha256": _sha_bytes(anchor_rgb.tobytes()),
        "conditioning_rgb_sha256": _sha_bytes(conditioning_rgb.tobytes()),
    }


def _validate_temporal_geometry(value: Any, *, context: str) -> None:
    if value != {"frame_count": FRAME_COUNT, "frame_rate": FRAME_RATE}:
        raise Atomic1000MergeError(f"{context} temporal geometry differs")


def _validate_row_shape(row: Mapping[str, Any], *, epoch: int, index: int) -> str:
    if set(row) != _ROW_KEYS:
        delta = sorted(set(row) ^ _ROW_KEYS)
        raise Atomic1000MergeError(
            f"epoch {epoch} row {index} dataset schema is open: {delta}"
        )
    iid = row.get("iid")
    if not isinstance(iid, str) or _IID_RE.fullmatch(iid) is None:
        raise Atomic1000MergeError(f"epoch {epoch} row {index} IID is invalid")
    if (
        row.get("schema_version") != ROW_SCHEMA
        or row.get("lineage") != "atomic_new_wan"
        or row.get("primary_training_label_field")
        != "atomic_action_instruction"
        or row.get("wan_edit_instruction_txt_role")
        != "generation_prompt_not_primary_training_label"
        or row.get("decoded_target_frame0_override_required") is not True
        or row.get("target_mp4_decoded_frame0_pixel_equality_claimed") is not False
    ):
        raise Atomic1000MergeError(f"dataset row identity/role policy differs iid={iid}")
    fields = (
        "atomic_action_instruction",
        "camera_instruction",
        "preservation_instruction",
        "full_edit_instruction",
        "wan_generation_prompt",
    )
    texts: dict[str, str] = {}
    for field in fields:
        text = _text(row.get(field), context=f"row {iid} {field}")
        if _sha_text(text) != _sha_field(
            row.get(field + "_sha256"), context=f"row {iid} {field}_sha256"
        ):
            raise Atomic1000MergeError(f"row {iid} {field} digest differs")
        texts[field] = text
    expected_full = (
        f"{texts['atomic_action_instruction']} {texts['camera_instruction']} "
        f"{texts['preservation_instruction']}"
    )
    if texts["full_edit_instruction"] != expected_full:
        raise Atomic1000MergeError(f"row {iid} full instruction composition differs")
    _validate_temporal_geometry(
        row.get("source_temporal_geometry"), context=f"row {iid} source"
    )
    _validate_temporal_geometry(
        row.get("target_temporal_geometry"), context=f"row {iid} target"
    )
    return iid


def _validate_planner(
    planner: Mapping[str, Any],
    *,
    row: Mapping[str, Any],
    planner_path: Path,
    cache: dict[Path, str],
) -> str:
    iid = str(row["iid"])
    if set(planner) != _PLANNER_KEYS or planner.get("schema_version") != PLANNER_PASSED_SCHEMA:
        raise Atomic1000MergeError(f"planner passed schema differs iid={iid}")
    group_id = _text(planner.get("group_id"), context=f"planner group iid={iid}", maximum=2000)
    _text(planner.get("family"), context=f"planner family iid={iid}", maximum=2000)
    if (
        planner.get("iid") != iid
        or planner.get("edit_instruction") != row["wan_generation_prompt"]
        or planner.get("edit_instruction_sha256")
        != row["wan_generation_prompt_sha256"]
        or planner.get("source_video_sha256") != row["source_video_sha256"]
        or planner.get("action_change_substantive") is not True
        or planner.get("all_dynamic_subjects_covered") is not True
        or planner.get("camera_covered") is not True
        or planner.get("human_review_status") != "pending"
        or planner.get("generation_authorized") is not False
        or planner.get("production_eligible") is not False
    ):
        raise Atomic1000MergeError(f"planner row binding/policy differs iid={iid}")
    _sha_field(planner.get("qwen_record_digest"), context=f"planner record digest iid={iid}")
    geometry = planner.get("strict_temporal_geometry")
    if (
        not isinstance(geometry, dict)
        or set(geometry) != {
            "frame_count",
            "fps",
            "timeline_span_seconds",
            "width",
            "height",
        }
        or geometry.get("frame_count") != FRAME_COUNT
        or geometry.get("fps") != FRAME_RATE
        or isinstance(geometry.get("timeline_span_seconds"), bool)
        or not isinstance(geometry.get("timeline_span_seconds"), (int, float))
        or not math.isclose(float(geometry["timeline_span_seconds"]), 3.2, abs_tol=1e-9)
        or type(geometry.get("width")) is not int
        or type(geometry.get("height")) is not int
        or geometry["width"] <= 0
        or geometry["height"] <= 0
    ):
        raise Atomic1000MergeError(f"planner strict temporal geometry differs iid={iid}")
    source_path = planner.get("resolved_source_video")
    anchor_path = planner.get("resolved_anchor_image")
    if not isinstance(source_path, str) or not isinstance(anchor_path, str):
        raise Atomic1000MergeError(f"planner media paths differ iid={iid}")
    original_source = _plain_file(
        Path(source_path), context=f"planner resolved source iid={iid}"
    )
    original_anchor = _plain_file(
        Path(anchor_path), context=f"planner resolved anchor iid={iid}"
    )
    source_sha = _sha_field(
        planner.get("source_video_sha256"), context=f"planner source SHA iid={iid}"
    )
    anchor_sha = _sha_field(
        planner.get("anchor_sha256"), context=f"planner anchor SHA iid={iid}"
    )
    if _sha_file(original_source, cache) != source_sha:
        raise Atomic1000MergeError(f"planner original source digest differs iid={iid}")
    if _sha_file(original_anchor, cache) != anchor_sha:
        raise Atomic1000MergeError(f"planner original anchor digest differs iid={iid}")
    if anchor_sha != row["strict_source_frame0_anchor_png_sha256"]:
        raise Atomic1000MergeError(f"planner/source-I0 anchor digest differs iid={iid}")
    # The dataset row hash-binds this exact planner byte stream.  This is the
    # fail-closed group lineage used for cross-epoch uniqueness.
    if _sha_file(planner_path, cache) != row["planner_passed_sha256"]:
        raise Atomic1000MergeError(f"planner artifact digest differs iid={iid}")
    return group_id


def _validate_atomic_result(
    atomic: Mapping[str, Any], *, row: Mapping[str, Any], planner_path: Path
) -> None:
    iid = str(row["iid"])
    if set(atomic) != _ATOMIC_RESULT_KEYS or atomic.get("schema_version") != ATOMIC_RESULT_SCHEMA:
        raise Atomic1000MergeError(f"atomic result schema differs iid={iid}")
    if (
        atomic.get("iid") != iid
        or atomic.get("status") != "ok"
        or atomic.get("error") is not None
        or not isinstance(atomic.get("rewrite"), dict)
        or atomic.get("source_passed_path") != str(planner_path)
        or atomic.get("source_passed_sha256") != row["planner_passed_sha256"]
        or atomic.get("source_frame_grid_generation_prompt")
        != row["wan_generation_prompt"]
        or atomic.get("source_frame_grid_generation_prompt_sha256")
        != row["wan_generation_prompt_sha256"]
    ):
        raise Atomic1000MergeError(f"atomic result identity/lineage differs iid={iid}")
    for field in (
        "atomic_action_instruction",
        "camera_instruction",
        "preservation_instruction",
        "full_edit_instruction",
    ):
        if (
            atomic.get(field) != row[field]
            or atomic.get(field + "_sha256") != row[field + "_sha256"]
        ):
            raise Atomic1000MergeError(f"atomic result {field} differs iid={iid}")
    digest = _sha_field(atomic.get("record_digest"), context=f"atomic record digest iid={iid}")
    if digest != _object_digest(atomic, omit="record_digest"):
        raise Atomic1000MergeError(f"atomic record digest differs iid={iid}")


def _validate_wan_result(
    wan: Mapping[str, Any],
    *,
    row: Mapping[str, Any],
    sidecar_pixels: Mapping[str, str],
) -> None:
    iid = str(row["iid"])
    policy = wan.get("first_frame_policy")
    outputs = wan.get("outputs")
    if (
        wan.get("schema_version") != WAN_RESULT_SCHEMA
        or wan.get("iid") != iid
        or not isinstance(policy, dict)
        or not isinstance(outputs, dict)
        or policy.get("policy_version") != FIRST_FRAME_POLICY
        or policy.get("tensor_frame0_overridden_before_encoding") is not True
        or policy.get("conditioning_tensor_dtype") != "float32"
        or policy.get("preencode_frame0_matches_png_pixels") is not True
        or policy.get("mp4_codec_is_lossy") is not True
        or policy.get("mp4_decode_pixel_equality_claimed") is not False
        or policy.get("preencode_frame0_pixel_sha256")
        != sidecar_pixels["conditioning_rgb_sha256"]
        or policy.get("lossless_png_pixel_sha256")
        != sidecar_pixels["conditioning_rgb_sha256"]
    ):
        raise Atomic1000MergeError(f"Wan strict frame-zero policy differs iid={iid}")
    parent = Path(str(row["wan_result"])).parent
    bindings = (
        (
            "preview_mp4",
            "preview_mp4_sha256",
            Path(str(row["target_video"])),
            row["target_video_sha256"],
        ),
        (
            "conditioning_anchor_original",
            "conditioning_anchor_original_sha256",
            Path(str(row["strict_source_frame0_anchor_png"])),
            row["strict_source_frame0_anchor_png_sha256"],
        ),
        (
            "conditioning_frame0_float32",
            "conditioning_frame0_float32_sha256",
            Path(str(row["strict_target_frame0_float32_npy"])),
            row["strict_target_frame0_float32_npy_sha256"],
        ),
        (
            "conditioning_frame0_png",
            "conditioning_frame0_png_sha256",
            Path(str(row["strict_target_frame0_png"])),
            row["strict_target_frame0_png_sha256"],
        ),
    )
    for name_field, sha_field, expected_path, expected_sha in bindings:
        name = outputs.get(name_field)
        if (
            not isinstance(name, str)
            or parent / name != expected_path
            or outputs.get(sha_field) != expected_sha
        ):
            raise Atomic1000MergeError(
                f"Wan output binding differs iid={iid} field={name_field}"
            )
    inputs = wan.get("inputs")
    prompt = wan.get("prompt")
    if (
        not isinstance(inputs, dict)
        or inputs.get("source_video_committed_path") != row["source_video"]
        or inputs.get("source_video_sha256") != row["source_video_sha256"]
        or not isinstance(prompt, dict)
        or prompt.get("field") != "edit_instruction"
        or prompt.get("text") != row["wan_generation_prompt"]
        or prompt.get("sha256") != row["wan_generation_prompt_sha256"]
    ):
        raise Atomic1000MergeError(f"Wan source/prompt lineage differs iid={iid}")
    digest = _sha_field(wan.get("result_digest"), context=f"Wan result digest iid={iid}")
    if digest != _object_digest(wan, omit="result_digest"):
        raise Atomic1000MergeError(f"Wan result digest differs iid={iid}")


def _validate_metadata(
    metadata: Mapping[str, Any],
    *,
    row: Mapping[str, Any],
    metadata_path: Path,
    cache: dict[Path, str],
) -> int:
    iid = str(row["iid"])
    if set(metadata) != _METADATA_KEYS or metadata.get("schema_version") != SAMPLE_METADATA_SCHEMA:
        raise Atomic1000MergeError(f"sample metadata schema differs iid={iid}")
    if (
        metadata.get("iid") != iid
        or metadata.get("primary_training_label_field")
        != "atomic_action_instruction"
        or metadata.get("wan_generation_prompt_field")
        != "planner_passed.edit_instruction"
        or metadata.get("wan_generation_prompt_is_training_label") is not False
        or metadata.get("edit_instruction_txt_role")
        != "generation_only_not_training_label"
        or metadata.get("wan_generation_prompt_txt_role")
        != "generation_only_not_training_label"
        or metadata.get("source_video_sha256") != row["source_video_sha256"]
    ):
        raise Atomic1000MergeError(f"sample metadata identity/roles differ iid={iid}")
    for field in (
        "atomic_action_instruction",
        "camera_instruction",
        "preservation_instruction",
        "full_edit_instruction",
        "wan_generation_prompt",
    ):
        if (
            metadata.get(field) != row[field]
            or metadata.get(field + "_sha256") != row[field + "_sha256"]
        ):
            raise Atomic1000MergeError(f"sample metadata {field} differs iid={iid}")
    digest = _sha_field(
        metadata.get("metadata_digest"), context=f"sample metadata digest iid={iid}"
    )
    if digest != _object_digest(metadata, omit="metadata_digest"):
        raise Atomic1000MergeError(f"sample metadata object digest differs iid={iid}")
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, dict):
        raise Atomic1000MergeError(f"sample metadata artifacts differ iid={iid}")
    fixed = {
        "wan_generation_prompt.txt",
        "atomic_action_instruction.txt",
        "camera_instruction.txt",
        "preservation_instruction.txt",
        "full_edit_instruction.txt",
        "planner_passed.jsonl",
        "atomic_result.json",
        "atomic_admission.json",
        "edit_instruction.txt",
        "preview.mp4",
        "result.json",
    }
    source_names = [name for name in artifacts if name.startswith("source_video.")]
    if len(source_names) != 1 or set(artifacts) != fixed | {source_names[0]}:
        raise Atomic1000MergeError(f"sample metadata artifact closure differs iid={iid}")
    verified = 0
    for name, record in artifacts.items():
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "bytes"}:
            raise Atomic1000MergeError(f"metadata artifact record differs iid={iid} name={name}")
        raw_path = record.get("path")
        if not isinstance(raw_path, str):
            raise Atomic1000MergeError(f"metadata artifact path differs iid={iid} name={name}")
        path = _plain_file(Path(raw_path), context=f"metadata artifact iid={iid} name={name}")
        if path.parent != metadata_path.parent or path.name != name:
            raise Atomic1000MergeError(f"metadata artifact locality differs iid={iid} name={name}")
        if type(record.get("bytes")) is not int or record["bytes"] != path.stat().st_size:
            raise Atomic1000MergeError(f"metadata artifact byte count differs iid={iid} name={name}")
        expected = _sha_field(record.get("sha256"), context=f"metadata artifact SHA iid={iid} name={name}")
        if _sha_file(path, cache) != expected:
            raise Atomic1000MergeError(f"metadata artifact SHA differs iid={iid} name={name}")
        verified += 1
    expected_paths = {
        source_names[0]: row["source_video"],
        "preview.mp4": row["target_video"],
        "result.json": row["wan_result"],
    }
    for name, expected_path in expected_paths.items():
        if artifacts[name]["path"] != expected_path:
            raise Atomic1000MergeError(f"metadata primary artifact path differs iid={iid} name={name}")
    if Path(str(metadata["artifacts"]["planner_passed.jsonl"]["path"])).read_bytes() != Path(
        str(row["planner_passed"])
    ).read_bytes():
        raise Atomic1000MergeError(f"metadata planner copy differs iid={iid}")
    if Path(str(metadata["artifacts"]["atomic_result.json"]["path"])).read_bytes() != Path(
        str(row["atomic_result"])
    ).read_bytes():
        raise Atomic1000MergeError(f"metadata atomic-result copy differs iid={iid}")
    if Path(str(metadata["artifacts"]["edit_instruction.txt"]["path"])).read_bytes() != row[
        "wan_generation_prompt"
    ].encode("utf-8"):
        raise Atomic1000MergeError(f"metadata generation-prompt file differs iid={iid}")
    return verified


def _validate_epoch_summary(
    summary: Mapping[str, Any], *, expected_rows: int, manifest_sha: str, epoch: int
) -> None:
    if set(summary) != _EPOCH_SUMMARY_KEYS:
        raise Atomic1000MergeError(f"epoch {epoch} summary schema is open")
    if (
        summary.get("schema_version") != EPOCH_SUMMARY_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("minimum_success") != expected_rows
        or summary.get("total_rows") != expected_rows
        or summary.get("new_wan_rows") != expected_rows
        or summary.get("legacy_reused_rows") != 0
        or summary.get("manifest_sha256") != manifest_sha
        or summary.get("primary_training_label_field")
        != "atomic_action_instruction"
        or summary.get("wan_generation_prompt_is_separate") is not True
    ):
        raise Atomic1000MergeError(f"epoch {epoch} summary binding differs")


def _write_create_only(path: Path, raw: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _artifact_record(path: Path, raw: bytes, *, rows: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": str(path),
        "sha256": _sha_bytes(raw),
        "bytes": len(raw),
    }
    if rows is not None:
        value["rows"] = rows
    return value


def merge_epochs(
    specs: Sequence[EpochSpec],
    *,
    output_root: Path,
    ffprobe: Path,
    required_targets: Sequence[int] = PRODUCTION_EPOCH_TARGETS,
    expected_total: int = PRODUCTION_TOTAL,
) -> dict[str, Any]:
    """Validate and create one exact-size, cross-epoch dataset publication."""

    targets = tuple(spec.expected_rows for spec in specs)
    if targets != tuple(required_targets) or sum(targets) != expected_total:
        raise Atomic1000MergeError(
            f"epoch targets must be exactly {tuple(required_targets)}, observed {targets}"
        )
    if expected_total <= 0:
        raise Atomic1000MergeError("expected total must be positive")
    roots = [spec.run_root for spec in specs]
    if len(set(roots)) != len(roots):
        raise Atomic1000MergeError("epoch run roots must be unique")
    ffprobe = _plain_file(ffprobe, context="ffprobe")
    if not os.access(ffprobe, os.X_OK):
        raise Atomic1000MergeError("ffprobe is not executable")
    if not output_root.is_absolute():
        raise Atomic1000MergeError("output root must be absolute")
    _plain_dir(output_root.parent, context="output parent")
    if output_root.exists() or output_root.is_symlink():
        raise Atomic1000MergeError(f"output root already exists: {output_root}")

    cache: dict[Path, str] = {}
    manifest_parts: list[bytes] = []
    epoch_records: list[dict[str, Any]] = []
    seen_iids: dict[str, tuple[int, int]] = {}
    seen_groups: dict[str, tuple[int, int, str]] = {}
    metadata_artifact_count = 0

    for epoch, spec in enumerate(specs):
        root = _plain_dir(spec.run_root, context=f"epoch {epoch} run root")
        manifest_path = _plain_file(
            root / EPOCH_MANIFEST_NAME, context=f"epoch {epoch} manifest"
        )
        summary_path = _plain_file(
            root / EPOCH_SUMMARY_NAME, context=f"epoch {epoch} summary"
        )
        manifest_raw = manifest_path.read_bytes()
        summary_raw = summary_path.read_bytes()
        manifest_sha = _sha_bytes(manifest_raw)
        summary_sha = _sha_bytes(summary_raw)
        summary = _load_json_bytes(summary_raw, context=f"epoch {epoch} summary")
        _validate_epoch_summary(
            summary,
            expected_rows=spec.expected_rows,
            manifest_sha=manifest_sha,
            epoch=epoch,
        )
        lines = manifest_raw.splitlines(keepends=True)
        if len(lines) != spec.expected_rows or any(
            not line.endswith(b"\n") or not line[:-1] for line in lines
        ):
            raise Atomic1000MergeError(f"epoch {epoch} manifest row count/JSONL closure differs")
        for index, line in enumerate(lines):
            row = _load_json_bytes(line[:-1], context=f"epoch {epoch} row {index}")
            if _canonical(row) + b"\n" != line:
                raise Atomic1000MergeError(
                    f"epoch {epoch} row {index} is not canonical JSONL"
                )
            iid = _validate_row_shape(row, epoch=epoch, index=index)
            previous_iid = seen_iids.get(iid)
            if previous_iid is not None:
                raise Atomic1000MergeError(
                    f"duplicate IID {iid}: epoch/row {previous_iid} and {(epoch, index)}"
                )
            paths = {
                path_field: _bound_artifact(
                    row,
                    path_field,
                    sha_field,
                    iid=iid,
                    cache=cache,
                )
                for path_field, sha_field in _ROW_ARTIFACTS
            }
            source_probe = _probe_video(paths["source_video"], ffprobe=ffprobe)
            target_probe = _probe_video(paths["target_video"], ffprobe=ffprobe)
            if source_probe != row["source_temporal_geometry"]:
                raise Atomic1000MergeError(f"fresh source ffprobe differs iid={iid}")
            if target_probe != row["target_temporal_geometry"]:
                raise Atomic1000MergeError(f"fresh target ffprobe differs iid={iid}")

            planner = _single_jsonl(
                paths["planner_passed"], context=f"planner passed iid={iid}"
            )
            group_id = _validate_planner(
                planner,
                row=row,
                planner_path=paths["planner_passed"],
                cache=cache,
            )
            previous_group = seen_groups.get(group_id)
            if previous_group is not None:
                raise Atomic1000MergeError(
                    f"duplicate planner group_id {group_id}: {previous_group} and {(epoch, index, iid)}"
                )

            atomic = _load_json_bytes(
                paths["atomic_result"].read_bytes(), context=f"atomic result iid={iid}"
            )
            _validate_atomic_result(
                atomic, row=row, planner_path=paths["planner_passed"]
            )
            sidecar_pixels = _verify_strict_sidecars(
                source_video=paths["source_video"],
                source_anchor=paths["strict_source_frame0_anchor_png"],
                conditioning_npy=paths["strict_target_frame0_float32_npy"],
                conditioning_png=paths["strict_target_frame0_png"],
                iid=iid,
            )
            wan = _load_json_bytes(
                paths["wan_result"].read_bytes(), context=f"Wan result iid={iid}"
            )
            _validate_wan_result(wan, row=row, sidecar_pixels=sidecar_pixels)
            metadata = _load_json_bytes(
                paths["sample_metadata"].read_bytes(),
                context=f"sample metadata iid={iid}",
            )
            metadata_artifact_count += _validate_metadata(
                metadata,
                row=row,
                metadata_path=paths["sample_metadata"],
                cache=cache,
            )
            seen_iids[iid] = (epoch, index)
            seen_groups[group_id] = (epoch, index, iid)
        manifest_parts.append(manifest_raw)
        epoch_records.append(
            {
                "epoch_index": epoch,
                "run_root": str(root),
                "expected_rows": spec.expected_rows,
                "manifest": str(manifest_path),
                "manifest_sha256": manifest_sha,
                "manifest_bytes": len(manifest_raw),
                "manifest_rows": len(lines),
                "summary": str(summary_path),
                "summary_sha256": summary_sha,
                "summary_bytes": len(summary_raw),
            }
        )

    if len(seen_iids) != expected_total or len(seen_groups) != expected_total:
        raise Atomic1000MergeError(
            f"merged uniqueness/row total differs: iids={len(seen_iids)} groups={len(seen_groups)}"
        )
    manifest_payload = b"".join(manifest_parts)
    if len(manifest_payload.splitlines()) != expected_total:
        raise Atomic1000MergeError("merged manifest is not exact-size")
    manifest_sha = _sha_bytes(manifest_payload)
    summary_value: dict[str, Any] = {
        "schema_version": MERGED_SUMMARY_SCHEMA,
        "status": "complete",
        "total_rows": expected_total,
        "new_wan_rows": expected_total,
        "epoch_count": len(specs),
        "epoch_targets": list(targets),
        "epochs": epoch_records,
        "manifest_sha256": manifest_sha,
        "iid_unique": True,
        "group_id_unique": True,
        "group_id_derivation": "sha256_bound_planner_passed.group_id",
        "primary_training_label_field": "atomic_action_instruction",
        "wan_generation_prompt_is_separate": True,
        "strict_target_frame0_override_required_for_every_row": True,
        "target_mp4_decoded_frame0_pixel_equality_claimed": False,
        "source_target_temporal_geometry": {
            "frame_count": FRAME_COUNT,
            "frame_rate": FRAME_RATE,
        },
        "row_artifacts_sha256_verified": expected_total * len(_ROW_ARTIFACTS),
        "sample_metadata_artifacts_sha256_verified": metadata_artifact_count,
    }
    summary_payload = _pretty(summary_value)

    output_root.mkdir(mode=0o700)
    try:
        manifest_output = output_root / OUTPUT_MANIFEST_NAME
        summary_output = output_root / OUTPUT_SUMMARY_NAME
        done_output = output_root / OUTPUT_DONE_NAME
        _write_create_only(manifest_output, manifest_payload)
        _write_create_only(summary_output, summary_payload)
        done_value: dict[str, Any] = {
            "schema_version": MERGED_DONE_SCHEMA,
            "status": "complete",
            "total_rows": expected_total,
            "epoch_count": len(specs),
            "epoch_targets": list(targets),
            "epochs": epoch_records,
            "artifacts": {
                OUTPUT_MANIFEST_NAME: _artifact_record(
                    manifest_output, manifest_payload, rows=expected_total
                ),
                OUTPUT_SUMMARY_NAME: _artifact_record(
                    summary_output, summary_payload
                ),
            },
            "iid_unique": True,
            "group_id_unique": True,
            "strict_artifact_and_temporal_revalidation": True,
            "done_digest": None,
        }
        done_value["done_digest"] = _object_digest(done_value, omit="done_digest")
        done_payload = _pretty(done_value)
        _write_create_only(done_output, done_payload)
        descriptor = os.open(output_root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(output_root, 0o500)
    except Exception:
        # Never remove or replace a partial create-only publication.  Its
        # presence is intentional evidence that publication did not close.
        raise
    return done_value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge eight immutable Goku atomic epochs into exact1000"
    )
    parser.add_argument(
        "--epoch",
        action="append",
        nargs=2,
        required=True,
        metavar=("RUN_ROOT", "EXPECTED_ROWS"),
        help="repeat in immutable epoch order; production targets are 128x7 then 104",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ffprobe", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    specs: list[EpochSpec] = []
    for root, count_raw in args.epoch:
        try:
            count = int(count_raw)
        except ValueError as error:
            raise SystemExit(f"invalid --epoch EXPECTED_ROWS: {count_raw}") from error
        if count <= 0 or str(count) != count_raw:
            raise SystemExit(f"invalid --epoch EXPECTED_ROWS: {count_raw}")
        specs.append(EpochSpec(Path(root), count))
    try:
        result = merge_epochs(
            specs,
            output_root=args.output_root,
            ffprobe=args.ffprobe,
        )
    except Atomic1000MergeError as error:
        raise SystemExit(f"goku atomic1000 merge failed: {error}") from error
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
