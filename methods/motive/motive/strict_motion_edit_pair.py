"""Strict loader for the motion-edit pairs emitted by ``wan22_i2v_batch``.

The generated target is an H.264 preview.  Its decoded first frame is *not*
expected, required, or claimed to be pixel-identical to the source.  Instead,
this loader validates the generated manifest and ``result.json`` sidecar, then
decodes both videos and replaces frame zero of both tensors with one shared,
lossless conditioning artifact.

Validation also closes the semantic chain rather than trusting filenames:
OpenCV-decoded source I0 must equal the copied original-anchor PNG pixel for
pixel, and the float32 sidecar must match a runner-equivalent
``RGB -> to_tensor -> [-1,1] -> bicubic`` reconstruction within ``1e-6``
absolute error (bitwise equality is recorded separately).

The returned training convention matches ``lucy.video``:

* ``source_video`` and ``target_video`` are float32 Torch tensors;
* their shape is ``[T, C, H, W]`` and their RGB range is nominally ``[-1, 1]``;
* decoded video frames are RGB-resized with Pillow LANCZOS; and
* the preferred float32 conditioning tensor is resized, only when necessary,
  with Torch bicubic interpolation and ``align_corners=False``.  It is not
  clamped, preserving the bound conditioning values.

``sample_mode="first"`` streams only until the requested strided frames have
been collected.  ``uniform`` uses a count-only pass and an indexed decode pass;
neither mode materializes an entire long source video.

The ordinary ``lucy.data.KiwiEditDataset`` must not be used to re-open the
returned source/target paths when strict frame-zero identity is required: it
would decode the lossy MP4 again without applying this replacement.  Use
``StrictMotionEditDataset`` directly; :func:`to_lucy_sample` only adapts its
already-loaded sample for Lucy's existing collator.

CLI examples::

    python -m motive.strict_motion_edit_pair contract
    python -m motive.strict_motion_edit_pair audit \
        --manifest /path/to/generated_manifest.jsonl \
        --width 832 --height 480 --num-frames 49
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


GENERATED_MANIFEST_SCHEMA = "motive-wan22-i2v-generated-target-v1"
SAMPLE_RESULT_SCHEMA = "motive-wan22-i2v-sample-v1"
FIRST_FRAME_POLICY = "wan22-i2v-strict-preencode-frame0-v1"
TEMPORAL_POLICY = "wan22-i2v-source-timebase-preserving-v1"
APPROVAL_SCHEMA = "motive-goku-action-anchor-approval-v1"
APPROVED_MANIFEST_ROLE = "approved_generation"
REQUIRED_AUTHORIZATION_MODE = "bound_human_approval"
MODEL_SAMPLE_FPS = 16
MAX_DURATION_DELTA_FRAMES = 1
LOADER_CONTRACT_SCHEMA = "motive-strict-motion-edit-pair-loader-v2"
CONDITIONING_RECONSTRUCTION_ATOL = 1e-6
INVALID_BATCH_MARKER = "INVALID_DO_NOT_TRAIN.json"
INVALID_BATCH_ANCESTOR_LIMIT = 5
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_IID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class StrictMotionEditPairError(RuntimeError):
    """A generated pair or its lossless frame-zero binding is invalid."""


@dataclass(frozen=True)
class StrictPairRecord:
    """Fully validated file bindings for one generated pair."""

    iid: str
    group_id: str
    edit_instruction: str
    absolute_target_prompt: str
    source_video: Path
    target_video: Path
    conditioning_anchor_original: Path
    conditioning_frame0_float32: Path
    conditioning_frame0_png: Path
    result_json: Path
    result_digest: str
    conditioning_shape: tuple[int, int, int]
    source_i0_rgb_sha256: str
    conditioning_reconstruction_bitwise_equal: bool
    conditioning_reconstruction_max_abs_error: float
    source_frame_count: int
    source_fps: Fraction
    source_duration_seconds: float
    target_frame_count: int
    target_fps: Fraction
    target_duration_seconds: float
    generated_row: Mapping[str, Any]


def _reject_constant(value: str) -> None:
    raise StrictMotionEditPairError(
        f"non-finite JSON constant is forbidden: {value}"
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrictMotionEditPairError(
                f"duplicate JSON object key: {key!r}"
            )
        result[key] = value
    return result


def _parse_json(raw: bytes, *, context: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise StrictMotionEditPairError(f"{context} is not UTF-8") from error
    try:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        if isinstance(error, StrictMotionEditPairError):
            raise
        raise StrictMotionEditPairError(
            f"{context} is not strict JSON: {error}"
        ) from error


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise StrictMotionEditPairError(
            f"value is not canonical JSON: {error}"
        ) from error


def _object_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _require_string(value: Any, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise StrictMotionEditPairError(
            f"{context} must be a non-empty canonical string"
        )
    return value


def _require_sha256(value: Any, *, context: str) -> str:
    digest = _require_string(value, context=context)
    if _SHA256_RE.fullmatch(digest) is None:
        raise StrictMotionEditPairError(
            f"{context} must be a lowercase SHA-256"
        )
    return digest


def _validate_approval_record(
    value: Any,
    *,
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StrictMotionEditPairError(
            f"{context} must be a closed approval object"
        )
    required = {
        "schema_version",
        "approval_digest",
        "approval_file_sha256",
        "proposal_sha256",
        "reviewer_id",
        "reviewed_at_utc",
        "decision",
        "reason",
    }
    if set(value) != required:
        raise StrictMotionEditPairError(
            f"{context} keys differ from the closed approval schema"
        )
    if value.get("schema_version") != APPROVAL_SCHEMA:
        raise StrictMotionEditPairError(
            f"{context} schema_version must be {APPROVAL_SCHEMA!r}"
        )
    for field in (
        "approval_digest",
        "approval_file_sha256",
        "proposal_sha256",
    ):
        _require_sha256(value.get(field), context=f"{context}.{field}")
    for field in ("reviewer_id", "reviewed_at_utc", "reason"):
        _require_string(value.get(field), context=f"{context}.{field}")
    if value.get("decision") != "approved":
        raise StrictMotionEditPairError(
            f"{context}.decision must be exactly 'approved'"
        )
    return dict(value)


def _regular_file(path: Path, *, context: str) -> Path:
    path = path.expanduser()
    if path.is_symlink() or not path.is_file():
        raise StrictMotionEditPairError(
            f"{context} must be a regular non-symlink file: {path}"
        )
    if path.stat().st_size <= 0:
        raise StrictMotionEditPairError(f"{context} is empty: {path}")
    return path.resolve(strict=True)


def _resolve_file(value: Any, *, base_dir: Path, context: str) -> Path:
    raw = _require_string(value, context=context)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return _regular_file(path, context=context)


def _reject_invalid_batch_marker(start: Path) -> None:
    """Fail closed when this path belongs to a quarantined generation batch."""

    current = start.expanduser().resolve()
    for _ in range(INVALID_BATCH_ANCESTOR_LIMIT):
        marker = current / INVALID_BATCH_MARKER
        if marker.exists() or marker.is_symlink():
            raise StrictMotionEditPairError(
                "generated data is quarantined by "
                f"{INVALID_BATCH_MARKER}: {marker}"
            )
        parent = current.parent
        if parent == current:
            break
        current = parent


def _positive_int(value: Any, *, context: str) -> int:
    if type(value) is not int or value <= 0:
        raise StrictMotionEditPairError(f"{context} must be a positive integer")
    return value


def _positive_finite_float(
    value: Any,
    *,
    context: str,
    allow_zero: bool = False,
) -> float:
    if type(value) not in {int, float}:
        raise StrictMotionEditPairError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (
        result < 0 if allow_zero else result <= 0
    ):
        raise StrictMotionEditPairError(
            f"{context} must be a "
            f"{'non-negative' if allow_zero else 'positive'} finite number"
        )
    return result


def _temporal_grid(value: Any, *, context: str) -> tuple[int, Fraction, float]:
    if not isinstance(value, Mapping):
        raise StrictMotionEditPairError(f"{context} must be an object")
    frames = _positive_int(value.get("frames"), context=f"{context}.frames")
    rate_text = _require_string(
        value.get("frame_rate"),
        context=f"{context}.frame_rate",
    )
    try:
        rate = Fraction(rate_text)
    except (ValueError, ZeroDivisionError) as error:
        raise StrictMotionEditPairError(
            f"{context}.frame_rate is not a rational rate: {rate_text!r}"
        ) from error
    if rate <= 0:
        raise StrictMotionEditPairError(
            f"{context}.frame_rate must be positive"
        )
    duration = _positive_finite_float(
        value.get("duration_seconds"),
        context=f"{context}.duration_seconds",
    )
    nominal_duration = frames / float(rate)
    one_frame = 1.0 / float(rate)
    if abs(duration - nominal_duration) > one_frame + 1e-9:
        raise StrictMotionEditPairError(
            f"{context} duration is inconsistent with frames/FPS: "
            f"frames={frames} fps={rate} duration={duration}"
        )
    return frames, rate, duration


def _validate_temporal_alignment(
    *,
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Any],
) -> tuple[tuple[int, Fraction, float], tuple[int, Fraction, float]]:
    source = _temporal_grid(
        inputs.get("source_video_ffprobe"),
        context="result_json inputs.source_video_ffprobe",
    )
    target = _temporal_grid(
        outputs.get("preview_mp4_ffprobe"),
        context="result_json outputs.preview_mp4_ffprobe",
    )
    source_frames, source_fps, source_duration = source
    target_frames, target_fps, target_duration = target
    if source_frames != target_frames:
        raise StrictMotionEditPairError(
            "source/target temporal frame-count mismatch: "
            f"source={source_frames} target={target_frames}"
        )
    if source_fps != target_fps:
        raise StrictMotionEditPairError(
            "source/target temporal FPS mismatch: "
            f"source={source_fps} target={target_fps}"
        )
    tolerance = 1.0 / float(source_fps)
    if abs(source_duration - target_duration) > tolerance + 1e-9:
        raise StrictMotionEditPairError(
            "source/target temporal duration mismatch: "
            f"source={source_duration} target={target_duration} "
            f"tolerance={tolerance}"
        )
    return source, target


def _validate_pair_temporal_policy(
    value: Any,
    *,
    source: tuple[int, Fraction, float],
    target: tuple[int, Fraction, float],
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StrictMotionEditPairError(f"{context} must be an object")
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
    if set(value) != required:
        raise StrictMotionEditPairError(
            f"{context} keys differ from the closed temporal schema"
        )
    expected_equal = {
        "policy_version": TEMPORAL_POLICY,
        "model_sample_fps": MODEL_SAMPLE_FPS,
        "model_sample_fps_role": "diffusion_configuration_only",
        "output_container_rate_source": "source_video",
        "frame_count_equal": True,
        "frame_rate_equal": True,
        "duration_match_tolerance_frames": MAX_DURATION_DELTA_FRAMES,
        "duration_within_tolerance": True,
    }
    for field, expected in expected_equal.items():
        if value.get(field) != expected:
            raise StrictMotionEditPairError(
                f"{context}.{field} must be {expected!r}"
            )

    for name, temporal in (("source", source), ("target", target)):
        endpoint = value.get(name)
        if not isinstance(endpoint, Mapping) or set(endpoint) != {
            "frame_count",
            "frame_rate",
            "duration_seconds",
        }:
            raise StrictMotionEditPairError(
                f"{context}.{name} is not a closed temporal endpoint"
            )
        frames, rate, duration = temporal
        if endpoint.get("frame_count") != frames:
            raise StrictMotionEditPairError(
                f"{context}.{name}.frame_count differs from ffprobe"
            )
        try:
            endpoint_rate = Fraction(
                _require_string(
                    endpoint.get("frame_rate"),
                    context=f"{context}.{name}.frame_rate",
                )
            )
        except (ValueError, ZeroDivisionError) as error:
            raise StrictMotionEditPairError(
                f"{context}.{name}.frame_rate is invalid"
            ) from error
        if endpoint_rate != rate:
            raise StrictMotionEditPairError(
                f"{context}.{name}.frame_rate differs from ffprobe"
            )
        endpoint_duration = _positive_finite_float(
            endpoint.get("duration_seconds"),
            context=f"{context}.{name}.duration_seconds",
        )
        if not math.isclose(
            endpoint_duration,
            duration,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise StrictMotionEditPairError(
                f"{context}.{name}.duration_seconds differs from ffprobe"
            )

    duration_delta = abs(source[2] - target[2])
    duration_delta_frames = duration_delta * float(source[1])
    tolerance_seconds = MAX_DURATION_DELTA_FRAMES / float(source[1])
    numeric_expected = {
        "duration_delta_seconds": duration_delta,
        "duration_delta_frames": duration_delta_frames,
        "duration_match_tolerance_seconds": tolerance_seconds,
    }
    for field, expected in numeric_expected.items():
        actual = _positive_finite_float(
            value.get(field),
            context=f"{context}.{field}",
            allow_zero=True,
        )
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
            raise StrictMotionEditPairError(
                f"{context}.{field} differs from the bound temporal grid"
            )
    return dict(value)


def _verify_file_hash(
    path: Path,
    expected: Any,
    *,
    context: str,
) -> str:
    expected_digest = _require_sha256(expected, context=f"{context} hash")
    actual = _sha256_file(path)
    if actual != expected_digest:
        raise StrictMotionEditPairError(
            f"{context} hash mismatch: expected={expected_digest} actual={actual}"
        )
    return expected_digest


def read_generated_manifest(
    manifest_path: str | Path,
) -> tuple[Path, list[dict[str, Any]]]:
    """Read strict JSONL and reject truncation, blanks, or duplicate IIDs."""

    path = _regular_file(
        Path(manifest_path),
        context="generated manifest",
    )
    _reject_invalid_batch_marker(path.parent)
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise StrictMotionEditPairError(
            "generated manifest must end with a newline"
        )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            raise StrictMotionEditPairError(
                f"generated manifest contains a blank line at {line_number}"
            )
        value = _parse_json(
            line,
            context=f"generated manifest {path}:{line_number}",
        )
        if not isinstance(value, dict):
            raise StrictMotionEditPairError(
                f"generated manifest row {line_number} is not an object"
            )
        iid = _require_string(
            value.get("iid"),
            context=f"generated manifest row {line_number} iid",
        )
        if _IID_RE.fullmatch(iid) is None or iid in {".", ".."}:
            raise StrictMotionEditPairError(
                f"generated manifest row {line_number} has unsafe iid={iid!r}"
            )
        if iid in seen:
            raise StrictMotionEditPairError(
                f"duplicate generated manifest iid: {iid}"
            )
        seen.add(iid)
        rows.append(value)
    if not rows:
        raise StrictMotionEditPairError("generated manifest is empty")
    return path, rows


def _load_result(path: Path) -> dict[str, Any]:
    value = _parse_json(path.read_bytes(), context=f"result sidecar {path}")
    if not isinstance(value, dict):
        raise StrictMotionEditPairError(
            f"result sidecar must contain one object: {path}"
        )
    claimed = _require_sha256(
        value.get("result_digest"),
        context="result_json result_digest",
    )
    bound = dict(value)
    del bound["result_digest"]
    actual = _object_digest(bound)
    if claimed != actual:
        raise StrictMotionEditPairError(
            f"result_json digest mismatch: expected={claimed} actual={actual}"
        )
    return value


def _bind_result_output(
    *,
    row: Mapping[str, Any],
    result: Mapping[str, Any],
    sample_dir: Path,
    base_dir: Path,
    row_path_field: str,
    row_hash_field: str,
    result_path_field: str,
    result_hash_field: str,
) -> Path:
    outputs = result.get("outputs")
    if not isinstance(outputs, Mapping):
        raise StrictMotionEditPairError("result_json outputs must be an object")
    basename = _require_string(
        outputs.get(result_path_field),
        context=f"result_json outputs.{result_path_field}",
    )
    if Path(basename).name != basename or basename in {".", ".."}:
        raise StrictMotionEditPairError(
            f"result_json outputs.{result_path_field} must be one basename"
        )
    committed_path = _regular_file(
        sample_dir / basename,
        context=f"result output {result_path_field}",
    )
    row_path = _resolve_file(
        row.get(row_path_field),
        base_dir=base_dir,
        context=f"generated row {row_path_field}",
    )
    if row_path != committed_path:
        raise StrictMotionEditPairError(
            f"{row_path_field} does not resolve to its result_json output"
        )
    row_hash = _require_sha256(
        row.get(row_hash_field),
        context=f"generated row {row_hash_field}",
    )
    result_hash = _require_sha256(
        outputs.get(result_hash_field),
        context=f"result_json outputs.{result_hash_field}",
    )
    if row_hash != result_hash:
        raise StrictMotionEditPairError(
            f"{row_hash_field} differs from result_json binding"
        )
    _verify_file_hash(
        committed_path,
        row_hash,
        context=f"result output {result_path_field}",
    )
    return committed_path


def _validate_conditioning_artifacts(
    source_path: Path,
    anchor_path: Path,
    float32_path: Path,
    png_path: Path,
    *,
    inputs: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> tuple[tuple[int, int, int], str, bool, float]:
    try:
        import cv2
        import numpy as np
        from PIL import Image
        import torch
        import torch.nn.functional as torch_functional
    except ImportError as error:
        raise StrictMotionEditPairError(
            "conditioning validation requires OpenCV, NumPy, Pillow, and Torch"
        ) from error

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise StrictMotionEditPairError(
            f"OpenCV could not open source video: {source_path}"
        )
    try:
        ok, source_i0_bgr = capture.read()
    finally:
        capture.release()
    if not ok or source_i0_bgr is None or source_i0_bgr.size == 0:
        raise StrictMotionEditPairError(
            f"OpenCV could not decode exact source frame zero: {source_path}"
        )
    source_i0 = cv2.cvtColor(source_i0_bgr, cv2.COLOR_BGR2RGB)

    try:
        with Image.open(anchor_path) as image:
            anchor_format = image.format
            anchor_rgb = np.asarray(image.convert("RGB"))
    except Exception as error:
        raise StrictMotionEditPairError(
            f"cannot load original conditioning anchor: {anchor_path}: {error}"
        ) from error
    if anchor_format != "PNG" or anchor_rgb.dtype != np.uint8:
        raise StrictMotionEditPairError(
            "conditioning_anchor_original must be a lossless RGB PNG"
        )
    if not np.array_equal(source_i0, anchor_rgb):
        raise StrictMotionEditPairError(
            "conditioning_anchor_original pixels do not equal the exact "
            "OpenCV-decoded source frame zero"
        )
    anchor_height, anchor_width = anchor_rgb.shape[:2]
    source_i0_digest = hashlib.sha256(
        source_i0.tobytes(order="C")
    ).hexdigest()
    if inputs.get("anchor_rgb_sha256") != source_i0_digest:
        raise StrictMotionEditPairError(
            "result_json inputs.anchor_rgb_sha256 does not bind source I0"
        )
    if (
        inputs.get("anchor_width") != anchor_width
        or inputs.get("anchor_height") != anchor_height
    ):
        raise StrictMotionEditPairError(
            "result_json anchor dimensions disagree with source I0"
        )

    try:
        array = np.load(float32_path, allow_pickle=False)
    except Exception as error:
        raise StrictMotionEditPairError(
            f"cannot load conditioning float32 NPY: {float32_path}: {error}"
        ) from error
    if (
        not isinstance(array, np.ndarray)
        or array.dtype != np.dtype("<f4")
        or array.ndim != 3
        or array.shape[0] != 3
        or not array.flags.c_contiguous
        or not np.isfinite(array).all()
    ):
        raise StrictMotionEditPairError(
            "conditioning float32 NPY must be finite C-contiguous "
            "little-endian float32 with shape [3,H,W]"
        )
    channels, height, width = (int(value) for value in array.shape)
    if height <= 0 or width <= 0:
        raise StrictMotionEditPairError(
            "conditioning float32 NPY has an empty spatial dimension"
        )

    try:
        with Image.open(png_path) as image:
            png = np.asarray(image.convert("RGB"))
    except Exception as error:
        raise StrictMotionEditPairError(
            f"cannot load conditioning PNG: {png_path}: {error}"
        ) from error
    if png.shape != (height, width, 3) or png.dtype != np.uint8:
        raise StrictMotionEditPairError(
            "conditioning PNG shape/dtype does not match float32 NPY"
        )
    projected = (
        np.rint(
            (array.transpose(1, 2, 0) + np.float32(1.0))
            * np.float32(127.5)
        )
        .clip(0, 255)
        .astype(np.uint8)
    )
    if not np.array_equal(projected, png):
        raise StrictMotionEditPairError(
            "conditioning PNG pixels are not the bound display projection "
            "of conditioning_frame0_float32"
        )
    pixel_digest = hashlib.sha256(
        png.tobytes(order="C")
    ).hexdigest()
    for field in (
        "preencode_frame0_pixel_sha256",
        "lossless_png_pixel_sha256",
    ):
        if _require_sha256(
            policy.get(field),
            context=f"result_json first_frame_policy.{field}",
        ) != pixel_digest:
            raise StrictMotionEditPairError(
                f"result_json first_frame_policy.{field} mismatch"
            )
    if policy.get("conditioning_tensor_shape") != [
        channels,
        height,
        width,
    ]:
        raise StrictMotionEditPairError(
            "result_json conditioning_tensor_shape disagrees with NPY"
        )
    if policy.get("conditioning_tensor_dtype") != "float32":
        raise StrictMotionEditPairError(
            "result_json conditioning_tensor_dtype must be float32"
        )

    try:
        anchor_tensor = torch.from_numpy(anchor_rgb.copy())
    except RuntimeError as error:
        if "Numpy is not available" not in str(error):
            raise
        anchor_tensor = torch.tensor(anchor_rgb.tolist(), dtype=torch.uint8)
    anchor_tensor = (
        anchor_tensor.permute(2, 0, 1)
        .contiguous()
        .to(dtype=torch.float32)
        .div(255.0)
        .sub(0.5)
        .div(0.5)
    )
    reconstructed = torch_functional.interpolate(
        anchor_tensor[None].cpu(),
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    )[0].contiguous()
    try:
        recorded = torch.from_numpy(array.copy())
    except RuntimeError as error:
        if "Numpy is not available" not in str(error):
            raise
        recorded = torch.tensor(array.tolist(), dtype=torch.float32)
    bitwise_equal = bool(torch.equal(reconstructed, recorded))
    max_abs_error = float(
        (reconstructed - recorded).abs().max().item()
    )
    if not math.isfinite(max_abs_error) or (
        max_abs_error > CONDITIONING_RECONSTRUCTION_ATOL
    ):
        raise StrictMotionEditPairError(
            "conditioning_frame0_float32 is not derived from the exact "
            "source-I0 anchor by the runner bicubic transform: "
            f"max_abs_error={max_abs_error:.9g} "
            f"atol={CONDITIONING_RECONSTRUCTION_ATOL:.9g}"
        )
    return (
        (channels, height, width),
        source_i0_digest,
        bitwise_equal,
        max_abs_error,
    )


def validate_generated_row(
    row: Mapping[str, Any],
    *,
    base_dir: str | Path,
) -> StrictPairRecord:
    """Validate hashes, result bindings, and the strict frame-zero policy."""

    if not isinstance(row, Mapping):
        raise StrictMotionEditPairError("generated row must be an object")
    base = Path(base_dir).expanduser().resolve()
    _reject_invalid_batch_marker(base)
    if row.get("schema_version") != GENERATED_MANIFEST_SCHEMA:
        raise StrictMotionEditPairError(
            f"generated row schema_version must be {GENERATED_MANIFEST_SCHEMA!r}"
        )
    iid = _require_string(row.get("iid"), context="generated row iid")
    if _IID_RE.fullmatch(iid) is None or iid in {".", ".."}:
        raise StrictMotionEditPairError(f"unsafe generated row iid={iid!r}")
    group_id = _require_string(
        row.get("group_id"),
        context="generated row group_id",
    )
    _require_string(
        row.get("action_category"),
        context="generated row action_category",
    )
    _require_string(
        row.get("target_action_verb"),
        context="generated row target_action_verb",
    )
    edit_instruction = _require_string(
        row.get("edit_instruction"),
        context="generated row edit_instruction",
    )
    absolute_target_prompt = _require_string(
        row.get("absolute_target_prompt"),
        context="generated row absolute_target_prompt",
    )
    seed = row.get("seed")
    if type(seed) is not int or seed < 0:
        raise StrictMotionEditPairError(
            "generated row seed must be a non-negative integer"
        )
    authorization_mode = _require_string(
        row.get("authorization_mode"),
        context="generated row authorization_mode",
    )
    if authorization_mode != REQUIRED_AUTHORIZATION_MODE:
        raise StrictMotionEditPairError(
            "generated row is not backed by a proposal-bound human approval"
        )
    if (
        row.get("manifest_role") != APPROVED_MANIFEST_ROLE
        or row.get("production_eligible") is not True
        or row.get("human_review_status") != "approved"
        or row.get("generation_authorized") is not True
    ):
        raise StrictMotionEditPairError(
            "generated row is not explicitly approved for production"
        )
    row_approval = _validate_approval_record(
        row.get("approval"),
        context="generated row approval",
    )
    if row.get("action_change_substantive") != "yes":
        raise StrictMotionEditPairError(
            "generated row action_change_substantive must be exactly 'yes'"
        )
    if row.get("first_frame_policy") != FIRST_FRAME_POLICY:
        raise StrictMotionEditPairError(
            "generated row first_frame_policy mismatch"
        )
    if row.get("mp4_decode_pixel_equality_claimed") is not False:
        raise StrictMotionEditPairError(
            "generated row must explicitly deny MP4 decoded pixel equality"
        )

    source = _resolve_file(
        row.get("source_video"),
        base_dir=base,
        context="generated row source_video",
    )
    _verify_file_hash(
        source,
        row.get("source_video_sha256"),
        context="source video",
    )
    result_path = _resolve_file(
        row.get("result_json"),
        base_dir=base,
        context="generated row result_json",
    )
    # A manifest copied outside a quarantined run must not bypass the batch
    # marker.  Re-check the committed sample location, not only ``base_dir``.
    _reject_invalid_batch_marker(result_path.parent)
    result = _load_result(result_path)
    row_result_digest = _require_sha256(
        row.get("result_digest"),
        context="generated row result_digest",
    )
    if row_result_digest != result["result_digest"]:
        raise StrictMotionEditPairError(
            "generated row result_digest differs from result_json"
        )
    if result.get("schema_version") != SAMPLE_RESULT_SCHEMA:
        raise StrictMotionEditPairError(
            f"result_json schema_version must be {SAMPLE_RESULT_SCHEMA!r}"
        )
    for field, expected in (("iid", iid), ("group_id", group_id)):
        if result.get(field) != expected:
            raise StrictMotionEditPairError(
                f"result_json {field} differs from generated row"
            )
    for field in ("seed", "authorization_mode"):
        if result.get(field) != row.get(field):
            raise StrictMotionEditPairError(
                f"result_json {field} differs from generated row"
            )
    if (
        result.get("manifest_role") != APPROVED_MANIFEST_ROLE
        or result.get("production_eligible") is not True
        or result.get("generation_authorized_in_manifest") is not True
        or result.get("human_review_status_at_generation") != "approved"
    ):
        raise StrictMotionEditPairError(
            "result_json lacks bound production approval provenance"
        )
    result_approval = _validate_approval_record(
        result.get("approval"),
        context="result_json approval",
    )
    if _canonical_bytes(result_approval) != _canonical_bytes(row_approval):
        raise StrictMotionEditPairError(
            "result_json approval differs from generated row"
        )
    if result.get("action_change_substantive") != "yes":
        raise StrictMotionEditPairError(
            "result_json action_change_substantive must be exactly 'yes'"
        )
    prompt = result.get("prompt")
    if not isinstance(prompt, Mapping):
        raise StrictMotionEditPairError("result_json prompt must be an object")
    if (
        prompt.get("edit_instruction") != edit_instruction
        or prompt.get("text") != absolute_target_prompt
        or prompt.get("field") != "absolute_target_prompt"
    ):
        raise StrictMotionEditPairError(
            "result_json prompt differs from generated row"
        )
    inputs = result.get("inputs")
    if not isinstance(inputs, Mapping):
        raise StrictMotionEditPairError("result_json inputs must be an object")
    if inputs.get("source_video_sha256") != row.get("source_video_sha256"):
        raise StrictMotionEditPairError(
            "result_json source video hash differs from generated row"
        )
    result_source = _resolve_file(
        inputs.get("source_video_resolved_path"),
        base_dir=result_path.parent,
        context="result_json source_video_resolved_path",
    )
    if result_source != source:
        raise StrictMotionEditPairError(
            "result_json source path differs from generated row"
        )
    outputs = result.get("outputs")
    if not isinstance(outputs, Mapping):
        raise StrictMotionEditPairError("result_json outputs must be an object")
    source_temporal, target_temporal = _validate_temporal_alignment(
        inputs=inputs,
        outputs=outputs,
    )
    result_temporal_policy = _validate_pair_temporal_policy(
        result.get("temporal_policy"),
        source=source_temporal,
        target=target_temporal,
        context="result_json temporal_policy",
    )
    row_temporal_policy = _validate_pair_temporal_policy(
        row.get("temporal_policy"),
        source=source_temporal,
        target=target_temporal,
        context="generated row temporal_policy",
    )
    if _canonical_bytes(result_temporal_policy) != _canonical_bytes(
        row_temporal_policy
    ):
        raise StrictMotionEditPairError(
            "generated row temporal_policy differs from result_json"
        )

    sample_dir = result_path.parent
    target = _bind_result_output(
        row=row,
        result=result,
        sample_dir=sample_dir,
        base_dir=base,
        row_path_field="target_preview_mp4",
        row_hash_field="target_preview_mp4_sha256",
        result_path_field="preview_mp4",
        result_hash_field="preview_mp4_sha256",
    )
    anchor = _bind_result_output(
        row=row,
        result=result,
        sample_dir=sample_dir,
        base_dir=base,
        row_path_field="conditioning_anchor_original",
        row_hash_field="conditioning_anchor_original_sha256",
        result_path_field="conditioning_anchor_original",
        result_hash_field="conditioning_anchor_original_sha256",
    )
    if (
        inputs.get("anchor_sha256")
        != row.get("conditioning_anchor_original_sha256")
    ):
        raise StrictMotionEditPairError(
            "result_json inputs.anchor_sha256 differs from committed anchor"
        )
    float32_path = _bind_result_output(
        row=row,
        result=result,
        sample_dir=sample_dir,
        base_dir=base,
        row_path_field="conditioning_frame0_float32",
        row_hash_field="conditioning_frame0_float32_sha256",
        result_path_field="conditioning_frame0_float32",
        result_hash_field="conditioning_frame0_float32_sha256",
    )
    png_path = _bind_result_output(
        row=row,
        result=result,
        sample_dir=sample_dir,
        base_dir=base,
        row_path_field="conditioning_frame0_png",
        row_hash_field="conditioning_frame0_png_sha256",
        result_path_field="conditioning_frame0_png",
        result_hash_field="conditioning_frame0_png_sha256",
    )

    policy = result.get("first_frame_policy")
    if not isinstance(policy, Mapping):
        raise StrictMotionEditPairError(
            "result_json first_frame_policy must be an object"
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
            raise StrictMotionEditPairError(
                f"result_json first_frame_policy.{field} must be {expected!r}"
            )
    (
        conditioning_shape,
        source_i0_digest,
        reconstruction_bitwise_equal,
        reconstruction_max_abs_error,
    ) = _validate_conditioning_artifacts(
        source,
        anchor,
        float32_path,
        png_path,
        inputs=inputs,
        policy=policy,
    )
    return StrictPairRecord(
        iid=iid,
        group_id=group_id,
        edit_instruction=edit_instruction,
        absolute_target_prompt=absolute_target_prompt,
        source_video=source,
        target_video=target,
        conditioning_anchor_original=anchor,
        conditioning_frame0_float32=float32_path,
        conditioning_frame0_png=png_path,
        result_json=result_path,
        result_digest=row_result_digest,
        conditioning_shape=conditioning_shape,
        source_i0_rgb_sha256=source_i0_digest,
        conditioning_reconstruction_bitwise_equal=(
            reconstruction_bitwise_equal
        ),
        conditioning_reconstruction_max_abs_error=(
            reconstruction_max_abs_error
        ),
        source_frame_count=source_temporal[0],
        source_fps=source_temporal[1],
        source_duration_seconds=source_temporal[2],
        target_frame_count=target_temporal[0],
        target_fps=target_temporal[1],
        target_duration_seconds=target_temporal[2],
        generated_row=dict(row),
    )


def _iter_decoded_frames(path: Path):
    try:
        import imageio.v3 as iio
    except ImportError as error:
        raise StrictMotionEditPairError(
            "video loading requires imageio"
        ) from error
    return iio.imiter(path)


def _sample_decoded_frames(
    path: Path,
    *,
    num_frames: int,
    frame_stride: int,
    sample_mode: str,
    short_video_mode: str,
) -> list[Any]:
    """Select frames with O(num_frames) memory.

    ``first`` stops decoding as soon as enough strided frames are available.
    ``uniform`` makes a count-only pass followed by a selected-index pass,
    trading one extra decode for bounded memory on long/high-resolution clips.
    """

    try:
        if sample_mode == "first":
            selected = []
            for decoded_index, frame in enumerate(
                _iter_decoded_frames(path)
            ):
                if decoded_index % frame_stride != 0:
                    continue
                selected.append(frame)
                if len(selected) == num_frames:
                    break
        elif sample_mode == "uniform":
            decoded_count = sum(1 for _ in _iter_decoded_frames(path))
            available = (
                decoded_count + frame_stride - 1
            ) // frame_stride
            if available >= num_frames:
                if num_frames == 1:
                    sampled_positions = [0]
                else:
                    sampled_positions = [
                        round(
                            index
                            * (available - 1)
                            / (num_frames - 1)
                        )
                        for index in range(num_frames)
                    ]
                decoded_indices = {
                    position * frame_stride
                    for position in sampled_positions
                }
                selected = [
                    frame
                    for decoded_index, frame in enumerate(
                        _iter_decoded_frames(path)
                    )
                    if decoded_index in decoded_indices
                ]
            else:
                selected = [
                    frame
                    for decoded_index, frame in enumerate(
                        _iter_decoded_frames(path)
                    )
                    if decoded_index % frame_stride == 0
                ]
        else:
            raise StrictMotionEditPairError(
                "sample_mode must be first or uniform"
            )
    except StrictMotionEditPairError:
        raise
    except Exception as error:
        raise StrictMotionEditPairError(
            f"could not decode video {path}: {error}"
        ) from error

    if not selected:
        raise StrictMotionEditPairError(f"{path} has no decodable frames")
    if len(selected) < num_frames:
        missing = num_frames - len(selected)
        if short_video_mode == "error":
            raise StrictMotionEditPairError(
                f"{path} has {len(selected)} sampled frames, "
                f"need {num_frames}"
            )
        if short_video_mode == "pad":
            selected = [*selected, *([selected[-1]] * missing)]
        elif short_video_mode == "loop":
            original = list(selected)
            while len(selected) < num_frames:
                selected.extend(original)
            selected = selected[:num_frames]
        else:
            raise StrictMotionEditPairError(
                "short_video_mode must be error, pad, or loop"
            )
    return selected


def _decode_video_tensor(
    path: Path,
    *,
    width: int,
    height: int,
    num_frames: int,
    frame_stride: int,
    sample_mode: str,
    short_video_mode: str,
):
    """Decode using the same RGB/LANCZOS/TCHW convention as Lucy."""

    try:
        import numpy as np
        from PIL import Image
        import torch
    except ImportError as error:
        raise StrictMotionEditPairError(
            "video loading requires NumPy, Pillow, and Torch"
        ) from error
    frames = _sample_decoded_frames(
        path,
        num_frames=num_frames,
        frame_stride=frame_stride,
        sample_mode=sample_mode,
        short_video_mode=short_video_mode,
    )

    resampling = getattr(Image, "Resampling", Image).LANCZOS
    arrays = []
    for frame in frames:
        image = Image.fromarray(frame).convert("RGB")
        image = image.resize((width, height), resampling)
        arrays.append(np.asarray(image, dtype=np.float32) / 127.5 - 1.0)
    stacked = np.stack(arrays, axis=0)
    try:
        tensor = torch.from_numpy(stacked)
    except RuntimeError as error:
        # Some diagnostic environments pair a NumPy-1-built Torch wheel with
        # NumPy 2.  The production environment is pinned consistently, but
        # construction through Python values keeps the contract auditable.
        if "Numpy is not available" not in str(error):
            raise
        tensor = torch.tensor(stacked.tolist(), dtype=torch.float32)
    return tensor.permute(0, 3, 1, 2).contiguous()


def _load_conditioning_tensor(
    record: StrictPairRecord,
    *,
    width: int,
    height: int,
    sidecar: str,
):
    try:
        import numpy as np
        from PIL import Image
        import torch
        import torch.nn.functional as torch_functional
    except ImportError as error:
        raise StrictMotionEditPairError(
            "conditioning loading requires NumPy, Pillow, and Torch"
        ) from error
    if sidecar == "float32":
        array = np.load(
            record.conditioning_frame0_float32,
            allow_pickle=False,
        )
        try:
            conditioning = torch.from_numpy(array.copy())
        except RuntimeError as error:
            if "Numpy is not available" not in str(error):
                raise
            conditioning = torch.tensor(array.tolist(), dtype=torch.float32)
        artifact = record.conditioning_frame0_float32
    elif sidecar == "png":
        with Image.open(record.conditioning_frame0_png) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32)
        normalized = (array / 127.5 - 1.0).transpose(2, 0, 1).copy()
        try:
            conditioning = torch.from_numpy(normalized)
        except RuntimeError as error:
            if "Numpy is not available" not in str(error):
                raise
            conditioning = torch.tensor(
                normalized.tolist(),
                dtype=torch.float32,
            )
        artifact = record.conditioning_frame0_png
    else:
        raise StrictMotionEditPairError(
            "sidecar must be 'float32' or 'png'"
        )
    if conditioning.dtype != torch.float32:
        conditioning = conditioning.float()
    if tuple(conditioning.shape[-2:]) != (height, width):
        conditioning = torch_functional.interpolate(
            conditioning[None],
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )[0]
    return conditioning.contiguous(), artifact


def _load_validated_record(
    record: StrictPairRecord,
    *,
    width: int,
    height: int,
    num_frames: int,
    frame_stride: int,
    sample_mode: str,
    short_video_mode: str,
    sidecar: str,
) -> dict[str, Any]:
    if width <= 0 or height <= 0 or num_frames <= 0 or frame_stride <= 0:
        raise StrictMotionEditPairError(
            "width, height, num_frames, and frame_stride must be positive"
        )
    source = _decode_video_tensor(
        record.source_video,
        width=width,
        height=height,
        num_frames=num_frames,
        frame_stride=frame_stride,
        sample_mode=sample_mode,
        short_video_mode=short_video_mode,
    )
    target = _decode_video_tensor(
        record.target_video,
        width=width,
        height=height,
        num_frames=num_frames,
        frame_stride=frame_stride,
        sample_mode=sample_mode,
        short_video_mode=short_video_mode,
    )
    expected_shape = (num_frames, 3, height, width)
    if tuple(source.shape) != expected_shape or tuple(target.shape) != expected_shape:
        raise StrictMotionEditPairError(
            "decoded source/target tensor shape violates [T,3,H,W] contract"
        )
    conditioning, artifact = _load_conditioning_tensor(
        record,
        width=width,
        height=height,
        sidecar=sidecar,
    )
    source = source.clone()
    target = target.clone()
    source[0].copy_(conditioning)
    target[0].copy_(conditioning)
    try:
        import torch
    except ImportError as error:
        raise StrictMotionEditPairError("Torch is required") from error
    if not torch.equal(source[0], target[0]):
        raise StrictMotionEditPairError(
            "internal error: strict source/target frame-zero replacement differs"
        )
    return {
        "iid": record.iid,
        "group_id": record.group_id,
        "prompt": record.edit_instruction,
        "edit_instruction": record.edit_instruction,
        "absolute_target_prompt": record.absolute_target_prompt,
        "operation": "motion_action_edit",
        "sample_type": "wan22_strict_first_frame",
        "source_video": source,
        "target_video": target,
        "src_path": str(record.source_video),
        "tgt_path": str(record.target_video),
        "conditioning_frame0_path": str(artifact),
        "conditioning_frame0_kind": sidecar,
        "first_frame_policy": FIRST_FRAME_POLICY,
        "strict_frame0_replacement_applied": True,
        "mp4_decode_pixel_equality_claimed": False,
        "result_json": str(record.result_json),
        "result_digest": record.result_digest,
        "source_i0_rgb_sha256": record.source_i0_rgb_sha256,
        "conditioning_reconstruction_bitwise_equal": (
            record.conditioning_reconstruction_bitwise_equal
        ),
        "conditioning_reconstruction_max_abs_error": (
            record.conditioning_reconstruction_max_abs_error
        ),
        "conditioning_reconstruction_atol": (
            CONDITIONING_RECONSTRUCTION_ATOL
        ),
        "temporal_alignment": {
            "frame_count": record.source_frame_count,
            "fps": f"{record.source_fps.numerator}/{record.source_fps.denominator}",
            "source_duration_seconds": record.source_duration_seconds,
            "target_duration_seconds": record.target_duration_seconds,
            "duration_tolerance_seconds": 1.0 / float(record.source_fps),
            "frame_count_equal": (
                record.source_frame_count == record.target_frame_count
            ),
            "fps_equal": record.source_fps == record.target_fps,
        },
    }


def load_strict_motion_edit_pair(
    row: Mapping[str, Any],
    *,
    base_dir: str | Path,
    width: int,
    height: int,
    num_frames: int = 49,
    frame_stride: int = 1,
    sample_mode: str = "first",
    short_video_mode: str = "error",
    sidecar: str = "float32",
) -> dict[str, Any]:
    """Validate and load one pair, then enforce exact tensor frame-zero identity."""

    record = validate_generated_row(row, base_dir=base_dir)
    return _load_validated_record(
        record,
        width=width,
        height=height,
        num_frames=num_frames,
        frame_stride=frame_stride,
        sample_mode=sample_mode,
        short_video_mode=short_video_mode,
        sidecar=sidecar,
    )


class StrictMotionEditDataset:
    """Map-style dataset over a completed Wan generated manifest.

    All rows, result sidecars, hashes, and conditioning artifacts are validated
    eagerly once at construction.  ``__getitem__`` decodes media and applies
    the shared frame-zero replacement.  It never silently skips a bad sample.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        width: int,
        height: int,
        num_frames: int = 49,
        frame_stride: int = 1,
        sample_mode: str = "first",
        short_video_mode: str = "error",
        sidecar: str = "float32",
        lucy_compatible: bool = False,
    ) -> None:
        if width <= 0 or height <= 0 or num_frames <= 0 or frame_stride <= 0:
            raise StrictMotionEditPairError(
                "width, height, num_frames, and frame_stride must be positive"
            )
        if sample_mode not in {"first", "uniform"}:
            raise StrictMotionEditPairError(
                "sample_mode must be first or uniform"
            )
        if short_video_mode not in {"error", "pad", "loop"}:
            raise StrictMotionEditPairError(
                "short_video_mode must be error, pad, or loop"
            )
        if sidecar not in {"float32", "png"}:
            raise StrictMotionEditPairError(
                "sidecar must be 'float32' or 'png'"
            )
        manifest, rows = read_generated_manifest(manifest_path)
        self.manifest_path = manifest
        self.records = [
            validate_generated_row(row, base_dir=manifest.parent)
            for row in rows
        ]
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.sample_mode = sample_mode
        self.short_video_mode = short_video_mode
        self.sidecar = sidecar
        self.lucy_compatible = lucy_compatible

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = _load_validated_record(
            self.records[index],
            width=self.width,
            height=self.height,
            num_frames=self.num_frames,
            frame_stride=self.frame_stride,
            sample_mode=self.sample_mode,
            short_video_mode=self.short_video_mode,
            sidecar=self.sidecar,
        )
        return to_lucy_sample(sample) if self.lucy_compatible else sample


def to_lucy_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Add the fields consumed by ``lucy.data.collate_video_edit_batch``.

    This adapts tensors that have *already* received strict frame-zero
    replacement.  It does not convert the row back into a normal Kiwi manifest.
    """

    try:
        from lucy.concepts import parse_edit_prompt
    except ImportError as error:
        raise StrictMotionEditPairError(
            "Lucy compatibility requires importing lucy.concepts"
        ) from error
    prompt = _require_string(sample.get("prompt"), context="sample prompt")
    concepts = parse_edit_prompt(prompt)
    result = dict(sample)
    result.update(
        {
            "concept_text": concepts.as_text(),
            "concept_atoms": concepts.atoms,
            "base_video": None,
            "anchor_video": None,
            "history_mask": None,
            "teacher_source_video": None,
            "base_path": None,
            "anchor_path": None,
            "history_mask_path": None,
            "teacher_source_path": None,
            "initial_teacher_edit_scale": 1.0,
            "initial_teacher_content_scale": 1.0,
            "initial_teacher_detail_scale": 1.0,
            "initial_teacher_temporal_scale": 1.0,
        }
    )
    return result


def loader_contract() -> dict[str, Any]:
    """Return the machine-readable training contract shown by the CLI."""

    return {
        "schema_version": LOADER_CONTRACT_SCHEMA,
        "input": {
            "generated_manifest_schema": GENERATED_MANIFEST_SCHEMA,
            "result_schema": SAMPLE_RESULT_SCHEMA,
            "required_first_frame_policy": FIRST_FRAME_POLICY,
            "invalid_batch_marker": INVALID_BATCH_MARKER,
            "required_authorization_mode": REQUIRED_AUTHORIZATION_MODE,
            "required_manifest_role": APPROVED_MANIFEST_ROLE,
            "required_approval_schema": APPROVAL_SCHEMA,
            "required_temporal_policy": TEMPORAL_POLICY,
            "hash_validation": (
                "source, target preview, original anchor, float32 NPY, PNG, "
                "and result object digest"
            ),
        },
        "temporal_alignment": {
            "source_target_frame_count_equal": True,
            "source_target_fps_equal": True,
            "source_target_duration_tolerance": "at most one source frame",
            "sampling_grid": "shared frame indices on an equal FPS grid",
        },
        "output": {
            "source_video": "torch.float32 [T,3,H,W]",
            "target_video": "torch.float32 [T,3,H,W]",
            "decoded_frame_resize": "Pillow RGB LANCZOS to (width,height)",
            "default_num_frames": 49,
            "default_temporal_sampling": "first, stride=1, short=error",
            "decode_memory": (
                "first stops after selected frames; uniform uses a count pass "
                "and an indexed pass; both retain O(T) frames"
            ),
        },
        "frame_zero": {
            "default_artifact": "conditioning_frame0_float32.npy",
            "alternative_artifact": "conditioning_frame0.png",
            "artifact_resize": (
                "torch bicubic align_corners=False only when artifact H/W "
                "differs from requested H/W"
            ),
            "source_and_target_override": True,
            "post_override_torch_equal": True,
            "source_i0_to_anchor_check": (
                "OpenCV exact decoded I0 equals original anchor PNG pixels"
            ),
            "anchor_to_float32_check": (
                "runner-equivalent RGB to_tensor, [-1,1], Torch bicubic"
            ),
            "anchor_to_float32_atol": CONDITIONING_RECONSTRUCTION_ATOL,
            "mp4_decoded_native_pixel_equality_claimed": False,
            "mp4_decoded_native_pixel_equality_required": False,
        },
        "limitations": [
            "Semantic target-vs-source action validity must be certified upstream.",
            "Do not re-open target paths through the ordinary Kiwi loader.",
        ],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "contract",
        help="print the machine-readable loader contract",
    )
    audit = subparsers.add_parser(
        "audit",
        help="validate and decode generated pairs without writing files",
    )
    audit.add_argument("--manifest", required=True, type=Path)
    audit.add_argument("--width", required=True, type=int)
    audit.add_argument("--height", required=True, type=int)
    audit.add_argument("--num-frames", default=49, type=int)
    audit.add_argument("--frame-stride", default=1, type=int)
    audit.add_argument("--sample-mode", choices=("first", "uniform"), default="first")
    audit.add_argument(
        "--short-video-mode",
        choices=("error", "pad", "loop"),
        default="error",
    )
    audit.add_argument("--sidecar", choices=("float32", "png"), default="float32")
    audit.add_argument("--limit", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "contract":
        print(json.dumps(loader_contract(), ensure_ascii=False, sort_keys=True))
        return 0
    if args.limit is not None and args.limit <= 0:
        raise StrictMotionEditPairError("--limit must be positive")
    manifest, rows = read_generated_manifest(args.manifest)
    selected = rows[: args.limit] if args.limit is not None else rows
    records = [
        validate_generated_row(row, base_dir=manifest.parent)
        for row in selected
    ]
    for record in records:
        sample = _load_validated_record(
            record,
            width=args.width,
            height=args.height,
            num_frames=args.num_frames,
            frame_stride=args.frame_stride,
            sample_mode=args.sample_mode,
            short_video_mode=args.short_video_mode,
            sidecar=args.sidecar,
        )
        if tuple(sample["source_video"].shape) != tuple(
            sample["target_video"].shape
        ):
            raise StrictMotionEditPairError(
                f"source/target shape mismatch for iid={record.iid}"
            )
    print(
        json.dumps(
            {
                "schema_version": LOADER_CONTRACT_SCHEMA,
                "manifest": str(manifest),
                "manifest_rows": len(rows),
                "audited_rows": len(records),
                "shape_per_pair": [
                    args.num_frames,
                    3,
                    args.height,
                    args.width,
                ],
                "sidecar": args.sidecar,
                "strict_frame0_replacement_applied": True,
                "source_i0_anchor_pixel_equal": True,
                "conditioning_reconstruction_atol": (
                    CONDITIONING_RECONSTRUCTION_ATOL
                ),
                "conditioning_reconstruction_max_abs_error": max(
                    record.conditioning_reconstruction_max_abs_error
                    for record in records
                ),
                "conditioning_reconstruction_bitwise_equal_rows": sum(
                    record.conditioning_reconstruction_bitwise_equal
                    for record in records
                ),
                "mp4_decoded_native_pixel_equality_claimed": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
