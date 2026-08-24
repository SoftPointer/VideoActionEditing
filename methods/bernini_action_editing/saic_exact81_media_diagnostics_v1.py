#!/usr/bin/env python3
"""Hash-bound exact81 decoded-media diagnostics with permanently zero authority.

This module compares one caller hash-pinned local source video with one local
candidate video.  It does not prove dataset registration or external
provenance.  It strictly decodes all 81 frames at 25 fps, computes all 80
camera-compensated motion transitions through
``methods/motive/motive/geometry.py``, and reports
simple full-81 technical diagnostics.  Optical flow is computed internally; no
mask, track, pose, trajectory, or caller-supplied flow artifact is accepted.

The output is diagnostic evidence only.  It has no identity, appearance,
background, semantic non-target, event, source-binding, or inverse evaluator,
and can never authorize selection, training, or an optimizer step.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


METHOD_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = METHOD_ROOT.parents[1]
MOTIVE_ROOT = REPOSITORY_ROOT / "methods" / "motive"
if str(MOTIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTIVE_ROOT))

from motive.geometry import MotionConfig, analyze_video  # noqa: E402
import decoded_temporal_event_evaluator_v1 as _decoded_evaluator  # noqa: E402

decode_exact81_rgb24 = _decoded_evaluator.decode_exact81_rgb24


SCHEMA_VERSION = "bernini-saic-exact81-media-diagnostics-v1"
FRAME_COUNT = 81
FPS = 25
TRANSITION_COUNT = FRAME_COUNT - 1
UNAVAILABLE_AXES = (
    "identity",
    "appearance",
    "background",
    "non_target",
    "event",
    "source_bind",
    "inverse",
)
DIAGNOSTIC_AXES = ("camera", "technical", "temporal_consistency")


def _schema_version(_literal: str = "bernini-saic-exact81-media-diagnostics-v1") -> str:
    """Return a captured literal, independent of rebindable public snapshots."""

    return _literal


def _frame_count(_literal: int = 81) -> int:
    return _literal


def _fps(_literal: int = 25) -> int:
    return _literal


def _transition_count(_literal: int = 80) -> int:
    return _literal


def _decoder_implementation_path(
    _literal: Path = Path(_decoded_evaluator.__file__).resolve(strict=True),
) -> Path:
    return _literal


def _own_implementation_path(
    _literal: Path = Path(__file__).resolve(strict=True),
) -> Path:
    return _literal


def _geometry_implementation_path(
    _literal: Path = (MOTIVE_ROOT / "motive" / "geometry.py").resolve(strict=True),
) -> Path:
    return _literal


def _authority_contract() -> dict[str, bool]:
    """Rebuild the non-authoritative contract from literals.

    Public module attributes are convenient for inspection but are not a
    Python trust root: a caller can rebind them in-process.  Build and replay
    therefore never derive authority from those attributes.
    """

    return {
        "measurement_runtime_qualified": False,
        "candidate_selection_allowed": False,
        "training_allowed": False,
        "optimizer_step_allowed": False,
        "absolute_action_editing_success_claimed": False,
    }


def _input_closure_contract() -> dict[str, bool]:
    return {
        "source_video_read": True,
        "candidate_video_read": True,
        "decoded_exact81_whole_frames_read": True,
        "external_mask_read": False,
        "external_track_read": False,
        "external_pose_read": False,
        "external_flow_read": False,
        "external_trajectory_read": False,
        "internally_computed_optical_flow_diagnostic_only": True,
    }


def _availability_contract() -> dict[str, str]:
    return {
        "identity": "unavailable",
        "appearance": "unavailable",
        "background": "unavailable",
        "non_target": "unavailable",
        "event": "unavailable",
        "source_bind": "unavailable",
        "inverse": "unavailable",
        "camera": "diagnostic_only",
        "technical": "diagnostic_only",
        "temporal_consistency": "diagnostic_only",
    }


# Read-only snapshots for callers and documentation.  They are deliberately
# not consumed by build/validation, so rebinding a module global cannot weaken
# the emitted or accepted contract.
AUTHORITY = MappingProxyType(_authority_contract())
INPUT_CLOSURE = MappingProxyType(_input_closure_contract())


class SAICExact81DiagnosticError(RuntimeError):
    """A media, runtime, numerical, serialization, or authority check failed."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise SAICExact81DiagnosticError("diagnostic is not canonical JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SAICExact81DiagnosticError(f"{label} must be lowercase SHA-256")
    return value


def _plain_absolute_file(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise SAICExact81DiagnosticError(f"{label} must be absolute")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise SAICExact81DiagnosticError(f"{label} is unavailable") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise SAICExact81DiagnosticError(f"{label} must be one non-symlink file")
    return path.resolve(strict=True)


def _file_signature(row: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        row.st_dev,
        row.st_ino,
        row.st_size,
        row.st_mtime_ns,
        row.st_ctime_ns,
    )


def _stable_file_binding(
    value: str | Path, *, expected_sha256: str, label: str
) -> dict[str, Any]:
    path = _plain_absolute_file(value, label=label)
    expected = _sha(expected_sha256, label=f"{label} expected hash")
    before = path.lstat()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    digest = hashlib.sha256()
    try:
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or _file_signature(opened) != _file_signature(before)
            ):
                raise SAICExact81DiagnosticError(f"{label} changed before hashing")
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise SAICExact81DiagnosticError(f"failed to hash {label}") from error
    after = path.lstat()
    observed = digest.hexdigest()
    if (
        not stat.S_ISREG(after.st_mode)
        or _file_signature(before) != _file_signature(after)
    ):
        raise SAICExact81DiagnosticError(f"{label} changed while hashing")
    if observed != expected:
        raise SAICExact81DiagnosticError(f"{label} hash differs")
    return {"path": str(path), "sha256": observed, "bytes": int(opened.st_size)}


@contextmanager
def _verified_media_snapshot(
    value: str | Path, *, expected_sha256: str, label: str
):
    """Yield a private snapshot containing exactly the hash-bound media bytes.

    Decoders and the geometry implementation open paths themselves.  Measuring
    the caller's path after a preflight hash would therefore leave a TOCTOU
    window.  Copying from the same verified descriptor closes that window; a
    postflight check additionally requires the caller path to return to the
    same content before a receipt is emitted.
    """

    path = _plain_absolute_file(value, label=label)
    expected = _sha(expected_sha256, label=f"{label} expected hash")
    before = path.lstat()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _file_signature(opened) != _file_signature(before)
        ):
            raise SAICExact81DiagnosticError(f"{label} changed before snapshot")
        with tempfile.TemporaryDirectory(prefix="saic-exact81-media-") as directory:
            suffix = path.suffix if path.suffix else ".bin"
            snapshot = Path(directory) / f"{label.replace(' ', '-')}{suffix}"
            output_flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
            )
            output = os.open(snapshot, output_flags, 0o400)
            digest = hashlib.sha256()
            copied = 0
            try:
                while True:
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    offset = 0
                    while offset < len(chunk):
                        offset += os.write(output, chunk[offset:])
                    copied += len(chunk)
                os.fsync(output)
            finally:
                os.close(output)
            after = path.lstat()
            if (
                not stat.S_ISREG(after.st_mode)
                or _file_signature(before) != _file_signature(after)
            ):
                raise SAICExact81DiagnosticError(f"{label} changed during snapshot")
            observed = digest.hexdigest()
            if observed != expected:
                raise SAICExact81DiagnosticError(f"{label} hash differs")
            binding = {
                "path": str(path),
                "sha256": observed,
                "bytes": int(copied),
            }
            try:
                yield binding, snapshot
            finally:
                _stable_file_binding(
                    path,
                    expected_sha256=observed,
                    label=f"{label} postflight",
                )
    except OSError as error:
        raise SAICExact81DiagnosticError(f"failed to snapshot {label}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _tool_identity(name: str) -> dict[str, str]:
    found = shutil.which(name)
    if found is None:
        raise SAICExact81DiagnosticError(f"required executable is unavailable: {name}")
    path = _plain_absolute_file(Path(found).resolve(), label=name)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    completed = subprocess.run(
        (str(path), "-version"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise SAICExact81DiagnosticError(f"cannot identify executable: {name}")
    first_line = completed.stdout.decode("utf-8", errors="replace").splitlines()
    if not first_line:
        raise SAICExact81DiagnosticError(f"empty executable identity: {name}")
    return {"path": str(path), "sha256": digest, "version_line": first_line[0]}


def _runtime_identity() -> dict[str, Any]:
    implementation_path = _own_implementation_path()
    geometry_path = _geometry_implementation_path()
    decoder_path = _decoder_implementation_path()
    return {
        "python": platform.python_version(),
        "numpy": str(np.__version__),
        "opencv": str(cv2.__version__),
        "implementation_path": str(implementation_path),
        "implementation_sha256": hashlib.sha256(implementation_path.read_bytes()).hexdigest(),
        "geometry_path": str(geometry_path),
        "geometry_sha256": hashlib.sha256(geometry_path.read_bytes()).hexdigest(),
        "decoded_evaluator_path": str(decoder_path),
        "decoded_evaluator_sha256": hashlib.sha256(decoder_path.read_bytes()).hexdigest(),
        "ffmpeg": _tool_identity("ffmpeg"),
        "ffprobe": _tool_identity("ffprobe"),
    }


def _postflight_runtime_identity(runtime: Mapping[str, Any]) -> None:
    bindings = (
        ("implementation_path", "implementation_sha256", "diagnostic implementation"),
        ("geometry_path", "geometry_sha256", "geometry implementation"),
        (
            "decoded_evaluator_path",
            "decoded_evaluator_sha256",
            "decoded evaluator implementation",
        ),
    )
    try:
        for path_key, hash_key, label in bindings:
            _stable_file_binding(
                runtime[path_key], expected_sha256=runtime[hash_key], label=label
            )
        for tool in ("ffmpeg", "ffprobe"):
            _stable_file_binding(
                runtime[tool]["path"],
                expected_sha256=runtime[tool]["sha256"],
                label=f"{tool} executable",
            )
    except (KeyError, TypeError) as error:
        raise SAICExact81DiagnosticError("runtime identity is malformed") from error


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise SAICExact81DiagnosticError(f"{label} must be a real number, not bool/text")
    result = float(value)
    if not math.isfinite(result):
        raise SAICExact81DiagnosticError(f"{label} is non-finite")
    return result


def _metric_number(value: Any, *, label: str) -> int | float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise SAICExact81DiagnosticError(f"{label} must be numeric, not bool/text")
    if isinstance(value, numbers.Integral):
        return int(value)
    return _finite(value, label=label)


def _decode_media(path: Path, runtime: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    frames, metadata = decode_exact81_rgb24(
        path,
        ffmpeg=runtime["ffmpeg"]["path"],
        ffprobe=runtime["ffprobe"]["path"],
    )
    if (
        len(frames) != _frame_count()
        or type(metadata.get("frame_count")) is not int
        or metadata.get("frame_count") != _frame_count()
        or type(metadata.get("fps")) is not int
        or metadata.get("fps") != _fps()
    ):
        raise SAICExact81DiagnosticError("decoded media is not exact81@25fps")
    height = metadata.get("height")
    width = metadata.get("width")
    if type(height) is not int or type(width) is not int or height <= 0 or width <= 0:
        raise SAICExact81DiagnosticError("decoded media geometry differs")
    frame_size = height * width * 3
    if any(type(frame) is not bytes or len(frame) != frame_size for frame in frames):
        raise SAICExact81DiagnosticError("decoded RGB frame bytes differ")
    decoded_hash = hashlib.sha256(b"".join(frames)).hexdigest()
    claimed_hash = _sha(
        metadata.get("decoded_rgb24_sha256"), label="decoded RGB hash"
    )
    if claimed_hash != decoded_hash:
        raise SAICExact81DiagnosticError("decoder RGB hash differs from decoded bytes")
    array = np.stack(
        [np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 3) for frame in frames]
    )
    decoder_contract = metadata.get("decoder_contract")
    if type(decoder_contract) is not str or not decoder_contract:
        raise SAICExact81DiagnosticError("decoder contract must be a non-empty string")
    return array, {
        "decoded_rgb24_sha256": decoded_hash,
        "frame_count": _frame_count(),
        "fps_numerator": _fps(),
        "fps_denominator": 1,
        "width": width,
        "height": height,
        "decoder_contract": decoder_contract,
    }


def _percentile(values: np.ndarray, q: float) -> float:
    return _finite(np.percentile(values, q), label=f"p{q:g}")


def _motion_diagnostics(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = MotionConfig(analysis_frames=_frame_count())
    analysis = analyze_video(path, config)
    metrics = analysis.metrics.to_dict()
    if (
        type(metrics.get("sampled_frames")) is not int
        or metrics.get("sampled_frames") != _frame_count()
        or type(metrics.get("source_frame_count")) is not int
        or metrics.get("source_frame_count") != _frame_count()
        or _finite(metrics.get("source_fps"), label="geometry source fps") != _fps()
        or len(analysis.frame_times) != _frame_count()
        or analysis.raw_flows.shape[0] != _transition_count()
        or analysis.global_flows.shape[0] != _transition_count()
        or analysis.residual_flows.shape[0] != _transition_count()
    ):
        raise SAICExact81DiagnosticError("geometry runtime did not analyze exact81/80 transitions")
    frame_times = np.asarray(analysis.frame_times, dtype=np.float64)
    dt = np.diff(frame_times)
    if dt.shape != (_transition_count(),) or np.any(~np.isfinite(dt)) or np.any(dt <= 0):
        raise SAICExact81DiagnosticError("geometry frame times differ")
    expected_dt = 1.0 / _fps()
    # geometry intentionally stores frame times as float32, so exact 1/25
    # cadence incurs sub-microsecond representation error near frame 80.
    if not bool(np.allclose(dt, expected_dt, rtol=0.0, atol=1.0e-6)):
        raise SAICExact81DiagnosticError("geometry cadence is not exact25fps")
    width = float(analysis.frames_gray.shape[2])
    transitions: list[dict[str, Any]] = []
    for index in range(_transition_count()):
        scale = width * dt[index]
        raw = np.asarray(analysis.raw_flows[index], dtype=np.float64) / scale
        global_flow = np.asarray(analysis.global_flows[index], dtype=np.float64) / scale
        residual = np.asarray(analysis.residual_flows[index], dtype=np.float64) / scale
        if not all(bool(np.isfinite(item).all()) for item in (raw, global_flow, residual)):
            raise SAICExact81DiagnosticError("geometry flow contains non-finite values")
        raw_speed = np.linalg.norm(raw, axis=-1)
        global_speed = np.linalg.norm(global_flow, axis=-1)
        residual_speed = np.linalg.norm(residual, axis=-1)
        transitions.append(
            {
                "from_frame": index,
                "to_frame": index + 1,
                "dt_seconds": _finite(dt[index], label="transition dt"),
                "global_mean_xy_widths_per_second": [
                    _finite(global_flow[..., 0].mean(), label="global mean x"),
                    _finite(global_flow[..., 1].mean(), label="global mean y"),
                ],
                "global_spatial_std_xy_widths_per_second": [
                    _finite(global_flow[..., 0].std(), label="global std x"),
                    _finite(global_flow[..., 1].std(), label="global std y"),
                ],
                "raw_speed_mean": _finite(raw_speed.mean(), label="raw speed mean"),
                "global_speed_mean": _finite(
                    global_speed.mean(), label="global speed mean"
                ),
                "global_speed_p90": _percentile(global_speed, 90.0),
                "residual_mean_xy_widths_per_second": [
                    _finite(residual[..., 0].mean(), label="residual mean x"),
                    _finite(residual[..., 1].mean(), label="residual mean y"),
                ],
                "residual_speed_mean": _finite(
                    residual_speed.mean(), label="residual speed mean"
                ),
                "residual_speed_p90": _percentile(residual_speed, 90.0),
                "residual_speed_p99": _percentile(residual_speed, 99.0),
                "residual_active_pixel_fraction": _finite(
                    np.mean(residual_speed >= config.active_speed_threshold),
                    label="residual active fraction",
                ),
            }
        )
    summary = {
        "motion_config": {
            "analysis_frames": _frame_count(),
            "resize_width": config.resize_width,
            "active_speed_threshold": config.active_speed_threshold,
        },
        "motion_label_diagnostic": analysis.label,
        "metrics": {
            key: _metric_number(value, label=f"geometry metric {key}")
            for key, value in metrics.items()
        },
        "sampled_frame_count": _frame_count(),
        "transition_count": _transition_count(),
    }
    return summary, transitions


def _technical_diagnostics(frames: np.ndarray) -> dict[str, Any]:
    if frames.dtype != np.uint8 or frames.shape[0] != _frame_count() or frames.ndim != 4:
        raise SAICExact81DiagnosticError("technical input is not 81 uint8 RGB frames")
    value = frames.astype(np.float32) / 255.0
    dx = value[:, :, 1:, :] - value[:, :, :-1, :]
    dy = value[:, 1:, :, :] - value[:, :-1, :, :]
    sharpness = 0.5 * (np.mean(dx * dx, axis=(1, 2, 3)) + np.mean(dy * dy, axis=(1, 2, 3)))
    clipped = np.logical_or(value <= (2.0 / 255.0), value >= (253.0 / 255.0))
    exposure = 1.0 - np.mean(clipped, axis=(1, 2, 3))
    frame_step = np.mean(np.abs(value[1:] - value[:-1]), axis=(1, 2, 3))
    rgb_mean = np.mean(value, axis=(1, 2, 3))
    second = rgb_mean[2:] - 2.0 * rgb_mean[1:-1] + rgb_mean[:-2]
    arrays = (sharpness, exposure, frame_step, rgb_mean, second)
    if any(np.any(~np.isfinite(item)) for item in arrays):
        raise SAICExact81DiagnosticError("technical diagnostic is non-finite")
    return {
        "frame_count": _frame_count(),
        "transition_count": _transition_count(),
        "sharpness_by_frame": [float(item) for item in sharpness],
        "exposure_score_by_frame": [float(item) for item in exposure],
        "mean_absolute_step_by_transition": [float(item) for item in frame_step],
        "global_rgb_mean_by_frame": [float(item) for item in rgb_mean],
        "global_rgb_second_difference": [float(item) for item in second],
        "summary": {
            "sharpness_mean": _finite(sharpness.mean(), label="sharpness mean"),
            "sharpness_p10": _percentile(sharpness, 10.0),
            "exposure_mean": _finite(exposure.mean(), label="exposure mean"),
            "exposure_min": _finite(exposure.min(), label="exposure min"),
            "frame_step_mean": _finite(frame_step.mean(), label="frame step mean"),
            "frame_step_p10": _percentile(frame_step, 10.0),
            "global_second_difference_abs_mean": _finite(
                np.abs(second).mean(), label="second difference mean"
            ),
        },
    }


def _camera_comparison(
    source: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if len(source) != _transition_count() or len(candidate) != _transition_count():
        raise SAICExact81DiagnosticError("camera trajectory is not full80")
    source_xy = np.asarray(
        [row["global_mean_xy_widths_per_second"] for row in source], dtype=np.float64
    )
    candidate_xy = np.asarray(
        [row["global_mean_xy_widths_per_second"] for row in candidate], dtype=np.float64
    )
    delta = candidate_xy - source_xy
    l2 = np.linalg.norm(delta, axis=1)
    dt = np.asarray([row["dt_seconds"] for row in source], dtype=np.float64)
    source_endpoint = np.sum(source_xy * dt[:, None], axis=0)
    candidate_endpoint = np.sum(candidate_xy * dt[:, None], axis=0)
    source_speed = np.asarray([row["global_speed_mean"] for row in source])
    candidate_speed = np.asarray([row["global_speed_mean"] for row in candidate])
    return {
        "transition_count": _transition_count(),
        "candidate_minus_source_global_mean_xy_by_transition": delta.tolist(),
        "global_mean_xy_l2_difference_by_transition": l2.tolist(),
        "global_mean_xy_l2_difference_mean": _finite(l2.mean(), label="camera delta mean"),
        "global_mean_xy_l2_difference_p90": _percentile(l2, 90.0),
        "global_mean_xy_l2_difference_max": _finite(l2.max(), label="camera delta max"),
        "global_speed_mean_absolute_difference": _finite(
            np.mean(np.abs(candidate_speed - source_speed)), label="camera speed difference"
        ),
        "source_cumulative_global_xy_widths": source_endpoint.tolist(),
        "candidate_cumulative_global_xy_widths": candidate_endpoint.tolist(),
        "cumulative_global_endpoint_l2_difference": _finite(
            np.linalg.norm(candidate_endpoint - source_endpoint),
            label="camera cumulative endpoint difference",
        ),
        "interpretation": "diagnostic_only_no_absolute_camera_pass_threshold",
    }


def _technical_comparison(
    source: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    source_summary = source["summary"]
    candidate_summary = candidate["summary"]
    epsilon = 1.0e-12
    source_sharp = source_summary["sharpness_mean"]
    candidate_sharp = candidate_summary["sharpness_mean"]
    sharpness_retention = (
        min(candidate_sharp / source_sharp, 1.0)
        if source_sharp > epsilon
        else float(candidate_sharp <= epsilon)
    )
    source_step = source_summary["frame_step_mean"]
    candidate_step = candidate_summary["frame_step_mean"]
    nonfreeze = (
        min(candidate_step / source_step, 1.0)
        if source_step > epsilon
        else float(candidate_step <= epsilon)
    )
    source_second = np.asarray(source["global_rgb_second_difference"], dtype=np.float64)
    candidate_second = np.asarray(candidate["global_rgb_second_difference"], dtype=np.float64)
    flicker = math.exp(-10.0 * float(np.mean(np.abs(candidate_second - source_second))))
    terms = np.asarray(
        [sharpness_retention, candidate_summary["exposure_mean"], nonfreeze, flicker],
        dtype=np.float64,
    )
    score = math.exp(float(np.mean(np.log(np.maximum(terms, epsilon)))))
    return {
        "sharpness_retention_diagnostic": _finite(
            sharpness_retention, label="sharpness retention"
        ),
        "candidate_exposure_diagnostic": _finite(
            candidate_summary["exposure_mean"], label="candidate exposure"
        ),
        "nonfreeze_retention_diagnostic": _finite(nonfreeze, label="nonfreeze"),
        "global_flicker_agreement_diagnostic": _finite(flicker, label="flicker"),
        "geometric_mean_technical_diagnostic": _finite(score, label="technical score"),
        "interpretation": "diagnostic_only_no_absolute_technical_pass_threshold",
    }


def build_diagnostic(
    *,
    source_video: str | Path,
    expected_source_sha256: str,
    candidate_video: str | Path,
    expected_candidate_sha256: str,
) -> dict[str, Any]:
    """Measure media directly; there are no caller-supplied score/authority slots."""

    with _verified_media_snapshot(
        source_video,
        expected_sha256=expected_source_sha256,
        label="source video",
    ) as (source, source_snapshot):
        with _verified_media_snapshot(
            candidate_video,
            expected_sha256=expected_candidate_sha256,
            label="candidate video",
        ) as (candidate, candidate_snapshot):
            return _build_diagnostic_from_snapshots(
                source=source,
                source_snapshot=source_snapshot,
                candidate=candidate,
                candidate_snapshot=candidate_snapshot,
            )


def _build_diagnostic_from_snapshots(
    *,
    source: Mapping[str, Any],
    source_snapshot: Path,
    candidate: Mapping[str, Any],
    candidate_snapshot: Path,
) -> dict[str, Any]:
    runtime = _runtime_identity()
    source_rgb, source_decode = _decode_media(source_snapshot, runtime)
    candidate_rgb, candidate_decode = _decode_media(candidate_snapshot, runtime)
    source_motion, source_transitions = _motion_diagnostics(source_snapshot)
    candidate_motion, candidate_transitions = _motion_diagnostics(candidate_snapshot)
    source_technical = _technical_diagnostics(source_rgb)
    candidate_technical = _technical_diagnostics(candidate_rgb)
    _postflight_runtime_identity(runtime)
    body = {
        "schema_version": _schema_version(),
        "media": {
            "source": {**source, "decode": source_decode},
            "candidate": {**candidate, "decode": candidate_decode},
        },
        "runtime": runtime,
        "input_closure": _input_closure_contract(),
        "availability": _availability_contract(),
        "source": {
            "motion_summary": source_motion,
            "transition_descriptors": source_transitions,
            "technical_full81": source_technical,
        },
        "candidate": {
            "motion_summary": candidate_motion,
            "transition_descriptors": candidate_transitions,
            "technical_full81": candidate_technical,
        },
        "comparisons": {
            "camera_trajectory": _camera_comparison(
                source_transitions, candidate_transitions
            ),
            "scene_cut_ratio_absolute_difference": _finite(
                abs(
                    candidate_motion["metrics"]["scene_cut_ratio"]
                    - source_motion["metrics"]["scene_cut_ratio"]
                ),
                label="scene cut difference",
            ),
            "temporal_energy_cv_absolute_difference": _finite(
                abs(
                    candidate_motion["metrics"]["temporal_energy_cv"]
                    - source_motion["metrics"]["temporal_energy_cv"]
                ),
                label="temporal CV difference",
            ),
            "technical": _technical_comparison(
                source_technical, candidate_technical
            ),
        },
        "authority": _authority_contract(),
        "remaining_gaps": [
            "no_external_dataset_registration_or_provenance",
            "no_identity_or_appearance_model",
            "no_semantic_background_or_non_target_localization",
            "no_qualified_exact81_event_observer",
            "no_wrong_or_dropped_source_binding_rollout",
            "no_real_inverse_rollout",
            "camera_and_technical_thresholds_not_calibrated",
        ],
    }
    return {**body, "diagnostic_digest": object_sha256(body)}


def _validate_permanent_contract(value: Mapping[str, Any]) -> None:
    if type(value.get("schema_version")) is not str or value.get(
        "schema_version"
    ) != _schema_version():
        raise SAICExact81DiagnosticError("diagnostic schema differs")
    if canonical_json_bytes(value.get("authority")) != canonical_json_bytes(
        _authority_contract()
    ):
        raise SAICExact81DiagnosticError("diagnostic cannot acquire authority")
    if canonical_json_bytes(value.get("input_closure")) != canonical_json_bytes(
        _input_closure_contract()
    ):
        raise SAICExact81DiagnosticError("diagnostic input closure differs")
    availability = value.get("availability")
    expected = _availability_contract()
    if canonical_json_bytes(availability) != canonical_json_bytes(expected):
        raise SAICExact81DiagnosticError("diagnostic availability differs")


def _validate_diagnostic_structure(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SAICExact81DiagnosticError("diagnostic root must be an object")
    row = dict(value)
    _validate_permanent_contract(row)
    digest = _sha(row.get("diagnostic_digest"), label="diagnostic digest")
    body = {key: item for key, item in row.items() if key != "diagnostic_digest"}
    if object_sha256(body) != digest:
        raise SAICExact81DiagnosticError("diagnostic digest differs")
    return json.loads(canonical_json_bytes(row).decode("ascii"))


def validate_diagnostic(value: Mapping[str, Any]) -> dict[str, Any]:
    """Replay bound media and reject even correctly re-signed changed metrics."""

    row = _validate_diagnostic_structure(value)
    try:
        source = row["media"]["source"]
        candidate = row["media"]["candidate"]
        replayed = build_diagnostic(
            source_video=source["path"],
            expected_source_sha256=source["sha256"],
            candidate_video=candidate["path"],
            expected_candidate_sha256=candidate["sha256"],
        )
    except (KeyError, TypeError) as error:
        raise SAICExact81DiagnosticError("diagnostic media binding differs") from error
    if canonical_json_bytes(replayed) != canonical_json_bytes(row):
        raise SAICExact81DiagnosticError("diagnostic differs from media replay")
    return row


def write_diagnostic_create_only(path: str | Path, value: Mapping[str, Any]) -> str:
    # A writer is not a validator, but it must never serialize a receipt that
    # weakens the permanent zero-authority contract or has an invalid digest.
    checked = _validate_diagnostic_structure(value)
    output = Path(path)
    if not output.is_absolute() or output == Path("/"):
        raise SAICExact81DiagnosticError("output must be a non-root absolute path")
    parent = output.parent.resolve(strict=True)
    if not parent.is_dir():
        raise SAICExact81DiagnosticError("output parent differs")
    raw = canonical_json_bytes(checked) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(output, flags, 0o400)
    except OSError as error:
        raise SAICExact81DiagnosticError("refusing to overwrite diagnostic") from error
    try:
        written = 0
        while written < len(raw):
            written += os.write(descriptor, raw[written:])
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return hashlib.sha256(raw).hexdigest()


def load_canonical_diagnostic(path: str | Path) -> dict[str, Any]:
    source = _plain_absolute_file(path, label="diagnostic file")
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SAICExact81DiagnosticError("diagnostic file is invalid ASCII JSON") from error
    if not isinstance(value, dict) or raw != canonical_json_bytes(value) + b"\n":
        raise SAICExact81DiagnosticError("diagnostic file is not canonical bytes")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--source-video", required=True)
    build.add_argument("--expected-source-sha256", required=True)
    build.add_argument("--candidate-video", required=True)
    build.add_argument("--expected-candidate-sha256", required=True)
    build.add_argument("--output", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--diagnostic", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        value = build_diagnostic(
            source_video=args.source_video,
            expected_source_sha256=args.expected_source_sha256,
            candidate_video=args.candidate_video,
            expected_candidate_sha256=args.expected_candidate_sha256,
        )
        _validate_diagnostic_structure(value)
        write_diagnostic_create_only(args.output, value)
        return 0
    if args.command == "validate":
        validate_diagnostic(load_canonical_diagnostic(args.diagnostic))
        return 0
    raise SAICExact81DiagnosticError("unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
