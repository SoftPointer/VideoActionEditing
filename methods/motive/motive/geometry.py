"""Camera-compensated motion extraction and conservative video screening.

This module is a deliberately inexpensive stage-0 filter.  It does not replace
Motive's model-gradient attribution.  Its purpose is to avoid spending one
backward pass on clips that are static, dominated by camera motion, corrupted
by cuts, or otherwise uninformative for action learning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class MotionConfig:
    """Configuration expressed in resolution- and frame-rate-aware units.

    Speeds are fractions of frame width per second.  For example, 0.01 means
    one percent of the frame width per second.
    """

    analysis_frames: int = 32
    resize_width: int = 256
    farneback_pyr_scale: float = 0.5
    farneback_levels: int = 4
    farneback_winsize: int = 21
    farneback_iterations: int = 3
    farneback_poly_n: int = 7
    farneback_poly_sigma: float = 1.5
    active_speed_threshold: float = 0.005
    static_residual_p90: float = 0.003
    static_active_fraction: float = 0.025
    camera_raw_speed: float = 0.003
    camera_explained_ratio: float = 0.70
    camera_residual_multiplier: float = 1.75
    max_scene_cut_ratio: float = 0.15
    max_scene_cuts: int = 0
    scene_cut_luma_delta: float = 0.28
    min_frames: int = 3
    eps: float = 1e-8

    def validate(self) -> None:
        if self.analysis_frames < self.min_frames:
            raise ValueError("analysis_frames must be >= min_frames")
        if self.resize_width < 32:
            raise ValueError("resize_width must be >= 32")
        if not 0.0 <= self.camera_explained_ratio <= 1.0:
            raise ValueError("camera_explained_ratio must be in [0, 1]")
        if self.max_scene_cuts < 0:
            raise ValueError("max_scene_cuts must be non-negative")


@dataclass(frozen=True)
class MotionMetrics:
    raw_speed_mean: float
    raw_speed_p90: float
    residual_speed_mean: float
    residual_speed_p90: float
    residual_speed_p99: float
    active_pixel_fraction: float
    active_frame_fraction: float
    camera_explained_ratio: float
    affine_inlier_ratio: float
    scene_cut_ratio: float
    temporal_energy_cv: float
    sampled_frames: int
    duration_seconds: float
    source_fps: float
    source_frame_count: int
    source_width: int
    source_height: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MotionAnalysis:
    """Full result, including arrays used to build an action descriptor."""

    path: Path
    label: str
    metrics: MotionMetrics
    frames_gray: np.ndarray
    frame_times: np.ndarray
    raw_flows: np.ndarray
    global_flows: np.ndarray
    residual_flows: np.ndarray

    def to_record(self) -> dict[str, Any]:
        return {
            "video": str(self.path),
            "motion_label": self.label,
            "motion_metrics": self.metrics.to_dict(),
        }


def _resize_gray(frame: np.ndarray, width: int) -> np.ndarray:
    height, original_width = frame.shape[:2]
    if original_width <= 0 or height <= 0:
        raise ValueError("invalid frame dimensions")
    target_height = max(2, int(round(height * width / original_width)))
    resized = cv2.resize(frame, (width, target_height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


def _uniform_indices(frame_count: int, count: int) -> list[int]:
    if frame_count <= 0:
        return []
    count = min(frame_count, count)
    indices = np.rint(np.linspace(0, frame_count - 1, num=count)).astype(np.int64)
    return list(dict.fromkeys(int(value) for value in indices))


def _read_sampled_frames(
    path: Path,
    config: MotionConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 16.0
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    source_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    source_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    frames: list[np.ndarray] = []
    indices: list[int] = []
    requested = _uniform_indices(frame_count, config.analysis_frames)
    if requested:
        for frame_index in requested:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            frames.append(_resize_gray(frame, config.resize_width))
            indices.append(frame_index)
    else:
        # Some containers expose no reliable frame count.  Decode a bounded
        # prefix rather than silently returning an empty analysis.
        frame_index = 0
        while len(frames) < config.analysis_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(_resize_gray(frame, config.resize_width))
            indices.append(frame_index)
            frame_index += 1
        frame_count = max(frame_count, frame_index)
    capture.release()

    if len(frames) < config.min_frames:
        raise RuntimeError(
            f"Video has only {len(frames)} readable sampled frames; "
            f"need at least {config.min_frames}: {path}"
        )
    if len({frame.shape for frame in frames}) != 1:
        raise RuntimeError(f"Video changes resolution during decoding: {path}")

    frame_times = np.asarray(indices, dtype=np.float32) / float(fps)
    metadata: dict[str, float | int] = {
        "fps": fps,
        "frame_count": frame_count,
        "width": source_width,
        "height": source_height,
    }
    return np.stack(frames), frame_times, metadata


def _dense_flow(previous: np.ndarray, current: np.ndarray, config: MotionConfig) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(
        previous,
        current,
        None,
        config.farneback_pyr_scale,
        config.farneback_levels,
        config.farneback_winsize,
        config.farneback_iterations,
        config.farneback_poly_n,
        config.farneback_poly_sigma,
        cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    ).astype(np.float32)


def _affine_global_flow(
    previous: np.ndarray,
    current: np.ndarray,
    dense_flow: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Estimate background camera flow with a robust partial affine model."""

    points = cv2.goodFeaturesToTrack(
        previous,
        maxCorners=400,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
    )
    matrix: np.ndarray | None = None
    inlier_ratio = 0.0
    if points is not None and len(points) >= 8:
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(
            previous,
            current,
            points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )
        if tracked is not None and status is not None:
            keep = status.reshape(-1).astype(bool)
            source = points.reshape(-1, 2)[keep]
            target = tracked.reshape(-1, 2)[keep]
            if len(source) >= 8:
                matrix, inliers = cv2.estimateAffinePartial2D(
                    source,
                    target,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=2.0,
                    maxIters=2000,
                    confidence=0.99,
                    refineIters=10,
                )
                if inliers is not None and len(inliers):
                    inlier_ratio = float(np.mean(inliers))

    height, width = previous.shape
    if matrix is None or not np.all(np.isfinite(matrix)) or inlier_ratio < 0.35:
        # Median dense flow is a robust translation estimate when background
        # occupies most of the frame.
        translation = np.median(dense_flow.reshape(-1, 2), axis=0)
        global_flow = np.empty_like(dense_flow)
        global_flow[..., 0] = translation[0]
        global_flow[..., 1] = translation[1]
        return global_flow, inlier_ratio

    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    projected_x = matrix[0, 0] * grid_x + matrix[0, 1] * grid_y + matrix[0, 2]
    projected_y = matrix[1, 0] * grid_x + matrix[1, 1] * grid_y + matrix[1, 2]
    return np.stack((projected_x - grid_x, projected_y - grid_y), axis=-1), inlier_ratio


def _safe_percentile(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, percentile))


def classify_motion(metrics: MotionMetrics, config: MotionConfig | None = None) -> str:
    """Classify a clip without deleting or moving it.

    Labels are conservative: ``dynamic_object`` means the clip passed a cheap
    eligibility screen, not that it is a high-influence Motive sample.
    """

    config = config or MotionConfig()
    transition_count = max(metrics.sampled_frames - 1, 1)
    detected_cuts = int(round(metrics.scene_cut_ratio * transition_count))
    if (
        metrics.scene_cut_ratio > config.max_scene_cut_ratio
        or detected_cuts > config.max_scene_cuts
    ):
        return "cut_or_decode_artifact"

    camera_only = (
        metrics.raw_speed_mean >= config.camera_raw_speed
        and metrics.camera_explained_ratio >= config.camera_explained_ratio
        and metrics.residual_speed_p90
        < config.static_residual_p90 * config.camera_residual_multiplier
    )
    if camera_only:
        return "camera_only"

    static = (
        metrics.residual_speed_p90 < config.static_residual_p90
        and metrics.active_pixel_fraction < config.static_active_fraction
    )
    if static:
        return "static"
    return "dynamic_object"


def analyze_video(
    path: str | Path,
    config: MotionConfig | None = None,
) -> MotionAnalysis:
    """Decode a bounded sample and compute camera-compensated dense flow."""

    config = config or MotionConfig()
    config.validate()
    video_path = Path(path).expanduser()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    frames, frame_times, metadata = _read_sampled_frames(video_path, config)
    raw_flows: list[np.ndarray] = []
    global_flows: list[np.ndarray] = []
    residual_flows: list[np.ndarray] = []
    affine_inliers: list[float] = []
    scene_cuts: list[float] = []

    for previous, current in zip(frames[:-1], frames[1:]):
        dense = _dense_flow(previous, current, config)
        global_flow, inlier_ratio = _affine_global_flow(previous, current, dense)
        residual = dense - global_flow
        raw_flows.append(dense)
        global_flows.append(global_flow)
        residual_flows.append(residual)
        affine_inliers.append(inlier_ratio)
        scene_cuts.append(float(np.mean(cv2.absdiff(previous, current))) / 255.0)

    raw = np.stack(raw_flows)
    global_array = np.stack(global_flows)
    residual = np.stack(residual_flows)
    dt = np.maximum(np.diff(frame_times), 1.0 / float(metadata["fps"]))
    width = float(frames.shape[2])
    scale = width * dt[:, None, None]
    raw_speed = np.linalg.norm(raw, axis=-1) / scale
    residual_speed = np.linalg.norm(residual, axis=-1) / scale

    raw_energy = float(np.sum(raw_speed**2))
    residual_energy = float(np.sum(residual_speed**2))
    camera_explained = float(
        np.clip(1.0 - residual_energy / (raw_energy + config.eps), 0.0, 1.0)
    )
    frame_energy = np.mean(residual_speed, axis=(1, 2))
    frame_energy_mean = float(np.mean(frame_energy))
    temporal_cv = float(np.std(frame_energy) / (frame_energy_mean + config.eps))
    scene_cut_ratio = float(
        np.mean(np.asarray(scene_cuts) >= config.scene_cut_luma_delta)
    )
    active_pixels = residual_speed >= config.active_speed_threshold
    active_frames = np.mean(residual_speed, axis=(1, 2)) >= config.static_residual_p90

    duration = float(frame_times[-1] - frame_times[0])
    metrics = MotionMetrics(
        raw_speed_mean=float(np.mean(raw_speed)),
        raw_speed_p90=_safe_percentile(raw_speed, 90.0),
        residual_speed_mean=float(np.mean(residual_speed)),
        residual_speed_p90=_safe_percentile(residual_speed, 90.0),
        residual_speed_p99=_safe_percentile(residual_speed, 99.0),
        active_pixel_fraction=float(np.mean(active_pixels)),
        active_frame_fraction=float(np.mean(active_frames)),
        camera_explained_ratio=camera_explained,
        affine_inlier_ratio=float(np.mean(affine_inliers)),
        scene_cut_ratio=scene_cut_ratio,
        temporal_energy_cv=temporal_cv,
        sampled_frames=int(len(frames)),
        duration_seconds=duration,
        source_fps=float(metadata["fps"]),
        source_frame_count=int(metadata["frame_count"]),
        source_width=int(metadata["width"]),
        source_height=int(metadata["height"]),
    )
    label = classify_motion(metrics, config)
    return MotionAnalysis(
        path=video_path,
        label=label,
        metrics=metrics,
        frames_gray=frames,
        frame_times=frame_times,
        raw_flows=raw,
        global_flows=global_array,
        residual_flows=residual,
    )


def normalize_motion_magnitude(
    magnitude: np.ndarray,
    *,
    mode: str = "robust",
    low_percentile: float = 5.0,
    high_percentile: float = 95.0,
    absolute_gate: float | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Map motion magnitude to [0, 1] with an explicit zero-motion guard.

    ``mode="motive"`` reproduces clip-wise min-max normalization from the
    paper.  ``mode="robust"`` is recommended for noisy, mostly-static corpora.
    """

    values = np.asarray(magnitude, dtype=np.float32)
    if values.size == 0:
        raise ValueError("magnitude must not be empty")
    if absolute_gate is not None and float(np.percentile(values, 99.0)) < absolute_gate:
        return np.zeros_like(values)
    if mode == "motive":
        low = float(np.min(values))
        high = float(np.max(values))
    elif mode == "robust":
        low = float(np.percentile(values, low_percentile))
        # Quantiling over all pixels erases a real actor occupying less than
        # (100-high_percentile)% of the clip. Estimate the high anchor only
        # from values measurably above the robust floor.
        active_values = values[values > low + eps]
        high = (
            float(np.percentile(active_values, high_percentile))
            if active_values.size
            else low
        )
    else:
        raise ValueError(f"unsupported normalization mode: {mode}")
    if high - low <= eps:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low + eps), 0.0, 1.0)


def delta_motion_mask(
    source_flow: np.ndarray,
    target_flow: np.ndarray,
    *,
    mode: str = "robust",
    absolute_gate: float | None = None,
) -> np.ndarray:
    """Create an action-edit mask from vector flow change, not magnitude alone."""

    source = np.asarray(source_flow, dtype=np.float32)
    target = np.asarray(target_flow, dtype=np.float32)
    if source.shape != target.shape or source.shape[-1] != 2:
        raise ValueError(
            "source_flow and target_flow must share shape [..., H, W, 2]"
        )
    delta = np.linalg.norm(target - source, axis=-1)
    return normalize_motion_magnitude(
        delta,
        mode=mode,
        absolute_gate=absolute_gate,
    )
