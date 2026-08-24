#!/usr/bin/env python3
"""Fail-closed decoded-video validity gate for action-editing evaluation.

This module intentionally has no model, training, or video-container
dependencies.  It consumes the decoded RGB tensor that an inference/evaluation
process already holds and decides whether that tensor is safe to publish for
visual review.

The gate is specifically designed not to be fooled by a source-clamped frame
zero.  Every decoded frame is inspected, frame-to-frame continuity is measured
after the phase-zero transition, and four temporal segments are reported.  A
normal first frame followed by black frames, coloured spatial noise, temporal
noise, frozen decoder output, non-finite pixels, or an incomplete decode is a
hard failure.

Inputs are either ``uint8`` RGB in [0, 255] or floating RGB in [0, 1], with
shape ``[frames, height, width, 3]``.  The default contract is exact81.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Optional

import numpy as np


SCHEMA_VERSION = "bernini-action-editing-visual-validity-gate-v1"


class VisualValidityError(RuntimeError):
    """Raised when a caller requires a publishable decoded video."""


class VisualValidityInputError(ValueError):
    """Raised internally for a malformed or incomplete decoded tensor."""


@dataclass(frozen=True)
class VisualValidityThresholds:
    """Predeclared engineering limits for catastrophic decode failures.

    These are not action-quality or identity-recognition thresholds.  They
    only reject media that is structurally invalid or visibly degenerate.
    Thresholds are expressed on RGB values normalized to [0, 1].
    """

    expected_frame_count: int = 81
    minimum_height: int = 16
    minimum_width: int = 16
    temporal_segment_count: int = 4

    dark_pixel_luma: float = 0.04
    bright_pixel_luma: float = 0.96
    catastrophic_dark_pixel_fraction: float = 0.75
    catastrophic_bright_pixel_fraction: float = 0.97
    relative_dark_luma_ratio: float = 0.45
    minimum_post_onset_median_luma_ratio: float = 0.45
    maximum_post_onset_median_luma_ratio: float = 3.0

    flat_frame_luma_std: float = 0.005
    flat_frame_neighbor_l1: float = 0.003
    spatial_noise_lag: int = 4
    spatial_noise_neighbor_l1: float = 0.16
    spatial_noise_max_channel_correlation: float = 0.65
    spatial_noise_laplacian_l1: float = 0.12

    maximum_post_onset_temporal_median_l1: float = 0.065
    catastrophic_single_transition_l1: float = 0.55
    maximum_phase0_to_frame1_l1: float = 0.38
    frozen_transition_l1: float = 0.0005
    frozen_transition_fraction: float = 0.95


DEFAULT_THRESHOLDS = VisualValidityThresholds()


def _finite_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise VisualValidityInputError("computed visual metric is not finite")
    return result


def _normalise_video(
    value: Any,
    *,
    label: str,
    thresholds: VisualValidityThresholds,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise VisualValidityInputError(f"{label} must be a numpy ndarray")
    if value.ndim != 4 or value.shape[-1] != 3:
        raise VisualValidityInputError(
            f"{label} must have shape [frames,height,width,3], got {value.shape}"
        )
    frame_count, height, width, _ = (int(item) for item in value.shape)
    if frame_count != thresholds.expected_frame_count:
        raise VisualValidityInputError(
            f"{label} must be exact{thresholds.expected_frame_count}, got {frame_count}"
        )
    if height < thresholds.minimum_height or width < thresholds.minimum_width:
        raise VisualValidityInputError(
            f"{label} geometry is too small: {height}x{width}"
        )
    if np.issubdtype(value.dtype, np.bool_) or np.issubdtype(
        value.dtype, np.complexfloating
    ):
        raise VisualValidityInputError(f"{label} dtype is not real RGB: {value.dtype}")

    if np.issubdtype(value.dtype, np.integer):
        minimum = int(np.min(value))
        maximum = int(np.max(value))
        if minimum < 0 or maximum > 255:
            raise VisualValidityInputError(
                f"{label} integer pixels must be in [0,255], got [{minimum},{maximum}]"
            )
        result = value.astype(np.float32) / 255.0
    elif np.issubdtype(value.dtype, np.floating):
        result = value.astype(np.float32, copy=False)
        if not bool(np.all(np.isfinite(result))):
            raise VisualValidityInputError(f"{label} contains NaN or infinity")
        minimum = float(np.min(result))
        maximum = float(np.max(result))
        tolerance = 1e-6
        if minimum < -tolerance or maximum > 1.0 + tolerance:
            raise VisualValidityInputError(
                f"{label} float pixels must be in [0,1], got [{minimum},{maximum}]"
            )
        if minimum < 0.0 or maximum > 1.0:
            result = np.clip(result, 0.0, 1.0)
    else:
        raise VisualValidityInputError(f"{label} dtype is not numeric RGB: {value.dtype}")

    if not bool(np.all(np.isfinite(result))):
        raise VisualValidityInputError(f"{label} contains non-finite decoded pixels")
    return np.ascontiguousarray(result)


def _minimum_channel_correlation(frame: np.ndarray) -> float:
    pixels = frame.reshape(-1, 3).astype(np.float64)
    centred = pixels - np.mean(pixels, axis=0, keepdims=True)
    energy = np.sqrt(np.sum(centred * centred, axis=0))
    correlations: list[float] = []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        denominator = float(energy[left] * energy[right])
        # A constant channel is not evidence of independently coloured noise.
        correlation = 1.0 if denominator <= 1e-12 else float(
            np.dot(centred[:, left], centred[:, right]) / denominator
        )
        correlations.append(max(-1.0, min(1.0, correlation)))
    return min(correlations)


def _neighbor_l1(frame: np.ndarray, lag: int) -> float:
    horizontal = float(np.mean(np.abs(frame[:, lag:, :] - frame[:, :-lag, :])))
    vertical = float(np.mean(np.abs(frame[lag:, :, :] - frame[:-lag, :, :])))
    return 0.5 * (horizontal + vertical)


def _laplacian_l1(frame: np.ndarray) -> float:
    centre = frame[1:-1, 1:-1, :]
    residual = (
        4.0 * centre
        - frame[:-2, 1:-1, :]
        - frame[2:, 1:-1, :]
        - frame[1:-1, :-2, :]
        - frame[1:-1, 2:, :]
    )
    return float(np.mean(np.abs(residual)) / 4.0)


def _frame_metrics(
    frame: np.ndarray,
    *,
    frame_index: int,
    phase0_luma_mean: Optional[float],
    thresholds: VisualValidityThresholds,
) -> dict[str, Any]:
    luma = np.tensordot(
        frame,
        np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32),
        axes=([-1], [0]),
    )
    luma_mean = _finite_float(np.mean(luma))
    luma_std = _finite_float(np.std(luma))
    dark_fraction = _finite_float(
        np.mean(luma <= thresholds.dark_pixel_luma)
    )
    bright_fraction = _finite_float(
        np.mean(luma >= thresholds.bright_pixel_luma)
    )
    lag = min(
        thresholds.spatial_noise_lag,
        int(frame.shape[0]) - 1,
        int(frame.shape[1]) - 1,
    )
    neighbor_lag1 = _finite_float(_neighbor_l1(frame, 1))
    neighbor_lagged = _finite_float(_neighbor_l1(frame, lag))
    laplacian = _finite_float(_laplacian_l1(frame))
    channel_correlation = _finite_float(_minimum_channel_correlation(frame))
    chroma_amplitude = _finite_float(
        np.mean(np.max(frame, axis=-1) - np.min(frame, axis=-1))
    )
    luma_ratio = (
        1.0
        if phase0_luma_mean is None
        else luma_mean / max(phase0_luma_mean, 1e-8)
    )

    is_black = dark_fraction >= thresholds.catastrophic_dark_pixel_fraction
    if phase0_luma_mean is not None:
        is_black = is_black or (
            luma_ratio <= thresholds.relative_dark_luma_ratio
            and dark_fraction >= 0.50
        )
    is_whiteout = bright_fraction >= thresholds.catastrophic_bright_pixel_fraction
    is_flat = (
        luma_std <= thresholds.flat_frame_luma_std
        and neighbor_lag1 <= thresholds.flat_frame_neighbor_l1
    )
    is_spatial_noise = (
        neighbor_lagged >= thresholds.spatial_noise_neighbor_l1
        and channel_correlation <= thresholds.spatial_noise_max_channel_correlation
    ) or laplacian >= thresholds.spatial_noise_laplacian_l1

    return {
        "frame_index": frame_index,
        "luma_mean": luma_mean,
        "luma_std": luma_std,
        "luma_ratio_to_phase0": _finite_float(luma_ratio),
        "dark_pixel_fraction": dark_fraction,
        "bright_pixel_fraction": bright_fraction,
        "chroma_amplitude_mean": chroma_amplitude,
        "neighbor_l1_lag1": neighbor_lag1,
        "neighbor_l1_lagged": neighbor_lagged,
        "laplacian_l1": laplacian,
        "minimum_channel_correlation": channel_correlation,
        "is_black": bool(is_black),
        "is_whiteout": bool(is_whiteout),
        "is_flat": bool(is_flat),
        "is_spatial_color_noise": bool(is_spatial_noise),
    }


def _global_ssim(left: np.ndarray, right: np.ndarray) -> float:
    x = left.astype(np.float64)
    y = right.astype(np.float64)
    c1, c2 = 0.01**2, 0.03**2
    numerator = (2.0 * x.mean() * y.mean() + c1) * (
        2.0 * np.mean((x - x.mean()) * (y - y.mean())) + c2
    )
    denominator = (x.mean() ** 2 + y.mean() ** 2 + c1) * (
        x.var() + y.var() + c2
    )
    return _finite_float(numerator / denominator) if denominator > 0 else 1.0


def _segment_ranges(frame_count: int, segment_count: int) -> list[tuple[int, int]]:
    if segment_count <= 0 or segment_count > frame_count:
        raise VisualValidityInputError("temporal_segment_count is invalid")
    boundaries = np.array_split(np.arange(frame_count), segment_count)
    return [(int(segment[0]), int(segment[-1])) for segment in boundaries]


def _failure(code: str, message: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {"code": code, "message": message, "evidence": dict(evidence)}


def _input_failure_report(
    error: Exception,
    thresholds: VisualValidityThresholds,
) -> dict[str, Any]:
    failure = _failure(
        "input_contract_violation",
        "decoded video did not satisfy the exact RGB input contract",
        {"error": str(error)},
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "fail",
        "passed": False,
        "publishable": False,
        "fail_closed": True,
        "input_contract_passed": False,
        "phase0_only": False,
        "metric_scope": "all decoded frames plus all temporal transitions",
        "thresholds": asdict(thresholds),
        "failure_codes": [failure["code"]],
        "failures": [failure],
    }


def evaluate_visual_validity(
    frames: Any,
    *,
    reference_frames: Optional[Any] = None,
    thresholds: VisualValidityThresholds = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Return a JSON-safe, fail-closed validity report for decoded RGB frames.

    ``reference_frames`` is optional.  When supplied, source-preservation
    proxies are computed for frame zero, all 81 frames, every post-onset frame,
    and each temporal segment.  They are intentionally reported rather than
    treated as an identity recognizer.
    """

    try:
        video = _normalise_video(frames, label="candidate", thresholds=thresholds)
        reference = None
        if reference_frames is not None:
            reference = _normalise_video(
                reference_frames, label="reference", thresholds=thresholds
            )
            if reference.shape != video.shape:
                raise VisualValidityInputError(
                    f"reference shape {reference.shape} differs from candidate {video.shape}"
                )
        ranges = _segment_ranges(
            int(video.shape[0]), thresholds.temporal_segment_count
        )
    except (TypeError, ValueError, FloatingPointError) as error:
        return _input_failure_report(error, thresholds)

    phase0_probe = _frame_metrics(
        video[0],
        frame_index=0,
        phase0_luma_mean=None,
        thresholds=thresholds,
    )
    phase0_luma_mean = float(phase0_probe["luma_mean"])
    frame_metrics = [phase0_probe]
    frame_metrics.extend(
        _frame_metrics(
            video[index],
            frame_index=index,
            phase0_luma_mean=phase0_luma_mean,
            thresholds=thresholds,
        )
        for index in range(1, int(video.shape[0]))
    )

    temporal_steps = np.asarray(
        [
            float(np.mean(np.abs(video[index] - video[index - 1])))
            for index in range(1, int(video.shape[0]))
        ],
        dtype=np.float64,
    )
    if not bool(np.all(np.isfinite(temporal_steps))):
        return _input_failure_report(
            VisualValidityInputError("temporal metrics contain non-finite values"),
            thresholds,
        )
    post_temporal_steps = temporal_steps[1:]
    post_frame_metrics = frame_metrics[1:]
    post_luma_ratio_median = _finite_float(
        np.median([item["luma_ratio_to_phase0"] for item in post_frame_metrics])
    )
    post_temporal_median = _finite_float(np.median(post_temporal_steps))
    frozen_fraction = _finite_float(
        np.mean(post_temporal_steps <= thresholds.frozen_transition_l1)
    )

    dark_indices = [
        item["frame_index"] for item in post_frame_metrics if item["is_black"]
    ]
    whiteout_indices = [
        item["frame_index"] for item in post_frame_metrics if item["is_whiteout"]
    ]
    flat_indices = [
        item["frame_index"] for item in post_frame_metrics if item["is_flat"]
    ]
    noise_indices = [
        item["frame_index"]
        for item in post_frame_metrics
        if item["is_spatial_color_noise"]
    ]

    failures: list[dict[str, Any]] = []
    if phase0_probe["is_black"] or phase0_probe["is_whiteout"] or phase0_probe["is_flat"]:
        failures.append(
            _failure(
                "phase0_unusable",
                "frame zero is itself blank, clipped, or flat",
                {"frame_metrics": phase0_probe},
            )
        )
    if dark_indices:
        worst = max(
            dark_indices,
            key=lambda index: frame_metrics[index]["dark_pixel_fraction"],
        )
        failures.append(
            _failure(
                "post_onset_blackout",
                "one or more post-onset frames collapsed toward black",
                {
                    "frame_indices": dark_indices,
                    "worst_frame": worst,
                    "worst_dark_pixel_fraction": frame_metrics[worst][
                        "dark_pixel_fraction"
                    ],
                },
            )
        )
    if whiteout_indices:
        failures.append(
            _failure(
                "post_onset_whiteout",
                "one or more post-onset frames collapsed toward white",
                {"frame_indices": whiteout_indices},
            )
        )
    if flat_indices:
        failures.append(
            _failure(
                "post_onset_flat_decode",
                "one or more post-onset frames lost essentially all spatial detail",
                {"frame_indices": flat_indices},
            )
        )
    if noise_indices:
        failures.append(
            _failure(
                "post_onset_spatial_color_noise",
                "one or more post-onset frames have catastrophic spatial noise statistics",
                {"frame_indices": noise_indices},
            )
        )
    if post_luma_ratio_median < thresholds.minimum_post_onset_median_luma_ratio:
        failures.append(
            _failure(
                "post_onset_median_luma_collapse",
                "the post-onset trajectory is much darker than its valid first frame",
                {"post_onset_median_luma_ratio": post_luma_ratio_median},
            )
        )
    if post_luma_ratio_median > thresholds.maximum_post_onset_median_luma_ratio:
        failures.append(
            _failure(
                "post_onset_median_luma_explosion",
                "the post-onset trajectory is much brighter than its valid first frame",
                {"post_onset_median_luma_ratio": post_luma_ratio_median},
            )
        )
    if post_temporal_median > thresholds.maximum_post_onset_temporal_median_l1:
        failures.append(
            _failure(
                "post_onset_temporal_incoherence",
                "post-onset frames change with noise-like temporal energy",
                {"post_onset_temporal_median_l1": post_temporal_median},
            )
        )
    if frozen_fraction >= thresholds.frozen_transition_fraction:
        failures.append(
            _failure(
                "post_onset_temporal_freeze",
                "the decoder emitted an essentially frozen post-onset trajectory",
                {"frozen_transition_fraction": frozen_fraction},
            )
        )
    phase0_to_frame1 = _finite_float(temporal_steps[0])
    if phase0_to_frame1 > thresholds.maximum_phase0_to_frame1_l1:
        failures.append(
            _failure(
                "phase0_to_frame1_catastrophic_jump",
                "the apparently valid frame zero is followed by a catastrophic decode jump",
                {"phase0_to_frame1_l1": phase0_to_frame1},
            )
        )
    maximum_transition = _finite_float(np.max(temporal_steps))
    if maximum_transition > thresholds.catastrophic_single_transition_l1:
        destination = int(np.argmax(temporal_steps)) + 1
        failures.append(
            _failure(
                "catastrophic_single_transition",
                "at least one decoded transition exceeds the catastrophic limit",
                {
                    "destination_frame": destination,
                    "transition_l1": maximum_transition,
                },
            )
        )

    reference_l1: Optional[list[float]] = None
    reference_ssim: Optional[list[float]] = None
    if reference is not None:
        reference_l1 = [
            _finite_float(np.mean(np.abs(video[index] - reference[index])))
            for index in range(int(video.shape[0]))
        ]
        reference_ssim = [
            _global_ssim(video[index], reference[index])
            for index in range(int(video.shape[0]))
        ]

    segments: list[dict[str, Any]] = []
    for segment_index, (start, end) in enumerate(ranges):
        segment_frames = frame_metrics[start : end + 1]
        transition_values = temporal_steps[start:end]
        segment: dict[str, Any] = {
            "segment_index": segment_index,
            "frame_start": start,
            "frame_end_inclusive": end,
            "frame_count": end - start + 1,
            "luma_mean": _finite_float(
                np.mean([item["luma_mean"] for item in segment_frames])
            ),
            "maximum_dark_pixel_fraction": _finite_float(
                max(item["dark_pixel_fraction"] for item in segment_frames)
            ),
            "spatial_color_noise_frame_fraction": _finite_float(
                np.mean([item["is_spatial_color_noise"] for item in segment_frames])
            ),
            "temporal_step_l1_median": (
                _finite_float(np.median(transition_values))
                if len(transition_values)
                else 0.0
            ),
        }
        if reference_l1 is not None and reference_ssim is not None:
            segment.update(
                reference_l1_mean=_finite_float(
                    np.mean(reference_l1[start : end + 1])
                ),
                reference_global_ssim_mean=_finite_float(
                    np.mean(reference_ssim[start : end + 1])
                ),
            )
        segments.append(segment)

    trajectory: dict[str, Any] = {
        "evaluated_frame_count": int(video.shape[0]),
        "post_onset_frame_count": int(video.shape[0]) - 1,
        "evaluated_transition_count": int(len(temporal_steps)),
        "all_frames_evaluated": True,
        "phase0_luma_mean": phase0_luma_mean,
        "post_onset_median_luma_ratio": post_luma_ratio_median,
        "phase0_to_frame1_l1": phase0_to_frame1,
        "post_onset_temporal_median_l1": post_temporal_median,
        "maximum_temporal_transition_l1": maximum_transition,
        "frozen_transition_fraction": frozen_fraction,
        "black_frame_indices": dark_indices,
        "whiteout_frame_indices": whiteout_indices,
        "flat_frame_indices": flat_indices,
        "spatial_color_noise_frame_indices": noise_indices,
        "temporal_segments": segments,
        "frame_metrics": frame_metrics,
    }
    if reference_l1 is not None and reference_ssim is not None:
        trajectory["reference_metrics"] = {
            "metric_scope": "phase0, full trajectory, post-onset trajectory, and segments",
            "phase0_l1": reference_l1[0],
            "phase0_global_ssim": reference_ssim[0],
            "full_trajectory_l1_mean": _finite_float(np.mean(reference_l1)),
            "full_trajectory_global_ssim_mean": _finite_float(
                np.mean(reference_ssim)
            ),
            "post_onset_l1_mean": _finite_float(np.mean(reference_l1[1:])),
            "post_onset_global_ssim_mean": _finite_float(
                np.mean(reference_ssim[1:])
            ),
            "post_onset_global_ssim_minimum": _finite_float(
                np.min(reference_ssim[1:])
            ),
        }

    failure_codes = [item["code"] for item in failures]
    passed = not failures
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if passed else "fail",
        "passed": passed,
        "publishable": passed,
        "fail_closed": True,
        "input_contract_passed": True,
        "phase0_only": False,
        "metric_scope": "all decoded frames plus all temporal transitions",
        "thresholds": asdict(thresholds),
        "failure_codes": failure_codes,
        "failures": failures,
        "trajectory": trajectory,
    }


def require_visual_validity(
    frames: Any,
    *,
    reference_frames: Optional[Any] = None,
    thresholds: VisualValidityThresholds = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Return the report or raise, preventing publication on any gate failure."""

    report = evaluate_visual_validity(
        frames, reference_frames=reference_frames, thresholds=thresholds
    )
    if not report["publishable"]:
        codes = ", ".join(report["failure_codes"])
        raise VisualValidityError(f"decoded video is not publishable: {codes}")
    return report


__all__ = [
    "DEFAULT_THRESHOLDS",
    "SCHEMA_VERSION",
    "VisualValidityError",
    "VisualValidityInputError",
    "VisualValidityThresholds",
    "evaluate_visual_validity",
    "require_visual_validity",
]
