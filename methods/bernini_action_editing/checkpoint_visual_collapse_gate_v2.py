#!/usr/bin/env python3
"""Automatic exact-81 visual-collapse gate for checkpoint sweeps.

This gate is deliberately narrower than a semantic video evaluator.  It
answers whether a decoded checkpoint result is visually usable, rather than
whether the requested action edit is correct.  Every frame and transition is
measured.  A source video is required and a frozen-base decode is optional but
strongly recommended.

Thresholds are not hand tuned per checkpoint.  ``calibrate`` learns strict
separating thresholds from a labelled clean-base/collapsed cohort (the current
Heldout8 corpus is the reference cohort), fingerprints the calibration, and
replays both classes through the final decision rule.  ``evaluate`` refuses an
invalid or non-separating calibration and emits one JSON report suitable for a
multi-checkpoint sweep.

Only NumPy is needed for tensor evaluation.  The CLI uses ffmpeg/ffprobe to
decode all 81 frames at a fixed analysis geometry.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np


SCHEMA_VERSION = "bernini-checkpoint-visual-collapse-gate-v2"
CALIBRATION_SCHEMA_VERSION = (
    "bernini-checkpoint-visual-collapse-heldout-calibration-v2"
)
EXPECTED_FRAME_COUNT = 81
ANALYSIS_WIDTH = 192
ANALYSIS_HEIGHT = 144


class CollapseGateError(RuntimeError):
    """Raised for a malformed input, calibration, or decoded video."""


@dataclass(frozen=True)
class HardFailureThresholds:
    """Conservative non-learned limits for blank and frozen media."""

    dark_luma: float = 0.035
    bright_luma: float = 0.965
    frame_dark_pixel_fraction: float = 0.98
    frame_bright_pixel_fraction: float = 0.995
    trajectory_bad_frame_fraction: float = 0.25
    flat_luma_std: float = 0.003
    flat_spatial_gradient_l1: float = 0.002
    frozen_transition_l1: float = 0.0005
    frozen_transition_fraction: float = 0.95


HARD_FAILURE_THRESHOLDS = HardFailureThresholds()


@dataclass(frozen=True)
class CalibrationExample:
    """One clean-base/collapsed triplet used to fit a cohort calibration."""

    case_id: str
    source: np.ndarray
    frozen_base: np.ndarray
    collapsed: np.ndarray
    identities: Optional[Mapping[str, Any]] = None


_METRIC_DIRECTIONS: dict[str, str] = {
    # Absolute source-only signals.
    "temporal_frame_l1_median": "high",
    "temporal_frame_l1_p90": "high",
    "temporal_global_rgb_l1_median": "high",
    "temporal_global_rgb_l1_p90": "high",
    "source_l1_mean": "high",
    "source_global_ssim_mean": "low",
    "source_edge_correlation_mean": "low",
    "spatial_gradient_l1_median": "high",
    "spatial_laplacian_l1_median": "high",
    # Frozen-base differential signals.
    "candidate_base_l1_mean": "high",
    "source_l1_excess_over_base": "high",
    "source_ssim_drop_from_base": "high",
    "source_edge_correlation_drop_from_base": "high",
    "temporal_frame_l1_ratio_to_base": "high",
    "temporal_global_rgb_l1_ratio_to_base": "high",
    "spatial_gradient_log_ratio_abs_vs_base": "high",
}

_REQUIRED_SEPARATING_METRICS = (
    "temporal_frame_l1_median",
    "temporal_global_rgb_l1_median",
    "source_l1_mean",
    "candidate_base_l1_mean",
    "source_l1_excess_over_base",
    "source_ssim_drop_from_base",
)


def _finite(value: Any, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise CollapseGateError(f"{label} is not finite")
    return result


def _json_safe(value: Any) -> Any:
    """Convert NumPy scalars/arrays recursively and reject NaN/Inf."""

    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return _finite(value, label="JSON number")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_video(value: Any, *, label: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise CollapseGateError(f"{label} must be a numpy ndarray")
    if value.ndim != 4 or value.shape[-1] != 3:
        raise CollapseGateError(
            f"{label} must have shape [frames,height,width,3], got {value.shape}"
        )
    if int(value.shape[0]) != EXPECTED_FRAME_COUNT:
        raise CollapseGateError(
            f"{label} must contain exactly {EXPECTED_FRAME_COUNT} frames, "
            f"got {value.shape[0]}"
        )
    if int(value.shape[1]) < 16 or int(value.shape[2]) < 16:
        raise CollapseGateError(f"{label} geometry is too small: {value.shape[1:3]}")
    if np.issubdtype(value.dtype, np.bool_) or np.issubdtype(
        value.dtype, np.complexfloating
    ):
        raise CollapseGateError(f"{label} has an invalid RGB dtype: {value.dtype}")
    if np.issubdtype(value.dtype, np.integer):
        minimum, maximum = int(np.min(value)), int(np.max(value))
        if minimum < 0 or maximum > 255:
            raise CollapseGateError(
                f"{label} integer RGB must be in [0,255], got [{minimum},{maximum}]"
            )
        result = value.astype(np.float32) / 255.0
    elif np.issubdtype(value.dtype, np.floating):
        result = value.astype(np.float32, copy=False)
        if not bool(np.all(np.isfinite(result))):
            raise CollapseGateError(f"{label} contains NaN or infinity")
        minimum, maximum = float(np.min(result)), float(np.max(result))
        if minimum < -1e-6 or maximum > 1.0 + 1e-6:
            raise CollapseGateError(
                f"{label} float RGB must be in [0,1], got [{minimum},{maximum}]"
            )
        result = np.clip(result, 0.0, 1.0)
    else:
        raise CollapseGateError(f"{label} has a non-numeric RGB dtype: {value.dtype}")
    return np.ascontiguousarray(result)


def _global_ssim_per_frame(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x = left.astype(np.float64)
    y = right.astype(np.float64)
    axes = (1, 2, 3)
    mean_x = np.mean(x, axis=axes)
    mean_y = np.mean(y, axis=axes)
    centred_x = x - mean_x[:, None, None, None]
    centred_y = y - mean_y[:, None, None, None]
    variance_x = np.mean(centred_x * centred_x, axis=axes)
    variance_y = np.mean(centred_y * centred_y, axis=axes)
    covariance = np.mean(centred_x * centred_y, axis=axes)
    c1, c2 = 0.01**2, 0.03**2
    numerator = (2.0 * mean_x * mean_y + c1) * (2.0 * covariance + c2)
    denominator = (
        (mean_x * mean_x + mean_y * mean_y + c1)
        * (variance_x + variance_y + c2)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.ones_like(numerator),
        where=denominator > 1e-15,
    )


def _luma(video: np.ndarray) -> np.ndarray:
    return np.tensordot(
        video,
        np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32),
        axes=([-1], [0]),
    )


def _edge_magnitude(luma: np.ndarray) -> np.ndarray:
    horizontal = luma[:, :-1, 1:] - luma[:, :-1, :-1]
    vertical = luma[:, 1:, :-1] - luma[:, :-1, :-1]
    return np.sqrt(horizontal * horizontal + vertical * vertical)


def _correlation_per_frame(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x = left.reshape(left.shape[0], -1).astype(np.float64)
    y = right.reshape(right.shape[0], -1).astype(np.float64)
    x -= np.mean(x, axis=1, keepdims=True)
    y -= np.mean(y, axis=1, keepdims=True)
    numerator = np.sum(x * y, axis=1)
    denominator = np.sqrt(np.sum(x * x, axis=1) * np.sum(y * y, axis=1))
    # Two flat edge maps carry no positive structural evidence.
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )


def _entropy_per_frame(luma: np.ndarray, bins: int = 32) -> np.ndarray:
    result = []
    denominator = math.log2(bins)
    for frame in luma:
        counts = np.histogram(frame, bins=bins, range=(0.0, 1.0))[0].astype(
            np.float64
        )
        probabilities = counts[counts > 0.0] / max(float(np.sum(counts)), 1.0)
        result.append(-float(np.sum(probabilities * np.log2(probabilities))) / denominator)
    return np.asarray(result, dtype=np.float64)


def _reference_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    label: str,
) -> dict[str, Any]:
    l1 = np.mean(np.abs(candidate - reference), axis=(1, 2, 3))
    ssim = _global_ssim_per_frame(candidate, reference)
    edge_correlation = _correlation_per_frame(
        _edge_magnitude(_luma(candidate)), _edge_magnitude(_luma(reference))
    )
    return {
        "label": label,
        "frame_count": EXPECTED_FRAME_COUNT,
        "per_frame_l1": l1,
        "per_frame_global_ssim": ssim,
        "per_frame_edge_correlation": edge_correlation,
        "l1_mean": _finite(np.mean(l1), label=f"{label} L1 mean"),
        "l1_median": _finite(np.median(l1), label=f"{label} L1 median"),
        "global_ssim_mean": _finite(np.mean(ssim), label=f"{label} SSIM mean"),
        "global_ssim_minimum": _finite(np.min(ssim), label=f"{label} SSIM minimum"),
        "edge_correlation_mean": _finite(
            np.mean(edge_correlation), label=f"{label} edge correlation mean"
        ),
    }


def compute_visual_features(
    source_frames: Any,
    candidate_frames: Any,
    *,
    frozen_base_frames: Optional[Any] = None,
) -> dict[str, Any]:
    """Measure every candidate frame and transition.

    Tensor callers must supply already aligned videos with equal geometry.  The
    CLI performs deterministic ffmpeg scaling before calling this function.
    """

    source = _normalise_video(source_frames, label="source")
    candidate = _normalise_video(candidate_frames, label="candidate")
    if source.shape != candidate.shape:
        raise CollapseGateError(
            f"source shape {source.shape} differs from candidate {candidate.shape}"
        )
    frozen_base = None
    if frozen_base_frames is not None:
        frozen_base = _normalise_video(frozen_base_frames, label="frozen_base")
        if frozen_base.shape != candidate.shape:
            raise CollapseGateError(
                f"frozen_base shape {frozen_base.shape} differs from candidate "
                f"{candidate.shape}"
            )

    candidate_luma = _luma(candidate)
    horizontal = np.abs(candidate_luma[:, :, 1:] - candidate_luma[:, :, :-1])
    vertical = np.abs(candidate_luma[:, 1:, :] - candidate_luma[:, :-1, :])
    spatial_gradient = 0.5 * (
        np.mean(horizontal, axis=(1, 2)) + np.mean(vertical, axis=(1, 2))
    )
    centre = candidate_luma[:, 1:-1, 1:-1]
    laplacian = np.mean(
        np.abs(
            4.0 * centre
            - candidate_luma[:, :-2, 1:-1]
            - candidate_luma[:, 2:, 1:-1]
            - candidate_luma[:, 1:-1, :-2]
            - candidate_luma[:, 1:-1, 2:]
        ),
        axis=(1, 2),
    ) / 4.0
    luma_mean = np.mean(candidate_luma, axis=(1, 2))
    luma_std = np.std(candidate_luma, axis=(1, 2))
    dark_fraction = np.mean(
        candidate_luma <= HARD_FAILURE_THRESHOLDS.dark_luma, axis=(1, 2)
    )
    bright_fraction = np.mean(
        candidate_luma >= HARD_FAILURE_THRESHOLDS.bright_luma, axis=(1, 2)
    )
    chroma_amplitude = np.mean(
        np.max(candidate, axis=-1) - np.min(candidate, axis=-1), axis=(1, 2)
    )
    entropy = _entropy_per_frame(candidate_luma)

    temporal_l1 = np.mean(
        np.abs(candidate[1:] - candidate[:-1]), axis=(1, 2, 3)
    )
    frame_rgb_mean = np.mean(candidate, axis=(1, 2))
    temporal_global_rgb_l1 = np.mean(
        np.abs(frame_rgb_mean[1:] - frame_rgb_mean[:-1]), axis=1
    )

    source_metrics = _reference_metrics(candidate, source, label="candidate_to_source")
    base_metrics: Optional[dict[str, Any]] = None
    base_source_metrics: Optional[dict[str, Any]] = None
    base_temporal_l1: Optional[np.ndarray] = None
    base_temporal_global_rgb_l1: Optional[np.ndarray] = None
    base_spatial_gradient: Optional[np.ndarray] = None
    if frozen_base is not None:
        base_metrics = _reference_metrics(
            candidate, frozen_base, label="candidate_to_frozen_base"
        )
        base_source_metrics = _reference_metrics(
            frozen_base, source, label="frozen_base_to_source"
        )
        base_temporal_l1 = np.mean(
            np.abs(frozen_base[1:] - frozen_base[:-1]), axis=(1, 2, 3)
        )
        base_rgb_mean = np.mean(frozen_base, axis=(1, 2))
        base_temporal_global_rgb_l1 = np.mean(
            np.abs(base_rgb_mean[1:] - base_rgb_mean[:-1]), axis=1
        )
        base_luma = _luma(frozen_base)
        base_spatial_gradient = 0.5 * (
            np.mean(np.abs(base_luma[:, :, 1:] - base_luma[:, :, :-1]), axis=(1, 2))
            + np.mean(
                np.abs(base_luma[:, 1:, :] - base_luma[:, :-1, :]), axis=(1, 2)
            )
        )

    frame_rows = []
    for index in range(EXPECTED_FRAME_COUNT):
        frame_rows.append(
            {
                "frame_index": index,
                "luma_mean": _finite(luma_mean[index], label="frame luma mean"),
                "luma_std": _finite(luma_std[index], label="frame luma std"),
                "dark_pixel_fraction": _finite(
                    dark_fraction[index], label="dark pixel fraction"
                ),
                "bright_pixel_fraction": _finite(
                    bright_fraction[index], label="bright pixel fraction"
                ),
                "chroma_amplitude_mean": _finite(
                    chroma_amplitude[index], label="chroma amplitude"
                ),
                "luma_entropy_normalised": _finite(
                    entropy[index], label="luma entropy"
                ),
                "spatial_gradient_l1": _finite(
                    spatial_gradient[index], label="spatial gradient"
                ),
                "spatial_laplacian_l1": _finite(
                    laplacian[index], label="spatial laplacian"
                ),
                "source_l1": _finite(
                    source_metrics["per_frame_l1"][index], label="source L1"
                ),
                "source_global_ssim": _finite(
                    source_metrics["per_frame_global_ssim"][index],
                    label="source SSIM",
                ),
                "source_edge_correlation": _finite(
                    source_metrics["per_frame_edge_correlation"][index],
                    label="source edge correlation",
                ),
            }
        )
        if base_metrics is not None:
            frame_rows[-1].update(
                base_l1=_finite(
                    base_metrics["per_frame_l1"][index], label="base L1"
                ),
                base_global_ssim=_finite(
                    base_metrics["per_frame_global_ssim"][index], label="base SSIM"
                ),
                base_edge_correlation=_finite(
                    base_metrics["per_frame_edge_correlation"][index],
                    label="base edge correlation",
                ),
            )

    transition_rows = []
    for destination in range(1, EXPECTED_FRAME_COUNT):
        row = {
            "source_frame_index": destination - 1,
            "destination_frame_index": destination,
            "frame_l1": _finite(
                temporal_l1[destination - 1], label="transition L1"
            ),
            "global_rgb_mean_l1": _finite(
                temporal_global_rgb_l1[destination - 1],
                label="transition global RGB L1",
            ),
        }
        if base_temporal_l1 is not None and base_temporal_global_rgb_l1 is not None:
            row.update(
                frozen_base_frame_l1=_finite(
                    base_temporal_l1[destination - 1], label="base transition L1"
                ),
                frozen_base_global_rgb_mean_l1=_finite(
                    base_temporal_global_rgb_l1[destination - 1],
                    label="base transition RGB L1",
                ),
            )
        transition_rows.append(row)

    scalars: dict[str, float] = {
        "temporal_frame_l1_median": _finite(
            np.median(temporal_l1), label="temporal L1 median"
        ),
        "temporal_frame_l1_p90": _finite(
            np.percentile(temporal_l1, 90), label="temporal L1 p90"
        ),
        "temporal_global_rgb_l1_median": _finite(
            np.median(temporal_global_rgb_l1), label="temporal RGB L1 median"
        ),
        "temporal_global_rgb_l1_p90": _finite(
            np.percentile(temporal_global_rgb_l1, 90), label="temporal RGB L1 p90"
        ),
        "source_l1_mean": _finite(source_metrics["l1_mean"], label="source L1 mean"),
        "source_global_ssim_mean": _finite(
            source_metrics["global_ssim_mean"], label="source SSIM mean"
        ),
        "source_edge_correlation_mean": _finite(
            source_metrics["edge_correlation_mean"],
            label="source edge correlation mean",
        ),
        "spatial_gradient_l1_median": _finite(
            np.median(spatial_gradient), label="spatial gradient median"
        ),
        "spatial_laplacian_l1_median": _finite(
            np.median(laplacian), label="spatial laplacian median"
        ),
        "luma_entropy_median": _finite(np.median(entropy), label="entropy median"),
        "chroma_amplitude_median": _finite(
            np.median(chroma_amplitude), label="chroma amplitude median"
        ),
    }

    if (
        frozen_base is not None
        and base_metrics is not None
        and base_source_metrics is not None
        and base_temporal_l1 is not None
        and base_temporal_global_rgb_l1 is not None
        and base_spatial_gradient is not None
    ):
        epsilon = 1e-6
        candidate_gradient = float(np.median(spatial_gradient))
        frozen_gradient = float(np.median(base_spatial_gradient))
        scalars.update(
            candidate_base_l1_mean=_finite(
                base_metrics["l1_mean"], label="candidate/base L1 mean"
            ),
            candidate_base_global_ssim_mean=_finite(
                base_metrics["global_ssim_mean"], label="candidate/base SSIM mean"
            ),
            candidate_base_edge_correlation_mean=_finite(
                base_metrics["edge_correlation_mean"],
                label="candidate/base edge correlation mean",
            ),
            frozen_base_source_l1_mean=_finite(
                base_source_metrics["l1_mean"], label="base/source L1 mean"
            ),
            frozen_base_source_global_ssim_mean=_finite(
                base_source_metrics["global_ssim_mean"],
                label="base/source SSIM mean",
            ),
            source_l1_excess_over_base=_finite(
                source_metrics["l1_mean"] - base_source_metrics["l1_mean"],
                label="source L1 excess",
            ),
            source_ssim_drop_from_base=_finite(
                base_source_metrics["global_ssim_mean"]
                - source_metrics["global_ssim_mean"],
                label="source SSIM drop",
            ),
            source_edge_correlation_drop_from_base=_finite(
                base_source_metrics["edge_correlation_mean"]
                - source_metrics["edge_correlation_mean"],
                label="source edge correlation drop",
            ),
            temporal_frame_l1_ratio_to_base=_finite(
                np.median(temporal_l1)
                / max(float(np.median(base_temporal_l1)), epsilon),
                label="temporal L1 ratio",
            ),
            temporal_global_rgb_l1_ratio_to_base=_finite(
                np.median(temporal_global_rgb_l1)
                / max(float(np.median(base_temporal_global_rgb_l1)), epsilon),
                label="temporal RGB ratio",
            ),
            spatial_gradient_log_ratio_abs_vs_base=_finite(
                abs(math.log((candidate_gradient + epsilon) / (frozen_gradient + epsilon))),
                label="spatial gradient log ratio",
            ),
        )

    black_indices = np.flatnonzero(
        dark_fraction >= HARD_FAILURE_THRESHOLDS.frame_dark_pixel_fraction
    ).astype(int)
    white_indices = np.flatnonzero(
        bright_fraction >= HARD_FAILURE_THRESHOLDS.frame_bright_pixel_fraction
    ).astype(int)
    flat_indices = np.flatnonzero(
        (luma_std <= HARD_FAILURE_THRESHOLDS.flat_luma_std)
        & (spatial_gradient <= HARD_FAILURE_THRESHOLDS.flat_spatial_gradient_l1)
    ).astype(int)
    frozen_transition_fraction = _finite(
        np.mean(temporal_l1 <= HARD_FAILURE_THRESHOLDS.frozen_transition_l1),
        label="frozen transition fraction",
    )
    hard_artifacts = {
        "black_frame_indices": black_indices.tolist(),
        "white_frame_indices": white_indices.tolist(),
        "flat_frame_indices": flat_indices.tolist(),
        "black_frame_fraction": _finite(
            len(black_indices) / EXPECTED_FRAME_COUNT, label="black frame fraction"
        ),
        "white_frame_fraction": _finite(
            len(white_indices) / EXPECTED_FRAME_COUNT, label="white frame fraction"
        ),
        "flat_frame_fraction": _finite(
            len(flat_indices) / EXPECTED_FRAME_COUNT, label="flat frame fraction"
        ),
        "frozen_transition_fraction": frozen_transition_fraction,
    }

    return _json_safe(
        {
            "metric_scope": "all 81 frames and all 80 adjacent transitions",
            "all_frames_evaluated": True,
            "all_transitions_evaluated": True,
            "evaluated_frame_count": EXPECTED_FRAME_COUNT,
            "evaluated_transition_count": EXPECTED_FRAME_COUNT - 1,
            "analysis_geometry": {
                "height": int(candidate.shape[1]),
                "width": int(candidate.shape[2]),
            },
            "frozen_base_supplied": frozen_base is not None,
            "scalars": scalars,
            "hard_artifacts": hard_artifacts,
            "frame_metrics": frame_rows,
            "transition_metrics": transition_rows,
        }
    )


def _fit_threshold(
    clean_values: Sequence[float],
    collapsed_values: Sequence[float],
    *,
    direction: str,
) -> dict[str, Any]:
    clean = [_finite(value, label="clean calibration value") for value in clean_values]
    collapsed = [
        _finite(value, label="collapsed calibration value") for value in collapsed_values
    ]
    if not clean or not collapsed:
        raise CollapseGateError("calibration classes must both be non-empty")
    if direction == "high":
        clean_extreme = max(clean)
        collapsed_extreme = min(collapsed)
        margin = collapsed_extreme - clean_extreme
    elif direction == "low":
        clean_extreme = min(clean)
        collapsed_extreme = max(collapsed)
        margin = clean_extreme - collapsed_extreme
    else:
        raise CollapseGateError(f"unknown threshold direction: {direction}")
    threshold = 0.5 * (clean_extreme + collapsed_extreme)
    return {
        "direction": direction,
        "threshold": _finite(threshold, label="calibration threshold"),
        "clean_extreme": _finite(clean_extreme, label="clean extreme"),
        "collapsed_extreme": _finite(collapsed_extreme, label="collapsed extreme"),
        "strict_separation_margin": _finite(margin, label="separation margin"),
        "fully_separating": bool(margin > 0.0),
        "clean_values": clean,
        "collapsed_values": collapsed,
    }


def _metric_triggered(value: float, row: Mapping[str, Any]) -> bool:
    threshold = _finite(row["threshold"], label="threshold")
    direction = row.get("direction")
    if direction == "high":
        return value > threshold
    if direction == "low":
        return value < threshold
    raise CollapseGateError(f"invalid calibration direction: {direction!r}")


def _validate_calibration(calibration: Mapping[str, Any]) -> dict[str, Any]:
    value = _json_safe(dict(calibration))
    if value.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise CollapseGateError("calibration schema differs")
    if value.get("expected_frame_count") != EXPECTED_FRAME_COUNT:
        raise CollapseGateError("calibration frame count differs")
    geometry = value.get("analysis_geometry")
    if (
        not isinstance(geometry, Mapping)
        or isinstance(geometry.get("width"), bool)
        or isinstance(geometry.get("height"), bool)
        or not isinstance(geometry.get("width"), int)
        or not isinstance(geometry.get("height"), int)
        or geometry["width"] < 16
        or geometry["height"] < 16
    ):
        raise CollapseGateError("calibration analysis geometry is invalid")
    if value.get("hard_failure_thresholds") != asdict(HARD_FAILURE_THRESHOLDS):
        raise CollapseGateError("calibration hard-failure thresholds differ")
    if value.get("required_separating_metrics") != list(
        _REQUIRED_SEPARATING_METRICS
    ):
        raise CollapseGateError("calibration required metric set differs")
    thresholds = value.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise CollapseGateError("calibration thresholds are missing")
    for metric in _REQUIRED_SEPARATING_METRICS:
        row = thresholds.get(metric)
        if not isinstance(row, Mapping) or row.get("fully_separating") is not True:
            raise CollapseGateError(
                f"required calibration metric is not separating: {metric}"
            )
        if row.get("direction") != _METRIC_DIRECTIONS[metric]:
            raise CollapseGateError(
                f"required calibration metric direction differs: {metric}"
            )
    if value.get("self_check", {}).get("passed") is not True:
        raise CollapseGateError("calibration self-check did not pass")
    supplied = value.get("calibration_fingerprint")
    unsigned = dict(value)
    unsigned.pop("calibration_fingerprint", None)
    expected = _object_sha256(unsigned)
    if supplied != expected:
        raise CollapseGateError("calibration fingerprint differs")
    return value


def _decision_from_features(
    features: Mapping[str, Any], calibration: Mapping[str, Any]
) -> dict[str, Any]:
    if features.get("analysis_geometry") != calibration.get("analysis_geometry"):
        raise CollapseGateError(
            "evaluated feature geometry differs from calibration geometry"
        )
    thresholds = calibration["thresholds"]
    scalars = features["scalars"]
    triggered: dict[str, bool] = {}
    for metric, row in thresholds.items():
        if row.get("fully_separating") is not True or metric not in scalars:
            continue
        triggered[metric] = _metric_triggered(
            _finite(scalars[metric], label=f"feature {metric}"), row
        )

    evidence_families: dict[str, dict[str, Any]] = {}

    def family(name: str, metrics: Sequence[str], minimum_votes: int = 1) -> None:
        available = [metric for metric in metrics if metric in triggered]
        hits = [metric for metric in available if triggered[metric]]
        evidence_families[name] = {
            "triggered": len(hits) >= minimum_votes,
            "minimum_votes": minimum_votes,
            "available_metrics": available,
            "triggered_metrics": hits,
        }

    family(
        "temporal_incoherence",
        ("temporal_frame_l1_median", "temporal_frame_l1_p90"),
    )
    family(
        "temporal_color_incoherence",
        (
            "temporal_global_rgb_l1_median",
            "temporal_global_rgb_l1_p90",
        ),
    )
    family(
        "source_structure_divergence",
        (
            "source_l1_mean",
            "source_global_ssim_mean",
            "source_edge_correlation_mean",
        ),
    )
    family(
        "spatial_texture_anomaly",
        (
            "spatial_gradient_l1_median",
            "spatial_laplacian_l1_median",
            "spatial_gradient_log_ratio_abs_vs_base",
        ),
    )
    family(
        "frozen_base_relative_divergence",
        (
            "candidate_base_l1_mean",
            "source_l1_excess_over_base",
            "source_ssim_drop_from_base",
            "source_edge_correlation_drop_from_base",
        ),
        minimum_votes=2,
    )

    hard = features["hard_artifacts"]
    hard_failures: list[str] = []
    bad_fraction = HARD_FAILURE_THRESHOLDS.trajectory_bad_frame_fraction
    if hard["black_frame_fraction"] >= bad_fraction:
        hard_failures.append("trajectory_blackout")
    if hard["white_frame_fraction"] >= bad_fraction:
        hard_failures.append("trajectory_whiteout")
    if hard["flat_frame_fraction"] >= bad_fraction:
        hard_failures.append("trajectory_flat_decode")
    if hard["frozen_transition_fraction"] >= HARD_FAILURE_THRESHOLDS.frozen_transition_fraction:
        hard_failures.append("trajectory_freeze")

    temporal_triggered = (
        evidence_families["temporal_incoherence"]["triggered"]
        or evidence_families["temporal_color_incoherence"]["triggered"]
    )
    support_triggered = sum(
        int(evidence_families[name]["triggered"])
        for name in (
            "source_structure_divergence",
            "spatial_texture_anomaly",
            "frozen_base_relative_divergence",
        )
    )
    learned_collapse = bool(temporal_triggered and support_triggered >= 1)
    collapsed = bool(hard_failures or learned_collapse)
    failure_codes = list(hard_failures)
    if learned_collapse:
        failure_codes.append("calibrated_visual_collapse")
    return {
        "passed": not collapsed,
        "collapsed": collapsed,
        "failure_codes": failure_codes,
        "hard_failure_codes": hard_failures,
        "learned_collapse": learned_collapse,
        "decision_rule": (
            "hard blank/flat/freeze failure OR "
            "(temporal or temporal-color incoherence AND at least one "
            "structural/spatial/base-relative support family)"
        ),
        "triggered_metrics": sorted(
            metric for metric, did_trigger in triggered.items() if did_trigger
        ),
        "metric_threshold_results": triggered,
        "evidence_families": evidence_families,
    }


def calibrate_examples(
    examples: Iterable[CalibrationExample],
    *,
    cohort_name: str,
) -> dict[str, Any]:
    """Fit and self-test a calibration from clean/collapsed triplets."""

    rows = list(examples)
    if len(rows) < 2:
        raise CollapseGateError("calibration requires at least two independent cases")
    case_ids = [row.case_id for row in rows]
    if len(set(case_ids)) != len(case_ids):
        raise CollapseGateError("calibration case IDs must be unique")

    clean_features: list[dict[str, Any]] = []
    collapsed_features: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    for row in rows:
        clean = compute_visual_features(
            row.source, row.frozen_base, frozen_base_frames=row.frozen_base
        )
        collapsed = compute_visual_features(
            row.source, row.collapsed, frozen_base_frames=row.frozen_base
        )
        clean_features.append(clean)
        collapsed_features.append(collapsed)
        example_rows.append(
            {
                "case_id": row.case_id,
                "identities": _json_safe(row.identities or {}),
                "clean_scalars": clean["scalars"],
                "collapsed_scalars": collapsed["scalars"],
            }
        )

    geometries = {
        (
            feature["analysis_geometry"]["width"],
            feature["analysis_geometry"]["height"],
        )
        for feature in clean_features + collapsed_features
    }
    if len(geometries) != 1:
        raise CollapseGateError("calibration examples have inconsistent geometry")
    analysis_width, analysis_height = next(iter(geometries))

    thresholds = {}
    for metric, direction in _METRIC_DIRECTIONS.items():
        if not all(metric in feature["scalars"] for feature in clean_features):
            continue
        if not all(metric in feature["scalars"] for feature in collapsed_features):
            continue
        thresholds[metric] = _fit_threshold(
            [feature["scalars"][metric] for feature in clean_features],
            [feature["scalars"][metric] for feature in collapsed_features],
            direction=direction,
        )

    missing = [
        metric
        for metric in _REQUIRED_SEPARATING_METRICS
        if metric not in thresholds or not thresholds[metric]["fully_separating"]
    ]
    if missing:
        raise CollapseGateError(
            "reference cohort does not strictly separate required metrics: "
            + ", ".join(missing)
        )

    unsigned: dict[str, Any] = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "cohort_name": cohort_name,
        "case_count": len(rows),
        "case_ids": case_ids,
        "expected_frame_count": EXPECTED_FRAME_COUNT,
        "analysis_geometry": {"width": analysis_width, "height": analysis_height},
        "metric_scope": "all 81 frames and all 80 adjacent transitions",
        "hard_failure_thresholds": asdict(HARD_FAILURE_THRESHOLDS),
        "required_separating_metrics": list(_REQUIRED_SEPARATING_METRICS),
        "thresholds": thresholds,
        "examples": example_rows,
    }

    # The temporary self-check field is sufficient for the decision function;
    # the final, exact counts replace it immediately afterwards.
    temporary = dict(unsigned)
    temporary["self_check"] = {"passed": True}
    clean_decisions = [
        _decision_from_features(feature, temporary) for feature in clean_features
    ]
    collapsed_decisions = [
        _decision_from_features(feature, temporary) for feature in collapsed_features
    ]
    self_check = {
        "clean_expected_pass_count": len(rows),
        "clean_actual_pass_count": sum(
            int(decision["passed"]) for decision in clean_decisions
        ),
        "collapsed_expected_fail_count": len(rows),
        "collapsed_actual_fail_count": sum(
            int(not decision["passed"]) for decision in collapsed_decisions
        ),
        "per_case": [
            {
                "case_id": rows[index].case_id,
                "clean_passed": clean_decisions[index]["passed"],
                "collapsed_failed": not collapsed_decisions[index]["passed"],
                "collapsed_failure_codes": collapsed_decisions[index]["failure_codes"],
            }
            for index in range(len(rows))
        ],
    }
    self_check["passed"] = bool(
        self_check["clean_actual_pass_count"] == len(rows)
        and self_check["collapsed_actual_fail_count"] == len(rows)
    )
    if not self_check["passed"]:
        raise CollapseGateError("calibration decision-rule replay did not separate cohort")
    unsigned["self_check"] = self_check
    result = _json_safe(unsigned)
    result["calibration_fingerprint"] = _object_sha256(result)
    return result


def evaluate_visual_collapse(
    source_frames: Any,
    candidate_frames: Any,
    *,
    calibration: Mapping[str, Any],
    frozen_base_frames: Optional[Any] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Return a JSON-safe fail-closed report for one checkpoint/sample."""

    report_prefix = {
        "schema_version": SCHEMA_VERSION,
        "fail_closed": True,
        "expected_frame_count": EXPECTED_FRAME_COUNT,
        "metadata": _json_safe(metadata or {}),
    }
    try:
        verified_calibration = _validate_calibration(calibration)
        features = compute_visual_features(
            source_frames,
            candidate_frames,
            frozen_base_frames=frozen_base_frames,
        )
        decision = _decision_from_features(features, verified_calibration)
    except (CollapseGateError, TypeError, ValueError, FloatingPointError) as error:
        return {
            **report_prefix,
            "status": "error",
            "passed": False,
            "publishable": False,
            "collapsed": None,
            "input_contract_passed": False,
            "failure_codes": ["input_or_calibration_contract_violation"],
            "error": str(error),
        }

    return _json_safe(
        {
            **report_prefix,
            "status": "pass" if decision["passed"] else "fail",
            "passed": decision["passed"],
            "publishable": decision["passed"],
            "collapsed": decision["collapsed"],
            "input_contract_passed": True,
            "failure_codes": decision["failure_codes"],
            "decision": decision,
            "calibration": {
                "schema_version": verified_calibration["schema_version"],
                "cohort_name": verified_calibration["cohort_name"],
                "case_count": verified_calibration["case_count"],
                "calibration_fingerprint": verified_calibration[
                    "calibration_fingerprint"
                ],
            },
            "features": features,
        }
    )


def _run_json(command: Sequence[str], *, label: str) -> dict[str, Any]:
    result = subprocess.run(
        list(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise CollapseGateError(f"{label} failed ({result.returncode}): {error}")
    try:
        value = json.loads(result.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CollapseGateError(f"{label} did not emit JSON") from error
    if not isinstance(value, dict):
        raise CollapseGateError(f"{label} JSON root is not an object")
    return value


def _parse_rate(value: str) -> float:
    pieces = value.split("/", 1)
    try:
        if len(pieces) == 2:
            denominator = float(pieces[1])
            return float(pieces[0]) / denominator if denominator else 0.0
        return float(value)
    except ValueError:
        return 0.0


def decode_video_exact81(
    path: Path,
    *,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode every frame to 192x144 RGB and return a byte identity."""

    source = path.expanduser().resolve(strict=True)
    probe = _run_json(
        (
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames",
            "-of",
            "json",
            str(source),
        ),
        label=f"ffprobe {source}",
    )
    streams = probe.get("streams")
    if not isinstance(streams, list) or len(streams) != 1:
        raise CollapseGateError(f"{source} must expose exactly one selected video stream")
    stream = streams[0]
    result = subprocess.run(
        (
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-vf",
            f"scale={ANALYSIS_WIDTH}:{ANALYSIS_HEIGHT}:flags=area",
            "-vsync",
            "0",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise CollapseGateError(f"ffmpeg decode failed for {source}: {error}")
    frame_bytes = ANALYSIS_WIDTH * ANALYSIS_HEIGHT * 3
    if len(result.stdout) % frame_bytes:
        raise CollapseGateError(f"ffmpeg emitted a partial RGB frame for {source}")
    frame_count = len(result.stdout) // frame_bytes
    if frame_count != EXPECTED_FRAME_COUNT:
        raise CollapseGateError(
            f"{source} decoded to {frame_count} frames, expected {EXPECTED_FRAME_COUNT}"
        )
    frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(
        EXPECTED_FRAME_COUNT, ANALYSIS_HEIGHT, ANALYSIS_WIDTH, 3
    )
    rate_text = str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0")
    identity = {
        "path": str(source),
        "sha256": _file_sha256(source),
        "size_bytes": source.stat().st_size,
        "codec_name": stream.get("codec_name"),
        "original_width": int(stream.get("width", 0)),
        "original_height": int(stream.get("height", 0)),
        "reported_frame_count": stream.get("nb_frames"),
        "decoded_frame_count": frame_count,
        "reported_frame_rate": rate_text,
        "reported_frame_rate_float": _finite(
            _parse_rate(rate_text), label="reported frame rate"
        ),
        "analysis_width": ANALYSIS_WIDTH,
        "analysis_height": ANALYSIS_HEIGHT,
    }
    return frames, identity


def calibrate_video_corpus(
    corpus_dir: Path,
    *,
    collapsed_suffix: str = "v16r3",
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> dict[str, Any]:
    """Discover ``caseNN-{source,base,<collapsed>}.mp4`` triplets."""

    root = corpus_dir.expanduser().resolve(strict=True)
    pattern = re.compile(r"^(case[0-9]+)-source\.mp4$")
    rows: list[CalibrationExample] = []
    for source_path in sorted(root.glob("case*-source.mp4")):
        match = pattern.match(source_path.name)
        if not match:
            continue
        case_id = match.group(1)
        base_path = root / f"{case_id}-base.mp4"
        collapsed_path = root / f"{case_id}-{collapsed_suffix}.mp4"
        if not base_path.is_file() or not collapsed_path.is_file():
            raise CollapseGateError(f"incomplete calibration triplet for {case_id}")
        source, source_identity = decode_video_exact81(
            source_path, ffmpeg=ffmpeg, ffprobe=ffprobe
        )
        base, base_identity = decode_video_exact81(
            base_path, ffmpeg=ffmpeg, ffprobe=ffprobe
        )
        collapsed, collapsed_identity = decode_video_exact81(
            collapsed_path, ffmpeg=ffmpeg, ffprobe=ffprobe
        )
        rows.append(
            CalibrationExample(
                case_id=case_id,
                source=source,
                frozen_base=base,
                collapsed=collapsed,
                identities={
                    "source": source_identity,
                    "frozen_base": base_identity,
                    "collapsed": collapsed_identity,
                },
            )
        )
    if not rows:
        raise CollapseGateError(f"no calibration triplets found in {root}")
    return calibrate_examples(
        rows,
        cohort_name=f"{root.name}:{collapsed_suffix}:clean-base-vs-collapsed",
    )


def load_calibration(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.expanduser().resolve(strict=True).read_text("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CollapseGateError(f"could not read calibration JSON: {path}") from error
    if not isinstance(value, dict):
        raise CollapseGateError("calibration JSON root must be an object")
    return _validate_calibration(value)


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        _json_safe(value),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(destination.parent),
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate = subparsers.add_parser(
        "calibrate", help="fit thresholds from clean-base/collapsed triplets"
    )
    calibrate.add_argument("--corpus-dir", type=Path, required=True)
    calibrate.add_argument("--collapsed-suffix", default="v16r3")
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--ffmpeg", default="ffmpeg")
    calibrate.add_argument("--ffprobe", default="ffprobe")

    evaluate = subparsers.add_parser("evaluate", help="gate one checkpoint sample")
    evaluate.add_argument("--source", type=Path, required=True)
    evaluate.add_argument("--candidate", type=Path, required=True)
    evaluate.add_argument("--frozen-base", type=Path)
    evaluate.add_argument("--calibration", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--sample-id", required=True)
    evaluate.add_argument("--checkpoint-step", type=int, required=True)
    evaluate.add_argument("--checkpoint-label")
    evaluate.add_argument("--ffmpeg", default="ffmpeg")
    evaluate.add_argument("--ffprobe", default="ffprobe")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "calibrate":
            calibration = calibrate_video_corpus(
                args.corpus_dir,
                collapsed_suffix=args.collapsed_suffix,
                ffmpeg=args.ffmpeg,
                ffprobe=args.ffprobe,
            )
            write_json_atomic(args.output, calibration)
            return 0

        calibration = load_calibration(args.calibration)
        source, source_identity = decode_video_exact81(
            args.source, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe
        )
        candidate, candidate_identity = decode_video_exact81(
            args.candidate, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe
        )
        frozen_base = None
        frozen_base_identity = None
        if args.frozen_base is not None:
            frozen_base, frozen_base_identity = decode_video_exact81(
                args.frozen_base, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe
            )
        metadata = {
            "sample_id": args.sample_id,
            "checkpoint_step": args.checkpoint_step,
            "checkpoint_label": args.checkpoint_label
            or f"checkpoint-{args.checkpoint_step:08d}",
            "inputs": {
                "source": source_identity,
                "candidate": candidate_identity,
                "frozen_base": frozen_base_identity,
            },
        }
        report = evaluate_visual_collapse(
            source,
            candidate,
            calibration=calibration,
            frozen_base_frames=frozen_base,
            metadata=metadata,
        )
        write_json_atomic(args.output, report)
        return 0 if report["passed"] else 2
    except (CollapseGateError, OSError, ValueError) as error:
        if args.command == "evaluate":
            report = {
                "schema_version": SCHEMA_VERSION,
                "status": "error",
                "passed": False,
                "publishable": False,
                "collapsed": None,
                "fail_closed": True,
                "failure_codes": ["evaluation_runtime_error"],
                "metadata": {
                    "sample_id": args.sample_id,
                    "checkpoint_step": args.checkpoint_step,
                    "checkpoint_label": args.checkpoint_label,
                },
                "error": str(error),
            }
            write_json_atomic(args.output, report)
            return 2
        raise SystemExit(str(error))


__all__ = [
    "ANALYSIS_HEIGHT",
    "ANALYSIS_WIDTH",
    "CALIBRATION_SCHEMA_VERSION",
    "EXPECTED_FRAME_COUNT",
    "HARD_FAILURE_THRESHOLDS",
    "SCHEMA_VERSION",
    "CalibrationExample",
    "CollapseGateError",
    "calibrate_examples",
    "calibrate_video_corpus",
    "compute_visual_features",
    "decode_video_exact81",
    "evaluate_visual_collapse",
    "load_calibration",
    "write_json_atomic",
]


if __name__ == "__main__":
    raise SystemExit(main())
