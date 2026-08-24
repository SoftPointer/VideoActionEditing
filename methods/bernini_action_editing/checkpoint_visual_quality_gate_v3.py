#!/usr/bin/env python3
"""Fail-closed, two-scale video quality gate for Bernini checkpoints.

This is a new gate and a new evidence schema.  It deliberately does not read,
rewrite, upgrade, or impersonate ``checkpoint_visual_collapse_gate_v2``.  V2
only detects catastrophic collapse; this gate additionally measures blur,
noise-like texture, route-off structural corruption, and freezing.

The four failure families are non-compensating: a confirmed trigger in any
family fails the sample.  The outcome is three-state.  Raw low-SSIM and high
temporal-residual conditions without independent artifact support are
``unresolved``: fail-closed for promotion, but never misreported as visual
collapse.  That guard is necessary because legitimate large actions can differ
strongly from a frozen base and fast source motion can have a large residual.

The CLI decodes exactly 81 frames at both 192x144 and 384x288.  Every frame and
all 80 adjacent transitions are represented in the report.  NumPy is the only
Python dependency; ffmpeg and ffprobe are used by the CLI.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Optional, Sequence

import numpy as np


SCHEMA_VERSION = "bernini-checkpoint-visual-quality-gate-v3"
EXPECTED_FRAME_COUNT = 81
ANALYSIS_SCALES = ((192, 144), (384, 288))
WINDOW_SIZE = 11
SALIENT_TILE_SIZE = 16


class QualityGateError(RuntimeError):
    """Raised when media or gate inputs violate the v3 contract."""


@dataclass(frozen=True)
class QualityThresholds:
    # NOISE candidates from the labelled-replay audit.
    hp_kurtosis_max: float = 5.0
    spectral_flatness_max: float = 0.12
    chroma_hp_ratio_max: float = 1.50
    motion_compensated_residual_ratio_max: float = 1.80
    # BLUR.
    frame_hp_retention_p10_min: float = 0.55
    frame_laplacian_var_retention_p10_min: float = 0.25
    salient_tile_retention_min: float = 0.50
    salient_bad_tile_fraction: float = 0.25
    salient_bad_frame_fraction: float = 0.20
    # ROUTEOFF_STRUCTURE raw candidates.  These are not sufficient alone.
    windowed_ssim_to_base_min: float = 0.70
    global_ssim_to_base_min: float = 0.85
    edge_correlation_to_base_min: float = 0.50
    structure_support_tile_retention_min: float = 0.65
    structure_support_bad_frame_fraction: float = 0.20
    base_source_global_ssim_for_structure_reference_min: float = 0.30
    # FREEZE.
    near_duplicate_windowed_ssim_min: float = 0.995
    near_duplicate_fraction_min: float = 0.90
    near_duplicate_excess_over_base_min: float = 0.50


THRESHOLDS = QualityThresholds()


def _finite(value: Any, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise QualityGateError(f"{label} is not finite")
    return result


def _json_safe(value: Any) -> Any:
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _scale_key(width: int, height: int) -> str:
    return f"{width}x{height}"


def _normalise_video(value: Any, *, label: str, width: int, height: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise QualityGateError(f"{label} must be a numpy ndarray")
    expected = (EXPECTED_FRAME_COUNT, height, width, 3)
    if value.shape != expected:
        raise QualityGateError(f"{label} must have shape {expected}, got {value.shape}")
    if np.issubdtype(value.dtype, np.bool_) or np.issubdtype(
        value.dtype, np.complexfloating
    ):
        raise QualityGateError(f"{label} has invalid RGB dtype {value.dtype}")
    if np.issubdtype(value.dtype, np.integer):
        minimum, maximum = int(np.min(value)), int(np.max(value))
        if minimum < 0 or maximum > 255:
            raise QualityGateError(f"{label} integer RGB is outside [0,255]")
        return np.ascontiguousarray(value.astype(np.float32) / 255.0)
    if np.issubdtype(value.dtype, np.floating):
        result = value.astype(np.float32, copy=False)
        if not bool(np.all(np.isfinite(result))):
            raise QualityGateError(f"{label} contains NaN or infinity")
        if float(np.min(result)) < -1e-6 or float(np.max(result)) > 1.0 + 1e-6:
            raise QualityGateError(f"{label} floating RGB is outside [0,1]")
        return np.ascontiguousarray(np.clip(result, 0.0, 1.0))
    raise QualityGateError(f"{label} has non-numeric RGB dtype {value.dtype}")


def _validate_scale_maps(
    source_by_scale: Mapping[str, Any],
    candidate_by_scale: Mapping[str, Any],
    base_by_scale: Mapping[str, Any],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    expected_keys = {_scale_key(width, height) for width, height in ANALYSIS_SCALES}
    for label, value in (
        ("source", source_by_scale),
        ("candidate", candidate_by_scale),
        ("frozen_base", base_by_scale),
    ):
        if not isinstance(value, Mapping) or set(value) != expected_keys:
            raise QualityGateError(
                f"{label} scales must be exactly {sorted(expected_keys)}"
            )
    result = {}
    for width, height in ANALYSIS_SCALES:
        key = _scale_key(width, height)
        result[key] = (
            _normalise_video(source_by_scale[key], label=f"source[{key}]", width=width, height=height),
            _normalise_video(candidate_by_scale[key], label=f"candidate[{key}]", width=width, height=height),
            _normalise_video(base_by_scale[key], label=f"frozen_base[{key}]", width=width, height=height),
        )
    return result


def _luma(video: np.ndarray) -> np.ndarray:
    return (
        0.2126 * video[..., 0]
        + 0.7152 * video[..., 1]
        + 0.0722 * video[..., 2]
    ).astype(np.float32)


def _highpass3(value: np.ndarray) -> np.ndarray:
    padded = np.pad(value, ((0, 0), (1, 1), (1, 1)), mode="reflect")
    local_mean = np.zeros_like(value, dtype=np.float32)
    height, width = value.shape[1:]
    for dy in range(3):
        for dx in range(3):
            local_mean += padded[:, dy : dy + height, dx : dx + width]
    local_mean /= 9.0
    return value - local_mean


def _laplacian(value: np.ndarray) -> np.ndarray:
    padded = np.pad(value, ((0, 0), (1, 1), (1, 1)), mode="reflect")
    height, width = value.shape[1:]
    centre = padded[:, 1 : 1 + height, 1 : 1 + width]
    return (
        4.0 * centre
        - padded[:, :height, 1 : 1 + width]
        - padded[:, 2 : 2 + height, 1 : 1 + width]
        - padded[:, 1 : 1 + height, :width]
        - padded[:, 1 : 1 + height, 2 : 2 + width]
    )


def _gradient_magnitude(value: np.ndarray) -> np.ndarray:
    gx = value[:, 1:-1, 2:] - value[:, 1:-1, :-2]
    gy = value[:, 2:, 1:-1] - value[:, :-2, 1:-1]
    return np.sqrt(gx * gx + gy * gy)


def _correlation_per_frame(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x = left.reshape(left.shape[0], -1).astype(np.float64)
    y = right.reshape(right.shape[0], -1).astype(np.float64)
    x -= np.mean(x, axis=1, keepdims=True)
    y -= np.mean(y, axis=1, keepdims=True)
    numerator = np.sum(x * y, axis=1)
    denominator = np.sqrt(np.sum(x * x, axis=1) * np.sum(y * y, axis=1))
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )


def _global_ssim_per_frame(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x = left.reshape(left.shape[0], -1).astype(np.float64)
    y = right.reshape(right.shape[0], -1).astype(np.float64)
    mean_x = np.mean(x, axis=1)
    mean_y = np.mean(y, axis=1)
    var_x = np.var(x, axis=1)
    var_y = np.var(y, axis=1)
    covariance = np.mean(
        (x - mean_x[:, None]) * (y - mean_y[:, None]), axis=1
    )
    c1, c2 = 0.01**2, 0.03**2
    denominator = (mean_x * mean_x + mean_y * mean_y + c1) * (
        var_x + var_y + c2
    )
    return np.divide(
        (2.0 * mean_x * mean_y + c1) * (2.0 * covariance + c2),
        denominator,
        out=np.ones_like(denominator),
        where=denominator > 1e-15,
    )


def _box_mean(images: np.ndarray, window: int) -> np.ndarray:
    radius = window // 2
    padded = np.pad(
        images, ((0, 0), (radius, radius), (radius, radius)), mode="reflect"
    )
    integral = np.pad(
        np.cumsum(np.cumsum(padded, axis=1, dtype=np.float32), axis=2, dtype=np.float32),
        ((0, 0), (1, 0), (1, 0)),
        mode="constant",
    )
    sums = (
        integral[:, window:, window:]
        - integral[:, :-window, window:]
        - integral[:, window:, :-window]
        + integral[:, :-window, :-window]
    )
    return sums / float(window * window)


def _windowed_ssim_per_frame(
    left: np.ndarray, right: np.ndarray, *, window: int = WINDOW_SIZE
) -> np.ndarray:
    """Mean of a genuine local-window SSIM map for every frame."""

    if window % 2 != 1 or window < 3:
        raise QualityGateError("SSIM window must be odd and >=3")
    result = []
    # Chunking bounds peak RAM for the 384x288 scale.
    for start in range(0, left.shape[0], 4):
        x = left[start : start + 4].astype(np.float32, copy=False)
        y = right[start : start + 4].astype(np.float32, copy=False)
        mean_x = _box_mean(x, window)
        mean_y = _box_mean(y, window)
        var_x = np.maximum(_box_mean(x * x, window) - mean_x * mean_x, 0.0)
        var_y = np.maximum(_box_mean(y * y, window) - mean_y * mean_y, 0.0)
        covariance = _box_mean(x * y, window) - mean_x * mean_y
        c1, c2 = 0.01**2, 0.03**2
        denominator = (mean_x * mean_x + mean_y * mean_y + c1) * (
            var_x + var_y + c2
        )
        ssim = np.divide(
            (2.0 * mean_x * mean_y + c1) * (2.0 * covariance + c2),
            denominator,
            out=np.ones_like(denominator),
            where=denominator > 1e-12,
        )
        result.extend(np.mean(ssim, axis=(1, 2)).tolist())
    return np.asarray(result, dtype=np.float64)


def _highpass_kurtosis_per_frame(highpass: np.ndarray) -> np.ndarray:
    flattened = highpass.reshape(highpass.shape[0], -1).astype(np.float64)
    flattened -= np.mean(flattened, axis=1, keepdims=True)
    variance = np.mean(flattened * flattened, axis=1)
    fourth = np.mean(flattened**4, axis=1)
    return np.divide(
        fourth,
        variance * variance,
        out=np.full_like(variance, np.inf),
        where=variance > 1e-14,
    )


def _spectral_flatness_per_frame(highpass: np.ndarray) -> np.ndarray:
    result = []
    # Cap FFT geometry while retaining all 81 frames.
    stride = max(1, int(math.ceil(highpass.shape[2] / 192)))
    for frame in highpass[:, ::stride, ::stride]:
        power = np.abs(np.fft.rfft2(frame.astype(np.float64))) ** 2
        values = power.reshape(-1)[1:]
        arithmetic = float(np.mean(values))
        if arithmetic <= 1e-20:
            result.append(0.0)
        else:
            geometric = float(np.exp(np.mean(np.log(values + 1e-20))))
            result.append(geometric / arithmetic)
    return np.asarray(result, dtype=np.float64)


def _chroma_highpass_rms(video: np.ndarray) -> np.ndarray:
    red_green = video[..., 0] - video[..., 1]
    blue_green = video[..., 2] - video[..., 1]
    hp_rg = _highpass3(red_green)
    hp_bg = _highpass3(blue_green)
    return np.sqrt(np.mean(0.5 * (hp_rg * hp_rg + hp_bg * hp_bg), axis=(1, 2)))


def _salient_tile_retention(
    candidate_hp: np.ndarray,
    base_hp: np.ndarray,
    *,
    retention_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    height = (candidate_hp.shape[1] // SALIENT_TILE_SIZE) * SALIENT_TILE_SIZE
    width = (candidate_hp.shape[2] // SALIENT_TILE_SIZE) * SALIENT_TILE_SIZE
    candidate = candidate_hp[:, :height, :width]
    base = base_hp[:, :height, :width]
    frames = candidate.shape[0]
    candidate_energy = np.sqrt(
        np.mean(
            candidate.reshape(
                frames,
                height // SALIENT_TILE_SIZE,
                SALIENT_TILE_SIZE,
                width // SALIENT_TILE_SIZE,
                SALIENT_TILE_SIZE,
            )
            ** 2,
            axis=(2, 4),
        )
    )
    base_energy = np.sqrt(
        np.mean(
            base.reshape(
                frames,
                height // SALIENT_TILE_SIZE,
                SALIENT_TILE_SIZE,
                width // SALIENT_TILE_SIZE,
                SALIENT_TILE_SIZE,
            )
            ** 2,
            axis=(2, 4),
        )
    )
    result = []
    salient_counts = []
    for index in range(frames):
        threshold = max(float(np.percentile(base_energy[index], 75)), 1e-5)
        salient = base_energy[index] >= threshold
        # A requested action can move an edge without blurring it.  Match each
        # base-salient tile to the strongest candidate tile in a local 5x5
        # neighbourhood (32 analysis pixels in either direction) instead of
        # requiring coordinate identity.  This is still local retention, but
        # does not confuse ordinary pose/object displacement with blur.
        tile_map = candidate_energy[index]
        padded = np.pad(tile_map, ((2, 2), (2, 2)), mode="edge")
        local_maximum = np.zeros_like(tile_map)
        for dy in range(5):
            for dx in range(5):
                local_maximum = np.maximum(
                    local_maximum,
                    padded[dy : dy + tile_map.shape[0], dx : dx + tile_map.shape[1]],
                )
        retention = local_maximum / np.maximum(base_energy[index], 1e-6)
        result.append(float(np.mean(retention[salient] < retention_threshold)))
        salient_counts.append(int(np.sum(salient)))
    return np.asarray(result, dtype=np.float64), np.asarray(salient_counts, dtype=np.int64)


def _motion_compensated_residuals(luma: np.ndarray) -> np.ndarray:
    # Search integer translations on a bounded 96x72-ish representation.
    stride = max(1, int(math.ceil(luma.shape[2] / 96)))
    small = luma[:, ::stride, ::stride]
    result = []
    radius = 3
    for index in range(1, small.shape[0]):
        previous = small[index - 1]
        current = small[index]
        best = math.inf
        for dy in range(-radius, radius + 1):
            if dy >= 0:
                p_y, c_y = slice(dy, None), slice(None, -dy or None)
            else:
                p_y, c_y = slice(None, dy), slice(-dy, None)
            for dx in range(-radius, radius + 1):
                if dx >= 0:
                    p_x, c_x = slice(dx, None), slice(None, -dx or None)
                else:
                    p_x, c_x = slice(None, dx), slice(-dx, None)
                residual = float(
                    np.mean(np.abs(previous[p_y, p_x] - current[c_y, c_x]))
                )
                best = min(best, residual)
        result.append(best)
    return np.asarray(result, dtype=np.float64)


def _scale_features(
    source: np.ndarray,
    candidate: np.ndarray,
    frozen_base: np.ndarray,
    *,
    scale_key: str,
) -> dict[str, Any]:
    # Source is contractually aligned even though quality ratios use the frozen
    # base.  Keeping it in the signature prevents silent wrong-sample pairing.
    if source.shape != candidate.shape or source.shape != frozen_base.shape:
        raise QualityGateError(f"unaligned input geometry at {scale_key}")

    source_luma = _luma(source)
    candidate_luma = _luma(candidate)
    base_luma = _luma(frozen_base)
    candidate_hp = _highpass3(candidate_luma)
    base_hp = _highpass3(base_luma)
    candidate_laplacian = _laplacian(candidate_luma)
    base_laplacian = _laplacian(base_luma)

    epsilon = 1e-8
    candidate_hp_rms = np.sqrt(np.mean(candidate_hp * candidate_hp, axis=(1, 2)))
    base_hp_rms = np.sqrt(np.mean(base_hp * base_hp, axis=(1, 2)))
    hp_retention = candidate_hp_rms / np.maximum(base_hp_rms, epsilon)
    candidate_lap_var = np.var(candidate_laplacian, axis=(1, 2))
    base_lap_var = np.var(base_laplacian, axis=(1, 2))
    lap_retention = candidate_lap_var / np.maximum(base_lap_var, epsilon)
    kurtosis = _highpass_kurtosis_per_frame(candidate_hp)
    spectral_flatness = _spectral_flatness_per_frame(candidate_hp)
    candidate_chroma_hp = _chroma_highpass_rms(candidate)
    base_chroma_hp = _chroma_highpass_rms(frozen_base)
    chroma_ratio_per_frame = candidate_chroma_hp / np.maximum(base_chroma_hp, epsilon)

    salient_bad_fraction, salient_count = _salient_tile_retention(
        candidate_hp,
        base_hp,
        retention_threshold=THRESHOLDS.salient_tile_retention_min,
    )
    structure_bad_fraction, _ = _salient_tile_retention(
        candidate_hp,
        base_hp,
        retention_threshold=THRESHOLDS.structure_support_tile_retention_min,
    )

    windowed_ssim = _windowed_ssim_per_frame(candidate_luma, base_luma)
    global_ssim = _global_ssim_per_frame(candidate_luma, base_luma)
    candidate_source_global_ssim = _global_ssim_per_frame(
        candidate_luma, source_luma
    )
    base_source_global_ssim = _global_ssim_per_frame(base_luma, source_luma)
    edge_correlation = _correlation_per_frame(
        _gradient_magnitude(candidate_luma), _gradient_magnitude(base_luma)
    )
    candidate_duplicate_ssim = _windowed_ssim_per_frame(
        candidate_luma[:-1], candidate_luma[1:]
    )
    base_duplicate_ssim = _windowed_ssim_per_frame(
        base_luma[:-1], base_luma[1:]
    )
    candidate_mc = _motion_compensated_residuals(candidate_luma)
    base_mc = _motion_compensated_residuals(base_luma)
    mc_ratio = float(np.median(candidate_mc)) / max(float(np.median(base_mc)), epsilon)

    scalars = {
        "hp_kurtosis_median": _finite(np.median(kurtosis), label="HP kurtosis"),
        "spectral_flatness_median": _finite(
            np.median(spectral_flatness), label="spectral flatness"
        ),
        "chroma_hp_ratio_to_base": _finite(
            np.percentile(chroma_ratio_per_frame, 90), label="chroma HP ratio"
        ),
        "motion_compensated_temporal_residual_ratio_to_base": _finite(
            mc_ratio, label="motion-compensated residual ratio"
        ),
        "frame_hp_rms_retention_to_base_p10": _finite(
            np.percentile(hp_retention, 10), label="HP retention p10"
        ),
        "frame_laplacian_var_retention_to_base_p10": _finite(
            np.percentile(lap_retention, 10), label="Laplacian retention p10"
        ),
        "base_salient_low_retention_frame_fraction": _finite(
            np.mean(salient_bad_fraction > THRESHOLDS.salient_bad_tile_fraction),
            label="salient low-retention frame fraction",
        ),
        "structure_support_low_retention_frame_fraction": _finite(
            np.mean(structure_bad_fraction > THRESHOLDS.salient_bad_tile_fraction),
            label="structure support frame fraction",
        ),
        "candidate_base_windowed_ssim_mean": _finite(
            np.mean(windowed_ssim), label="windowed SSIM"
        ),
        "candidate_base_global_ssim_mean": _finite(
            np.mean(global_ssim), label="global SSIM"
        ),
        "candidate_base_edge_correlation_mean": _finite(
            np.mean(edge_correlation), label="edge correlation"
        ),
        "candidate_source_global_ssim_mean": _finite(
            np.mean(candidate_source_global_ssim), label="candidate/source SSIM"
        ),
        "base_source_global_ssim_mean": _finite(
            np.mean(base_source_global_ssim), label="base/source SSIM"
        ),
        "candidate_near_duplicate_transition_fraction": _finite(
            np.mean(candidate_duplicate_ssim >= THRESHOLDS.near_duplicate_windowed_ssim_min),
            label="candidate duplicate fraction",
        ),
        "base_near_duplicate_transition_fraction": _finite(
            np.mean(base_duplicate_ssim >= THRESHOLDS.near_duplicate_windowed_ssim_min),
            label="base duplicate fraction",
        ),
    }
    scalars["near_duplicate_fraction_excess_over_base"] = _finite(
        scalars["candidate_near_duplicate_transition_fraction"]
        - scalars["base_near_duplicate_transition_fraction"],
        label="duplicate excess",
    )

    frame_metrics = []
    for index in range(EXPECTED_FRAME_COUNT):
        frame_metrics.append(
            {
                "frame_index": index,
                "hp_rms": _finite(candidate_hp_rms[index], label="frame HP RMS"),
                "base_hp_rms": _finite(base_hp_rms[index], label="base HP RMS"),
                "hp_retention_to_base": _finite(hp_retention[index], label="HP retention"),
                "laplacian_variance": _finite(candidate_lap_var[index], label="lap variance"),
                "base_laplacian_variance": _finite(base_lap_var[index], label="base lap variance"),
                "laplacian_var_retention_to_base": _finite(
                    lap_retention[index], label="lap retention"
                ),
                "hp_kurtosis": _finite(kurtosis[index], label="HP kurtosis"),
                "spectral_flatness": _finite(
                    spectral_flatness[index], label="spectral flatness"
                ),
                "chroma_hp_ratio_to_base": _finite(
                    chroma_ratio_per_frame[index], label="chroma ratio"
                ),
                "base_salient_low_retention_tile_fraction": _finite(
                    salient_bad_fraction[index], label="salient tile fraction"
                ),
                "base_salient_tile_count": int(salient_count[index]),
                "candidate_base_windowed_ssim": _finite(
                    windowed_ssim[index], label="windowed SSIM"
                ),
                "candidate_base_global_ssim": _finite(
                    global_ssim[index], label="global SSIM"
                ),
                "candidate_base_edge_correlation": _finite(
                    edge_correlation[index], label="edge correlation"
                ),
            }
        )

    transition_metrics = []
    for index in range(EXPECTED_FRAME_COUNT - 1):
        transition_metrics.append(
            {
                "source_frame_index": index,
                "destination_frame_index": index + 1,
                "near_duplicate_windowed_ssim": _finite(
                    candidate_duplicate_ssim[index], label="duplicate SSIM"
                ),
                "base_near_duplicate_windowed_ssim": _finite(
                    base_duplicate_ssim[index], label="base duplicate SSIM"
                ),
                "motion_compensated_residual": _finite(
                    candidate_mc[index], label="MC residual"
                ),
                "base_motion_compensated_residual": _finite(
                    base_mc[index], label="base MC residual"
                ),
            }
        )

    return {
        "analysis_geometry": {
            "width": int(candidate.shape[2]),
            "height": int(candidate.shape[1]),
        },
        "windowed_ssim": {
            "implemented_as_local_map": True,
            "window_width": WINDOW_SIZE,
            "window_height": WINDOW_SIZE,
            "frame_reduction": "arithmetic mean of local SSIM map",
        },
        "scalars": scalars,
        "frame_metrics": frame_metrics,
        "transition_metrics": transition_metrics,
    }


def _decision(scale_features: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    per_scale = {}
    for key, features in scale_features.items():
        values = features["scalars"]
        noise_distribution = bool(
            values["hp_kurtosis_median"] < THRESHOLDS.hp_kurtosis_max
            and values["spectral_flatness_median"] < THRESHOLDS.spectral_flatness_max
        )
        chroma_noise = bool(
            values["chroma_hp_ratio_to_base"] > THRESHOLDS.chroma_hp_ratio_max
        )
        distribution_excess_support = bool(
            values["frame_hp_rms_retention_to_base_p10"] > 1.15
        )
        high_mc_raw = bool(
            values["motion_compensated_temporal_residual_ratio_to_base"]
            > THRESHOLDS.motion_compensated_residual_ratio_max
        )
        spatial_noise_support = bool(
            (noise_distribution and distribution_excess_support)
            or chroma_noise
            or values["frame_hp_rms_retention_to_base_p10"] > 1.50
        )
        # High residual alone is unsafe for fast-motion sources.  It is retained
        # as a raw trigger but needs an independent spatial noise signature.
        noise = bool(
            (noise_distribution and distribution_excess_support)
            or chroma_noise
            or (high_mc_raw and spatial_noise_support)
        )

        blur_global = bool(
            values["frame_hp_rms_retention_to_base_p10"]
            < THRESHOLDS.frame_hp_retention_p10_min
            and values["frame_laplacian_var_retention_to_base_p10"]
            < THRESHOLDS.frame_laplacian_var_retention_p10_min
        )
        blur_local = bool(
            values["base_salient_low_retention_frame_fraction"]
            >= THRESHOLDS.salient_bad_frame_fraction
        )
        blur = bool(blur_global or blur_local)

        low_windowed = bool(
            values["candidate_base_windowed_ssim_mean"]
            < THRESHOLDS.windowed_ssim_to_base_min
        )
        low_global_edge = bool(
            values["candidate_base_global_ssim_mean"]
            < THRESHOLDS.global_ssim_to_base_min
            and values["candidate_base_edge_correlation_mean"]
            < THRESHOLDS.edge_correlation_to_base_min
        )
        structure_raw = bool(low_windowed or low_global_edge)
        structure_spatial_support = bool(
            values["structure_support_low_retention_frame_fraction"]
            >= THRESHOLDS.structure_support_bad_frame_fraction
            or (noise_distribution and distribution_excess_support)
            or chroma_noise
            or blur_global
        )
        # Base-relative structure is a hard reference only when the frozen base
        # itself remains sufficiently aligned with the source.  Otherwise a
        # low candidate/base SSIM can simply reflect a legitimate large action;
        # in that regime an independent spatial artifact is required.
        base_structure_reference_eligible = bool(
            values["base_source_global_ssim_mean"]
            >= THRESHOLDS.base_source_global_ssim_for_structure_reference_min
        )
        routeoff_structure = bool(structure_raw and structure_spatial_support)
        routeoff_unresolved = bool(structure_raw and not structure_spatial_support)

        freeze = bool(
            values["candidate_near_duplicate_transition_fraction"]
            >= THRESHOLDS.near_duplicate_fraction_min
            and values["near_duplicate_fraction_excess_over_base"]
            >= THRESHOLDS.near_duplicate_excess_over_base_min
        )
        per_scale[key] = {
            "NOISE": {
                "triggered": noise,
                "unresolved": bool(high_mc_raw and not spatial_noise_support),
                "raw_conditions": {
                    "low_kurtosis_and_low_spectral_flatness": noise_distribution,
                    "distribution_has_hp_excess_over_base": distribution_excess_support,
                    "chroma_hp_ratio_above_1p50": chroma_noise,
                    "motion_compensated_residual_ratio_above_1p80": high_mc_raw,
                    "motion_residual_has_spatial_support": bool(high_mc_raw and spatial_noise_support),
                },
            },
            "BLUR": {
                "triggered": blur,
                "unresolved": False,
                "raw_conditions": {
                    "p10_hp_below_0p55_and_laplacian_below_0p25": blur_global,
                    "salient_tile_bad_frame_fraction_at_least_0p20": blur_local,
                },
            },
            "ROUTEOFF_STRUCTURE": {
                "triggered": routeoff_structure,
                "unresolved": routeoff_unresolved,
                "raw_candidate_triggered": structure_raw,
                "independent_spatial_artifact_support": structure_spatial_support,
                "base_structure_reference_eligible": base_structure_reference_eligible,
                "raw_conditions": {
                    "windowed_ssim_below_0p70": low_windowed,
                    "global_ssim_below_0p85_and_edge_corr_below_0p50": low_global_edge,
                },
            },
            "FREEZE": {
                "triggered": freeze,
                "unresolved": False,
                "raw_conditions": {
                    "candidate_near_duplicate_fraction_at_least_0p90": bool(
                        values["candidate_near_duplicate_transition_fraction"]
                        >= THRESHOLDS.near_duplicate_fraction_min
                    ),
                    "near_duplicate_excess_over_base_at_least_0p50": bool(
                        values["near_duplicate_fraction_excess_over_base"]
                        >= THRESHOLDS.near_duplicate_excess_over_base_min
                    ),
                },
            },
        }

    families = {}
    for family in ("NOISE", "BLUR", "ROUTEOFF_STRUCTURE", "FREEZE"):
        triggered_scales = [
            key for key, decisions in per_scale.items() if decisions[family]["triggered"]
        ]
        families[family] = {
            "triggered": bool(triggered_scales),
            "triggered_scales": triggered_scales,
            "unresolved": any(
                bool(decisions[family].get("unresolved"))
                for decisions in per_scale.values()
            ),
            "unresolved_scales": [
                key
                for key, decisions in per_scale.items()
                if decisions[family].get("unresolved")
            ],
            "per_scale": {key: decisions[family] for key, decisions in per_scale.items()},
        }
    failure_codes = [f"quality_{name.lower()}" for name, row in families.items() if row["triggered"]]
    unresolved_codes = [
        f"quality_{name.lower()}_requires_external_verifier"
        for name, row in families.items()
        if row["unresolved"] and not row["triggered"]
    ]
    if failure_codes:
        outcome = "fail"
    elif unresolved_codes:
        outcome = "unresolved"
    else:
        outcome = "pass"
    return {
        "outcome": outcome,
        "passed": outcome == "pass",
        "hard_artifact_failure": outcome == "fail",
        "unresolved": outcome == "unresolved",
        "failure_codes": failure_codes,
        "unresolved_codes": unresolved_codes,
        "decision_rule": "NOISE OR BLUR OR ROUTEOFF_STRUCTURE OR FREEZE",
        "family_combination": "non-compensating OR",
        "evidence_families": families,
    }


def evaluate_visual_quality(
    source_frames_by_scale: Mapping[str, Any],
    candidate_frames_by_scale: Mapping[str, Any],
    *,
    frozen_base_frames_by_scale: Mapping[str, Any],
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Evaluate one aligned source/candidate/base triplet, failing closed."""

    prefix = {
        "schema_version": SCHEMA_VERSION,
        "fail_closed": True,
        "expected_frame_count": EXPECTED_FRAME_COUNT,
        "required_analysis_scales": [
            {"width": width, "height": height} for width, height in ANALYSIS_SCALES
        ],
        "thresholds": asdict(THRESHOLDS),
        "metadata": _json_safe(metadata or {}),
    }
    try:
        inputs = _validate_scale_maps(
            source_frames_by_scale,
            candidate_frames_by_scale,
            frozen_base_frames_by_scale,
        )
        features = {}
        for key, (source, candidate, frozen_base) in inputs.items():
            features[key] = _scale_features(
                source, candidate, frozen_base, scale_key=key
            )
        decision = _decision(features)
    except (QualityGateError, TypeError, ValueError, FloatingPointError) as error:
        return {
            **prefix,
            "status": "error",
            "passed": False,
            "publishable": False,
            "input_contract_passed": False,
            "failure_codes": ["quality_gate_input_contract_violation"],
            "error": str(error),
        }

    return _json_safe(
        {
            **prefix,
            "status": decision["outcome"],
            "passed": decision["passed"],
            "publishable": decision["passed"],
            "hard_artifact_failure": decision["hard_artifact_failure"],
            "unresolved": decision["unresolved"],
            "input_contract_passed": True,
            "failure_codes": decision["failure_codes"],
            "unresolved_codes": decision["unresolved_codes"],
            "decision": decision,
            "features": {
                "metric_scope": "all 81 frames and all 80 adjacent transitions at both scales",
                "all_frames_evaluated": True,
                "all_transitions_evaluated": True,
                "evaluated_frame_count_per_scale": EXPECTED_FRAME_COUNT,
                "evaluated_transition_count_per_scale": EXPECTED_FRAME_COUNT - 1,
                "scales": features,
            },
        }
    )


def _run_json(command: Sequence[str], *, label: str) -> dict[str, Any]:
    result = subprocess.run(
        list(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise QualityGateError(f"{label} failed ({result.returncode}): {error}")
    try:
        value = json.loads(result.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualityGateError(f"{label} did not emit JSON") from error
    if not isinstance(value, dict):
        raise QualityGateError(f"{label} JSON root is not an object")
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


def decode_video_exact81_multiscale(
    path: Path,
    *,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
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
        raise QualityGateError(f"{source} must expose exactly one video stream")
    stream = streams[0]
    decoded = {}
    for width, height in ANALYSIS_SCALES:
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
                f"scale={width}:{height}:flags=area",
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
            raise QualityGateError(f"ffmpeg decode failed for {source}: {error}")
        frame_bytes = width * height * 3
        if len(result.stdout) % frame_bytes:
            raise QualityGateError(f"ffmpeg emitted a partial RGB frame for {source}")
        frame_count = len(result.stdout) // frame_bytes
        if frame_count != EXPECTED_FRAME_COUNT:
            raise QualityGateError(
                f"{source} decoded to {frame_count} frames, expected {EXPECTED_FRAME_COUNT}"
            )
        decoded[_scale_key(width, height)] = np.frombuffer(
            result.stdout, dtype=np.uint8
        ).reshape(EXPECTED_FRAME_COUNT, height, width, 3)
    rate_text = str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0")
    identity = {
        "path": str(source),
        "sha256": _file_sha256(source),
        "size_bytes": source.stat().st_size,
        "codec_name": stream.get("codec_name"),
        "original_width": int(stream.get("width", 0)),
        "original_height": int(stream.get("height", 0)),
        "reported_frame_count": stream.get("nb_frames"),
        "decoded_frame_count_per_scale": EXPECTED_FRAME_COUNT,
        "reported_frame_rate": rate_text,
        "reported_frame_rate_float": _finite(_parse_rate(rate_text), label="frame rate"),
        "analysis_scales": [
            {"width": width, "height": height} for width, height in ANALYSIS_SCALES
        ],
    }
    return decoded, identity


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        _json_safe(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
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
    parser.add_argument("evaluate", nargs="?")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--frozen-base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--checkpoint-label")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        source, source_identity = decode_video_exact81_multiscale(
            args.source, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe
        )
        candidate, candidate_identity = decode_video_exact81_multiscale(
            args.candidate, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe
        )
        frozen_base, base_identity = decode_video_exact81_multiscale(
            args.frozen_base, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe
        )
        report = evaluate_visual_quality(
            source,
            candidate,
            frozen_base_frames_by_scale=frozen_base,
            metadata={
                "sample_id": args.sample_id,
                "checkpoint_step": args.checkpoint_step,
                "checkpoint_label": args.checkpoint_label
                or f"checkpoint-{args.checkpoint_step:08d}",
                "inputs": {
                    "source": source_identity,
                    "candidate": candidate_identity,
                    "frozen_base": base_identity,
                },
            },
        )
        write_json_atomic(args.output, report)
        return 0 if report["passed"] else 2
    except (QualityGateError, OSError, ValueError) as error:
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "error",
            "passed": False,
            "publishable": False,
            "fail_closed": True,
            "input_contract_passed": False,
            "failure_codes": ["quality_gate_runtime_error"],
            "metadata": {
                "sample_id": args.sample_id,
                "checkpoint_step": args.checkpoint_step,
                "checkpoint_label": args.checkpoint_label,
            },
            "error": str(error),
        }
        write_json_atomic(args.output, report)
        return 2


__all__ = [
    "ANALYSIS_SCALES",
    "EXPECTED_FRAME_COUNT",
    "QualityGateError",
    "QualityThresholds",
    "SCHEMA_VERSION",
    "THRESHOLDS",
    "decode_video_exact81_multiscale",
    "evaluate_visual_quality",
    "write_json_atomic",
]


if __name__ == "__main__":
    raise SystemExit(main())
