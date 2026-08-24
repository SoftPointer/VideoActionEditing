#!/usr/bin/env python3
"""Render the frozen E00 v15b source-role observer diagnostics.

The report is deliberately diagnostic.  It visualizes raw per-block affinity,
control margins, and hand-audited source-instance ROIs, but it never converts
those maps into an attention route or an action-success label.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
from safetensors.numpy import load_file


SCHEMA_VERSION = "bernini-e00-source-instance-role-overlay-review-v15b"
SOURCE_SHA256 = "888789206a3120c0780be8961dee7fdda520502cb95f573fe211b269aaea53de"
PROBE_RECEIPT_FILE_SHA256 = (
    "1048ba5c86102311bbc57a2353ce1234ca45275428cada4a13c550050c669078"
)
DIAGNOSTIC_TENSOR_FILE_SHA256 = (
    "1ac1d1643b71aae3275660d45035a73817c980a8c5dde952458acfb1c6bc94c2"
)
R6_PROBE_RECEIPT_FILE_SHA256 = (
    "8f081c990edd84a64ca35e78ca1de3d4ea6cf4b80bfcdec70bf54c51dc9ed959"
)
R6_DIAGNOSTIC_TENSOR_FILE_SHA256 = (
    "2535193d41a3405460bd152cd77bc61db7ef8ea6ba7cefd98f514f0787acc553"
)
R4_PROFILE = "r4_legacy_averaged_null"
R6_PROFILE = "r6_explicit_null64"
ROLE_NAMES = ("agent", "old_actor", "new_actor", "recipient", "support")
VESSELS = ("old_actor", "new_actor", "recipient")
BLOCKS = (4, 9, 14, 19, 24)
DISPLAY_PHASES = (0, 5, 10, 15, 20)
GRID_HEIGHT = 37
GRID_WIDTH = 25
DISPLAY_HEIGHT = 592
DISPLAY_WIDTH = 400

# These are visual-audit ROIs in the 37x25 patch grid (x0,y0,x1,y1,
# exclusive).  They describe the real source only and are forbidden as model
# inputs.  They are intentionally broad enough for the observed vessel motion.
ROIS = {
    "old_actor": (7, 8, 18, 21),
    "new_actor": (8, 16, 21, 29),
    "recipient": (0, 25, 6, 33),
}
ROLE_COLORS_BGR = {
    "old_actor": (60, 80, 245),
    "new_actor": (75, 220, 90),
    "recipient": (235, 185, 45),
}


class V15BReviewError(RuntimeError):
    """Fail-closed review input or render error."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V15BReviewError(f"cannot parse {path}") from error
    if not isinstance(value, Mapping):
        raise V15BReviewError(f"{path} is not one JSON object")
    return value


def _validate_inputs(
    receipt_path: Path, tensor_path: Path, source_path: Path
) -> tuple[Mapping[str, Any], Mapping[str, np.ndarray], list[np.ndarray], str]:
    if _sha256_file(source_path) != SOURCE_SHA256:
        raise V15BReviewError("source video SHA differs")
    receipt_file_sha256 = _sha256_file(receipt_path)
    tensor_file_sha256 = _sha256_file(tensor_path)
    identity = (receipt_file_sha256, tensor_file_sha256)
    if identity == (PROBE_RECEIPT_FILE_SHA256, DIAGNOSTIC_TENSOR_FILE_SHA256):
        profile = R4_PROFILE
    elif identity == (
        R6_PROBE_RECEIPT_FILE_SHA256,
        R6_DIAGNOSTIC_TENSOR_FILE_SHA256,
    ):
        profile = R6_PROFILE
    elif receipt_file_sha256 not in (
        PROBE_RECEIPT_FILE_SHA256,
        R6_PROBE_RECEIPT_FILE_SHA256,
    ):
        raise V15BReviewError("probe receipt file SHA differs")
    else:
        raise V15BReviewError("diagnostic tensor file SHA differs")
    receipt = _read_json(receipt_path)
    diagnostics = receipt.get("diagnostics")
    expected_schema = (
        "bernini-source-owned-instance-role-sp4-probe-v15b"
        if profile == R4_PROFILE
        else "bernini-source-owned-instance-role-null64-sp4-probe-v15b-r6"
    )
    if (
        receipt.get("schema_version") != expected_schema
        or receipt.get("role_names") != list(ROLE_NAMES)
        or receipt.get("localization_semantically_certified") is not False
        or receipt.get("action_success_certified") is not False
        or not isinstance(diagnostics, Mapping)
        or diagnostics.get("file_sha256") != tensor_file_sha256
    ):
        raise V15BReviewError("probe receipt/tensor identity differs")
    expected_gates = receipt.get("mechanical_gates")
    if not isinstance(expected_gates, Mapping) or not expected_gates or not all(
        value is True for value in expected_gates.values()
    ):
        raise V15BReviewError("frozen observer mechanical gates did not all pass")
    tensors = load_file(str(tensor_path))
    if profile == R4_PROFILE:
        expected_keys = {
            "aggregate_affinity",
            "aggregate_null_affinity",
            "aggregate_shuffled_affinity",
            "aggregate_group_masks_u8",
        }
        for block in BLOCKS:
            expected_keys.update(
                {
                    f"block_{block:02d}_affinity",
                    f"block_{block:02d}_null_affinity",
                    f"block_{block:02d}_shuffled_affinity",
                }
            )
    else:
        expected_keys = {
            "aggregate_affinity",
            "aggregate_legacy_null_affinity",
            "aggregate_null_span_affinity",
            "aggregate_shuffled_affinity",
            "calibration_exploratory_track_masks_u8",
            "calibration_standardized_role_maps",
            "calibration_strict_aggregate_masks_u8",
            "calibration_strict_block_masks_u8",
        }
        for block in BLOCKS:
            expected_keys.update(
                {
                    f"block_{block:02d}_affinity",
                    f"block_{block:02d}_legacy_null_affinity",
                    f"block_{block:02d}_null_span_affinity",
                    f"block_{block:02d}_shuffled_affinity",
                }
            )
    if set(tensors) != expected_keys:
        raise V15BReviewError("diagnostic tensor registry differs")
    for name, value in tensors.items():
        expected = (5, 21, 37, 25)
        if "legacy_null_affinity" in name or (
            profile == R4_PROFILE
            and "null_affinity" in name
            and "shuffled" not in name
        ):
            expected = (21, 37, 25)
        elif "null_span_affinity" in name:
            expected = (64, 21, 37, 25)
        elif name in (
            "calibration_exploratory_track_masks_u8",
            "calibration_standardized_role_maps",
            "calibration_strict_block_masks_u8",
        ):
            expected = (5, 5, 21, 37, 25)
        elif name in (
            "aggregate_group_masks_u8",
            "calibration_strict_aggregate_masks_u8",
        ):
            expected = (5, 21, 37, 25)
        if tuple(value.shape) != expected or not np.isfinite(value).all():
            raise V15BReviewError(f"diagnostic tensor geometry/value differs: {name}")
        if ("masks_u8" in name or name == "aggregate_group_masks_u8") and not set(
            np.unique(value).tolist()
        ).issubset({0, 1}):
            raise V15BReviewError(f"diagnostic mask values differ: {name}")

    capture = cv2.VideoCapture(str(source_path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT)))
    capture.release()
    if len(frames) != 81:
        raise V15BReviewError("source video must contain exactly 81 frames")
    return receipt, tensors, frames, profile


def _top_indices(value: np.ndarray, fraction: float) -> np.ndarray:
    flat = value.reshape(-1)
    count = max(1, int(math.ceil(flat.size * fraction)))
    return np.argpartition(flat, flat.size - count)[flat.size - count :]


def _roi_mask(role: str) -> np.ndarray:
    if role not in ROIS:
        raise V15BReviewError("ROI requested for non-vessel role")
    x0, y0, x1, y1 = ROIS[role]
    mask = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    a = left.reshape(-1).astype(np.float64)
    b = right.reshape(-1).astype(np.float64)
    if float(a.std()) == 0.0 or float(b.std()) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _role_phase_row(
    *,
    block: str,
    phase: int,
    role: str,
    real: np.ndarray,
    null: np.ndarray,
    shuffled: np.ndarray,
) -> dict[str, Any]:
    flat = real.reshape(-1).astype(np.float64)
    median = float(np.median(flat))
    mad = float(np.median(np.abs(flat - median)))
    robust_scale = max(1e-12, 1.4826 * mad)
    top5 = _top_indices(real, 0.05)
    top5_median = float(np.median(flat[top5]))
    real_null = real.astype(np.float64) - null.astype(np.float64)
    real_shuffle = real.astype(np.float64) - shuffled.astype(np.float64)
    row: dict[str, Any] = {
        "block": block,
        "phase": phase,
        "video_frame": phase * 4,
        "role": role,
        "raw_min": float(flat.min()),
        "raw_median": median,
        "raw_max": float(flat.max()),
        "raw_std": float(flat.std()),
        "raw_mad": mad,
        "top5_median": top5_median,
        "top5_median_robust_z": (top5_median - median) / robust_scale,
        "real_null_min": float(real_null.min()),
        "real_null_median": float(np.median(real_null)),
        "real_null_max": float(real_null.max()),
        "real_null_top5site_median": float(np.median(real_null.reshape(-1)[top5])),
        "real_shuffle_min": float(real_shuffle.min()),
        "real_shuffle_median": float(np.median(real_shuffle)),
        "real_shuffle_max": float(real_shuffle.max()),
        "real_shuffle_top5site_median": float(
            np.median(real_shuffle.reshape(-1)[top5])
        ),
    }
    if role in VESSELS:
        roi = _roi_mask(role).reshape(-1)
        peak = int(flat.argmax())
        row["peak_row"] = peak // GRID_WIDTH
        row["peak_col"] = peak % GRID_WIDTH
        row["peak_in_roi"] = bool(roi[peak])
        area = float(roi.mean())
        for label, fraction in (("top2", 0.02), ("top5", 0.05), ("top8", 0.08)):
            indices = _top_indices(real, fraction)
            hit = float(roi[indices].mean())
            row[f"{label}_roi_hit_rate"] = hit
            row[f"{label}_roi_enrichment"] = hit / area
        row["roi_area_fraction"] = area
    return row


def _pair_phase_row(
    *, block: str, phase: int, first: str, second: str, maps: np.ndarray
) -> dict[str, Any]:
    left = maps[ROLE_NAMES.index(first), phase]
    right = maps[ROLE_NAMES.index(second), phase]
    left_top = set(_top_indices(left, 0.08).tolist())
    right_top = set(_top_indices(right, 0.08).tolist())
    union = left_top | right_top
    iou = len(left_top & right_top) / len(union) if union else 0.0
    return {
        "block": block,
        "phase": phase,
        "video_frame": phase * 4,
        "first_role": first,
        "second_role": second,
        "pearson_corr": _safe_corr(left, right),
        "top8_percent_iou": iou,
    }


def _block_tensors(
    tensors: Mapping[str, np.ndarray]
) -> Iterable[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    yield (
        "aggregate",
        tensors["aggregate_affinity"],
        tensors["aggregate_null_affinity"],
        tensors["aggregate_shuffled_affinity"],
    )
    for block in BLOCKS:
        yield (
            str(block),
            tensors[f"block_{block:02d}_affinity"],
            tensors[f"block_{block:02d}_null_affinity"],
            tensors[f"block_{block:02d}_shuffled_affinity"],
        )


def _compute_metrics(tensors: Mapping[str, np.ndarray]) -> Mapping[str, Any]:
    role_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for block, real, null, shuffled in _block_tensors(tensors):
        for phase in range(21):
            for role_index, role in enumerate(ROLE_NAMES):
                role_rows.append(
                    _role_phase_row(
                        block=block,
                        phase=phase,
                        role=role,
                        real=real[role_index, phase],
                        null=null[phase],
                        shuffled=shuffled[role_index, phase],
                    )
                )
            for first_index, first in enumerate(VESSELS):
                for second in VESSELS[first_index + 1 :]:
                    pair_rows.append(
                        _pair_phase_row(
                            block=block,
                            phase=phase,
                            first=first,
                            second=second,
                            maps=real,
                        )
                    )
    summaries: dict[str, Any] = {}
    for block in ("aggregate", *(str(item) for item in BLOCKS)):
        summaries[block] = {}
        for role in VESSELS:
            rows = [
                item
                for item in role_rows
                if item["block"] == block and item["role"] == role
            ]
            summaries[block][role] = {
                "phases": len(rows),
                "peak_in_roi_phases": sum(bool(item["peak_in_roi"]) for item in rows),
                "mean_top2_roi_hit_rate": float(
                    np.mean([item["top2_roi_hit_rate"] for item in rows])
                ),
                "mean_top5_roi_hit_rate": float(
                    np.mean([item["top5_roi_hit_rate"] for item in rows])
                ),
                "mean_top8_roi_hit_rate": float(
                    np.mean([item["top8_roi_hit_rate"] for item in rows])
                ),
                "median_top5_robust_z": float(
                    np.median([item["top5_median_robust_z"] for item in rows])
                ),
                "median_raw_std": float(np.median([item["raw_std"] for item in rows])),
                "median_top5site_real_null": float(
                    np.median([item["real_null_top5site_median"] for item in rows])
                ),
                "median_top5site_real_shuffle": float(
                    np.median([item["real_shuffle_top5site_median"] for item in rows])
                ),
            }
    return {
        "role_phase_metrics": role_rows,
        "vessel_pair_phase_metrics": pair_rows,
        "block_role_summaries": summaries,
    }


def _spatial_median_mad(value: np.ndarray) -> np.ndarray:
    """Standardize each final HxW map without spatial supervision."""

    array = np.asarray(value, dtype=np.float32)
    median = np.median(array, axis=(-2, -1), keepdims=True)
    mad = np.median(np.abs(array - median), axis=(-2, -1), keepdims=True)
    scale = np.float32(1.4826) * mad
    return np.divide(
        array - median,
        scale,
        out=np.zeros_like(array, dtype=np.float32),
        where=scale > np.float32(1e-12),
    )


def _null_strength_percentile(real_strength: float, null_strengths: np.ndarray) -> float:
    values = np.asarray(null_strengths, dtype=np.float64)
    if values.shape != (64,) or not np.isfinite(values).all():
        raise V15BReviewError("r6 null strength distribution differs")
    below = float(np.sum(values < real_strength))
    equal = float(np.sum(values == real_strength))
    return (below + 0.5 * equal) / 64.0


def _r6_block_tensors(
    tensors: Mapping[str, np.ndarray]
) -> Iterable[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    yield (
        "aggregate",
        tensors["aggregate_affinity"],
        tensors["aggregate_null_span_affinity"],
        tensors["aggregate_shuffled_affinity"],
        tensors["calibration_strict_aggregate_masks_u8"],
    )
    for offset, block in enumerate(BLOCKS):
        yield (
            str(block),
            tensors[f"block_{block:02d}_affinity"],
            tensors[f"block_{block:02d}_null_span_affinity"],
            tensors[f"block_{block:02d}_shuffled_affinity"],
            tensors["calibration_strict_block_masks_u8"][offset],
        )


def _role_phase_row_r6(
    *,
    block: str,
    phase: int,
    role: str,
    real: np.ndarray,
    null_bank: np.ndarray,
    shuffled: np.ndarray,
    strict_mask: np.ndarray,
) -> dict[str, Any]:
    """Summarize r6 using scalar peak-strength null comparisons.

    The 64 null maps are never averaged and never subtracted pointwise.  ROI
    fields below are post-hoc human-audit diagnostics only.
    """

    flat = real.reshape(-1).astype(np.float64)
    median = float(np.median(flat))
    mad = float(np.median(np.abs(flat - median)))
    robust_scale = max(1e-12, 1.4826 * mad)
    top5 = _top_indices(real, 0.05)
    top5_median = float(np.median(flat[top5]))
    real_z = _spatial_median_mad(real)
    null_z = _spatial_median_mad(null_bank)
    null_strengths = np.max(null_z, axis=(-2, -1)).astype(np.float64)
    real_strength = float(real_z.max())
    real_shuffle = real.astype(np.float64) - shuffled.astype(np.float64)
    row: dict[str, Any] = {
        "block": block,
        "phase": phase,
        "video_frame": phase * 4,
        "role": role,
        "raw_min": float(flat.min()),
        "raw_median": median,
        "raw_max": float(flat.max()),
        "raw_std": float(flat.std()),
        "raw_mad": mad,
        "top5_median": top5_median,
        "top5_median_robust_z": (top5_median - median) / robust_scale,
        "real_peak_robust_z": real_strength,
        "null_peak_min_robust_z": float(null_strengths.min()),
        "null_peak_median_robust_z": float(np.median(null_strengths)),
        "null_peak_p95_robust_z": float(np.quantile(null_strengths, 0.95)),
        "null_peak_max_robust_z": float(null_strengths.max()),
        "real_peak_minus_null_p95_robust_z": real_strength
        - float(np.quantile(null_strengths, 0.95)),
        "real_peak_null_percentile": _null_strength_percentile(
            real_strength, null_strengths
        ),
        "real_shuffle_min": float(real_shuffle.min()),
        "real_shuffle_median": float(np.median(real_shuffle)),
        "real_shuffle_max": float(real_shuffle.max()),
        "real_shuffle_top5site_median": float(
            np.median(real_shuffle.reshape(-1)[top5])
        ),
        "strict_mask_pixels": int(np.asarray(strict_mask).sum()),
    }
    if role in VESSELS:
        roi = _roi_mask(role).reshape(-1)
        peak = int(flat.argmax())
        row["peak_row"] = peak // GRID_WIDTH
        row["peak_col"] = peak % GRID_WIDTH
        row["peak_in_roi"] = bool(roi[peak])
        area = float(roi.mean())
        for label, fraction in (("top2", 0.02), ("top5", 0.05), ("top8", 0.08)):
            indices = _top_indices(real, fraction)
            hit = float(roi[indices].mean())
            row[f"{label}_roi_hit_rate"] = hit
            row[f"{label}_roi_enrichment"] = hit / area
        mask_flat = np.asarray(strict_mask, dtype=bool).reshape(-1)
        row["strict_mask_roi_hit_rate"] = (
            float(roi[mask_flat].mean()) if bool(mask_flat.any()) else None
        )
        row["roi_area_fraction"] = area
    return row


def _compute_metrics_r6(tensors: Mapping[str, np.ndarray]) -> Mapping[str, Any]:
    role_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for block, real, null_bank, shuffled, strict_masks in _r6_block_tensors(tensors):
        for phase in range(21):
            for role_index, role in enumerate(ROLE_NAMES):
                role_rows.append(
                    _role_phase_row_r6(
                        block=block,
                        phase=phase,
                        role=role,
                        real=real[role_index, phase],
                        null_bank=null_bank[:, phase],
                        shuffled=shuffled[role_index, phase],
                        strict_mask=strict_masks[role_index, phase],
                    )
                )
            for first_index, first in enumerate(VESSELS):
                for second in VESSELS[first_index + 1 :]:
                    pair_rows.append(
                        _pair_phase_row(
                            block=block,
                            phase=phase,
                            first=first,
                            second=second,
                            maps=real,
                        )
                    )
    summaries: dict[str, Any] = {}
    for block in ("aggregate", *(str(item) for item in BLOCKS)):
        summaries[block] = {}
        for role in VESSELS:
            rows = [
                item
                for item in role_rows
                if item["block"] == block and item["role"] == role
            ]
            strict_hits = [
                item["strict_mask_roi_hit_rate"]
                for item in rows
                if item["strict_mask_roi_hit_rate"] is not None
            ]
            summaries[block][role] = {
                "phases": len(rows),
                "peak_in_roi_phases": sum(bool(item["peak_in_roi"]) for item in rows),
                "mean_top2_roi_hit_rate": float(
                    np.mean([item["top2_roi_hit_rate"] for item in rows])
                ),
                "mean_top5_roi_hit_rate": float(
                    np.mean([item["top5_roi_hit_rate"] for item in rows])
                ),
                "mean_top8_roi_hit_rate": float(
                    np.mean([item["top8_roi_hit_rate"] for item in rows])
                ),
                "median_top5_robust_z": float(
                    np.median([item["top5_median_robust_z"] for item in rows])
                ),
                "median_raw_std": float(np.median([item["raw_std"] for item in rows])),
                "median_real_peak_null_percentile": float(
                    np.median([item["real_peak_null_percentile"] for item in rows])
                ),
                "null95_pass_phases": sum(
                    item["real_peak_null_percentile"] >= 0.95 for item in rows
                ),
                "median_real_peak_minus_null_p95_robust_z": float(
                    np.median(
                        [item["real_peak_minus_null_p95_robust_z"] for item in rows]
                    )
                ),
                "median_top5site_real_shuffle": float(
                    np.median([item["real_shuffle_top5site_median"] for item in rows])
                ),
                "strict_mask_pixels": int(
                    sum(item["strict_mask_pixels"] for item in rows)
                ),
                "strict_mask_phases": int(
                    sum(item["strict_mask_pixels"] > 0 for item in rows)
                ),
                "mean_strict_mask_roi_hit_rate": (
                    float(np.mean(strict_hits)) if strict_hits else None
                ),
            }
    return {
        "role_phase_metrics": role_rows,
        "vessel_pair_phase_metrics": pair_rows,
        "block_role_summaries": summaries,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise V15BReviewError("refusing to write empty metric table")
    fields = list(rows[0])
    fields.extend(
        sorted({key for row in rows for key in row}.difference(fields))
    )
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _draw_rois(frame: np.ndarray) -> np.ndarray:
    result = frame.copy()
    for role in VESSELS:
        x0, y0, x1, y1 = ROIS[role]
        p0 = (round(x0 * DISPLAY_WIDTH / GRID_WIDTH), round(y0 * DISPLAY_HEIGHT / GRID_HEIGHT))
        p1 = (round(x1 * DISPLAY_WIDTH / GRID_WIDTH), round(y1 * DISPLAY_HEIGHT / GRID_HEIGHT))
        color = ROLE_COLORS_BGR[role]
        cv2.rectangle(result, p0, p1, color, 3)
        cv2.putText(result, role, (p0[0] + 3, max(22, p0[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return result


def _robust_normalize(value: np.ndarray) -> np.ndarray:
    flat = value.reshape(-1).astype(np.float32)
    low = float(np.percentile(flat, 50.0))
    high = float(np.percentile(flat, 98.0))
    if high <= low + 1e-12:
        return np.zeros_like(value, dtype=np.float32)
    return np.clip((value.astype(np.float32) - low) / (high - low), 0.0, 1.0)


def _role_overlay(frame: np.ndarray, affinity: np.ndarray, role: str) -> np.ndarray:
    norm = _robust_normalize(affinity)
    heat = cv2.applyColorMap(
        cv2.resize((norm * 255).astype(np.uint8), (DISPLAY_WIDTH, DISPLAY_HEIGHT)),
        cv2.COLORMAP_TURBO,
    )
    alpha = cv2.resize(norm, (DISPLAY_WIDTH, DISPLAY_HEIGHT))[:, :, None] * 0.62
    result = (frame.astype(np.float32) * (1.0 - alpha) + heat.astype(np.float32) * alpha).astype(np.uint8)
    x0, y0, x1, y1 = ROIS[role]
    p0 = (round(x0 * DISPLAY_WIDTH / GRID_WIDTH), round(y0 * DISPLAY_HEIGHT / GRID_HEIGHT))
    p1 = (round(x1 * DISPLAY_WIDTH / GRID_WIDTH), round(y1 * DISPLAY_HEIGHT / GRID_HEIGHT))
    cv2.rectangle(result, p0, p1, ROLE_COLORS_BGR[role], 3)
    peak = np.unravel_index(int(affinity.argmax()), affinity.shape)
    peak_xy = (
        round((peak[1] + 0.5) * DISPLAY_WIDTH / GRID_WIDTH),
        round((peak[0] + 0.5) * DISPLAY_HEIGHT / GRID_HEIGHT),
    )
    cv2.drawMarker(result, peak_xy, (255, 255, 255), cv2.MARKER_CROSS, 20, 3)
    return result


def _control_overlay(
    frame: np.ndarray,
    real: np.ndarray,
    null: np.ndarray,
    shuffled: np.ndarray,
    phase: int,
) -> np.ndarray:
    vessel = np.stack(
        [
            real[ROLE_NAMES.index(role), phase]
            - np.maximum(null[phase], shuffled[ROLE_NAMES.index(role), phase])
            for role in VESSELS
        ],
        axis=0,
    )
    winner = vessel.argmax(axis=0)
    strength = _robust_normalize(vessel.max(axis=0))
    color = np.zeros((GRID_HEIGHT, GRID_WIDTH, 3), dtype=np.uint8)
    for index, role in enumerate(VESSELS):
        color[winner == index] = ROLE_COLORS_BGR[role]
    color = cv2.resize(color, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_NEAREST)
    alpha = cv2.resize(strength, (DISPLAY_WIDTH, DISPLAY_HEIGHT))[:, :, None] * 0.72
    return (frame.astype(np.float32) * (1.0 - alpha) + color.astype(np.float32) * alpha).astype(np.uint8)


def _draw_strict_mask_outline(
    image: np.ndarray, mask: np.ndarray, role: str
) -> np.ndarray:
    result = image.copy()
    resized = cv2.resize(
        np.asarray(mask, dtype=np.uint8),
        (DISPLAY_WIDTH, DISPLAY_HEIGHT),
        interpolation=cv2.INTER_NEAREST,
    )
    contours, _ = cv2.findContours(resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(result, contours, -1, ROLE_COLORS_BGR[role], 4)
    return result


def _r6_null_shuffle_overlay(
    frame: np.ndarray,
    real: np.ndarray,
    null_bank: np.ndarray,
    shuffled: np.ndarray,
    strict_masks: np.ndarray,
    phase: int,
) -> np.ndarray:
    """Display-only null-percentile + shuffle audit.

    Null controls are reduced only to a 64-sample peak-strength percentile;
    they are never averaged/subtracted pointwise.  Colored fill shows the
    strongest vessel standardized-real minus standardized-shuffle map among
    roles whose peak clears the preregistered 95th null percentile.  Contours
    show saved strict-calibration candidates.
    """

    vessel_real = np.stack(
        [real[ROLE_NAMES.index(role), phase] for role in VESSELS], axis=0
    )
    vessel_shuffle = np.stack(
        [shuffled[ROLE_NAMES.index(role), phase] for role in VESSELS], axis=0
    )
    real_z = _spatial_median_mad(vessel_real)
    shuffle_z = _spatial_median_mad(vessel_shuffle)
    null_z = _spatial_median_mad(null_bank[:, phase])
    null_strengths = np.max(null_z, axis=(-2, -1))
    eligible = np.asarray(
        [
            _null_strength_percentile(float(real_z[index].max()), null_strengths)
            >= 0.95
            for index in range(len(VESSELS))
        ],
        dtype=bool,
    )
    scores = real_z - shuffle_z
    scores[~eligible] = np.float32(-np.inf)
    if not bool(eligible.any()):
        result = frame.copy()
    else:
        winner = scores.argmax(axis=0)
        maximum = scores.max(axis=0)
        strength = _robust_normalize(
            np.where(np.isfinite(maximum), maximum, np.float32(0.0))
        )
        color = np.zeros((GRID_HEIGHT, GRID_WIDTH, 3), dtype=np.uint8)
        for index, role in enumerate(VESSELS):
            color[(winner == index) & eligible[index]] = ROLE_COLORS_BGR[role]
        color = cv2.resize(
            color, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_NEAREST
        )
        alpha = cv2.resize(strength, (DISPLAY_WIDTH, DISPLAY_HEIGHT))[:, :, None] * 0.66
        result = (
            frame.astype(np.float32) * (1.0 - alpha)
            + color.astype(np.float32) * alpha
        ).astype(np.uint8)
    for role in VESSELS:
        result = _draw_strict_mask_outline(
            result, strict_masks[ROLE_NAMES.index(role), phase], role
        )
    return result


def _render_images(
    output: Path,
    tensors: Mapping[str, np.ndarray],
    frames: Sequence[np.ndarray],
    role_rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    media = output / "media"
    media.mkdir(parents=True, exist_ok=False)
    rows_by_key = {
        (str(item["block"]), int(item["phase"]), str(item["role"])): item
        for item in role_rows
    }
    cards: list[Mapping[str, Any]] = []
    for block, real, null, shuffled in _block_tensors(tensors):
        for phase in DISPLAY_PHASES:
            frame = frames[phase * 4]
            names = []
            source_name = f"block_{block}_phase_{phase:02d}_source.jpg"
            cv2.imwrite(str(media / source_name), _draw_rois(frame), [cv2.IMWRITE_JPEG_QUALITY, 90])
            names.append(source_name)
            metrics = []
            for role in VESSELS:
                role_name = f"block_{block}_phase_{phase:02d}_{role}.jpg"
                cv2.imwrite(
                    str(media / role_name),
                    _role_overlay(frame, real[ROLE_NAMES.index(role), phase], role),
                    [cv2.IMWRITE_JPEG_QUALITY, 90],
                )
                names.append(role_name)
                metrics.append(rows_by_key[(block, phase, role)])
            control_name = f"block_{block}_phase_{phase:02d}_control.jpg"
            cv2.imwrite(
                str(media / control_name),
                _control_overlay(frame, real, null, shuffled, phase),
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )
            names.append(control_name)
            cards.append(
                {
                    "block": block,
                    "phase": phase,
                    "frame": phase * 4,
                    "images": names,
                    "metrics": metrics,
                }
            )
    if len(cards) != 30 or any(len(item["images"]) != 5 for item in cards):
        raise V15BReviewError("review grid cardinality differs")
    files = tuple(media.glob("*.jpg"))
    if len(files) != 150:
        raise V15BReviewError("review must contain exactly 150 rendered images")
    return cards


def _render_images_r6(
    output: Path,
    tensors: Mapping[str, np.ndarray],
    frames: Sequence[np.ndarray],
    role_rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    media = output / "media"
    media.mkdir(parents=True, exist_ok=False)
    rows_by_key = {
        (str(item["block"]), int(item["phase"]), str(item["role"])): item
        for item in role_rows
    }
    cards: list[Mapping[str, Any]] = []
    for block, real, null_bank, shuffled, strict_masks in _r6_block_tensors(tensors):
        for phase in DISPLAY_PHASES:
            frame = frames[phase * 4]
            names: list[str] = []
            source_name = f"block_{block}_phase_{phase:02d}_source.jpg"
            cv2.imwrite(
                str(media / source_name),
                _draw_rois(frame),
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )
            names.append(source_name)
            metrics = []
            for role in VESSELS:
                role_index = ROLE_NAMES.index(role)
                role_name = f"block_{block}_phase_{phase:02d}_{role}.jpg"
                role_image = _role_overlay(frame, real[role_index, phase], role)
                role_image = _draw_strict_mask_outline(
                    role_image, strict_masks[role_index, phase], role
                )
                cv2.imwrite(
                    str(media / role_name),
                    role_image,
                    [cv2.IMWRITE_JPEG_QUALITY, 90],
                )
                names.append(role_name)
                metrics.append(rows_by_key[(block, phase, role)])
            control_name = f"block_{block}_phase_{phase:02d}_null_shuffle.jpg"
            cv2.imwrite(
                str(media / control_name),
                _r6_null_shuffle_overlay(
                    frame, real, null_bank, shuffled, strict_masks, phase
                ),
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )
            names.append(control_name)
            cards.append(
                {
                    "block": block,
                    "phase": phase,
                    "frame": phase * 4,
                    "images": names,
                    "metrics": metrics,
                }
            )
    if len(cards) != 30 or any(len(item["images"]) != 5 for item in cards):
        raise V15BReviewError("r6 review grid cardinality differs")
    files = tuple(media.glob("*.jpg"))
    if len(files) != 150:
        raise V15BReviewError("r6 review must contain exactly 150 rendered images")
    return cards


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _build_html(
    *, receipt: Mapping[str, Any], metrics: Mapping[str, Any], cards: Sequence[Mapping[str, Any]]
) -> str:
    summary = metrics["block_role_summaries"]
    headers = ("Source + fixed audit ROIs", "old_actor #1", "new_actor #2", "recipient #3", "real−max(null,shuffle)")
    sections = []
    for block in ("aggregate", *(str(item) for item in BLOCKS)):
        block_cards = [item for item in cards if item["block"] == block]
        table_rows = []
        for role in VESSELS:
            value = summary[block][role]
            table_rows.append(
                "<tr>"
                f"<th>{html.escape(role)}</th>"
                f"<td>{value['peak_in_roi_phases']}/21</td>"
                f"<td>{_fmt(value['mean_top2_roi_hit_rate'])}</td>"
                f"<td>{_fmt(value['mean_top5_roi_hit_rate'])}</td>"
                f"<td>{_fmt(value['median_top5_robust_z'])}</td>"
                f"<td>{_fmt(value['median_raw_std'], 5)}</td>"
                f"<td>{_fmt(value['median_top5site_real_null'], 5)}</td>"
                f"<td>{_fmt(value['median_top5site_real_shuffle'], 5)}</td>"
                "</tr>"
            )
        grids = []
        for card in block_cards:
            cells = []
            for index, image_name in enumerate(card["images"]):
                label = headers[index]
                detail = ""
                if 1 <= index <= 3:
                    row = card["metrics"][index - 1]
                    detail = (
                        f"peak ROI={_fmt(row['peak_in_roi'])} · Top2 ROI={_fmt(row['top2_roi_hit_rate'])} · "
                        f"z={_fmt(row['top5_median_robust_z'])} · std={_fmt(row['raw_std'],5)}"
                    )
                cells.append(
                    "<figure><figcaption>"
                    + html.escape(label)
                    + ("<small>" + html.escape(detail) + "</small>" if detail else "")
                    + f"</figcaption><img loading='lazy' src='media/{html.escape(image_name)}'></figure>"
                )
            grids.append(
                f"<h3>phase {card['phase']} · source frame {card['frame']}</h3>"
                "<div class='grid'>" + "".join(cells) + "</div>"
            )
        sections.append(
            f"<section><h2>Block {html.escape(block)}</h2>"
            "<table><thead><tr><th>role</th><th>peak ROI phases</th><th>mean Top2 ROI</th>"
            "<th>mean Top5 ROI</th><th>median Top5 robust-z</th><th>median std</th>"
            "<th>median top-site real−null</th><th>median top-site real−shuffle</th></tr></thead><tbody>"
            + "".join(table_rows)
            + "</tbody></table>"
            + "".join(grids)
            + "</section>"
        )
    mask = receipt["diagnostics"]["mask_diagnostic"]
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>E00 v15b source-role observer</title>
<style>
body{{margin:0;background:#f6f2e8;color:#18231f;font:15px/1.45 system-ui,sans-serif}}main{{max-width:1760px;margin:auto;padding:18px}}
h1,h2,h3{{margin:.4em 0}}.verdict{{background:#fee4dc;border:2px solid #c6472f;border-radius:12px;padding:14px;font-weight:700}}
.note{{background:#fff;border:1px solid #cfc6b2;border-radius:10px;padding:12px;margin:12px 0}}section{{background:#fffaf2;border:1px solid #d8cfbd;border-radius:14px;padding:14px;margin:18px 0}}
.grid{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;align-items:start}}figure{{margin:0;border:1px solid #cfc6b2;border-radius:8px;overflow:hidden;background:#fff}}
figcaption{{display:block;padding:7px;font-weight:700;min-height:50px}}figcaption small{{display:block;color:#65716c;font-weight:500}}img{{display:block;width:100%;height:auto}}
table{{width:100%;border-collapse:collapse;margin:10px 0 18px;font-size:13px}}th,td{{border:1px solid #d6cebd;padding:5px;text-align:right}}th:first-child,td:first-child{{text-align:left}}code{{word-break:break-all}}@media(max-width:950px){{.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
</style></head><body><main>
<h1>E00 v15b · frozen source-role observer</h1>
<div class="verdict">FAIL-CLOSED for routing/training: strict vessel masks are empty in all 21 phases. Mechanical observer integrity passed, but localization is not certified.</div>
<div class="note">This page separates scale/control-gate failure from raw localization. Heatmaps are raw source-Q × source-caption-K affinities normalized only for display. White cross = raw peak; colored box = source-only manual audit ROI. ROI usage is <code>human_audit_not_algorithm</code>: no ROI enters weighting, calibration, masking, routing, or a model call. The final column colors the strongest vessel-specific real−max(null, shuffled) margin (red #1, green #2, blue #3). No anchor map or anchor position is used.</div>
<div class="note">receipt <code>{html.escape(str(receipt['receipt_sha256']))}</code> · frozen output <code>{html.escape(str(receipt['frozen_output_sha256']))}</code> · strict candidate qualified={html.escape(str(mask['mechanical_candidate_qualified']))}</div>
{''.join(sections)}
</main></body></html>"""


def _build_html_r6(
    *,
    receipt: Mapping[str, Any],
    metrics: Mapping[str, Any],
    cards: Sequence[Mapping[str, Any]],
) -> str:
    summary = metrics["block_role_summaries"]
    headers = (
        "Source + fixed audit ROIs",
        "old_actor #1",
        "new_actor #2",
        "recipient #3",
        "null95 + shuffle audit",
    )
    sections = []
    for block in ("aggregate", *(str(item) for item in BLOCKS)):
        block_cards = [item for item in cards if item["block"] == block]
        table_rows = []
        for role in VESSELS:
            value = summary[block][role]
            table_rows.append(
                "<tr>"
                f"<th>{html.escape(role)}</th>"
                f"<td>{value['peak_in_roi_phases']}/21</td>"
                f"<td>{_fmt(value['mean_top2_roi_hit_rate'])}</td>"
                f"<td>{_fmt(value['median_top5_robust_z'])}</td>"
                f"<td>{_fmt(value['median_raw_std'], 5)}</td>"
                f"<td>{_fmt(value['median_real_peak_null_percentile'])}</td>"
                f"<td>{value['null95_pass_phases']}/21</td>"
                f"<td>{_fmt(value['median_real_peak_minus_null_p95_robust_z'])}</td>"
                f"<td>{_fmt(value['median_top5site_real_shuffle'], 5)}</td>"
                f"<td>{value['strict_mask_pixels']} px / {value['strict_mask_phases']} ph</td>"
                f"<td>{_fmt(value['mean_strict_mask_roi_hit_rate'])}</td>"
                "</tr>"
            )
        grids = []
        for card in block_cards:
            cells = []
            for index, image_name in enumerate(card["images"]):
                label = headers[index]
                detail = ""
                if 1 <= index <= 3:
                    row = card["metrics"][index - 1]
                    detail = (
                        f"peak ROI={_fmt(row['peak_in_roi'])} · Top2 ROI={_fmt(row['top2_roi_hit_rate'])} · "
                        f"z={_fmt(row['top5_median_robust_z'])} · null pct={_fmt(row['real_peak_null_percentile'])} · "
                        f"strict={row['strict_mask_pixels']}px"
                    )
                elif index == 4:
                    detail = "fill=real−shuffle after scalar null95; contour=saved strict candidate"
                cells.append(
                    "<figure><figcaption>"
                    + html.escape(label)
                    + ("<small>" + html.escape(detail) + "</small>" if detail else "")
                    + f"</figcaption><img loading='lazy' src='media/{html.escape(image_name)}'></figure>"
                )
            grids.append(
                f"<h3>phase {card['phase']} · source frame {card['frame']}</h3>"
                "<div class='grid'>" + "".join(cells) + "</div>"
            )
        sections.append(
            f"<section><h2>Block {html.escape(block)}</h2>"
            "<table><thead><tr><th>role</th><th>peak ROI phases</th><th>mean Top2 ROI</th>"
            "<th>median Top5 robust-z</th><th>median raw std</th><th>median null percentile</th>"
            "<th>null95 pass</th><th>median peak−null-p95 z</th><th>median top-site real−shuffle</th>"
            "<th>strict candidate</th><th>strict ROI hit</th></tr></thead><tbody>"
            + "".join(table_rows)
            + "</tbody></table>"
            + "".join(grids)
            + "</section>"
        )
    calibration = receipt["diagnostics"]["calibration_receipt"]
    aggregate_summary = summary["aggregate"]
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>E00 v15b r6 explicit-null64 observer</title>
<style>
body{{margin:0;background:#f6f2e8;color:#18231f;font:15px/1.45 system-ui,sans-serif}}main{{max-width:1760px;margin:auto;padding:18px}}
h1,h2,h3{{margin:.4em 0}}.verdict{{background:#fee4dc;border:2px solid #c6472f;border-radius:12px;padding:14px;font-weight:700}}
.note{{background:#fff;border:1px solid #cfc6b2;border-radius:10px;padding:12px;margin:12px 0}}section{{background:#fffaf2;border:1px solid #d8cfbd;border-radius:14px;padding:14px;margin:18px 0;overflow-x:auto}}
.grid{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;align-items:start}}figure{{margin:0;border:1px solid #cfc6b2;border-radius:8px;overflow:hidden;background:#fff}}
figcaption{{display:block;padding:7px;font-weight:700;min-height:66px}}figcaption small{{display:block;color:#65716c;font-weight:500}}img{{display:block;width:100%;height:auto}}
table{{width:100%;border-collapse:collapse;margin:10px 0 18px;font-size:12px}}th,td{{border:1px solid #d6cebd;padding:5px;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}code{{word-break:break-all}}@media(max-width:950px){{.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
</style></head><body><main>
<h1>E00 v15b r6 · frozen source-role observer · explicit 64-null bank</h1>
<div class="verdict">NO-GO for routing/training. Mechanical observer gates passed, but the aggregate strict candidate has old_actor={aggregate_summary['old_actor']['strict_mask_pixels']} px, new_actor={aggregate_summary['new_actor']['strict_mask_pixels']} px, recipient={aggregate_summary['recipient']['strict_mask_pixels']} px. All three source vessels must localize reliably; old_actor is absent, and no semantic localization has been certified.</div>
<div class="note">The 64 preregistered non-special token/span maps are retained individually. “null percentile” compares scalar per-map peak strength; there is no averaged-null or pointwise real−null subtraction. The fifth column is display-only: colored fill is standardized real−shuffle for a vessel whose scalar peak clears null p95; colored contours are the saved calibration candidate. It is not a route.</div>
<div class="note">For the row named <code>aggregate</code>, raw heatmaps, null percentiles and shuffle values are the unweighted five-block mean. Saved strict contours use the preregistered role-specific block weighting (null percentile × temporal coherence). They are shown together for audit but are not the same aggregate statistic; the per-block sections are the direct apples-to-apples evidence.</div>
<div class="note">White cross = raw affinity peak; colored box = source-only manual audit ROI. ROI usage is <code>human_audit_not_algorithm</code>: ROI never enters standardization, null percentile, peer competition, track selection, masks, model calls, routing, training, or decoding. No anchor map or anchor absolute position is consumed.</div>
<div class="note">receipt <code>{html.escape(str(receipt['receipt_sha256']))}</code> · tensor <code>{html.escape(str(receipt['diagnostics']['file_sha256']))}</code> · calibration <code>{html.escape(str(calibration['receipt_sha256']))}</code> · strict total={calibration['strict_mask_pixel_count']} px · route_authorized=false</div>
{''.join(sections)}
</main></body></html>"""


def render(args: argparse.Namespace) -> Mapping[str, Any]:
    output = Path(args.output)
    if output.exists():
        raise V15BReviewError("review output already exists")
    output.mkdir(parents=True)
    receipt, tensors, frames, profile = _validate_inputs(
        Path(args.receipt), Path(args.tensors), Path(args.source)
    )
    metrics = (
        _compute_metrics(tensors)
        if profile == R4_PROFILE
        else _compute_metrics_r6(tensors)
    )
    _write_csv(output / "role_phase_metrics.csv", metrics["role_phase_metrics"])
    _write_csv(
        output / "vessel_pair_phase_metrics.csv",
        metrics["vessel_pair_phase_metrics"],
    )
    metrics_payload = {
        "schema_version": SCHEMA_VERSION,
        "review_profile": profile,
        "source_video_sha256": SOURCE_SHA256,
        "probe_receipt_file_sha256": _sha256_file(Path(args.receipt)),
        "diagnostic_tensor_file_sha256": _sha256_file(Path(args.tensors)),
        "probe_receipt_sha256": receipt["receipt_sha256"],
        "diagnostic_file_sha256": receipt["diagnostics"]["file_sha256"],
        "roi_grid_xyxy_exclusive": {key: list(value) for key, value in ROIS.items()},
        "roi_is_human_audit_only_not_model_input": True,
        "roi_usage": "human_audit_not_algorithm",
        "render_contract": {
            "grid_columns_max": 5,
            "grid_count": 30,
            "images_per_grid": 5,
            "rendered_image_count": 150,
        },
        "top8_definition": "top 8 percent of 925 spatial sites per block/phase/role",
        **metrics,
        "strict_mask_mechanical_candidate_qualified": (
            receipt["diagnostics"]["mask_diagnostic"][
                "mechanical_candidate_qualified"
            ]
            if profile == R4_PROFILE
            else receipt["diagnostics"]["calibration_receipt"][
                "mechanical_candidate_qualified"
            ]
        ),
        "null_bank_span_count": 0 if profile == R4_PROFILE else 64,
        "null_comparison": (
            "legacy_averaged_null_for_scale_diagnostic_only"
            if profile == R4_PROFILE
            else "64_individual_spatial_MAD_maps_scalar_peak_percentile_no_pointwise_subtraction"
        ),
        "aggregate_affinity_basis": "unweighted_mean_across_blocks",
        "strict_aggregate_mask_basis": (
            "legacy_probe_mask"
            if profile == R4_PROFILE
            else "role_specific_block_weights_from_null_percentile_times_temporal_coherence"
        ),
        "localization_semantically_certified": False,
        "route_authorized": False,
        "training_authorized": False,
        "decode_authorized": False,
    }
    with (output / "metrics.json").open("x", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    cards = (
        _render_images(output, tensors, frames, metrics["role_phase_metrics"])
        if profile == R4_PROFILE
        else _render_images_r6(
            output, tensors, frames, metrics["role_phase_metrics"]
        )
    )
    if len(cards) * 5 != 150:
        raise V15BReviewError("rendered review image count differs")
    page = (
        _build_html(receipt=receipt, metrics=metrics, cards=cards)
        if profile == R4_PROFILE
        else _build_html_r6(receipt=receipt, metrics=metrics, cards=cards)
    )
    (output / "index.html").write_text(page, encoding="utf-8")
    return {
        "index": str((output / "index.html").resolve()),
        "metrics": str((output / "metrics.json").resolve()),
        "role_csv": str((output / "role_phase_metrics.csv").resolve()),
        "pair_csv": str((output / "vessel_pair_phase_metrics.csv").resolve()),
        "image_count": len(cards) * 5,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--tensors", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = render(build_parser().parse_args(argv))
    except V15BReviewError as error:
        print(f"FAIL-CLOSED: {error}")
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
