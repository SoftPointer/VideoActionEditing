#!/usr/bin/env python3
"""Build a provenance-bound, review-only R7-P1 track overlay bundle.

This utility is intentionally downstream of the frozen R7 cache and P1
diagnostic.  It does not run a selector, change a threshold, train a model, or
authorize generation.  It only renders selected cache rows for human review.

The input cache must be the ``final/`` directory of a complete eight-shard P1
track-cache commit.  The diagnostic directory must contain the matching
``done.json``, ``summary.json``, and ``rows.jsonl`` commit.  Both commits are
revalidated with the original R7/P1 validators before any output is written.

Tracks are always drawn from ``*_normalized_tracks`` (the raw, pre-camera
coordinates), never from stabilized tracks.  Base and perturbed component
membership masks retain their original cache track indices and are only used
to color those raw tracks.  Exact cached source-frame indices are decoded from
the original video and resized to the cache's tracking size without geometric
stabilization.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


# Make the script executable directly from methods/motive/scripts without
# requiring an editable package installation.
_SCRIPT_PATH = Path(__file__).resolve()
_MOTIVE_ROOT = _SCRIPT_PATH.parents[1]
if str(_MOTIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MOTIVE_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from motive.r7_coherent_actor import CoherentActorConfig  # noqa: E402
from motive.r7_p1_diagnostic import (  # noqa: E402
    OUTPUT_DONE_NAME,
    OUTPUT_SUMMARY_NAME,
    ROWS_NAME,
    R7_P1_DIAGNOSTIC_DONE_SCHEMA,
    R7_P1_DIAGNOSTIC_ROW_SCHEMA,
    R7_P1_DIAGNOSTIC_SUMMARY_SCHEMA,
    DiagnosticGateConfig,
    DownstreamAuditConfig,
    P1DiagnosticConfig,
    build_diagnostic_contract,
    load_final_cache,
    validate_output_commit,
)
from motive.r7_preflight_extract import (  # noqa: E402
    VIDEO_FRAMES,
    _array_digest,
    _canonical_json,
    _file_digest,
    _object_digest,
    _safe_video_path,
)
from motive.r7_track_cache import (  # noqa: E402
    ARCHIVE_NAME,
    DONE_NAME,
    FINAL_DIR_NAME,
    MANIFEST_NAME,
    R7_TRACK_CACHE_FINAL_DONE_SCHEMA,
    R7_TRACK_CACHE_FINAL_SUMMARY_SCHEMA,
    SUMMARY_NAME,
)


REVIEW_BUNDLE_SCHEMA = "motive-r7-p1-human-review-bundle-v1"
REVIEW_ROW_SCHEMA = "motive-r7-p1-human-review-row-v1"
REVIEW_SUMMARY_SCHEMA = "motive-r7-p1-human-review-summary-v1"
REVIEW_MANIFEST_NAME = "review_manifest.jsonl"
REVIEW_SUMMARY_NAME = "summary.json"
VIDEOS_DIRECTORY_NAME = "videos"
REVIEW_SCOPE = (
    "human review visualization only; no threshold tuning, training, "
    "production decision, or generation authorization"
)
_IID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SIDES = ("target", "source")
_SIDE_INPUT_FIELDS = {"target": "tgt_video", "source": "src_video"}
_VISIBILITY_THRESHOLD = 0.5
_TRAIL_LENGTH = 6
_CODEC = "mp4v"

# OpenCV consumes BGR values.
_BACKGROUND_BGR = (150, 150, 150)
_BASE_BGR = (60, 220, 60)
_PERTURBED_BGR = (220, 70, 220)
_CURRENT_FRAME_BGR = (245, 245, 245)
_COLOR_CONTRACT = {
    "background": {
        "bgr": list(_BACKGROUND_BGR),
        "meaning": "cache tracks in neither selected component",
    },
    "base": {
        "bgr": list(_BASE_BGR),
        "meaning": "base selected-component membership",
    },
    "perturbed": {
        "bgr": list(_PERTURBED_BGR),
        "meaning": "audit perturbed selected-component membership",
    },
    "overlap": {
        "meaning": (
            "base green point/path plus perturbed magenta ring/path at the "
            "same raw cache track index"
        )
    },
}


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(_require_mapping(value, label=str(path)))


def _read_canonical_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number} is blank")
            value = json.loads(line)
            row = dict(
                _require_mapping(
                    value,
                    label=f"{path}:{line_number}",
                )
            )
            if line != _canonical_json(row) + "\n":
                raise ValueError(
                    f"{path}:{line_number} is not canonical JSONL"
                )
            rows.append(row)
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            dict(value),
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")


def _write_canonical_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(dict(row)) + "\n")


def _resolved_directory(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise NotADirectoryError(f"{label}: {resolved}")
    return resolved


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} is not a SHA-256 digest")
    return value


def _validate_cache_envelope(
    cache_final_directory: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = {
        "done": cache_final_directory / DONE_NAME,
        "summary": cache_final_directory / SUMMARY_NAME,
        "manifest": cache_final_directory / MANIFEST_NAME,
        "archive": cache_final_directory / ARCHIVE_NAME,
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    done = _read_json(paths["done"])
    summary = _read_json(paths["summary"])
    if done.get("schema_version") != R7_TRACK_CACHE_FINAL_DONE_SCHEMA:
        raise ValueError("cache final done schema differs")
    if summary.get("schema_version") != R7_TRACK_CACHE_FINAL_SUMMARY_SCHEMA:
        raise ValueError("cache final summary schema differs")
    if done.get("committed") is not True:
        raise ValueError("cache final commit is incomplete")
    digest_bindings = (
        ("archive_sha256", paths["archive"]),
        ("manifest_sha256", paths["manifest"]),
        ("summary_sha256", paths["summary"]),
    )
    for field, path in digest_bindings:
        expected = _require_sha256(done.get(field), label=f"cache {field}")
        if expected != _file_digest(path):
            raise ValueError(f"cache {field} differs")
    contract = _require_mapping(
        summary.get("contract"),
        label="cache final contract",
    )
    contract_sha256 = _object_digest(contract)
    if summary.get("contract_sha256") != contract_sha256:
        raise ValueError("cache summary contract digest differs")
    if done.get("contract_sha256") != contract_sha256:
        raise ValueError("cache done/summary contract digest differs")
    if done.get("rows") != summary.get("rows"):
        raise ValueError("cache done/summary row count differs")
    return done, summary


def _validate_diagnostic_envelope(
    diagnostic_directory: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    paths = {
        "done": diagnostic_directory / OUTPUT_DONE_NAME,
        "summary": diagnostic_directory / OUTPUT_SUMMARY_NAME,
        "rows": diagnostic_directory / ROWS_NAME,
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    done = _read_json(paths["done"])
    summary = _read_json(paths["summary"])
    if done.get("schema_version") != R7_P1_DIAGNOSTIC_DONE_SCHEMA:
        raise ValueError("diagnostic done schema differs")
    if summary.get("schema_version") != R7_P1_DIAGNOSTIC_SUMMARY_SCHEMA:
        raise ValueError("diagnostic summary schema differs")
    if done.get("committed") is not True:
        raise ValueError("diagnostic commit is incomplete")
    if (
        _require_sha256(
            done.get("rows_sha256"),
            label="diagnostic rows_sha256",
        )
        != _file_digest(paths["rows"])
    ):
        raise ValueError("diagnostic rows byte digest differs")
    if (
        _require_sha256(
            done.get("summary_sha256"),
            label="diagnostic summary_sha256",
        )
        != _file_digest(paths["summary"])
    ):
        raise ValueError("diagnostic summary byte digest differs")
    contract = _require_mapping(
        summary.get("contract"),
        label="diagnostic contract",
    )
    contract_sha256 = _object_digest(contract)
    if summary.get("contract_sha256") != contract_sha256:
        raise ValueError("diagnostic summary contract digest differs")
    if done.get("contract_sha256") != contract_sha256:
        raise ValueError("diagnostic done/summary contract digest differs")
    rows = _read_canonical_jsonl(paths["rows"])
    if done.get("rows") != len(rows) or summary.get("rows") != len(rows):
        raise ValueError("diagnostic committed row count differs")
    return done, summary, rows


def _config_from_diagnostic_contract(
    contract: Mapping[str, Any],
) -> P1DiagnosticConfig:
    selector = _require_mapping(
        _require_mapping(
            contract.get("selector"),
            label="diagnostic selector contract",
        ).get("config"),
        label="diagnostic selector config",
    )
    audit = _require_mapping(
        _require_mapping(
            contract.get("independent_audit"),
            label="diagnostic audit contract",
        ).get("config"),
        label="diagnostic audit config",
    )
    gate = _require_mapping(
        _require_mapping(
            contract.get("diagnostic_gate"),
            label="diagnostic gate contract",
        ).get("config"),
        label="diagnostic gate config",
    )
    seed = contract.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("diagnostic seed is not an integer")
    try:
        config = P1DiagnosticConfig(
            seed=seed,
            selector=CoherentActorConfig(**dict(selector)),
            audit=DownstreamAuditConfig(**dict(audit)),
            gate=DiagnosticGateConfig(**dict(gate)),
        )
    except TypeError as error:
        raise ValueError("diagnostic frozen config fields differ") from error
    config.validate()
    return config


def load_verified_inputs(
    *,
    cache_final_directory: Path,
    diagnostic_directory: Path,
    data_root: Path,
) -> dict[str, Any]:
    """Strongly validate the matching cache and diagnostic commits."""

    cache_final = _resolved_directory(
        cache_final_directory,
        label="cache final directory",
    )
    diagnostic = _resolved_directory(
        diagnostic_directory,
        label="diagnostic directory",
    )
    root = _resolved_directory(data_root, label="data root")
    if cache_final.name != FINAL_DIR_NAME:
        raise ValueError(
            f"cache input must be the {FINAL_DIR_NAME!r} directory"
        )
    cache_done, cache_summary = _validate_cache_envelope(cache_final)
    (
        diagnostic_done,
        diagnostic_summary,
        envelope_rows,
    ) = _validate_diagnostic_envelope(diagnostic)
    cache_contract = _require_mapping(
        cache_summary.get("contract"),
        label="cache contract",
    )
    contracted_data_root = Path(str(cache_contract.get("data_root", "")))
    if (
        not contracted_data_root.is_absolute()
        or contracted_data_root.expanduser().resolve(strict=True) != root
    ):
        raise ValueError(
            "provided data root differs from the frozen cache contract"
        )
    input_manifest_value = cache_contract.get("input_manifest")
    if not isinstance(input_manifest_value, str) or not input_manifest_value:
        raise ValueError("cache contract lacks input manifest")
    input_manifest = Path(input_manifest_value).expanduser().resolve(
        strict=True
    )
    if not input_manifest.is_file():
        raise FileNotFoundError(input_manifest)
    diagnostic_contract = _require_mapping(
        diagnostic_summary.get("contract"),
        label="diagnostic contract",
    )
    if diagnostic_contract.get("input_manifest") != str(input_manifest):
        raise ValueError("diagnostic/cache input manifest path differs")
    diagnostic_cache = _require_mapping(
        diagnostic_contract.get("cache"),
        label="diagnostic cache binding",
    )
    bound_final = Path(
        str(diagnostic_cache.get("final_directory", ""))
    ).expanduser()
    if (
        not bound_final.is_absolute()
        or bound_final.resolve(strict=True) != cache_final
    ):
        raise ValueError(
            "provided cache final directory differs from diagnostic binding"
        )

    # This validates the final commit, all eight exact source shards, modulo
    # ownership, row/array equality, and all cache artifact hashes.
    cache = load_final_cache(
        input_manifest=input_manifest,
        cache_root=cache_final.parent,
    )
    if dict(cache["contract"]) != dict(cache_contract):
        raise ValueError("cache envelope/core-validation contract differs")

    config = _config_from_diagnostic_contract(diagnostic_contract)
    expected_contract = build_diagnostic_contract(
        input_manifest=input_manifest,
        cache_root=cache_final.parent,
        cache=cache,
        config=config,
    )
    validated = validate_output_commit(
        output_directory=diagnostic,
        expected_contract=expected_contract,
        cache=cache,
        config=config,
    )
    if validated["done"] != diagnostic_done:
        raise ValueError("diagnostic done changed during validation")
    if validated["summary"] != diagnostic_summary:
        raise ValueError("diagnostic summary changed during validation")
    if validated["rows"] != envelope_rows:
        raise ValueError("diagnostic rows changed during validation")

    provenance = {
        "schema_version": REVIEW_BUNDLE_SCHEMA,
        "review_script": str(_SCRIPT_PATH),
        "review_script_sha256": _file_digest(_SCRIPT_PATH),
        "input_manifest": str(input_manifest),
        "input_manifest_sha256": _file_digest(input_manifest),
        "data_root": str(root),
        "cache": {
            "final_directory": str(cache_final),
            "done_sha256": _file_digest(cache_final / DONE_NAME),
            "summary_sha256": _file_digest(cache_final / SUMMARY_NAME),
            "manifest_sha256": _file_digest(cache_final / MANIFEST_NAME),
            "archive_sha256": _file_digest(cache_final / ARCHIVE_NAME),
            "contract_sha256": _object_digest(cache["contract"]),
            "strict_final_and_eight_source_shards_revalidated": True,
        },
        "diagnostic": {
            "directory": str(diagnostic),
            "done_sha256": _file_digest(diagnostic / OUTPUT_DONE_NAME),
            "summary_sha256": _file_digest(
                diagnostic / OUTPUT_SUMMARY_NAME
            ),
            "rows_sha256": _file_digest(diagnostic / ROWS_NAME),
            "contract_sha256": _object_digest(expected_contract),
            "rows_and_recomputed_summary_revalidated": True,
        },
    }
    return {
        "cache": cache,
        "diagnostic_rows": validated["rows"],
        "diagnostic_summary": validated["summary"],
        "config": config,
        "data_root": root,
        "provenance": provenance,
    }


def _read_iid_list(path: Path) -> tuple[list[str], dict[str, Any]]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    text = resolved.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError("IID JSON must be an array")
        values = value
        input_format = "json-array-v1"
    else:
        lines = text.splitlines()
        if any(not line.strip() for line in lines):
            raise ValueError("IID text list contains a blank line")
        values = [line.strip() for line in lines]
        input_format = "one-iid-per-line-v1"
    if not values:
        raise ValueError("IID list is empty")
    iids: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        if not isinstance(value, str) or not _IID_RE.fullmatch(value):
            raise ValueError(f"IID {index} is unsafe or malformed: {value!r}")
        if value in seen:
            raise ValueError(f"IID list contains duplicate {value!r}")
        seen.add(value)
        iids.append(value)
    return iids, {
        "path": str(resolved),
        "sha256": _file_digest(resolved),
        "format": input_format,
        "count": len(iids),
        "ordered_iids_sha256": _object_digest(iids),
    }


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_output_location(
    output_directory: Path,
    *,
    protected_directories: Sequence[Path],
) -> Path:
    output = output_directory.expanduser().resolve(strict=False)
    if output == Path(output.anchor):
        raise ValueError("output directory cannot be a filesystem root")
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite existing/partial review output: {output}"
        )
    for protected in protected_directories:
        resolved = protected.expanduser().resolve(strict=True)
        if (
            output == resolved
            or _is_within(output, resolved)
            or _is_within(resolved, output)
        ):
            raise ValueError(
                f"review output overlaps protected input: {output} vs "
                f"{resolved}"
            )
    return output


def _strict_mask(
    value: Any,
    *,
    track_count: int,
    label: str,
) -> np.ndarray:
    if value == []:
        return np.zeros(track_count, dtype=bool)
    if (
        not isinstance(value, list)
        or len(value) != track_count
        or any(not isinstance(item, bool) for item in value)
    ):
        raise ValueError(f"{label} is not a track-index-preserving bool mask")
    return np.asarray(value, dtype=bool)


def _normalized_to_pixel(
    points: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, int]:
    """Invert the cache normalization and clip only for rasterization."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 2:
        raise ValueError("normalized points must end in an x,y dimension")
    if not np.isfinite(values).all():
        raise ValueError("normalized points are non-finite")
    if width < 2 or height < 2:
        raise ValueError("raster size must be at least 2x2")
    scaled = np.rint(
        values * np.asarray([width, height], dtype=np.float64)
    ).astype(np.int64)
    outside = (
        (scaled[..., 0] < 0)
        | (scaled[..., 0] >= width)
        | (scaled[..., 1] < 0)
        | (scaled[..., 1] >= height)
    )
    scaled[..., 0] = np.clip(scaled[..., 0], 0, width - 1)
    scaled[..., 1] = np.clip(scaled[..., 1], 0, height - 1)
    return scaled, int(np.sum(outside))


def _draw_path_layer(
    frame: np.ndarray,
    pixels: np.ndarray,
    visibility: np.ndarray,
    *,
    frame_index: int,
    track_indices: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
    radius: int,
    ring: bool,
) -> None:
    first = max(0, frame_index - _TRAIL_LENGTH + 1)
    for track_index_value in track_indices:
        track_index = int(track_index_value)
        for time_index in range(first + 1, frame_index + 1):
            if (
                visibility[time_index - 1, track_index]
                < _VISIBILITY_THRESHOLD
                or visibility[time_index, track_index]
                < _VISIBILITY_THRESHOLD
            ):
                continue
            start = tuple(
                int(value) for value in pixels[time_index - 1, track_index]
            )
            stop = tuple(int(value) for value in pixels[time_index, track_index])
            cv2.line(
                frame,
                start,
                stop,
                color,
                thickness,
                lineType=cv2.LINE_AA,
            )
        if visibility[frame_index, track_index] < _VISIBILITY_THRESHOLD:
            continue
        point = tuple(int(value) for value in pixels[frame_index, track_index])
        cv2.circle(
            frame,
            point,
            radius,
            color,
            1 if ring else -1,
            lineType=cv2.LINE_AA,
        )
        if ring:
            cv2.drawMarker(
                frame,
                point,
                color,
                markerType=cv2.MARKER_CROSS,
                markerSize=max(3, radius),
                thickness=1,
                line_type=cv2.LINE_AA,
            )


def _draw_track_layers(
    frame: np.ndarray,
    normalized_tracks: np.ndarray,
    visibility: np.ndarray,
    *,
    frame_index: int,
    base_mask: np.ndarray,
    perturbed_mask: np.ndarray,
) -> int:
    """Draw track-index-preserving memberships on raw video coordinates."""

    tracks = np.asarray(normalized_tracks, dtype=np.float64)
    visible = np.asarray(visibility, dtype=np.float64)
    if (
        tracks.ndim != 3
        or tracks.shape[-1] != 2
        or visible.shape != tracks.shape[:2]
        or frame.ndim != 3
        or frame.shape[2] != 3
    ):
        raise ValueError("track overlay array shapes differ")
    frame_count, track_count, _ = tracks.shape
    if not 0 <= frame_index < frame_count:
        raise IndexError("overlay frame index is outside the cache")
    if base_mask.shape != (track_count,) or perturbed_mask.shape != (
        track_count,
    ):
        raise ValueError("component masks differ from raw track indices")
    if (
        not np.isfinite(tracks).all()
        or not np.isfinite(visible).all()
        or bool(((visible < 0.0) | (visible > 1.0)).any())
    ):
        raise ValueError("track overlay values are invalid")
    height, width = frame.shape[:2]
    pixels, clipped = _normalized_to_pixel(
        tracks,
        width=width,
        height=height,
    )
    background = ~(base_mask | perturbed_mask)
    _draw_path_layer(
        frame,
        pixels,
        visible,
        frame_index=frame_index,
        track_indices=np.flatnonzero(background),
        color=_BACKGROUND_BGR,
        thickness=1,
        radius=1,
        ring=False,
    )
    _draw_path_layer(
        frame,
        pixels,
        visible,
        frame_index=frame_index,
        track_indices=np.flatnonzero(base_mask),
        color=_BASE_BGR,
        thickness=2,
        radius=3,
        ring=False,
    )
    _draw_path_layer(
        frame,
        pixels,
        visible,
        frame_index=frame_index,
        track_indices=np.flatnonzero(perturbed_mask),
        color=_PERTURBED_BGR,
        thickness=2,
        radius=5,
        ring=True,
    )
    return clipped


def _strict_event_window(
    value: Any,
    *,
    frame_count: int,
    label: str,
) -> dict[str, Any] | None:
    if value is None:
        return None
    event = dict(_require_mapping(value, label=label))
    start = event.get("frame_start")
    stop = event.get("frame_stop")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(stop, bool)
        or not isinstance(stop, int)
        or not 0 <= start < stop <= frame_count
    ):
        raise ValueError(f"{label} frame interval differs")
    return event


def _audit_failure_axes(
    audit: Mapping[str, Any],
    *,
    thresholds: Mapping[str, Any],
) -> list[str]:
    if not bool(audit.get("comparison_available")):
        return []
    metrics = _require_mapping(audit.get("metrics"), label="audit metrics")
    comparisons = (
        ("mask", "actor_mask_iou", "actor_mask_iou_threshold", ">="),
        ("event", "event_window_iou", "event_window_iou_threshold", ">="),
        ("traj", "trajectory_rmse", "trajectory_rmse_threshold", "<="),
        (
            "track",
            "per_track_trajectory_rmse",
            "per_track_trajectory_rmse_threshold",
            "<=",
        ),
        ("energy", "energy_cosine", "energy_cosine_threshold", ">="),
        (
            "shape",
            "shape_profile_cosine",
            "shape_profile_cosine_threshold",
            ">=",
        ),
        (
            "duration",
            "event_duration_relative_error",
            "event_duration_relative_error_threshold",
            "<=",
        ),
    )
    failed: list[str] = []
    for short, metric_name, threshold_name, operator in comparisons:
        value = metrics.get(metric_name)
        threshold = thresholds.get(threshold_name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
        ):
            raise ValueError(f"audit metric/threshold {metric_name} differs")
        passed = (
            float(value) >= float(threshold)
            if operator == ">="
            else float(value) <= float(threshold)
        )
        if not passed:
            failed.append(short)
    return failed


def _display_lines(
    *,
    iid: str,
    side: str,
    side_record: Mapping[str, Any],
    audit: Mapping[str, Any] | None,
    audit_axes: Sequence[str],
) -> list[str]:
    ready = bool(side_record.get("diagnostic_ready"))
    score = side_record.get("score", 0.0)
    score_text = (
        f"{float(score):.6g}"
        if isinstance(score, (int, float))
        and not isinstance(score, bool)
        and math.isfinite(float(score))
        else "invalid"
    )
    lines = [
        f"IID {iid} | {side.upper()} | RAW cache tracks",
        "GRAY background | GREEN base | MAGENTA perturbed",
        f"BASE {'READY' if ready else 'NOT_READY'} | score={score_text}",
    ]
    if not ready:
        stage = side_record.get("failure_stage")
        reason = side_record.get("failure_reason")
        lines.append(f"FAIL {stage or '-'} / {reason or '-'}")
    event = side_record.get("event_window")
    if isinstance(event, Mapping):
        lines.append(
            "BASE EVENT "
            f"[{event.get('frame_start')},{event.get('frame_stop')})"
        )
    if audit is not None:
        if not bool(audit.get("eligible")):
            lines.append(f"AUDIT N/A | {audit.get('failure_reason') or '-'}")
        else:
            state = "PASS" if bool(audit.get("joint_pass")) else "FAIL"
            reason = audit.get("failure_reason") or "-"
            axes = ",".join(audit_axes) if audit_axes else "-"
            lines.append(f"AUDIT {state} | {reason} | axes={axes}")
            perturbed = audit.get("perturbed")
            if isinstance(perturbed, Mapping):
                perturbed_event = perturbed.get("event_window")
                if isinstance(perturbed_event, Mapping):
                    lines.append(
                        "PERT EVENT "
                        f"[{perturbed_event.get('frame_start')},"
                        f"{perturbed_event.get('frame_stop')})"
                    )
                if not bool(perturbed.get("diagnostic_ready")):
                    lines.append(
                        "PERT FAIL "
                        f"{perturbed.get('failure_stage') or '-'} / "
                        f"{perturbed.get('failure_reason') or '-'}"
                    )
    return lines


def _draw_text_panel(frame: np.ndarray, lines: Sequence[str]) -> None:
    height, width = frame.shape[:2]
    font_scale = max(0.32, min(0.55, width / 800.0))
    line_height = max(13, int(round(20 * font_scale / 0.55)))
    panel_height = min(height, 7 + line_height * len(lines))
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width - 1, panel_height - 1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0.0, dst=frame)
    approximate_characters = max(18, int(width / max(5.0, 8.5 * font_scale)))
    for index, line in enumerate(lines):
        clipped = (
            line
            if len(line) <= approximate_characters
            else line[: max(1, approximate_characters - 3)] + "..."
        )
        cv2.putText(
            frame,
            clipped,
            (5, 5 + line_height * (index + 1) - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _event_bar_coordinates(
    event: Mapping[str, Any] | None,
    *,
    frame_count: int,
    width: int,
) -> tuple[int, int] | None:
    if event is None:
        return None
    start = int(event["frame_start"])
    stop = int(event["frame_stop"])
    left = int(round(start / frame_count * (width - 1)))
    right = int(round(stop / frame_count * (width - 1)))
    return left, max(left + 1, min(width - 1, right))


def _draw_timeline(
    frame: np.ndarray,
    *,
    frame_index: int,
    frame_count: int,
    base_event: Mapping[str, Any] | None,
    perturbed_event: Mapping[str, Any] | None,
) -> None:
    height, width = frame.shape[:2]
    if height < 16 or width < 8:
        return
    y_base = height - 11
    y_perturbed = height - 5
    cv2.line(frame, (0, y_base), (width - 1, y_base), (45, 45, 45), 3)
    cv2.line(
        frame,
        (0, y_perturbed),
        (width - 1, y_perturbed),
        (45, 45, 45),
        3,
    )
    base_coordinates = _event_bar_coordinates(
        base_event,
        frame_count=frame_count,
        width=width,
    )
    if base_coordinates is not None:
        cv2.line(
            frame,
            (base_coordinates[0], y_base),
            (base_coordinates[1], y_base),
            _BASE_BGR,
            4,
        )
    perturbed_coordinates = _event_bar_coordinates(
        perturbed_event,
        frame_count=frame_count,
        width=width,
    )
    if perturbed_coordinates is not None:
        cv2.line(
            frame,
            (perturbed_coordinates[0], y_perturbed),
            (perturbed_coordinates[1], y_perturbed),
            _PERTURBED_BGR,
            4,
        )
    current_x = int(round(frame_index / max(1, frame_count - 1) * (width - 1)))
    cv2.line(
        frame,
        (current_x, height - 15),
        (current_x, height - 1),
        _CURRENT_FRAME_BGR,
        1,
    )


def _decode_cached_frames(
    path: Path,
    *,
    decode: Mapping[str, Any],
    frame_indices: np.ndarray,
    resized_size: tuple[int, int],
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Decode the exact cached indices and return raw, unwarped BGR frames."""

    indices = np.asarray(frame_indices)
    if (
        indices.ndim != 1
        or indices.dtype.kind not in "iu"
        or len(indices) < 2
        or bool((np.diff(indices) <= 0).any())
    ):
        raise ValueError("cached source frame indices differ")
    if decode.get("source_frame_indices") != indices.astype(int).tolist():
        raise ValueError("cache row/array source frame indices differ")
    source_size = decode.get("source_size")
    if (
        not isinstance(source_size, list)
        or len(source_size) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 2
            for value in source_size
        )
    ):
        raise ValueError("cache source size differs")
    expected_height, expected_width = (int(value) for value in source_size)
    expected_count = decode.get("source_frame_count")
    expected_fps = decode.get("source_fps")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count <= int(indices[-1])
        or isinstance(expected_fps, bool)
        or not isinstance(expected_fps, (int, float))
        or not math.isfinite(float(expected_fps))
        or float(expected_fps) <= 0.0
    ):
        raise ValueError("cache source count/FPS differs")
    render_height, render_width = resized_size
    if render_height < 2 or render_width < 2:
        raise ValueError("cache resized size differs")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"cannot open source video: {path}")
    frames: list[np.ndarray] = []
    try:
        observed_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        observed_fps = float(capture.get(cv2.CAP_PROP_FPS))
        observed_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        observed_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if (
            observed_count != expected_count
            or observed_width != expected_width
            or observed_height != expected_height
            or not math.isclose(
                observed_fps,
                float(expected_fps),
                rel_tol=1e-7,
                abs_tol=1e-7,
            )
        ):
            raise ValueError(
                "source video probe differs from frozen cache decode metadata"
            )
        for source_index_value in indices:
            source_index = int(source_index_value)
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, source_index):
                raise ValueError(
                    f"cannot seek exact cached frame {source_index}: {path}"
                )
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(
                    f"cannot decode exact cached frame {source_index}: {path}"
                )
            if frame.shape != (expected_height, expected_width, 3):
                raise ValueError(
                    f"decoded frame size differs at source frame {source_index}"
                )
            position = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
            if not math.isfinite(position) or not math.isclose(
                position,
                source_index + 1.0,
                rel_tol=0.0,
                abs_tol=0.5,
            ):
                raise ValueError(
                    f"backend did not confirm exact frame {source_index}"
                )
            if frame.shape[:2] != (render_height, render_width):
                interpolation = (
                    cv2.INTER_AREA
                    if render_height <= expected_height
                    and render_width <= expected_width
                    else cv2.INTER_LINEAR
                )
                frame = cv2.resize(
                    frame,
                    (render_width, render_height),
                    interpolation=interpolation,
                )
            if frame.shape != (render_height, render_width, 3):
                raise ValueError("rendered raw frame size differs")
            frames.append(np.ascontiguousarray(frame))
    finally:
        capture.release()
    if len(frames) != len(indices):
        raise ValueError("decoded frame count differs from cache")
    return frames, {
        "source_frame_count": expected_count,
        "source_fps": float(expected_fps),
        "source_size": [expected_height, expected_width],
        "rendered_content_size": [render_height, render_width],
        "decoded_frames": len(frames),
        "exact_seek_position_verified": True,
    }


def _sampled_playback_fps(frame_times: np.ndarray) -> float:
    times = np.asarray(frame_times, dtype=np.float64)
    if (
        times.ndim != 1
        or len(times) < 2
        or not np.isfinite(times).all()
        or bool((np.diff(times) <= 0.0).any())
    ):
        raise ValueError("cache frame times differ")
    fps = float((len(times) - 1) / (times[-1] - times[0]))
    if not math.isfinite(fps) or not 0.1 <= fps <= 240.0:
        raise ValueError("sampled playback FPS is invalid")
    return fps


def _encoded_size(content_size: tuple[int, int]) -> tuple[int, int]:
    height, width = content_size
    return height + height % 2, width + width % 2


def _pad_for_codec(
    frame: np.ndarray,
    *,
    encoded_size: tuple[int, int],
) -> np.ndarray:
    encoded_height, encoded_width = encoded_size
    height, width = frame.shape[:2]
    if encoded_height < height or encoded_width < width:
        raise ValueError("encoded frame size is smaller than content")
    if (encoded_height, encoded_width) == (height, width):
        return frame
    return cv2.copyMakeBorder(
        frame,
        0,
        encoded_height - height,
        0,
        encoded_width - width,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


def _probe_review_video(
    path: Path,
    *,
    expected_frames: int,
    expected_size: tuple[int, int],
    expected_fps: float,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"cannot reopen rendered review video: {path}")
    decoded = 0
    try:
        reported_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
        reported_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        reported_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if (
            reported_frames != expected_frames
            or (reported_height, reported_width) != expected_size
            or not math.isclose(
                reported_fps,
                expected_fps,
                rel_tol=1e-3,
                abs_tol=1e-2,
            )
        ):
            raise ValueError("rendered review video probe differs")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.shape != (
                expected_size[0],
                expected_size[1],
                3,
            ):
                raise ValueError("rendered review video frame size differs")
            decoded += 1
    finally:
        capture.release()
    if decoded != expected_frames:
        raise ValueError("rendered review video decoded frame count differs")
    return {
        "frame_count": decoded,
        "fps": reported_fps,
        "frame_size": [reported_height, reported_width],
        "codec": _CODEC,
    }


def _write_review_video(
    path: Path,
    *,
    raw_frames: Sequence[np.ndarray],
    normalized_tracks: np.ndarray,
    visibility: np.ndarray,
    base_mask: np.ndarray,
    perturbed_mask: np.ndarray,
    display_lines: Sequence[str],
    base_event: Mapping[str, Any] | None,
    perturbed_event: Mapping[str, Any] | None,
    playback_fps: float,
) -> dict[str, Any]:
    if not raw_frames:
        raise ValueError("cannot write an empty review video")
    content_size = tuple(int(value) for value in raw_frames[0].shape[:2])
    if any(frame.shape[:2] != content_size for frame in raw_frames):
        raise ValueError("raw review frames have inconsistent sizes")
    encoded_size = _encoded_size(content_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*_CODEC),
        playback_fps,
        (encoded_size[1], encoded_size[0]),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError("OpenCV mp4v VideoWriter could not be opened")
    try:
        for frame_index, raw_frame in enumerate(raw_frames):
            rendered = raw_frame.copy()
            _draw_track_layers(
                rendered,
                normalized_tracks,
                visibility,
                frame_index=frame_index,
                base_mask=base_mask,
                perturbed_mask=perturbed_mask,
            )
            _draw_text_panel(rendered, display_lines)
            _draw_timeline(
                rendered,
                frame_index=frame_index,
                frame_count=len(raw_frames),
                base_event=base_event,
                perturbed_event=perturbed_event,
            )
            writer.write(
                _pad_for_codec(rendered, encoded_size=encoded_size)
            )
    finally:
        writer.release()
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError("review video writer produced no bytes")
    probe = _probe_review_video(
        path,
        expected_frames=len(raw_frames),
        expected_size=encoded_size,
        expected_fps=playback_fps,
    )
    return {
        **probe,
        "content_frame_size": list(content_size),
        "codec_padding": {
            "bottom": encoded_size[0] - content_size[0],
            "right": encoded_size[1] - content_size[1],
        },
        "sha256": _file_digest(path),
        "bytes": path.stat().st_size,
    }


def _row_index(
    cache_rows: Sequence[Mapping[str, Any]],
    diagnostic_rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[int, Mapping[str, Any], Mapping[str, Any]]]:
    if len(cache_rows) != len(diagnostic_rows):
        raise ValueError("cache/diagnostic row count differs")
    output: dict[
        str,
        tuple[int, Mapping[str, Any], Mapping[str, Any]],
    ] = {}
    for array_index, (cache_row, diagnostic_row) in enumerate(
        zip(cache_rows, diagnostic_rows)
    ):
        iid = cache_row.get("iid")
        if (
            not isinstance(iid, str)
            or not _IID_RE.fullmatch(iid)
            or iid in output
            or diagnostic_row.get("iid") != iid
            or diagnostic_row.get("input_index")
            != cache_row.get("input_index")
        ):
            raise ValueError("cache/diagnostic IID binding differs")
        output[iid] = (array_index, cache_row, diagnostic_row)
    return output


def _render_review_item(
    *,
    stage_directory: Path,
    final_output_directory: Path,
    ordinal: int,
    side: str,
    array_index: int,
    cache: Mapping[str, Any],
    cache_row: Mapping[str, Any],
    diagnostic_row: Mapping[str, Any],
    data_root: Path,
    audit_thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    if side not in _SIDES:
        raise ValueError(f"unsupported review side: {side}")
    iid = str(cache_row["iid"])
    input_row = _require_mapping(
        cache_row.get("input_row"),
        label=f"cache row {iid} input_row",
    )
    input_field = _SIDE_INPUT_FIELDS[side]
    relative_video = input_row.get(input_field)
    if not isinstance(relative_video, str) or not relative_video:
        raise ValueError(f"cache row {iid} lacks {input_field}")
    source_video = _safe_video_path(data_root, relative_video)
    if not source_video.is_file():
        raise FileNotFoundError(source_video)
    cache_side = _require_mapping(
        cache_row.get(side),
        label=f"cache row {iid} {side}",
    )
    if cache_side.get("resolved_path") != str(source_video):
        raise ValueError(f"cache row {iid} {side} resolved path differs")
    expected_video_sha = _require_sha256(
        cache_side.get("video_sha256"),
        label=f"cache row {iid} {side} video_sha256",
    )
    before_video_sha = _file_digest(source_video)
    if before_video_sha != expected_video_sha:
        raise ValueError(f"source video bytes changed: {source_video}")
    decode = _require_mapping(
        cache_side.get("decode"),
        label=f"cache row {iid} {side} decode",
    )
    arrays = _require_mapping(cache.get("arrays"), label="cache arrays")
    raw_tracks = np.asarray(
        arrays[f"{side}_normalized_tracks"][array_index]
    )
    visibility = np.asarray(arrays[f"{side}_visibility"][array_index])
    frame_indices = np.asarray(
        arrays[f"{side}_source_frame_indices"][array_index]
    )
    frame_times = np.asarray(arrays[f"{side}_frame_times"][array_index])
    resized_array = np.asarray(
        arrays[f"{side}_resized_size"][array_index]
    )
    track_valid = bool(arrays[f"{side}_track_valid"][array_index])
    if (
        not track_valid
        or raw_tracks.ndim != 3
        or raw_tracks.shape[0] != VIDEO_FRAMES
        or raw_tracks.shape[2] != 2
        or visibility.shape != raw_tracks.shape[:2]
        or frame_indices.shape != (VIDEO_FRAMES,)
        or frame_times.shape != (VIDEO_FRAMES,)
        or resized_array.shape != (2,)
    ):
        raise ValueError(
            f"cache row {iid} {side} lacks a renderable track cache"
        )
    resized_size = (int(resized_array[0]), int(resized_array[1]))
    if decode.get("resized_size") != list(resized_size):
        raise ValueError(f"cache row {iid} {side} resized size differs")
    raw_frames, decode_probe = _decode_cached_frames(
        source_video,
        decode=decode,
        frame_indices=frame_indices,
        resized_size=resized_size,
    )
    after_video_sha = _file_digest(source_video)
    if after_video_sha != before_video_sha:
        raise ValueError(f"source video changed during decode: {source_video}")
    diagnostic_side = _require_mapping(
        diagnostic_row.get(side),
        label=f"diagnostic row {iid} {side}",
    )
    track_count = raw_tracks.shape[1]
    base_mask = _strict_mask(
        diagnostic_side.get("actor_track_mask"),
        track_count=track_count,
        label=f"diagnostic row {iid} {side} base mask",
    )
    audit: Mapping[str, Any] | None = None
    perturbed_record: Mapping[str, Any] | None = None
    if side == "target":
        audit = _require_mapping(
            diagnostic_row.get("target_audit"),
            label=f"diagnostic row {iid} target audit",
        )
        value = audit.get("perturbed")
        if value is not None:
            perturbed_record = _require_mapping(
                value,
                label=f"diagnostic row {iid} perturbed target",
            )
    perturbed_mask = (
        np.zeros(track_count, dtype=bool)
        if perturbed_record is None
        else _strict_mask(
            perturbed_record.get("actor_track_mask"),
            track_count=track_count,
            label=f"diagnostic row {iid} perturbed target mask",
        )
    )
    base_event = _strict_event_window(
        diagnostic_side.get("event_window"),
        frame_count=VIDEO_FRAMES,
        label=f"diagnostic row {iid} {side} base event",
    )
    perturbed_event = _strict_event_window(
        (
            None
            if perturbed_record is None
            else perturbed_record.get("event_window")
        ),
        frame_count=VIDEO_FRAMES,
        label=f"diagnostic row {iid} perturbed target event",
    )
    audit_axes = (
        []
        if audit is None
        else _audit_failure_axes(audit, thresholds=audit_thresholds)
    )
    display_lines = _display_lines(
        iid=iid,
        side=side,
        side_record=diagnostic_side,
        audit=audit,
        audit_axes=audit_axes,
    )
    playback_fps = _sampled_playback_fps(frame_times)
    relative_output = (
        Path(VIDEOS_DIRECTORY_NAME) / f"{iid}__{side}.mp4"
    )
    staged_video = stage_directory / relative_output
    video_artifact = _write_review_video(
        staged_video,
        raw_frames=raw_frames,
        normalized_tracks=raw_tracks,
        visibility=visibility,
        base_mask=base_mask,
        perturbed_mask=perturbed_mask,
        display_lines=display_lines,
        base_event=base_event,
        perturbed_event=perturbed_event,
        playback_fps=playback_fps,
    )
    _, out_of_bounds = _normalized_to_pixel(
        raw_tracks,
        width=resized_size[1],
        height=resized_size[0],
    )
    return {
        "schema_version": REVIEW_ROW_SCHEMA,
        "review_only": True,
        "ordinal": ordinal,
        "iid": iid,
        "input_index": cache_row["input_index"],
        "side": side,
        "label_type": diagnostic_row.get("label_type"),
        "negative_type": diagnostic_row.get("negative_type"),
        "positive": diagnostic_row.get("positive"),
        "action_signature": diagnostic_row.get("action_signature"),
        "source_video": {
            "input_field": input_field,
            "relative_path": relative_video,
            "resolved_path": str(source_video),
            "sha256": expected_video_sha,
            "sha256_rechecked_before_and_after_decode": True,
        },
        "decode": {
            "cache_record": dict(decode),
            "probe": decode_probe,
            "source_frame_indices_sha256": _array_digest(frame_indices),
            "frame_times_sha256": _array_digest(frame_times),
            "exact_cached_indices_used": True,
            "raw_frames_not_stabilized_or_warped": True,
        },
        "tracks": {
            "coordinate_source": (
                f"{side}_normalized_tracks; raw pre-camera cache coordinates"
            ),
            "coordinate_mapping": (
                "pixel_x=round(raw_normalized_x*cache_resized_width); "
                "pixel_y=round(raw_normalized_y*cache_resized_height); "
                "raster coordinates clipped only after inversion"
            ),
            "membership_mapping": (
                "base/perturbed bool-mask position equals immutable cache "
                "track index; no sorting or reindexing"
            ),
            "normalized_tracks_sha256": _array_digest(raw_tracks),
            "visibility_sha256": _array_digest(visibility),
            "track_count": track_count,
            "background_track_count": int(
                np.sum(~(base_mask | perturbed_mask))
            ),
            "base_track_indices": np.flatnonzero(base_mask).tolist(),
            "perturbed_track_indices": np.flatnonzero(
                perturbed_mask
            ).tolist(),
            "overlap_track_indices": np.flatnonzero(
                base_mask & perturbed_mask
            ).tolist(),
            "raw_coordinate_out_of_bounds_count": out_of_bounds,
            "visibility_threshold": _VISIBILITY_THRESHOLD,
            "trail_sampled_frames": _TRAIL_LENGTH,
            "colors": _COLOR_CONTRACT,
        },
        "diagnostic": {
            "cache_row_sha256": _object_digest(cache_row),
            "diagnostic_row_sha256": _object_digest(diagnostic_row),
            "camera_valid": diagnostic_row.get(f"{side}_camera_valid"),
            "base": {
                "diagnostic_ready": diagnostic_side.get(
                    "diagnostic_ready"
                ),
                "selector_ready": diagnostic_side.get("selector_ready"),
                "event_ready": diagnostic_side.get("event_ready"),
                "failure_stage": diagnostic_side.get("failure_stage"),
                "failure_reason": diagnostic_side.get("failure_reason"),
                "failure_detail": diagnostic_side.get("failure_detail"),
                "score": diagnostic_side.get("score"),
                "event_window": base_event,
            },
            "target_audit": (
                None
                if audit is None
                else {
                    "eligible": audit.get("eligible"),
                    "performed": audit.get("performed"),
                    "seed": audit.get("seed"),
                    "seed_derivation": audit.get("seed_derivation"),
                    "perturbation": audit.get("perturbation"),
                    "comparison_available": audit.get(
                        "comparison_available"
                    ),
                    "ready_consistent": audit.get("ready_consistent"),
                    "joint_pass": audit.get("joint_pass"),
                    "failure_reason": audit.get("failure_reason"),
                    "failed_axes": audit_axes,
                    "metrics": audit.get("metrics"),
                    "perturbed_diagnostic_ready": (
                        None
                        if perturbed_record is None
                        else perturbed_record.get("diagnostic_ready")
                    ),
                    "perturbed_failure_stage": (
                        None
                        if perturbed_record is None
                        else perturbed_record.get("failure_stage")
                    ),
                    "perturbed_failure_reason": (
                        None
                        if perturbed_record is None
                        else perturbed_record.get("failure_reason")
                    ),
                    "perturbed_event_window": perturbed_event,
                }
            ),
        },
        "output_video": {
            "relative_path": relative_output.as_posix(),
            "final_path": str(final_output_directory / relative_output),
            **video_artifact,
        },
        "scope": REVIEW_SCOPE,
        "formal_status": "INSUFFICIENT",
        "production_decision": False,
        "generation_authorized": False,
    }


def _validate_staged_bundle(
    stage_directory: Path,
    *,
    expected_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    manifest_path = stage_directory / REVIEW_MANIFEST_NAME
    summary_path = stage_directory / REVIEW_SUMMARY_NAME
    loaded_rows = _read_canonical_jsonl(manifest_path)
    if loaded_rows != [dict(row) for row in expected_rows]:
        raise ValueError("staged review manifest rows differ")
    loaded_summary = _read_json(summary_path)
    if loaded_summary != dict(summary):
        raise ValueError("staged review summary differs")
    if loaded_summary.get("committed") is not True:
        raise ValueError("staged review summary is not committed")
    if loaded_summary.get("review_manifest_sha256") != _file_digest(
        manifest_path
    ):
        raise ValueError("staged review manifest digest differs")
    expected_files = {
        REVIEW_MANIFEST_NAME,
        REVIEW_SUMMARY_NAME,
    }
    for row in loaded_rows:
        output_video = _require_mapping(
            row.get("output_video"),
            label="review output video",
        )
        relative_value = output_video.get("relative_path")
        if not isinstance(relative_value, str):
            raise ValueError("review output video path differs")
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("review output video path escapes bundle")
        video = (stage_directory / relative).resolve(strict=True)
        if not _is_within(video, stage_directory):
            raise ValueError("review output video escapes bundle")
        if output_video.get("sha256") != _file_digest(video):
            raise ValueError("review output video digest differs")
        if output_video.get("bytes") != video.stat().st_size:
            raise ValueError("review output video byte size differs")
        expected_files.add(relative.as_posix())
    actual_files = {
        path.relative_to(stage_directory).as_posix()
        for path in stage_directory.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError(
            "staged review file set differs; "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}"
        )


def build_review_bundle(
    *,
    cache_final_directory: Path,
    diagnostic_directory: Path,
    data_root: Path,
    iid_list: Path,
    output_directory: Path,
    include_source: bool = False,
) -> dict[str, Any]:
    """Validate inputs, render selected rows, and atomically publish a bundle."""

    # Refuse a visible output path before doing expensive cache validation.
    initial_output = output_directory.expanduser().resolve(strict=False)
    if initial_output.exists():
        raise FileExistsError(
            "refusing to overwrite existing/partial review output: "
            f"{initial_output}"
        )
    verified = load_verified_inputs(
        cache_final_directory=cache_final_directory,
        diagnostic_directory=diagnostic_directory,
        data_root=data_root,
    )
    iids, iid_provenance = _read_iid_list(iid_list)
    provenance = _require_mapping(
        verified.get("provenance"),
        label="verified provenance",
    )
    cache_final = Path(
        str(
            _require_mapping(
                provenance.get("cache"),
                label="verified cache provenance",
            )["final_directory"]
        )
    )
    diagnostic = Path(
        str(
            _require_mapping(
                provenance.get("diagnostic"),
                label="verified diagnostic provenance",
            )["directory"]
        )
    )
    root = Path(str(provenance["data_root"]))
    output = _validate_output_location(
        output_directory,
        protected_directories=(cache_final.parent, diagnostic, root),
    )
    cache = _require_mapping(verified.get("cache"), label="verified cache")
    cache_rows = cache.get("rows")
    diagnostic_rows = verified.get("diagnostic_rows")
    if not isinstance(cache_rows, Sequence) or isinstance(
        cache_rows, (str, bytes)
    ):
        raise ValueError("verified cache rows differ")
    if not isinstance(diagnostic_rows, Sequence) or isinstance(
        diagnostic_rows, (str, bytes)
    ):
        raise ValueError("verified diagnostic rows differ")
    index = _row_index(cache_rows, diagnostic_rows)
    missing = [iid for iid in iids if iid not in index]
    if missing:
        raise ValueError(f"requested IIDs are absent: {missing}")
    diagnostic_summary = _require_mapping(
        verified.get("diagnostic_summary"),
        label="verified diagnostic summary",
    )
    diagnostic_contract = _require_mapping(
        diagnostic_summary.get("contract"),
        label="verified diagnostic contract",
    )
    audit_thresholds = _require_mapping(
        _require_mapping(
            diagnostic_contract.get("independent_audit"),
            label="verified audit contract",
        ).get("config"),
        label="verified audit thresholds",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging_prefix = f".{output.name}.staging-"
    stale_staging = sorted(
        path
        for path in output.parent.iterdir()
        if path.name.startswith(staging_prefix)
    )
    if stale_staging:
        raise FileExistsError(
            "stale partial review staging exists; inspect it before retry: "
            + ", ".join(str(path) for path in stale_staging)
        )
    lock = output.parent / f".{output.name}.review-build.lock"
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise FileExistsError(
            f"review build lock already exists: {lock}"
        ) from error
    try:
        stage = Path(
            tempfile.mkdtemp(
                prefix=staging_prefix,
                dir=str(output.parent),
            )
        )
    except BaseException:
        lock.rmdir()
        raise
    rows: list[dict[str, Any]] = []
    sides = ("target", "source") if include_source else ("target",)
    try:
        ordinal = 0
        for iid in iids:
            array_index, cache_row, diagnostic_row = index[iid]
            for side in sides:
                rows.append(
                    _render_review_item(
                        stage_directory=stage,
                        final_output_directory=output,
                        ordinal=ordinal,
                        side=side,
                        array_index=array_index,
                        cache=cache,
                        cache_row=cache_row,
                        diagnostic_row=diagnostic_row,
                        data_root=Path(str(verified["data_root"])),
                        audit_thresholds=audit_thresholds,
                    )
                )
                ordinal += 1
        manifest_path = stage / REVIEW_MANIFEST_NAME
        _write_canonical_jsonl(manifest_path, rows)
        video_index = [
            {
                "iid": row["iid"],
                "side": row["side"],
                "relative_path": row["output_video"]["relative_path"],
                "sha256": row["output_video"]["sha256"],
                "bytes": row["output_video"]["bytes"],
            }
            for row in rows
        ]
        summary = {
            "schema_version": REVIEW_SUMMARY_SCHEMA,
            "bundle_schema_version": REVIEW_BUNDLE_SCHEMA,
            "committed": True,
            "review_only": True,
            "scope": REVIEW_SCOPE,
            "requested_iids": len(iids),
            "review_items": len(rows),
            "sides": list(sides),
            "include_source": bool(include_source),
            "review_manifest_name": REVIEW_MANIFEST_NAME,
            "review_manifest_sha256": _file_digest(manifest_path),
            "video_index_sha256": _object_digest(video_index),
            "video_index": video_index,
            "provenance": {
                **dict(provenance),
                "iid_list": iid_provenance,
                "output_directory": str(output),
                "input_commits_validated_before_output_write": True,
                "selected_video_bytes_rehashed_before_and_after_decode": True,
                "staging_directory_atomically_published": True,
            },
            "coordinate_contract": {
                "raw_tracks_only": True,
                "camera_stabilized_tracks_drawn": False,
                "track_indices_reordered": False,
                "content_frames": VIDEO_FRAMES,
                "codec_even_dimension_padding_only": True,
                "colors": _COLOR_CONTRACT,
            },
            "formal_status": "INSUFFICIENT",
            "production_decision": False,
            "generation_authorized": False,
        }
        _write_json(stage / REVIEW_SUMMARY_NAME, summary)
        _validate_staged_bundle(
            stage,
            expected_rows=rows,
            summary=summary,
        )
        if output.exists():
            raise FileExistsError(
                "review output appeared during build; refusing overwrite: "
                f"{output}"
            )
        os.rename(stage, output)
        lock.rmdir()
        return summary
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        try:
            lock.rmdir()
        except FileNotFoundError:
            pass
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-final-dir",
        type=Path,
        required=True,
        help="final/ directory of the complete P1 track cache",
    )
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--iid-list",
        type=Path,
        required=True,
        help="one IID per line, or a JSON array of IID strings",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--include-source",
        action="store_true",
        help="also render source; target is always rendered",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = build_review_bundle(
        cache_final_directory=args.cache_final_dir,
        diagnostic_directory=args.diagnostic_dir,
        data_root=args.data_root,
        iid_list=args.iid_list,
        output_directory=args.output_dir,
        include_source=args.include_source,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
