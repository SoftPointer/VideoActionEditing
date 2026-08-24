"""Immutable R7-P1 CoTracker observation cache.

R7-P0 showed that camera compensation works but its sparse top-outlier actor
selector does not.  Re-running CoTracker for every selector experiment would
be slow and would silently change the experimental input.  This module
therefore caches the *pre-selector* evidence:

* normalized CoTracker point tracks and visibility;
* exact decoded-frame indices and times;
* camera-compensated tracks and affine provenance when compensation succeeds.

The cache deliberately contains no actor labels, event boundaries, quality
decision, or generation authorization.  The inspected R5/P0 rows remain a
development set.  A downstream selector may use this cache for diagnostics,
but it must not turn the cache's existence into a production claim.

Each torchrun rank atomically commits a private shard.  ``finalize`` accepts
exactly eight complete shards, validates their hashes and modulo ownership,
then atomically commits one input-ordered archive.  ``--resume`` only validates
an existing complete commit; partial artifacts are never appended to.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .r7_preflight_extract import (
    DEFAULT_SEED,
    FINAL_WORLD_SIZE,
    GlobalExtractionError,
    PerVideoError,
    R7_VIDEO_SAMPLING,
    VIDEO_FRAMES,
    _array_digest,
    _atomic_json,
    _atomic_jsonl,
    _atomic_npz,
    _canonical_json,
    _file_digest,
    _object_digest,
    _read_r5_manifest,
    _safe_video_path,
    decode_video_fixed_frames,
    resolve_torchrun_coordinates,
)
from .r7_temporal_teacher import (
    LazyCoTrackerAdapter,
    TemporalTeacherConfig,
    TemporalTeacherError,
    robust_camera_compensation,
)


R7_TRACK_CACHE_SCHEMA = "motive-r7-p1-track-cache-v2"
R7_TRACK_CACHE_ROW_SCHEMA = "motive-r7-p1-track-cache-row-v2"
R7_TRACK_CACHE_SHARD_SUMMARY_SCHEMA = (
    "motive-r7-p1-track-cache-shard-summary-v2"
)
R7_TRACK_CACHE_SHARD_DONE_SCHEMA = "motive-r7-p1-track-cache-shard-done-v2"
R7_TRACK_CACHE_FINAL_SUMMARY_SCHEMA = (
    "motive-r7-p1-track-cache-final-summary-v2"
)
R7_TRACK_CACHE_FINAL_DONE_SCHEMA = "motive-r7-p1-track-cache-final-done-v2"
R7_TRACK_CACHE_PARTITION = "input-index-modulo-world-size-v1"
ARCHIVE_NAME = "track_cache.npz"
MANIFEST_NAME = "manifest.jsonl"
SUMMARY_NAME = "summary.json"
DONE_NAME = "done.json"
FINAL_DIR_NAME = "final"
SIDES = ("source", "target")
FORMAL_STATUS = "INSUFFICIENT"
COHORT_ROLE = "development"
COHORT_ID = "old181-r5-p0-inspected-dev-v1"
EXPECTED_COHORT_ROWS = 181
FORMAL_REASON = (
    "The inspected 181-row R5/P0 pseudo-labeled cohort is a development set "
    "without human event boundaries, actor masks, or a fresh locked split."
)
CACHE_SCOPE = "pre-selector observations for development diagnostics only"
MIN_SOURCE_TRACK_COVERAGE = 0.90
MIN_TARGET_TRACK_COVERAGE = 0.90
MIN_PAIRED_TRACK_COVERAGE = 0.85
OPERATIONAL_COVERAGE_POLICY = "all-181-rows-track-valid-fraction-v1"
TRACKER_ABI_FAILURE_REASONS = frozenset(
    {
        "invalid_shape",
        "invalid_dtype",
        "non_finite_input",
        "invalid_visibility",
        "invalid_frame_times",
        "invalid_frame_size",
        "invalid_pixel_range",
        "invalid_tracker_output",
        "missing_provenance",
        "insufficient_frames",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


def _rank_directory(root: Path, rank: int, world_size: int) -> Path:
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("require 0 <= rank < world_size")
    return (
        root.expanduser()
        / "shards"
        / f"rank-{rank:03d}-of-{world_size:03d}"
    )


def _artifact_paths(directory: Path) -> dict[str, Path]:
    return {
        "archive": directory / ARCHIVE_NAME,
        "manifest": directory / MANIFEST_NAME,
        "summary": directory / SUMMARY_NAME,
        "done": directory / DONE_NAME,
    }


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def validate_output_root(
    *,
    output_root: Path,
    input_manifest: Path,
    data_root: Path,
    tracker_checkpoint: Path,
    cotracker_root: Path,
    source_snapshot: Path | None = None,
) -> Path:
    """Reject output locations that could pollute or contain an input tree."""

    output = output_root.expanduser().resolve(strict=False)
    inputs = {
        "input_manifest": input_manifest.expanduser().resolve(strict=True),
        "data_root": data_root.expanduser().resolve(strict=True),
        "tracker_checkpoint": tracker_checkpoint.expanduser().resolve(
            strict=True
        ),
        "cotracker_root": cotracker_root.expanduser().resolve(strict=True),
    }
    if source_snapshot is not None:
        inputs["source_snapshot"] = source_snapshot.expanduser().resolve(
            strict=True
        )
    if output == Path(output.anchor):
        raise ValueError("output root cannot be a filesystem root")
    for name, value in inputs.items():
        if output == value or _is_within(output, value) or _is_within(
            value, output
        ):
            raise ValueError(
                f"output root overlaps protected {name}: {output} vs {value}"
            )
    if (output / FINAL_DIR_NAME / DONE_NAME).exists():
        raise FileExistsError(
            "completed final cache already exists; refuse an 8-GPU resume: "
            f"{output / FINAL_DIR_NAME / DONE_NAME}"
        )
    return output


def _run_git(root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ("git", "-C", str(root), *arguments),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        raise ValueError(f"cannot inspect CoTracker git source at {root}") from error
    return result.stdout.strip()


def cotracker_source_provenance(root: Path) -> dict[str, Any]:
    """Hash every importable Python source and require clean tracked bytes."""

    resolved = root.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    predictor = resolved / "cotracker" / "predictor.py"
    if not predictor.is_file():
        raise FileNotFoundError(predictor)
    inventory: list[dict[str, Any]] = []
    for path in sorted(resolved.rglob("*.py")):
        if not path.is_file():
            continue
        if path.is_symlink():
            raise ValueError(f"CoTracker Python source is a symlink: {path}")
        inventory.append(
            {
                "path": path.relative_to(resolved).as_posix(),
                "size": path.stat().st_size,
                "sha256": _file_digest(path),
            }
        )
    if not inventory:
        raise ValueError(f"CoTracker Python source bundle is empty: {resolved}")
    git_root = Path(
        _run_git(resolved, "rev-parse", "--show-toplevel")
    ).resolve(strict=True)
    git_head = _run_git(resolved, "rev-parse", "HEAD")
    if not _GIT_COMMIT_RE.fullmatch(git_head):
        raise ValueError("CoTracker git HEAD is not a full commit id")
    tracked_status = _run_git(
        resolved,
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
    )
    tracked_clean = tracked_status == ""
    if not tracked_clean:
        raise ValueError(
            "CoTracker has modified/deleted/staged tracked files; freeze it first"
        )
    return {
        "root": str(resolved),
        "git_toplevel": str(git_root),
        "git_head": git_head,
        "git_tracked_clean": True,
        "python_source_file_count": len(inventory),
        "python_source_files": inventory,
        "python_source_bundle_sha256": _object_digest(inventory),
    }


def _configure_determinism(seed: int, *, local_rank: int) -> dict[str, Any]:
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= 0xFFFFFFFF
    ):
        raise ValueError("seed must be an integer in [0, 2**32-1]")
    if local_rank < 0:
        raise ValueError("local_rank must be nonnegative")
    if os.environ.get("PYTHONHASHSEED") != str(seed):
        raise GlobalExtractionError(
            "PYTHONHASHSEED must equal --seed before Python starts"
        )
    try:
        import torch
    except ImportError as error:
        raise GlobalExtractionError("torch is not importable") from error
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        raise GlobalExtractionError("torch reports no CUDA/HIP device")
    if local_rank >= torch.cuda.device_count():
        raise GlobalExtractionError(
            f"local_rank={local_rank} exceeds visible devices"
        )
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return {
        "schema_version": "motive-r7-determinism-v1",
        "seed": seed,
        "rank_seed_policy": "identical-base-seed-on-all-eight-ranks-v1",
        "python_random_seeded": True,
        "numpy_seeded": True,
        "torch_cpu_seeded": True,
        "torch_all_visible_devices_seeded": True,
        "torch_deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_benchmark": bool(
            getattr(getattr(torch.backends, "cudnn", object()), "benchmark", False)
        ),
        "cudnn_deterministic": bool(
            getattr(
                getattr(torch.backends, "cudnn", object()),
                "deterministic",
                False,
            )
        ),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
    }


def runtime_provenance(
    *,
    local_rank: int,
    determinism: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        import cv2
        import torch
    except ImportError as error:
        raise GlobalExtractionError(
            "torch and OpenCV are required for runtime provenance"
        ) from error
    if local_rank < 0 or local_rank >= torch.cuda.device_count():
        raise GlobalExtractionError("invalid local GPU rank")
    properties = torch.cuda.get_device_properties(local_rank)
    capability = torch.cuda.get_device_capability(local_rank)
    build_config = (
        torch.__config__.show()
        if hasattr(torch, "__config__")
        else "unavailable"
    )
    hip_version = getattr(torch.version, "hip", None)
    return {
        "schema_version": "motive-r7-runtime-v1",
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": str(Path(sys.executable).resolve(strict=True)),
        "platform": platform.platform(),
        # Recent PyTorch releases expose ``torch.__version__`` as a
        # ``TorchVersion`` (a ``str`` subclass).  Contracts deliberately
        # admit JSON-native scalar types only, so normalize every library
        # version at the producer boundary rather than weakening validators.
        "numpy_version": str(np.__version__),
        "opencv_version": str(cv2.__version__),
        "torch_version": str(torch.__version__),
        "torch_hip_version": (
            None if hip_version is None else str(hip_version)
        ),
        "torch_build_config_sha256": _object_digest(build_config),
        "visible_device_count": int(torch.cuda.device_count()),
        "device_type": "cuda-hip" if hip_version else "cuda",
        "device_name": str(torch.cuda.get_device_name(local_rank)),
        "device_capability": [int(value) for value in capability],
        "device_total_memory": int(properties.total_memory),
        "determinism": dict(determinism),
    }


def _empty_arrays(
    row_count: int,
    *,
    track_count: int,
) -> dict[str, np.ndarray]:
    if row_count < 0:
        raise ValueError("row_count must be nonnegative")
    if track_count < 4:
        raise ValueError("track_count must be >= 4")
    arrays: dict[str, np.ndarray] = {
        "input_indices": np.zeros(row_count, dtype=np.int64),
        "positive": np.zeros(row_count, dtype=np.bool_),
    }
    for side in SIDES:
        arrays.update(
            {
                f"{side}_track_valid": np.zeros(
                    row_count, dtype=np.bool_
                ),
                f"{side}_camera_valid": np.zeros(
                    row_count, dtype=np.bool_
                ),
                f"{side}_normalized_tracks": np.zeros(
                    (row_count, VIDEO_FRAMES, track_count, 2),
                    dtype=np.float32,
                ),
                f"{side}_visibility": np.zeros(
                    (row_count, VIDEO_FRAMES, track_count),
                    dtype=np.float32,
                ),
                f"{side}_frame_times": np.zeros(
                    (row_count, VIDEO_FRAMES), dtype=np.float64
                ),
                f"{side}_source_frame_indices": np.zeros(
                    (row_count, VIDEO_FRAMES), dtype=np.int64
                ),
                f"{side}_resized_size": np.zeros(
                    (row_count, 2), dtype=np.int32
                ),
                f"{side}_source_fps": np.zeros(
                    row_count, dtype=np.float64
                ),
                f"{side}_stabilized_tracks": np.zeros(
                    (row_count, VIDEO_FRAMES, track_count, 2),
                    dtype=np.float32,
                ),
                f"{side}_transition_affines": np.zeros(
                    (row_count, VIDEO_FRAMES - 1, 2, 3),
                    dtype=np.float32,
                ),
                f"{side}_cumulative_affines": np.zeros(
                    (row_count, VIDEO_FRAMES, 2, 3),
                    dtype=np.float32,
                ),
                f"{side}_camera_crossfit_valid": np.zeros(
                    row_count, dtype=np.bool_
                ),
                f"{side}_camera_crossfit_raw_median": np.zeros(
                    row_count, dtype=np.float32
                ),
                f"{side}_camera_crossfit_residual_median": np.zeros(
                    row_count, dtype=np.float32
                ),
                f"{side}_camera_crossfit_residual_reduction": np.zeros(
                    row_count, dtype=np.float32
                ),
                f"{side}_background_residual_reduction": np.zeros(
                    row_count, dtype=np.float32
                ),
            }
        )
    return arrays


def _validate_array_contract(
    arrays: Mapping[str, np.ndarray],
    *,
    rows: int,
    track_count: int,
) -> None:
    expected = _empty_arrays(rows, track_count=track_count)
    if set(arrays) != set(expected):
        missing = sorted(set(expected) - set(arrays))
        extra = sorted(set(arrays) - set(expected))
        raise ValueError(
            f"track-cache arrays differ; missing={missing}, extra={extra}"
        )
    for name, template in expected.items():
        value = np.asarray(arrays[name])
        if value.shape != template.shape or value.dtype != template.dtype:
            raise ValueError(
                f"{name} differs: got {value.shape}/{value.dtype}, "
                f"expected {template.shape}/{template.dtype}"
            )
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
    indices = np.asarray(arrays["input_indices"], dtype=np.int64)
    if len(np.unique(indices)) != len(indices):
        raise ValueError("input_indices contain duplicates")
    for side in SIDES:
        tracked = np.asarray(arrays[f"{side}_track_valid"], dtype=bool)
        camera = np.asarray(arrays[f"{side}_camera_valid"], dtype=bool)
        crossfit = np.asarray(
            arrays[f"{side}_camera_crossfit_valid"], dtype=bool
        )
        if bool((camera & ~tracked).any()):
            raise ValueError(f"{side} camera_valid does not imply track_valid")
        if bool((crossfit & ~camera).any()):
            raise ValueError(
                f"{side} camera_crossfit_valid does not imply camera_valid"
            )
        visibility = np.asarray(arrays[f"{side}_visibility"])
        if bool(((visibility < 0.0) | (visibility > 1.0)).any()):
            raise ValueError(f"{side} visibility is outside [0,1]")
        if tracked.any():
            times = np.asarray(arrays[f"{side}_frame_times"])[tracked]
            if bool((np.diff(times, axis=1) <= 0.0).any()):
                raise ValueError(f"{side} valid frame times are not increasing")
            frame_indices = np.asarray(
                arrays[f"{side}_source_frame_indices"]
            )[tracked]
            if bool((np.diff(frame_indices, axis=1) <= 0).any()):
                raise ValueError(
                    f"{side} source frame indices are not increasing"
                )
            sizes = np.asarray(arrays[f"{side}_resized_size"])[tracked]
            if bool((sizes <= 1).any()):
                raise ValueError(f"{side} valid resized sizes are invalid")
            fps = np.asarray(arrays[f"{side}_source_fps"])[tracked]
            if bool((fps <= 0.0).any()):
                raise ValueError(f"{side} valid source FPS is not positive")
            expected_times = frame_indices.astype(np.float64) / fps[:, None]
            if not np.allclose(
                times,
                expected_times,
                rtol=1e-12,
                atol=1e-12,
            ):
                raise ValueError(
                    f"{side} frame times differ from source indices / FPS"
                )
        invalid = ~tracked
        for suffix in (
            "normalized_tracks",
            "visibility",
            "frame_times",
            "source_frame_indices",
            "resized_size",
            "source_fps",
            "stabilized_tracks",
            "transition_affines",
            "cumulative_affines",
        ):
            value = np.asarray(arrays[f"{side}_{suffix}"])[invalid]
            if value.size and bool((value != 0).any()):
                raise ValueError(
                    f"{side}_{suffix} is nonzero for invalid tracks"
                )
        camera_invalid = tracked & ~camera
        for suffix in (
            "stabilized_tracks",
            "transition_affines",
            "cumulative_affines",
        ):
            value = np.asarray(arrays[f"{side}_{suffix}"])[camera_invalid]
            if value.size and bool((value != 0).any()):
                raise ValueError(
                    f"{side}_{suffix} is nonzero when camera is invalid"
                )
        camera_metrics = (
            "camera_crossfit_raw_median",
            "camera_crossfit_residual_median",
            "camera_crossfit_residual_reduction",
            "background_residual_reduction",
        )
        not_camera = ~camera
        for suffix in camera_metrics:
            value = np.asarray(arrays[f"{side}_{suffix}"])[not_camera]
            if value.size and bool((value != 0).any()):
                raise ValueError(
                    f"{side}_{suffix} is nonzero when camera is invalid"
                )
        no_crossfit = ~crossfit
        for suffix in (
            "camera_crossfit_raw_median",
            "camera_crossfit_residual_median",
            "camera_crossfit_residual_reduction",
        ):
            value = np.asarray(arrays[f"{side}_{suffix}"])[no_crossfit]
            if value.size and bool((value != 0).any()):
                raise ValueError(
                    f"{side}_{suffix} is nonzero when crossfit is invalid"
                )
        raw = np.asarray(
            arrays[f"{side}_camera_crossfit_raw_median"],
            dtype=np.float64,
        )
        residual = np.asarray(
            arrays[f"{side}_camera_crossfit_residual_median"],
            dtype=np.float64,
        )
        reduction = np.asarray(
            arrays[f"{side}_camera_crossfit_residual_reduction"],
            dtype=np.float64,
        )
        background_reduction = np.asarray(
            arrays[f"{side}_background_residual_reduction"],
            dtype=np.float64,
        )
        if bool(((raw[crossfit] < 0.0) | (residual[crossfit] < 0.0)).any()):
            raise ValueError(f"{side} crossfit distances must be nonnegative")
        if bool(
            (
                (reduction[crossfit] < -1.0)
                | (reduction[crossfit] > 1.0)
            ).any()
        ) or bool(
            (
                (background_reduction[camera] < -1.0)
                | (background_reduction[camera] > 1.0)
            ).any()
        ):
            raise ValueError(f"{side} camera reduction is outside [-1,1]")
        if crossfit.any():
            expected_reduction = np.where(
                raw[crossfit] <= 1e-8,
                0.0,
                np.clip(
                    1.0 - residual[crossfit] / raw[crossfit],
                    -1.0,
                    1.0,
                ),
            )
            if not np.allclose(
                reduction[crossfit],
                expected_reduction,
                rtol=1e-5,
                atol=1e-6,
            ):
                raise ValueError(
                    f"{side} crossfit reduction differs from raw/residual"
                )
        identity = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        normalized = np.asarray(
            arrays[f"{side}_normalized_tracks"],
            dtype=np.float64,
        )
        stabilized = np.asarray(
            arrays[f"{side}_stabilized_tracks"],
            dtype=np.float64,
        )
        transitions = np.asarray(
            arrays[f"{side}_transition_affines"],
            dtype=np.float64,
        )
        cumulative = np.asarray(
            arrays[f"{side}_cumulative_affines"],
            dtype=np.float64,
        )
        for row_index in np.flatnonzero(camera):
            if not np.allclose(
                cumulative[row_index, 0],
                identity,
                rtol=1e-5,
                atol=1e-6,
            ):
                raise ValueError(
                    f"{side} cumulative affine does not start at identity"
                )
            for frame_index in range(VIDEO_FRAMES):
                cumulative_h = np.vstack(
                    (
                        cumulative[row_index, frame_index],
                        np.asarray([0.0, 0.0, 1.0]),
                    )
                )
                determinant = float(np.linalg.det(cumulative_h[:2, :2]))
                if abs(determinant) <= 1e-8:
                    raise ValueError(
                        f"{side} cumulative affine is singular"
                    )
                points_h = np.concatenate(
                    (
                        normalized[row_index, frame_index],
                        np.ones((track_count, 1), dtype=np.float64),
                    ),
                    axis=1,
                )
                reconstructed = (
                    points_h @ np.linalg.inv(cumulative_h).T
                )[:, :2]
                if not np.allclose(
                    stabilized[row_index, frame_index],
                    reconstructed,
                    rtol=2e-4,
                    atol=2e-5,
                ):
                    raise ValueError(
                        f"{side} stabilized/normalized coordinate relation differs"
                    )
                if frame_index == 0:
                    continue
                transition_h = np.vstack(
                    (
                        transitions[row_index, frame_index - 1],
                        np.asarray([0.0, 0.0, 1.0]),
                    )
                )
                if abs(float(np.linalg.det(transition_h[:2, :2]))) <= 1e-8:
                    raise ValueError(
                        f"{side} transition affine is singular"
                    )
                previous_h = np.vstack(
                    (
                        cumulative[row_index, frame_index - 1],
                        np.asarray([0.0, 0.0, 1.0]),
                    )
                )
                if not np.allclose(
                    cumulative_h,
                    transition_h @ previous_h,
                    rtol=2e-4,
                    atol=2e-5,
                ):
                    raise ValueError(
                        f"{side} transition/cumulative affine relation differs"
                    )


def _base_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(contract)
    value.pop("rank", None)
    value.pop("device", None)
    return value


def build_cache_contract(
    *,
    input_manifest: Path,
    data_root: Path,
    tracker_checkpoint: Path,
    cotracker_provenance: Mapping[str, Any],
    runtime: Mapping[str, Any],
    tracker_grid_size: int,
    rank: int,
    world_size: int,
    device: str,
    seed: int,
) -> dict[str, Any]:
    if tracker_grid_size < 2:
        raise ValueError("tracker_grid_size must be >= 2")
    if world_size != FINAL_WORLD_SIZE or not 0 <= rank < world_size:
        raise ValueError("track cache requires exactly eight ranks")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= 0xFFFFFFFF
    ):
        raise ValueError("seed must be an integer in [0, 2**32-1]")
    checkpoint = tracker_checkpoint.expanduser().resolve(strict=True)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    module = Path(__file__).resolve()
    dependencies = (
        module,
        module.with_name("r7_preflight_extract.py"),
        module.with_name("r7_temporal_teacher.py"),
    )
    return {
        "schema_version": R7_TRACK_CACHE_SCHEMA,
        "input_manifest": str(input_manifest.resolve(strict=True)),
        "input_manifest_sha256": _file_digest(input_manifest),
        "data_root": str(data_root.resolve(strict=True)),
        "rank": rank,
        "world_size": world_size,
        "partition": R7_TRACK_CACHE_PARTITION,
        "device": str(device),
        "seed": int(seed),
        "frames": VIDEO_FRAMES,
        "cohort_role": COHORT_ROLE,
        "cohort_id": COHORT_ID,
        "expected_cohort_rows": EXPECTED_COHORT_ROWS,
        "tracker": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _file_digest(checkpoint),
            "grid_size": int(tracker_grid_size),
            "track_count": int(tracker_grid_size) ** 2,
            "query_frame": 0,
            "backward_tracking": False,
            "source": dict(cotracker_provenance),
        },
        "runtime": dict(runtime),
        "camera_config": {
            key: value
            for key, value in vars(TemporalTeacherConfig()).items()
            if key.startswith("camera_")
            or key.startswith("minimum_camera_")
            or key in {
                "minimum_frames",
                "minimum_tracks",
                "visibility_threshold",
                "eps",
            }
        },
        "operational_coverage": {
            "policy": OPERATIONAL_COVERAGE_POLICY,
            "minimum_source_track_valid_fraction":
                MIN_SOURCE_TRACK_COVERAGE,
            "minimum_target_track_valid_fraction":
                MIN_TARGET_TRACK_COVERAGE,
            "minimum_paired_track_valid_fraction":
                MIN_PAIRED_TRACK_COVERAGE,
        },
        "cache_scope": CACHE_SCOPE,
        "formal_status": FORMAL_STATUS,
        "formal_reason": FORMAL_REASON,
        "production_decision": False,
        "generation_authorized": False,
        "implementation_sha256": {
            path.name: _file_digest(path) for path in dependencies
        },
    }


def _require_fail_closed(value: Mapping[str, Any], *, label: str) -> None:
    if (
        value.get("cohort_role") != COHORT_ROLE
        or value.get("cohort_id") != COHORT_ID
        or value.get("formal_status") != FORMAL_STATUS
        or value.get("formal_reason") != FORMAL_REASON
        or value.get("production_decision") is not False
        or value.get("generation_authorized") is not False
    ):
        raise ValueError(f"{label} fail-closed development semantics differ")


def _validate_contract_semantics(
    contract: Mapping[str, Any],
    *,
    final: bool,
) -> None:
    if contract.get("schema_version") != R7_TRACK_CACHE_SCHEMA:
        raise ValueError("track-cache contract schema differs")
    if contract.get("partition") != R7_TRACK_CACHE_PARTITION:
        raise ValueError("track-cache partition differs")
    if contract.get("world_size") != FINAL_WORLD_SIZE:
        raise ValueError("track-cache contract world size differs")
    if contract.get("frames") != VIDEO_FRAMES:
        raise ValueError("track-cache frame count differs")
    if contract.get("expected_cohort_rows") != EXPECTED_COHORT_ROWS:
        raise ValueError("track-cache cohort row count differs")
    if contract.get("cache_scope") != CACHE_SCOPE:
        raise ValueError("track-cache scope differs")
    _require_fail_closed(contract, label="contract")
    rank = contract.get("rank")
    device = contract.get("device")
    if final:
        if (
            rank != "merged"
            or device != "eight-shard-final"
            or contract.get("merge_world_size") != FINAL_WORLD_SIZE
            or contract.get("merge_policy")
            != "strict-input-index-order-v2"
        ):
            raise ValueError("track-cache final merge contract differs")
        source_hashes = contract.get("source_shard_done_sha256")
        if (
            not isinstance(source_hashes, list)
            or len(source_hashes) != FINAL_WORLD_SIZE
            or any(
                not isinstance(value, str)
                or not _SHA256_RE.fullmatch(value)
                for value in source_hashes
            )
        ):
            raise ValueError("track-cache source-shard hash registry differs")
    elif (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or not 0 <= rank < FINAL_WORLD_SIZE
        or device != f"cuda:{rank}"
    ):
        raise ValueError("track-cache shard rank/device contract differs")
    seed = contract.get("seed")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= 0xFFFFFFFF
    ):
        raise ValueError("track-cache seed differs")
    tracker = contract.get("tracker")
    if not isinstance(tracker, Mapping):
        raise ValueError("track-cache tracker contract is missing")
    grid_size = tracker.get("grid_size")
    if (
        isinstance(grid_size, bool)
        or not isinstance(grid_size, int)
        or grid_size < 2
        or tracker.get("track_count") != grid_size**2
        or tracker.get("query_frame") != 0
        or tracker.get("backward_tracking") is not False
    ):
        raise ValueError("track-cache tracker geometry contract differs")
    checkpoint = tracker.get("checkpoint")
    checkpoint_sha = tracker.get("checkpoint_sha256")
    if (
        not isinstance(checkpoint, str)
        or not Path(checkpoint).is_absolute()
        or not isinstance(checkpoint_sha, str)
        or not _SHA256_RE.fullmatch(checkpoint_sha)
    ):
        raise ValueError("track-cache checkpoint provenance differs")
    source = tracker.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("track-cache CoTracker source provenance is missing")
    inventory = source.get("python_source_files")
    inventory_paths = (
        [entry.get("path") for entry in inventory]
        if isinstance(inventory, list)
        and all(
            isinstance(entry, Mapping)
            and isinstance(entry.get("path"), str)
            for entry in inventory
        )
        else []
    )
    if (
        source.get("git_tracked_clean") is not True
        or not isinstance(source.get("git_head"), str)
        or not _GIT_COMMIT_RE.fullmatch(source["git_head"])
        or not isinstance(source.get("root"), str)
        or not Path(source["root"]).is_absolute()
        or not isinstance(source.get("git_toplevel"), str)
        or not Path(source["git_toplevel"]).is_absolute()
        or not isinstance(inventory, list)
        or not inventory
        or inventory_paths != sorted(set(inventory_paths))
        or "cotracker/predictor.py" not in inventory_paths
        or source.get("python_source_file_count") != len(inventory)
        or source.get("python_source_bundle_sha256")
        != _object_digest(inventory)
    ):
        raise ValueError("track-cache CoTracker source provenance differs")
    for entry in inventory:
        if (
            not isinstance(entry, Mapping)
            or not isinstance(entry.get("path"), str)
            or Path(entry["path"]).is_absolute()
            or ".." in Path(entry["path"]).parts
            or not isinstance(entry.get("size"), int)
            or entry["size"] < 0
            or not isinstance(entry.get("sha256"), str)
            or not _SHA256_RE.fullmatch(entry["sha256"])
        ):
            raise ValueError("track-cache CoTracker source inventory differs")
    runtime = contract.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("track-cache runtime provenance is missing")
    required_runtime_strings = (
        "python_version",
        "python_implementation",
        "python_executable",
        "platform",
        "numpy_version",
        "opencv_version",
        "torch_version",
        "device_type",
        "device_name",
    )
    if (
        runtime.get("schema_version") != "motive-r7-runtime-v1"
        or runtime.get("visible_device_count") != FINAL_WORLD_SIZE
        or runtime.get("device_type") != "cuda-hip"
        or not isinstance(runtime.get("torch_hip_version"), str)
        or not runtime["torch_hip_version"]
        or any(
            not isinstance(runtime.get(key), str) or not runtime[key]
            for key in required_runtime_strings
        )
        or not isinstance(runtime.get("device_total_memory"), int)
        or runtime["device_total_memory"] <= 0
        or not isinstance(runtime.get("device_capability"), list)
        or len(runtime["device_capability"]) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in runtime["device_capability"]
        )
        or not isinstance(runtime.get("torch_build_config_sha256"), str)
        or not _SHA256_RE.fullmatch(runtime["torch_build_config_sha256"])
    ):
        raise ValueError("track-cache runtime provenance differs")
    determinism = runtime.get("determinism")
    if (
        not isinstance(determinism, Mapping)
        or determinism.get("schema_version") != "motive-r7-determinism-v1"
        or determinism.get("seed") != seed
        or determinism.get("rank_seed_policy")
        != "identical-base-seed-on-all-eight-ranks-v1"
        or determinism.get("python_random_seeded") is not True
        or determinism.get("numpy_seeded") is not True
        or determinism.get("torch_cpu_seeded") is not True
        or determinism.get("torch_all_visible_devices_seeded") is not True
        or determinism.get("torch_deterministic_algorithms") is not True
        or determinism.get("cudnn_benchmark") is not False
        or determinism.get("cudnn_deterministic") is not True
        or determinism.get("python_hash_seed") != str(seed)
    ):
        raise ValueError("track-cache determinism provenance differs")
    expected_coverage = {
        "policy": OPERATIONAL_COVERAGE_POLICY,
        "minimum_source_track_valid_fraction": MIN_SOURCE_TRACK_COVERAGE,
        "minimum_target_track_valid_fraction": MIN_TARGET_TRACK_COVERAGE,
        "minimum_paired_track_valid_fraction": MIN_PAIRED_TRACK_COVERAGE,
    }
    if contract.get("operational_coverage") != expected_coverage:
        raise ValueError("track-cache operational coverage contract differs")
    implementation = contract.get("implementation_sha256")
    if (
        not isinstance(implementation, Mapping)
        or set(implementation)
        != {
            "r7_track_cache.py",
            "r7_preflight_extract.py",
            "r7_temporal_teacher.py",
        }
        or any(
            not isinstance(value, str) or not _SHA256_RE.fullmatch(value)
            for value in implementation.values()
        )
    ):
        raise ValueError("track-cache implementation provenance differs")


def _operational_coverage(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    rows = len(np.asarray(arrays["input_indices"]))
    if rows <= 0:
        raise ValueError("operational coverage requires nonempty arrays")
    source = np.asarray(arrays["source_track_valid"], dtype=bool)
    target = np.asarray(arrays["target_track_valid"], dtype=bool)
    paired = source & target
    return {
        "policy": OPERATIONAL_COVERAGE_POLICY,
        "rows": rows,
        "source_track_valid": int(source.sum()),
        "target_track_valid": int(target.sum()),
        "paired_track_valid": int(paired.sum()),
        "source_track_valid_fraction": float(source.mean()),
        "target_track_valid_fraction": float(target.mean()),
        "paired_track_valid_fraction": float(paired.mean()),
        "criteria": {
            "minimum_source_track_valid_fraction":
                MIN_SOURCE_TRACK_COVERAGE,
            "minimum_target_track_valid_fraction":
                MIN_TARGET_TRACK_COVERAGE,
            "minimum_paired_track_valid_fraction":
                MIN_PAIRED_TRACK_COVERAGE,
        },
        "passed": bool(
            float(source.mean()) >= MIN_SOURCE_TRACK_COVERAGE
            and float(target.mean()) >= MIN_TARGET_TRACK_COVERAGE
            and float(paired.mean()) >= MIN_PAIRED_TRACK_COVERAGE
        ),
    }


def _decode_record(decoded: Any) -> dict[str, Any]:
    return {
        "sampling_version": R7_VIDEO_SAMPLING,
        "decoded_frames": VIDEO_FRAMES,
        "source_frame_indices": decoded.source_frame_indices.tolist(),
        "source_fps": decoded.source_fps,
        "source_frame_count": decoded.source_frame_count,
        "source_size": list(decoded.source_size),
        "resized_size": list(decoded.resized_size),
    }


def _extract_side(
    *,
    path: Path,
    side: str,
    array_index: int,
    arrays: dict[str, np.ndarray],
    tracker: LazyCoTrackerAdapter,
    track_count: int,
    camera_config: TemporalTeacherConfig,
) -> dict[str, Any]:
    video_digest = _file_digest(path) if path.is_file() else None
    try:
        decoded = decode_video_fixed_frames(path)
    except PerVideoError as error:
        return {
            "status": "failed",
            "track_valid": False,
            "camera_valid": False,
            "failure_stage": "decode",
            "failure_reason": error.reason,
            "failure_message": str(error),
            "resolved_path": str(path),
            "video_sha256": video_digest,
        }
    try:
        observations = tracker.track(
            decoded.frames_rgb,
            frame_times=decoded.frame_times,
        )
    except TemporalTeacherError as error:
        if error.reason in TRACKER_ABI_FAILURE_REASONS:
            raise GlobalExtractionError(
                f"CoTracker ABI contract failed on {path}: {error.reason}"
            ) from error
        return {
            "status": "failed",
            "track_valid": False,
            "camera_valid": False,
            "failure_stage": "tracking",
            "failure_reason": error.reason,
            "failure_message": str(error),
            "resolved_path": str(path),
            "video_sha256": video_digest,
            "decode": _decode_record(decoded),
        }
    except (FileNotFoundError, ImportError, RuntimeError) as error:
        raise GlobalExtractionError(
            f"CoTracker runtime failed on {path}"
        ) from error
    if observations.tracks.shape != (VIDEO_FRAMES, track_count, 2):
        raise GlobalExtractionError(
            "CoTracker grid contract changed: "
            f"got {observations.tracks.shape}, expected "
            f"{(VIDEO_FRAMES, track_count, 2)}"
        )
    if observations.frame_size != decoded.resized_size:
        raise GlobalExtractionError(
            "CoTracker frame size differs from the decoded frame size"
        )
    if not np.array_equal(observations.frame_times, decoded.frame_times):
        raise GlobalExtractionError(
            "CoTracker frame times differ from decoded-frame times"
        )
    arrays[f"{side}_track_valid"][array_index] = True
    height, width = observations.frame_size
    # Match r7_temporal_teacher._normalize_tracks exactly so raw normalized
    # tracks and camera.stabilized_tracks share one coordinate system.
    scale = np.asarray([width, height], dtype=np.float32)
    normalized = observations.tracks.astype(np.float32) / scale[None, None]
    if not np.isfinite(normalized).all():
        raise GlobalExtractionError("normalized CoTracker tracks are non-finite")
    arrays[f"{side}_normalized_tracks"][array_index] = normalized
    arrays[f"{side}_visibility"][array_index] = observations.visibility
    arrays[f"{side}_frame_times"][array_index] = observations.frame_times
    arrays[f"{side}_source_frame_indices"][array_index] = (
        decoded.source_frame_indices
    )
    arrays[f"{side}_resized_size"][array_index] = [height, width]
    arrays[f"{side}_source_fps"][array_index] = decoded.source_fps
    try:
        camera = robust_camera_compensation(
            observations.tracks,
            observations.visibility,
            observations.frame_size,
            config=camera_config,
        )
    except TemporalTeacherError as error:
        return {
            "status": "track_only",
            "track_valid": True,
            "camera_valid": False,
            "failure_stage": "camera_compensation",
            "failure_reason": error.reason,
            "failure_message": str(error),
            "resolved_path": str(path),
            "video_sha256": video_digest,
            "decode": _decode_record(decoded),
            "tracker": {
                "backend": observations.backend,
                "provenance": dict(observations.provenance),
                "tracks": track_count,
            },
        }
    arrays[f"{side}_camera_valid"][array_index] = True
    if not np.array_equal(
        normalized,
        camera.normalized_tracks.astype(np.float32, copy=False),
    ):
        raise GlobalExtractionError(
            "cached normalized tracks differ from camera coordinates"
        )
    arrays[f"{side}_stabilized_tracks"][array_index] = (
        camera.stabilized_tracks
    )
    arrays[f"{side}_transition_affines"][array_index] = (
        camera.transition_affines
    )
    arrays[f"{side}_cumulative_affines"][array_index] = (
        camera.cumulative_affines
    )
    arrays[f"{side}_camera_crossfit_valid"][array_index] = (
        camera.crossfit_valid
    )
    arrays[f"{side}_camera_crossfit_raw_median"][array_index] = (
        camera.crossfit_raw_median
    )
    arrays[f"{side}_camera_crossfit_residual_median"][array_index] = (
        camera.crossfit_residual_median
    )
    arrays[f"{side}_camera_crossfit_residual_reduction"][array_index] = (
        camera.crossfit_residual_reduction
    )
    arrays[f"{side}_background_residual_reduction"][array_index] = (
        camera.background_residual_reduction
    )
    return {
        "status": "camera_ready",
        "track_valid": True,
        "camera_valid": True,
        "failure_stage": None,
        "failure_reason": None,
        "failure_message": None,
        "resolved_path": str(path),
        "video_sha256": video_digest,
        "decode": _decode_record(decoded),
        "tracker": {
            "backend": observations.backend,
            "provenance": dict(observations.provenance),
            "tracks": track_count,
        },
        "camera_crossfit": {
            "valid": bool(
                arrays[f"{side}_camera_crossfit_valid"][array_index]
            ),
            "raw_median": float(
                arrays[f"{side}_camera_crossfit_raw_median"][array_index]
            ),
            "residual_median": float(
                arrays[
                    f"{side}_camera_crossfit_residual_median"
                ][array_index]
            ),
            "residual_reduction": float(
                arrays[
                    f"{side}_camera_crossfit_residual_reduction"
                ][array_index]
            ),
        },
    }


def _commit(
    *,
    directory: Path,
    rows: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    contract: Mapping[str, Any],
    input_rows: int,
    final: bool,
) -> dict[str, Any]:
    _validate_contract_semantics(contract, final=final)
    if input_rows != EXPECTED_COHORT_ROWS:
        raise ValueError(
            f"track cache is fixed to {EXPECTED_COHORT_ROWS} development rows"
        )
    paths = _artifact_paths(directory)
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite track-cache artifacts: "
            + ", ".join(existing)
        )
    track_count = int(contract["tracker"]["track_count"])
    _validate_array_contract(
        arrays, rows=len(rows), track_count=track_count
    )
    canonical_rows = [dict(row) for row in rows]
    indices = [int(row["input_index"]) for row in canonical_rows]
    if indices != np.asarray(arrays["input_indices"]).tolist():
        raise ValueError("manifest/archive input index order differs")
    input_manifest = Path(str(contract["input_manifest"]))
    source_rows = _read_r5_manifest(input_manifest)
    if len(source_rows) != EXPECTED_COHORT_ROWS:
        raise ValueError("commit input manifest cohort size differs")
    expected_indices = (
        list(range(EXPECTED_COHORT_ROWS))
        if final
        else list(
            range(
                int(contract["rank"]),
                EXPECTED_COHORT_ROWS,
                FINAL_WORLD_SIZE,
            )
        )
    )
    if indices != expected_indices:
        raise ValueError("commit shard/final modulo coverage differs")
    for array_index, row in enumerate(canonical_rows):
        _validate_row_binding(
            row,
            input_row=source_rows[int(row["input_index"])],
            arrays=arrays,
            array_index=array_index,
            contract=contract,
            final=final,
            rehash_videos=False,
        )
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_npz(paths["archive"], arrays)
    _atomic_jsonl(paths["manifest"], canonical_rows)
    array_sha256 = {
        name: _array_digest(np.asarray(value))
        for name, value in sorted(arrays.items())
    }
    summary = {
        "schema_version": (
            R7_TRACK_CACHE_FINAL_SUMMARY_SCHEMA
            if final
            else R7_TRACK_CACHE_SHARD_SUMMARY_SCHEMA
        ),
        "rows": len(rows),
        "input_rows": input_rows,
        "track_count": track_count,
        "contract": dict(contract),
        "contract_sha256": _object_digest(contract),
        "input_indices_sha256": _object_digest(indices),
        "array_sha256": array_sha256,
        "counts": {
            side: {
                "track_valid": int(
                    np.sum(arrays[f"{side}_track_valid"])
                ),
                "camera_valid": int(
                    np.sum(arrays[f"{side}_camera_valid"])
                ),
                "camera_crossfit_valid": int(
                    np.sum(arrays[f"{side}_camera_crossfit_valid"])
                ),
            }
            for side in SIDES
        },
        "operational_coverage": _operational_coverage(arrays),
        "cohort_role": COHORT_ROLE,
        "cohort_id": COHORT_ID,
        "formal_status": FORMAL_STATUS,
        "formal_reason": FORMAL_REASON,
        "production_decision": False,
        "generation_authorized": False,
    }
    _atomic_json(paths["summary"], summary)
    done = {
        "schema_version": (
            R7_TRACK_CACHE_FINAL_DONE_SCHEMA
            if final
            else R7_TRACK_CACHE_SHARD_DONE_SCHEMA
        ),
        "committed": True,
        "rows": len(rows),
        "archive_sha256": _file_digest(paths["archive"]),
        "manifest_sha256": _file_digest(paths["manifest"]),
        "summary_sha256": _file_digest(paths["summary"]),
        "contract_sha256": summary["contract_sha256"],
        "cohort_role": COHORT_ROLE,
        "cohort_id": COHORT_ID,
        "formal_status": FORMAL_STATUS,
        "formal_reason": FORMAL_REASON,
        "production_decision": False,
        "generation_authorized": False,
    }
    _atomic_json(paths["done"], done)
    return done


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number} is blank")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            if line != _canonical_json(value) + "\n":
                raise ValueError(
                    f"{path}:{line_number} is not canonical JSONL"
                )
            rows.append(value)
    return rows


def _validate_decode_record(
    decode: Mapping[str, Any],
    *,
    arrays: Mapping[str, np.ndarray],
    side: str,
    array_index: int,
    bind_arrays: bool,
) -> None:
    expected_keys = {
        "sampling_version",
        "decoded_frames",
        "source_frame_indices",
        "source_fps",
        "source_frame_count",
        "source_size",
        "resized_size",
    }
    if set(decode) != expected_keys:
        raise ValueError(f"track-cache {side} decode fields differ")
    indices = decode.get("source_frame_indices")
    fps = decode.get("source_fps")
    source_count = decode.get("source_frame_count")
    source_size = decode.get("source_size")
    resized_size = decode.get("resized_size")
    if (
        decode.get("sampling_version") != R7_VIDEO_SAMPLING
        or decode.get("decoded_frames") != VIDEO_FRAMES
        or not isinstance(indices, list)
        or len(indices) != VIDEO_FRAMES
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in indices
        )
        or any(right <= left for left, right in zip(indices, indices[1:]))
        or isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not np.isfinite(fps)
        or fps <= 0.0
        or isinstance(source_count, bool)
        or not isinstance(source_count, int)
        or source_count <= indices[-1]
        or not isinstance(source_size, list)
        or len(source_size) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 1
            for value in source_size
        )
        or not isinstance(resized_size, list)
        or len(resized_size) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 1
            for value in resized_size
        )
    ):
        raise ValueError(f"track-cache {side} decode metadata differs")
    if not bind_arrays:
        return
    if indices != np.asarray(
        arrays[f"{side}_source_frame_indices"][array_index]
    ).tolist():
        raise ValueError(f"track-cache {side} decode/frame indices differ")
    if float(fps) != float(arrays[f"{side}_source_fps"][array_index]):
        raise ValueError(f"track-cache {side} decode/FPS differs")
    if resized_size != np.asarray(
        arrays[f"{side}_resized_size"][array_index]
    ).tolist():
        raise ValueError(f"track-cache {side} decode/resized size differs")


def _validate_row_binding(
    row: Mapping[str, Any],
    *,
    input_row: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    array_index: int,
    contract: Mapping[str, Any],
    final: bool,
    rehash_videos: bool,
) -> None:
    expected_keys = {
        "schema_version",
        "input_index",
        "shard_array_index",
        "shard_rank",
        "world_size",
        "iid",
        "input_row",
        "input_row_sha256",
        "input_digest",
        "prompt",
        "label_type",
        "negative_type",
        "positive",
        "action_signature",
        "source",
        "target",
        "paired_track_valid",
        "paired_camera_valid",
    }
    if final:
        expected_keys.add("final_array_index")
    if set(row) != expected_keys:
        raise ValueError(
            f"track-cache row {array_index} fields differ"
        )
    input_index = int(row["input_index"])
    shard_rank = input_index % FINAL_WORLD_SIZE
    if (
        row.get("schema_version") != R7_TRACK_CACHE_ROW_SCHEMA
        or row.get("input_index") != input_index
        or row.get("shard_rank") != shard_rank
        or row.get("world_size") != FINAL_WORLD_SIZE
        or row.get("shard_array_index")
        != input_index // FINAL_WORLD_SIZE
        or (final and row.get("final_array_index") != input_index)
        or row.get("input_row") != dict(input_row)
        or row.get("input_row_sha256") != _object_digest(input_row)
    ):
        raise ValueError(
            f"track-cache row {array_index} input/shard binding differs"
        )
    pilot = input_row["r5_pilot_label"]
    positive = pilot["class"] == "positive"
    expected_flattened = {
        "iid": input_row["iid"],
        "input_digest": input_row.get("input_digest"),
        "prompt": input_row.get("prompt"),
        "label_type": pilot["class"],
        "negative_type": pilot.get("negative_type"),
        "positive": positive,
        "action_signature": pilot.get("action_signature"),
    }
    for key, value in expected_flattened.items():
        if row.get(key) != value:
            raise ValueError(
                f"track-cache row {array_index} {key} differs from input"
            )
    if bool(arrays["positive"][array_index]) is not positive:
        raise ValueError(
            f"track-cache row {array_index} positive array differs from input"
        )
    side_flags: dict[str, tuple[bool, bool]] = {}
    data_root = Path(str(contract["data_root"]))
    tracker_contract = contract["tracker"]
    for side, input_field in (
        ("source", "src_video"),
        ("target", "tgt_video"),
    ):
        record = row.get(side)
        if not isinstance(record, Mapping):
            raise ValueError(f"track-cache row {array_index} lacks {side}")
        track_valid = bool(arrays[f"{side}_track_valid"][array_index])
        camera_valid = bool(arrays[f"{side}_camera_valid"][array_index])
        crossfit_valid = bool(
            arrays[f"{side}_camera_crossfit_valid"][array_index]
        )
        side_flags[side] = (track_valid, camera_valid)
        status = record.get("status")
        if status not in {"failed", "track_only", "camera_ready"}:
            raise ValueError(
                f"track-cache row {array_index} {side} status differs"
            )
        expected_status = (
            "camera_ready"
            if camera_valid
            else "track_only"
            if track_valid
            else "failed"
        )
        if (
            status != expected_status
            or record.get("track_valid") is not track_valid
            or record.get("camera_valid") is not camera_valid
        ):
            raise ValueError(
                f"track-cache row {array_index} {side} state differs"
            )
        common_keys = {
            "status",
            "track_valid",
            "camera_valid",
            "failure_stage",
            "failure_reason",
            "failure_message",
            "resolved_path",
            "video_sha256",
        }
        expected_side_keys = set(common_keys)
        if record.get("failure_stage") != "decode":
            expected_side_keys.add("decode")
        if track_valid:
            expected_side_keys.add("tracker")
        if camera_valid:
            expected_side_keys.add("camera_crossfit")
        if set(record) != expected_side_keys:
            raise ValueError(
                f"track-cache row {array_index} {side} fields differ"
            )
        expected_path = _safe_video_path(
            data_root, str(input_row[input_field])
        )
        if record.get("resolved_path") != str(expected_path):
            raise ValueError(
                f"track-cache row {array_index} {side} path differs"
            )
        video_digest = record.get("video_sha256")
        missing_video = (
            record.get("failure_stage") == "decode"
            and record.get("failure_reason") == "video_missing"
            and not expected_path.exists()
        )
        if video_digest is None and not missing_video:
            raise ValueError(
                f"track-cache row {array_index} {side} lacks video digest"
            )
        if video_digest is not None and (
            not isinstance(video_digest, str)
            or not _SHA256_RE.fullmatch(video_digest)
        ):
            raise ValueError(
                f"track-cache row {array_index} {side} video digest differs"
            )
        if rehash_videos and video_digest is not None:
            if (
                not expected_path.is_file()
                or _file_digest(expected_path) != video_digest
            ):
                raise ValueError(
                    f"track-cache video bytes changed: {expected_path}"
                )
        if status == "failed":
            if (
                record.get("failure_stage") not in {"decode", "tracking"}
                or not isinstance(record.get("failure_reason"), str)
                or not record["failure_reason"]
                or not isinstance(record.get("failure_message"), str)
                or not record["failure_message"]
            ):
                raise ValueError(
                    f"track-cache row {array_index} {side} failure differs"
                )
        elif status == "track_only":
            if (
                record.get("failure_stage") != "camera_compensation"
                or not isinstance(record.get("failure_reason"), str)
                or not record["failure_reason"]
                or not isinstance(record.get("failure_message"), str)
                or not record["failure_message"]
            ):
                raise ValueError(
                    f"track-cache row {array_index} {side} camera failure differs"
                )
        elif any(
            record.get(key) is not None
            for key in ("failure_stage", "failure_reason", "failure_message")
        ):
            raise ValueError(
                f"track-cache row {array_index} {side} success has failure data"
            )
        decode = record.get("decode")
        if record.get("failure_stage") == "decode":
            if decode is not None:
                raise ValueError(
                    f"track-cache row {array_index} {side} decode failure differs"
                )
        else:
            if not isinstance(decode, Mapping):
                raise ValueError(
                    f"track-cache row {array_index} {side} decode is missing"
                )
            _validate_decode_record(
                decode,
                arrays=arrays,
                side=side,
                array_index=array_index,
                bind_arrays=track_valid,
            )
        if track_valid:
            tracker = record.get("tracker")
            expected_device = f"cuda:{shard_rank}"
            if (
                not isinstance(tracker, Mapping)
                or set(tracker) != {"backend", "provenance", "tracks"}
                or tracker.get("backend") != "cotracker"
                or tracker.get("tracks") != tracker_contract["track_count"]
                or tracker.get("provenance")
                != {
                    "checkpoint": tracker_contract["checkpoint"],
                    "grid_size": tracker_contract["grid_size"],
                    "query_frame": 0,
                    "backward_tracking": False,
                    "device": expected_device,
                }
            ):
                raise ValueError(
                    f"track-cache row {array_index} {side} tracker provenance differs"
                )
        if camera_valid:
            camera_crossfit = record.get("camera_crossfit")
            expected_camera = {
                "valid": crossfit_valid,
                "raw_median": float(
                    arrays[f"{side}_camera_crossfit_raw_median"][array_index]
                ),
                "residual_median": float(
                    arrays[
                        f"{side}_camera_crossfit_residual_median"
                    ][array_index]
                ),
                "residual_reduction": float(
                    arrays[
                        f"{side}_camera_crossfit_residual_reduction"
                    ][array_index]
                ),
            }
            if not isinstance(camera_crossfit, Mapping) or dict(
                camera_crossfit
            ) != expected_camera:
                raise ValueError(
                    f"track-cache row {array_index} {side} crossfit differs"
                )
    if row.get("paired_track_valid") is not (
        side_flags["source"][0] and side_flags["target"][0]
    ):
        raise ValueError(
            f"track-cache row {array_index} paired track flag differs"
        )
    if row.get("paired_camera_valid") is not (
        side_flags["source"][1] and side_flags["target"][1]
    ):
        raise ValueError(
            f"track-cache row {array_index} paired camera flag differs"
        )


def validate_commit(
    directory: Path,
    *,
    expected_contract: Mapping[str, Any] | None = None,
    input_manifest: Path | None = None,
    rehash_videos: bool = False,
    final: bool = False,
    verify_source_shards: bool | None = None,
) -> dict[str, Any]:
    if input_manifest is None:
        raise ValueError("input_manifest is required for strong validation")
    manifest_path = input_manifest.expanduser().resolve(strict=True)
    input_rows = _read_r5_manifest(manifest_path)
    if len(input_rows) != EXPECTED_COHORT_ROWS:
        raise ValueError(
            f"track cache requires {EXPECTED_COHORT_ROWS} development rows"
        )
    paths = _artifact_paths(directory)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    done = _load_json(paths["done"])
    summary = _load_json(paths["summary"])
    expected_done_schema = (
        R7_TRACK_CACHE_FINAL_DONE_SCHEMA
        if final
        else R7_TRACK_CACHE_SHARD_DONE_SCHEMA
    )
    expected_summary_schema = (
        R7_TRACK_CACHE_FINAL_SUMMARY_SCHEMA
        if final
        else R7_TRACK_CACHE_SHARD_SUMMARY_SCHEMA
    )
    if done.get("schema_version") != expected_done_schema:
        raise ValueError("track-cache done schema differs")
    if summary.get("schema_version") != expected_summary_schema:
        raise ValueError("track-cache summary schema differs")
    if done.get("committed") is not True:
        raise ValueError("track-cache commit is not complete")
    _require_fail_closed(done, label="done")
    _require_fail_closed(summary, label="summary")
    for key, path_key in (
        ("archive_sha256", "archive"),
        ("manifest_sha256", "manifest"),
        ("summary_sha256", "summary"),
    ):
        if done.get(key) != _file_digest(paths[path_key]):
            raise ValueError(f"track-cache {key} differs")
    contract = summary.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("track-cache contract is missing")
    _validate_contract_semantics(contract, final=final)
    if summary.get("contract_sha256") != _object_digest(contract):
        raise ValueError("track-cache contract digest differs")
    if done.get("contract_sha256") != summary.get("contract_sha256"):
        raise ValueError("track-cache done/summary contract differs")
    if (
        expected_contract is not None
        and dict(contract) != dict(expected_contract)
    ):
        raise ValueError("track-cache contract differs from expected")
    rows = _load_manifest(paths["manifest"])
    with np.load(paths["archive"], allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    track_count = int(contract["tracker"]["track_count"])
    _validate_array_contract(
        arrays, rows=len(rows), track_count=track_count
    )
    if summary.get("rows") != len(rows) or done.get("rows") != len(rows):
        raise ValueError("track-cache row count differs")
    if (
        summary.get("input_rows") != EXPECTED_COHORT_ROWS
        or contract.get("input_manifest") != str(manifest_path)
        or contract.get("input_manifest_sha256") != _file_digest(manifest_path)
    ):
        raise ValueError("track-cache input manifest contract differs")
    indices = [int(row["input_index"]) for row in rows]
    if indices != arrays["input_indices"].tolist():
        raise ValueError("track-cache row/archive indices differ")
    if len(indices) != len(set(indices)):
        raise ValueError("track-cache manifest input indices duplicate")
    expected_indices = (
        list(range(EXPECTED_COHORT_ROWS))
        if final
        else list(
            range(
                int(contract["rank"]),
                EXPECTED_COHORT_ROWS,
                FINAL_WORLD_SIZE,
            )
        )
    )
    if indices != expected_indices:
        raise ValueError("track-cache shard/final modulo coverage differs")
    for array_index, row in enumerate(rows):
        _validate_row_binding(
            row,
            input_row=input_rows[int(row["input_index"])],
            arrays=arrays,
            array_index=array_index,
            contract=contract,
            final=final,
            rehash_videos=rehash_videos,
        )
    if summary.get("input_indices_sha256") != _object_digest(indices):
        raise ValueError("track-cache input-index digest differs")
    expected_array_sha = {
        name: _array_digest(np.asarray(value))
        for name, value in sorted(arrays.items())
    }
    if summary.get("array_sha256") != expected_array_sha:
        raise ValueError("track-cache array registry differs")
    expected_counts = {
        side: {
            "track_valid": int(
                np.sum(arrays[f"{side}_track_valid"])
            ),
            "camera_valid": int(
                np.sum(arrays[f"{side}_camera_valid"])
            ),
            "camera_crossfit_valid": int(
                np.sum(arrays[f"{side}_camera_crossfit_valid"])
            ),
        }
        for side in SIDES
    }
    if summary.get("counts") != expected_counts:
        raise ValueError("track-cache summary counts differ")
    coverage = _operational_coverage(arrays)
    if summary.get("operational_coverage") != coverage:
        raise ValueError("track-cache operational coverage differs")
    if final and coverage["passed"] is not True:
        raise ValueError("final track-cache operational coverage failed")
    result = {
        "done": done,
        "summary": summary,
        "contract": dict(contract),
        "rows": rows,
        "arrays": arrays,
    }
    verify_sources = final if verify_source_shards is None else bool(
        verify_source_shards
    )
    if final and verify_sources:
        _validate_final_source_shards(
            directory=directory,
            final_result=result,
            input_manifest=manifest_path,
            rehash_videos=rehash_videos,
        )
    return result


def _validate_final_source_shards(
    *,
    directory: Path,
    final_result: Mapping[str, Any],
    input_manifest: Path,
    rehash_videos: bool,
) -> None:
    output_root = directory.expanduser().resolve(strict=True).parent
    shard_results = [
        validate_commit(
            _rank_directory(output_root, rank, FINAL_WORLD_SIZE),
            input_manifest=input_manifest,
            rehash_videos=rehash_videos,
            final=False,
            verify_source_shards=False,
        )
        for rank in range(FINAL_WORLD_SIZE)
    ]
    final_contract = dict(final_result["contract"])
    expected_hashes = [
        _file_digest(
            _artifact_paths(
                _rank_directory(output_root, rank, FINAL_WORLD_SIZE)
            )["done"]
        )
        for rank in range(FINAL_WORLD_SIZE)
    ]
    if final_contract.get("source_shard_done_sha256") != expected_hashes:
        raise ValueError("final/source shard done hashes differ")
    final_base = dict(final_contract)
    for key in (
        "rank",
        "device",
        "merge_world_size",
        "merge_policy",
        "source_shard_done_sha256",
    ):
        final_base.pop(key, None)
    for rank, shard in enumerate(shard_results):
        if _base_contract(shard["contract"]) != final_base:
            raise ValueError(
                f"final/source shard {rank} base contract differs"
            )
    final_rows = final_result["rows"]
    final_arrays = final_result["arrays"]
    for rank, shard in enumerate(shard_results):
        shard_indices = np.asarray(
            shard["arrays"]["input_indices"], dtype=np.int64
        )
        expected = np.arange(
            rank,
            EXPECTED_COHORT_ROWS,
            FINAL_WORLD_SIZE,
            dtype=np.int64,
        )
        if not np.array_equal(shard_indices, expected):
            raise ValueError(f"source shard {rank} coverage differs")
        for local_index, input_index_value in enumerate(shard_indices):
            input_index = int(input_index_value)
            final_row = dict(final_rows[input_index])
            if final_row.pop("final_array_index", None) != input_index:
                raise ValueError(
                    f"final row {input_index} final index differs"
                )
            if final_row != shard["rows"][local_index]:
                raise ValueError(
                    f"final/source shard row differs at {input_index}"
                )
            for name, final_array in final_arrays.items():
                if not np.array_equal(
                    final_array[input_index],
                    shard["arrays"][name][local_index],
                ):
                    raise ValueError(
                        f"final/source shard array {name} differs at "
                        f"{input_index}"
                    )


def extract_rank(
    *,
    input_manifest: Path,
    data_root: Path,
    output_root: Path,
    tracker_checkpoint: Path,
    cotracker_root: Path,
    source_snapshot: Path | None,
    rank: int,
    world_size: int,
    local_rank: int,
    tracker_grid_size: int = 16,
    seed: int = DEFAULT_SEED,
    resume: bool = False,
) -> dict[str, Any]:
    if (
        world_size != FINAL_WORLD_SIZE
        or not 0 <= rank < world_size
        or local_rank != rank
    ):
        raise ValueError(
            "single-node track cache requires rank=local_rank in [0,8)"
        )
    manifest_path = input_manifest.expanduser().resolve(strict=True)
    root = data_root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    checkpoint = tracker_checkpoint.expanduser().resolve(strict=True)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    cotracker = cotracker_root.expanduser().resolve(strict=True)
    snapshot = (
        None
        if source_snapshot is None
        else source_snapshot.expanduser().resolve(strict=True)
    )
    output = validate_output_root(
        output_root=output_root,
        input_manifest=manifest_path,
        data_root=root,
        tracker_checkpoint=checkpoint,
        cotracker_root=cotracker,
        source_snapshot=snapshot,
    )
    input_rows = _read_r5_manifest(manifest_path)
    if len(input_rows) != EXPECTED_COHORT_ROWS:
        raise ValueError(
            f"track cache requires {EXPECTED_COHORT_ROWS} development rows"
        )
    determinism = _configure_determinism(seed, local_rank=local_rank)
    runtime = runtime_provenance(
        local_rank=local_rank,
        determinism=determinism,
    )
    source_provenance = cotracker_source_provenance(cotracker)
    device = f"cuda:{local_rank}"
    contract = build_cache_contract(
        input_manifest=manifest_path,
        data_root=root,
        tracker_checkpoint=checkpoint,
        cotracker_provenance=source_provenance,
        runtime=runtime,
        tracker_grid_size=tracker_grid_size,
        rank=rank,
        world_size=world_size,
        device=device,
        seed=seed,
    )
    directory = _rank_directory(output, rank, world_size)
    if (directory / DONE_NAME).exists():
        if not resume:
            raise FileExistsError(directory / DONE_NAME)
        return validate_commit(
            directory,
            expected_contract=contract,
            input_manifest=manifest_path,
            rehash_videos=True,
        )["done"]
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(
            f"partial track-cache shard cannot be resumed: {directory}"
        )
    selected = [
        (index, row)
        for index, row in enumerate(input_rows)
        if index % world_size == rank
    ]
    track_count = tracker_grid_size**2
    arrays = _empty_arrays(len(selected), track_count=track_count)
    tracker = LazyCoTrackerAdapter(
        checkpoint=checkpoint,
        device=device,
        grid_size=tracker_grid_size,
        backward_tracking=False,
    )
    camera_config = TemporalTeacherConfig()
    output_rows: list[dict[str, Any]] = []
    for array_index, (input_index, row) in enumerate(selected):
        iid = str(row["iid"])
        positive = row["r5_pilot_label"]["class"] == "positive"
        arrays["input_indices"][array_index] = input_index
        arrays["positive"][array_index] = positive
        results: dict[str, dict[str, Any]] = {}
        for side, field in (("source", "src_video"), ("target", "tgt_video")):
            path = _safe_video_path(root, str(row[field]))
            results[side] = _extract_side(
                path=path,
                side=side,
                array_index=array_index,
                arrays=arrays,
                tracker=tracker,
                track_count=track_count,
                camera_config=camera_config,
            )
        output_rows.append(
            {
                "schema_version": R7_TRACK_CACHE_ROW_SCHEMA,
                "input_index": input_index,
                "shard_array_index": array_index,
                "shard_rank": rank,
                "world_size": world_size,
                "iid": iid,
                "input_row": dict(row),
                "input_row_sha256": _object_digest(row),
                "input_digest": row.get("input_digest"),
                "prompt": row.get("prompt"),
                "label_type": row["r5_pilot_label"]["class"],
                "negative_type": row["r5_pilot_label"].get(
                    "negative_type"
                ),
                "positive": positive,
                "action_signature": row["r5_pilot_label"].get(
                    "action_signature"
                ),
                "source": results["source"],
                "target": results["target"],
                "paired_track_valid": bool(
                    results["source"]["track_valid"]
                    and results["target"]["track_valid"]
                ),
                "paired_camera_valid": bool(
                    results["source"]["camera_valid"]
                    and results["target"]["camera_valid"]
                ),
            }
        )
    return _commit(
        directory=directory,
        rows=output_rows,
        arrays=arrays,
        contract=contract,
        input_rows=len(input_rows),
        final=False,
    )


def finalize_shards(
    *,
    input_manifest: Path,
    output_root: Path,
    resume: bool = False,
) -> dict[str, Any]:
    manifest_path = input_manifest.expanduser().resolve(strict=True)
    input_rows = _read_r5_manifest(manifest_path)
    if len(input_rows) != EXPECTED_COHORT_ROWS:
        raise ValueError(
            f"track cache requires {EXPECTED_COHORT_ROWS} development rows"
        )
    final_directory = output_root.expanduser() / FINAL_DIR_NAME
    if (final_directory / DONE_NAME).exists():
        if not resume:
            raise FileExistsError(final_directory / DONE_NAME)
        return validate_commit(
            final_directory,
            input_manifest=manifest_path,
            final=True,
        )["done"]
    if final_directory.exists() and any(final_directory.iterdir()):
        raise FileExistsError(
            f"partial final track cache cannot be resumed: {final_directory}"
        )
    shard_results = [
        validate_commit(
            _rank_directory(output_root, rank, FINAL_WORLD_SIZE),
            input_manifest=manifest_path,
            rehash_videos=True,
        )
        for rank in range(FINAL_WORLD_SIZE)
    ]
    base_contracts = [
        _base_contract(result["contract"]) for result in shard_results
    ]
    if any(value != base_contracts[0] for value in base_contracts[1:]):
        raise ValueError("track-cache shard contracts differ")
    if int(base_contracts[0]["world_size"]) != FINAL_WORLD_SIZE:
        raise ValueError("track-cache finalization requires world size 8")
    tracker_contract = base_contracts[0]["tracker"]
    checkpoint = Path(str(tracker_contract["checkpoint"]))
    if (
        not checkpoint.is_file()
        or _file_digest(checkpoint)
        != tracker_contract["checkpoint_sha256"]
    ):
        raise ValueError(
            "CoTracker checkpoint changed before finalization"
        )
    source_root = Path(str(tracker_contract["source"]["root"]))
    if cotracker_source_provenance(source_root) != tracker_contract["source"]:
        raise ValueError(
            "CoTracker Python source changed before finalization"
        )
    track_count = int(base_contracts[0]["tracker"]["track_count"])
    arrays = _empty_arrays(len(input_rows), track_count=track_count)
    rows_by_index: dict[int, dict[str, Any]] = {}
    for rank, result in enumerate(shard_results):
        shard_rows = result["rows"]
        shard_arrays = result["arrays"]
        indices = np.asarray(shard_arrays["input_indices"], dtype=np.int64)
        expected = np.arange(rank, len(input_rows), FINAL_WORLD_SIZE)
        if not np.array_equal(indices, expected):
            raise ValueError(
                f"track-cache shard {rank} modulo coverage differs"
            )
        for local_index, input_index_value in enumerate(indices):
            input_index = int(input_index_value)
            if input_index in rows_by_index:
                raise ValueError(
                    f"duplicate track-cache input index {input_index}"
                )
            row = dict(shard_rows[local_index])
            row["final_array_index"] = input_index
            rows_by_index[input_index] = row
            for name in arrays:
                arrays[name][input_index] = shard_arrays[name][local_index]
    if set(rows_by_index) != set(range(len(input_rows))):
        raise ValueError("track-cache final coverage is incomplete")
    rows = [rows_by_index[index] for index in range(len(input_rows))]
    coverage = _operational_coverage(arrays)
    if coverage["passed"] is not True:
        raise ValueError(
            "refusing to commit final cache: operational coverage failed; "
            + _canonical_json(coverage)
        )
    final_contract = dict(base_contracts[0])
    final_contract.update(
        {
            "rank": "merged",
            "device": "eight-shard-final",
            "merge_world_size": FINAL_WORLD_SIZE,
            "merge_policy": "strict-input-index-order-v2",
            "source_shard_done_sha256": [
                _file_digest(
                    _artifact_paths(
                        _rank_directory(
                            output_root, rank, FINAL_WORLD_SIZE
                        )
                    )["done"]
                )
                for rank in range(FINAL_WORLD_SIZE)
            ],
        }
    )
    done = _commit(
        directory=final_directory,
        rows=rows,
        arrays=arrays,
        contract=final_contract,
        input_rows=len(input_rows),
        final=True,
    )
    validate_commit(
        final_directory,
        expected_contract=final_contract,
        input_manifest=manifest_path,
        rehash_videos=True,
        final=True,
    )
    return done


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--input-manifest", type=Path, required=True)
    preflight.add_argument("--data-root", type=Path, required=True)
    preflight.add_argument("--output-dir", type=Path, required=True)
    preflight.add_argument(
        "--cotracker-checkpoint", type=Path, required=True
    )
    preflight.add_argument("--cotracker-root", type=Path, required=True)
    preflight.add_argument("--source-snapshot", type=Path, required=True)
    extract = subparsers.add_parser("extract")
    extract.add_argument("--input-manifest", type=Path, required=True)
    extract.add_argument("--data-root", type=Path, required=True)
    extract.add_argument("--output-dir", type=Path, required=True)
    extract.add_argument("--cotracker-checkpoint", type=Path, required=True)
    extract.add_argument("--cotracker-root", type=Path, required=True)
    extract.add_argument("--source-snapshot", type=Path, required=True)
    extract.add_argument("--tracker-grid-size", type=int, default=16)
    extract.add_argument("--seed", type=int, default=DEFAULT_SEED)
    extract.add_argument("--rank", type=int)
    extract.add_argument("--world-size", type=int)
    extract.add_argument("--local-rank", type=int)
    extract.add_argument("--resume", action="store_true")
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--input-manifest", type=Path, required=True)
    finalize.add_argument("--output-dir", type=Path, required=True)
    finalize.add_argument("--resume", action="store_true")
    validate = subparsers.add_parser("validate")
    validate.add_argument("--input-manifest", type=Path, required=True)
    validate.add_argument("--output-dir", type=Path, required=True)
    validate.add_argument("--final", action="store_true")
    validate.add_argument("--rehash-videos", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "preflight":
        output = validate_output_root(
            output_root=args.output_dir,
            input_manifest=args.input_manifest,
            data_root=args.data_root,
            tracker_checkpoint=args.cotracker_checkpoint,
            cotracker_root=args.cotracker_root,
            source_snapshot=args.source_snapshot,
        )
        input_rows = _read_r5_manifest(args.input_manifest)
        if len(input_rows) != EXPECTED_COHORT_ROWS:
            raise ValueError(
                f"track cache requires {EXPECTED_COHORT_ROWS} development rows"
            )
        source = cotracker_source_provenance(args.cotracker_root)
        result = {
            "schema_version": "motive-r7-p1-track-cache-preflight-v1",
            "status": "ready",
            "rows": len(input_rows),
            "output_root": str(output),
            "input_manifest_sha256": _file_digest(args.input_manifest),
            "tracker_checkpoint_sha256": _file_digest(
                args.cotracker_checkpoint
            ),
            "cotracker_source_bundle_sha256":
                source["python_source_bundle_sha256"],
            "cohort_role": COHORT_ROLE,
            "cohort_id": COHORT_ID,
            "formal_status": FORMAL_STATUS,
            "formal_reason": FORMAL_REASON,
            "production_decision": False,
            "generation_authorized": False,
        }
    elif args.command == "extract":
        rank, world_size, local_rank = resolve_torchrun_coordinates(
            rank=args.rank,
            world_size=args.world_size,
            local_rank=args.local_rank,
        )
        result = extract_rank(
            input_manifest=args.input_manifest,
            data_root=args.data_root,
            output_root=args.output_dir,
            tracker_checkpoint=args.cotracker_checkpoint,
            cotracker_root=args.cotracker_root,
            source_snapshot=args.source_snapshot,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            tracker_grid_size=args.tracker_grid_size,
            seed=args.seed,
            resume=args.resume,
        )
    elif args.command == "finalize":
        result = finalize_shards(
            input_manifest=args.input_manifest,
            output_root=args.output_dir,
            resume=args.resume,
        )
    else:
        directory = (
            args.output_dir / FINAL_DIR_NAME
            if args.final
            else args.output_dir
        )
        result = validate_commit(
            directory,
            input_manifest=args.input_manifest,
            rehash_videos=args.rehash_videos,
            final=args.final,
        )["done"]
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "R7_TRACK_CACHE_SCHEMA",
    "R7_TRACK_CACHE_ROW_SCHEMA",
    "_empty_arrays",
    "_validate_array_contract",
    "build_cache_contract",
    "cotracker_source_provenance",
    "extract_rank",
    "finalize_shards",
    "main",
    "runtime_provenance",
    "validate_output_root",
    "validate_commit",
]
