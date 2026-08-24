"""Immutable, label-neutral CoTracker cache for the R7 candidate screen.

This module deliberately does not reuse the R5/P1 track-cache artifact
schemas.  It consumes one fully validated
``motive-r7-candidate-temporal-screen-v1`` commit, but does not copy its
pseudo-labels into the cache manifest or archive.  The only retained input
identity is the ordered source-row digest and IID.

The numerical observation core is shared with :mod:`motive.r7_track_cache`.
In particular, candidate arrays are made by removing the legacy ``positive``
array.  Validation temporarily adds an in-memory all-zero boolean array only
to invoke the label-independent geometric validator.  That compatibility
array is never serialized and has no label semantics.

Exactly eight ranks own rows by input-index modulo world size.  Every shard
and final directory is a create-only atomic commit with files mode ``0444``
and directory mode ``0555``.  Resume is validation-only.  Validation
rechecks the externally anchored input done SHA, all referenced media bytes,
the complete array/row binding, and (for final commits) all eight source
shards.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import r7_artifact_permissions as artifact_permissions
from . import r7_candidate_temporal_manifest as candidate_manifest
from . import r7_preflight_extract as preflight
from . import r7_temporal_teacher as temporal_teacher
from . import r7_track_cache as legacy_cache
from .r7_preflight_extract import (
    DEFAULT_SEED,
    FINAL_WORLD_SIZE,
    VIDEO_FRAMES,
    _array_digest,
    _canonical_json,
    _file_digest,
    _object_digest,
    _safe_video_path,
    resolve_torchrun_coordinates,
)
from .r7_temporal_teacher import (
    LazyCoTrackerAdapter,
    TemporalTeacherConfig,
)


CACHE_SCHEMA = "motive-r7-candidate-temporal-track-cache-v1"
ROW_SCHEMA = "motive-r7-candidate-temporal-track-cache-row-v1"
SHARD_SUMMARY_SCHEMA = (
    "motive-r7-candidate-temporal-track-cache-shard-summary-v1"
)
SHARD_DONE_SCHEMA = (
    "motive-r7-candidate-temporal-track-cache-shard-done-v1"
)
FINAL_SUMMARY_SCHEMA = (
    "motive-r7-candidate-temporal-track-cache-final-summary-v1"
)
FINAL_DONE_SCHEMA = (
    "motive-r7-candidate-temporal-track-cache-final-done-v1"
)
PARTITION = "input-index-modulo-world-size-v1"
MERGE_POLICY = "strict-input-index-order-v1"

ARCHIVE_NAME = "track_cache.npz"
MANIFEST_NAME = "manifest.jsonl"
SUMMARY_NAME = "summary.json"
DONE_NAME = "done.json"
FINAL_DIR_NAME = "final"
OUTPUT_NAMES = (ARCHIVE_NAME, MANIFEST_NAME, SUMMARY_NAME, DONE_NAME)
PAYLOAD_NAMES = (ARCHIVE_NAME, MANIFEST_NAME, SUMMARY_NAME)
SIDES = ("source", "target")

CACHE_SCOPE = (
    "pre-selector motion observations for candidate temporal screening; "
    "pseudo-labels are input provenance only and are not materialized"
)
LABEL_SEMANTICS = {
    "input_labels_are_pseudo": True,
    "input_labels_copied_to_cache": False,
    "positive_array_present": False,
    "cache_rows_are_label_neutral": True,
}
OPERATIONAL_COVERAGE_POLICY = (
    "all-input-rows-track-and-camera-valid-fraction-v2"
)
MIN_SOURCE_TRACK_COVERAGE = legacy_cache.MIN_SOURCE_TRACK_COVERAGE
MIN_TARGET_TRACK_COVERAGE = legacy_cache.MIN_TARGET_TRACK_COVERAGE
MIN_PAIRED_TRACK_COVERAGE = legacy_cache.MIN_PAIRED_TRACK_COVERAGE
MIN_SOURCE_CAMERA_COVERAGE = 0.90
MIN_TARGET_CAMERA_COVERAGE = 0.90
MIN_PAIRED_CAMERA_COVERAGE = 0.85

SAFETY_FIELDS = (
    "formal_evidence",
    "formal_split",
    "formal_report",
    "human_labels_asserted",
    "thresholds_human_calibrated",
    "training_authorized",
    "direct_training_supervision_allowed",
    "generation_authorized",
    "editing_authorized",
    "production_decision",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

_ROW_FIELDS = frozenset(
    {
        "schema_version",
        "input_index",
        "shard_array_index",
        "shard_rank",
        "world_size",
        "iid",
        "input_row_sha256",
        "input_artifact_digest",
        "source",
        "target",
        "paired_track_valid",
        "paired_camera_valid",
        *SAFETY_FIELDS,
    }
)
_INPUT_BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "directory",
        "manifest_sha256",
        "summary_sha256",
        "done_sha256",
        "artifact_digest",
        "rows",
        "row_order",
        "row_identity",
        "media_binding_digest",
    }
)
_BASE_CONTRACT_FIELDS = frozenset(
    {
        "schema_version",
        "input_binding",
        "data_root",
        "rank",
        "world_size",
        "partition",
        "device",
        "seed",
        "frames",
        "tracker",
        "runtime",
        "camera_config",
        "operational_coverage",
        "cache_scope",
        "label_semantics",
        "implementation_sha256",
        *SAFETY_FIELDS,
    }
)
_FINAL_CONTRACT_FIELDS = frozenset(
    {
        *_BASE_CONTRACT_FIELDS,
        "merge_world_size",
        "merge_policy",
        "source_shard_done_sha256",
    }
)
_SIDE_COMMON_FIELDS = frozenset(
    {
        "status",
        "track_valid",
        "camera_valid",
        "failure_stage",
        "failure_reason",
        "failure_message",
        "resolved_path",
        "video_sha256",
    }
)


@dataclass(frozen=True)
class _InputCommit:
    directory: Path
    rows: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]
    done: Mapping[str, Any]
    paths: Mapping[str, Path]
    snapshot: Mapping[str, Mapping[str, Any]]
    data_root: Path

    @property
    def done_sha256(self) -> str:
        return str(self.snapshot[candidate_manifest.DONE_NAME]["sha256"])

    @property
    def artifact_digest(self) -> str:
        return str(self.done["artifact_digest"])


def _safety_flags() -> dict[str, bool]:
    return {field: False for field in SAFETY_FIELDS}


def _require_sha256(value: Any, *, context: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _snapshot(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, path in sorted(paths.items()):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"input artifact is not a regular file: {path}")
        before = path.stat()
        digest = _file_digest(path)
        after = path.stat()
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise RuntimeError(f"input artifact changed while hashing: {path}")
        result[name] = {
            "sha256": digest,
            "bytes": int(before.st_size),
        }
    return result


def _assert_snapshot(value: _InputCommit) -> None:
    if _snapshot(value.paths) != {
        name: dict(record)
        for name, record in sorted(value.snapshot.items())
    }:
        raise RuntimeError("candidate temporal input changed during cache work")


def _load_input_commit(
    input_dir: Path,
    *,
    expected_done_sha256: str,
    expected_data_root: Path | None = None,
) -> _InputCommit:
    expected_done = _require_sha256(
        expected_done_sha256,
        context="expected candidate temporal done",
    )
    unresolved = input_dir.expanduser()
    if unresolved.is_symlink() or not unresolved.is_dir():
        raise FileNotFoundError(
            f"candidate temporal input must be a real directory: {unresolved}"
        )
    root = unresolved.resolve(strict=True)
    paths = {
        candidate_manifest.MANIFEST_NAME:
            root / candidate_manifest.MANIFEST_NAME,
        candidate_manifest.SUMMARY_NAME:
            root / candidate_manifest.SUMMARY_NAME,
        candidate_manifest.DONE_NAME:
            root / candidate_manifest.DONE_NAME,
    }
    before = _snapshot(paths)
    if before[candidate_manifest.DONE_NAME]["sha256"] != expected_done:
        raise ValueError("candidate temporal external done SHA differs")
    result = candidate_manifest.validate_candidate_temporal_manifest(root)
    after = _snapshot(paths)
    if after != before:
        raise RuntimeError("candidate temporal input changed during validation")
    returned_root = Path(result["directory"]).resolve(strict=True)
    if returned_root != root:
        raise ValueError("candidate temporal validator returned another root")
    rows = result.get("rows")
    summary = result.get("summary")
    done = result.get("done")
    if (
        not isinstance(rows, list)
        or not rows
        or not isinstance(summary, Mapping)
        or not isinstance(done, Mapping)
        or done.get("schema_version") != candidate_manifest.DONE_SCHEMA
        or summary.get("schema_version") != candidate_manifest.SUMMARY_SCHEMA
        or done.get("output_rows") != len(rows)
    ):
        raise ValueError("candidate temporal validator result differs")
    data_roots = {
        row["source_bindings"]["media"]["data_root"]
        for row in rows
    }
    if len(data_roots) != 1:
        raise ValueError("candidate temporal media roots differ")
    data_root = Path(next(iter(data_roots))).expanduser().resolve(strict=True)
    if not data_root.is_dir() or data_root.is_symlink():
        raise ValueError("candidate temporal data root is not a real directory")
    if expected_data_root is not None:
        live_root = expected_data_root.expanduser().resolve(strict=True)
        if live_root != data_root:
            raise ValueError("candidate temporal data root differs")
    return _InputCommit(
        directory=root,
        rows=tuple(dict(row) for row in rows),
        summary=dict(summary),
        done=dict(done),
        paths=paths,
        snapshot=after,
        data_root=data_root,
    )


def _input_binding(value: _InputCommit) -> dict[str, Any]:
    return {
        "schema_version": candidate_manifest.SUMMARY_SCHEMA,
        "directory": str(value.directory),
        "manifest_sha256":
            value.snapshot[candidate_manifest.MANIFEST_NAME]["sha256"],
        "summary_sha256":
            value.snapshot[candidate_manifest.SUMMARY_NAME]["sha256"],
        "done_sha256": value.done_sha256,
        "artifact_digest": value.artifact_digest,
        "rows": len(value.rows),
        "row_order": "ascending_candidate_row_index",
        "row_identity": "iid",
        "media_binding_digest": value.summary["media"]["binding_digest"],
    }


def _implementation_sha256() -> dict[str, str]:
    modules = (
        Path(__file__).resolve(strict=True),
        Path(artifact_permissions.__file__).resolve(strict=True),
        Path(candidate_manifest.__file__).resolve(strict=True),
        Path(legacy_cache.__file__).resolve(strict=True),
        Path(preflight.__file__).resolve(strict=True),
        Path(temporal_teacher.__file__).resolve(strict=True),
    )
    return {
        path.name: _file_digest(path)
        for path in sorted(modules, key=lambda item: item.name)
    }


def _camera_config() -> dict[str, Any]:
    return {
        key: value
        for key, value in vars(TemporalTeacherConfig()).items()
        if key.startswith("camera_")
        or key.startswith("minimum_camera_")
        or key
        in {
            "minimum_frames",
            "minimum_tracks",
            "visibility_threshold",
            "eps",
        }
    }


def _coverage_contract() -> dict[str, Any]:
    return {
        "policy": OPERATIONAL_COVERAGE_POLICY,
        "minimum_source_track_valid_fraction": MIN_SOURCE_TRACK_COVERAGE,
        "minimum_target_track_valid_fraction": MIN_TARGET_TRACK_COVERAGE,
        "minimum_paired_track_valid_fraction": MIN_PAIRED_TRACK_COVERAGE,
        "minimum_source_camera_valid_fraction":
            MIN_SOURCE_CAMERA_COVERAGE,
        "minimum_target_camera_valid_fraction":
            MIN_TARGET_CAMERA_COVERAGE,
        "minimum_paired_camera_valid_fraction":
            MIN_PAIRED_CAMERA_COVERAGE,
    }


def _empty_arrays(
    rows: int,
    *,
    track_count: int,
) -> dict[str, np.ndarray]:
    arrays = legacy_cache._empty_arrays(rows, track_count=track_count)
    positive = arrays.pop("positive", None)
    if (
        positive is None
        or positive.dtype != np.dtype("bool")
        or positive.shape != (rows,)
    ):
        raise RuntimeError("legacy compatibility array contract changed")
    return arrays


def _validate_array_contract(
    arrays: Mapping[str, np.ndarray],
    *,
    rows: int,
    track_count: int,
) -> None:
    expected = _empty_arrays(rows, track_count=track_count)
    if set(arrays) != set(expected):
        raise ValueError(
            "candidate track-cache arrays differ; "
            f"missing={sorted(set(expected) - set(arrays))}, "
            f"extra={sorted(set(arrays) - set(expected))}"
        )
    if "positive" in arrays:
        raise ValueError("candidate track cache must not materialize positive")
    geometry_view = {
        name: np.asarray(value) for name, value in arrays.items()
    }
    # Compatibility only: the legacy validator has no label-dependent
    # numerical checks, but its exact field set includes this array.
    geometry_view["positive"] = np.zeros(rows, dtype=np.bool_)
    legacy_cache._validate_array_contract(
        geometry_view,
        rows=rows,
        track_count=track_count,
    )


def build_cache_contract(
    *,
    input_dir: Path,
    expected_input_done_sha256: str,
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
    source = _load_input_commit(
        input_dir,
        expected_done_sha256=expected_input_done_sha256,
        expected_data_root=data_root,
    )
    if tracker_grid_size < 4:
        raise ValueError(
            "tracker_grid_size must be >= 4 so camera compensation has "
            "at least 16 tracks"
        )
    if world_size != FINAL_WORLD_SIZE or not 0 <= rank < world_size:
        raise ValueError("candidate track cache requires exactly eight ranks")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= 0xFFFFFFFF
    ):
        raise ValueError("seed must be an integer in [0, 2**32-1]")
    checkpoint = tracker_checkpoint.expanduser().resolve(strict=True)
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    contract = {
        "schema_version": CACHE_SCHEMA,
        "input_binding": _input_binding(source),
        "data_root": str(source.data_root),
        "rank": rank,
        "world_size": world_size,
        "partition": PARTITION,
        "device": str(device),
        "seed": int(seed),
        "frames": VIDEO_FRAMES,
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
        "camera_config": _camera_config(),
        "operational_coverage": _coverage_contract(),
        "cache_scope": CACHE_SCOPE,
        "label_semantics": dict(LABEL_SEMANTICS),
        "implementation_sha256": _implementation_sha256(),
        **_safety_flags(),
    }
    _validate_contract_semantics(contract, input_commit=source, final=False)
    return contract


def _runtime_contract_mismatch(
    field: str,
    value: Any,
    expected: str,
) -> None:
    rendered = repr(value)
    if len(rendered) > 160:
        rendered = rendered[:157] + "..."
    raise ValueError(
        "candidate runtime/determinism contract differs: "
        f"{field} expected {expected}; "
        f"got type={type(value).__name__} value={rendered}"
    )


def _validate_runtime_contract(
    runtime: Any,
    *,
    seed: int,
) -> None:
    if not isinstance(runtime, Mapping):
        _runtime_contract_mismatch(
            "runtime",
            runtime,
            "a mapping",
        )
    if runtime.get("schema_version") != "motive-r7-runtime-v1":
        _runtime_contract_mismatch(
            "runtime.schema_version",
            runtime.get("schema_version"),
            "'motive-r7-runtime-v1'",
        )
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
    for key in required_runtime_strings:
        value = runtime.get(key)
        if type(value) is not str or not value:
            _runtime_contract_mismatch(
                f"runtime.{key}",
                value,
                "a non-empty built-in str",
            )
    if runtime.get("device_type") != "cuda-hip":
        _runtime_contract_mismatch(
            "runtime.device_type",
            runtime.get("device_type"),
            "'cuda-hip'",
        )
    hip_version = runtime.get("torch_hip_version")
    if type(hip_version) is not str or not hip_version:
        _runtime_contract_mismatch(
            "runtime.torch_hip_version",
            hip_version,
            "a non-empty built-in str",
        )
    visible_devices = runtime.get("visible_device_count")
    if type(visible_devices) is not int or visible_devices != FINAL_WORLD_SIZE:
        _runtime_contract_mismatch(
            "runtime.visible_device_count",
            visible_devices,
            f"the built-in int {FINAL_WORLD_SIZE}",
        )
    total_memory = runtime.get("device_total_memory")
    if type(total_memory) is not int or total_memory <= 0:
        _runtime_contract_mismatch(
            "runtime.device_total_memory",
            total_memory,
            "a positive built-in int",
        )
    capability = runtime.get("device_capability")
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(type(value) is not int for value in capability)
    ):
        _runtime_contract_mismatch(
            "runtime.device_capability",
            capability,
            "a two-element list of built-in ints",
        )
    build_sha256 = runtime.get("torch_build_config_sha256")
    if (
        type(build_sha256) is not str
        or _SHA256_RE.fullmatch(build_sha256) is None
    ):
        _runtime_contract_mismatch(
            "runtime.torch_build_config_sha256",
            build_sha256,
            "a lowercase SHA-256 string",
        )
    determinism = runtime.get("determinism")
    if not isinstance(determinism, Mapping):
        _runtime_contract_mismatch(
            "runtime.determinism",
            determinism,
            "a mapping",
        )
    exact_determinism = {
        "schema_version": "motive-r7-determinism-v1",
        "seed": seed,
        "rank_seed_policy":
            "identical-base-seed-on-all-eight-ranks-v1",
        "python_random_seeded": True,
        "numpy_seeded": True,
        "torch_cpu_seeded": True,
        "torch_all_visible_devices_seeded": True,
        "python_hash_seed": str(seed),
        "torch_deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }
    for key, expected in exact_determinism.items():
        value = determinism.get(key)
        if value is not expected and value != expected:
            _runtime_contract_mismatch(
                f"runtime.determinism.{key}",
                value,
                repr(expected),
            )
        if isinstance(expected, bool) and value is not expected:
            _runtime_contract_mismatch(
                f"runtime.determinism.{key}",
                value,
                f"the bool singleton {expected!r}",
            )


def _validate_contract_semantics(
    contract: Mapping[str, Any],
    *,
    input_commit: _InputCommit,
    final: bool,
) -> None:
    expected_fields = (
        _FINAL_CONTRACT_FIELDS if final else _BASE_CONTRACT_FIELDS
    )
    if set(contract) != set(expected_fields):
        raise ValueError("candidate track-cache contract fields differ")
    if (
        contract.get("schema_version") != CACHE_SCHEMA
        or contract.get("input_binding") != _input_binding(input_commit)
        or contract.get("data_root") != str(input_commit.data_root)
        or contract.get("world_size") != FINAL_WORLD_SIZE
        or contract.get("partition") != PARTITION
        or contract.get("frames") != VIDEO_FRAMES
        or contract.get("camera_config") != _camera_config()
        or contract.get("operational_coverage") != _coverage_contract()
        or contract.get("cache_scope") != CACHE_SCOPE
        or contract.get("label_semantics") != LABEL_SEMANTICS
    ):
        raise ValueError("candidate track-cache contract semantics differ")
    if any(contract.get(field) is not False for field in SAFETY_FIELDS):
        raise ValueError("candidate track-cache safety gate asserted")
    rank = contract.get("rank")
    device = contract.get("device")
    if final:
        hashes = contract.get("source_shard_done_sha256")
        if (
            rank != "merged"
            or device != "eight-shard-final"
            or contract.get("merge_world_size") != FINAL_WORLD_SIZE
            or contract.get("merge_policy") != MERGE_POLICY
            or not isinstance(hashes, list)
            or len(hashes) != FINAL_WORLD_SIZE
            or any(
                type(value) is not str
                or _SHA256_RE.fullmatch(value) is None
                for value in hashes
            )
        ):
            raise ValueError("candidate final merge contract differs")
    elif (
        type(rank) is not int
        or not 0 <= rank < FINAL_WORLD_SIZE
        or device != f"cuda:{rank}"
    ):
        raise ValueError("candidate shard rank/device contract differs")
    seed = contract.get("seed")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= 0xFFFFFFFF
    ):
        raise ValueError("candidate track-cache seed differs")
    tracker = contract.get("tracker")
    if not isinstance(tracker, Mapping):
        raise ValueError("candidate tracker contract is missing")
    grid_size = tracker.get("grid_size")
    checkpoint = tracker.get("checkpoint")
    checkpoint_sha = tracker.get("checkpoint_sha256")
    if (
        type(grid_size) is not int
        or grid_size < 2
        or tracker.get("track_count") != grid_size**2
        or tracker.get("query_frame") != 0
        or tracker.get("backward_tracking") is not False
        or type(checkpoint) is not str
        or not Path(checkpoint).is_absolute()
        or type(checkpoint_sha) is not str
        or _SHA256_RE.fullmatch(checkpoint_sha) is None
    ):
        raise ValueError("candidate tracker geometry/provenance differs")
    checkpoint_path = Path(checkpoint)
    if (
        checkpoint_path.is_symlink()
        or not checkpoint_path.is_file()
        or _file_digest(checkpoint_path) != checkpoint_sha
    ):
        raise ValueError("candidate tracker checkpoint bytes differ")
    source = tracker.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("candidate CoTracker source provenance is missing")
    inventory = source.get("python_source_files")
    inventory_paths = (
        [entry.get("path") for entry in inventory]
        if isinstance(inventory, list)
        and all(
            isinstance(entry, Mapping)
            and type(entry.get("path")) is str
            for entry in inventory
        )
        else []
    )
    if (
        source.get("git_tracked_clean") is not True
        or type(source.get("root")) is not str
        or not Path(source["root"]).is_absolute()
        or type(source.get("git_head")) is not str
        or _GIT_COMMIT_RE.fullmatch(source["git_head"]) is None
        or type(source.get("git_toplevel")) is not str
        or not Path(source["git_toplevel"]).is_absolute()
        or not isinstance(inventory, list)
        or not inventory
        or inventory_paths != sorted(set(inventory_paths))
        or "cotracker/predictor.py" not in inventory_paths
        or source.get("python_source_file_count") != len(inventory)
        or source.get("python_source_bundle_sha256")
        != _object_digest(inventory)
    ):
        raise ValueError("candidate CoTracker source provenance differs")
    for entry in inventory:
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"path", "size", "sha256"}
            or type(entry.get("path")) is not str
            or Path(entry["path"]).is_absolute()
            or ".." in Path(entry["path"]).parts
            or isinstance(entry.get("size"), bool)
            or not isinstance(entry.get("size"), int)
            or entry["size"] < 0
            or type(entry.get("sha256")) is not str
            or _SHA256_RE.fullmatch(entry["sha256"]) is None
        ):
            raise ValueError(
                "candidate CoTracker source inventory differs"
            )
    _validate_runtime_contract(contract.get("runtime"), seed=seed)
    if contract.get("implementation_sha256") != _implementation_sha256():
        raise ValueError("candidate implementation bytes differ")


def _base_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(contract)
    for key in (
        "rank",
        "device",
        "merge_world_size",
        "merge_policy",
        "source_shard_done_sha256",
    ):
        value.pop(key, None)
    return value


def _rank_directory(root: Path, rank: int) -> Path:
    if type(rank) is not int or not 0 <= rank < FINAL_WORLD_SIZE:
        raise ValueError("rank must be in [0,8)")
    return (
        root.expanduser()
        / "shards"
        / f"rank-{rank:03d}-of-{FINAL_WORLD_SIZE:03d}"
    )


def _artifact_paths(directory: Path) -> dict[str, Path]:
    return {
        "archive": directory / ARCHIVE_NAME,
        "manifest": directory / MANIFEST_NAME,
        "summary": directory / SUMMARY_NAME,
        "done": directory / DONE_NAME,
    }


def _operational_coverage(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    rows = len(np.asarray(arrays["input_indices"]))
    source = np.asarray(arrays["source_track_valid"], dtype=bool)
    target = np.asarray(arrays["target_track_valid"], dtype=bool)
    paired = source & target
    source_camera = np.asarray(
        arrays["source_camera_valid"],
        dtype=bool,
    )
    target_camera = np.asarray(
        arrays["target_camera_valid"],
        dtype=bool,
    )
    paired_camera = source_camera & target_camera
    criteria = {
        "minimum_source_track_valid_fraction":
            MIN_SOURCE_TRACK_COVERAGE,
        "minimum_target_track_valid_fraction":
            MIN_TARGET_TRACK_COVERAGE,
        "minimum_paired_track_valid_fraction":
            MIN_PAIRED_TRACK_COVERAGE,
        "minimum_source_camera_valid_fraction":
            MIN_SOURCE_CAMERA_COVERAGE,
        "minimum_target_camera_valid_fraction":
            MIN_TARGET_CAMERA_COVERAGE,
        "minimum_paired_camera_valid_fraction":
            MIN_PAIRED_CAMERA_COVERAGE,
    }
    if rows == 0:
        return {
            "policy": OPERATIONAL_COVERAGE_POLICY,
            "rows": 0,
            "applicable": False,
            "source_track_valid": 0,
            "target_track_valid": 0,
            "paired_track_valid": 0,
            "source_camera_valid": 0,
            "target_camera_valid": 0,
            "paired_camera_valid": 0,
            "source_track_valid_fraction": None,
            "target_track_valid_fraction": None,
            "paired_track_valid_fraction": None,
            "source_camera_valid_fraction": None,
            "target_camera_valid_fraction": None,
            "paired_camera_valid_fraction": None,
            "criteria": criteria,
            "passed": None,
        }
    source_fraction = float(source.mean())
    target_fraction = float(target.mean())
    paired_fraction = float(paired.mean())
    source_camera_fraction = float(source_camera.mean())
    target_camera_fraction = float(target_camera.mean())
    paired_camera_fraction = float(paired_camera.mean())
    return {
        "policy": OPERATIONAL_COVERAGE_POLICY,
        "rows": rows,
        "applicable": True,
        "source_track_valid": int(source.sum()),
        "target_track_valid": int(target.sum()),
        "paired_track_valid": int(paired.sum()),
        "source_camera_valid": int(source_camera.sum()),
        "target_camera_valid": int(target_camera.sum()),
        "paired_camera_valid": int(paired_camera.sum()),
        "source_track_valid_fraction": source_fraction,
        "target_track_valid_fraction": target_fraction,
        "paired_track_valid_fraction": paired_fraction,
        "source_camera_valid_fraction": source_camera_fraction,
        "target_camera_valid_fraction": target_camera_fraction,
        "paired_camera_valid_fraction": paired_camera_fraction,
        "criteria": criteria,
        "passed": bool(
            source_fraction >= MIN_SOURCE_TRACK_COVERAGE
            and target_fraction >= MIN_TARGET_TRACK_COVERAGE
            and paired_fraction >= MIN_PAIRED_TRACK_COVERAGE
            and source_camera_fraction >= MIN_SOURCE_CAMERA_COVERAGE
            and target_camera_fraction >= MIN_TARGET_CAMERA_COVERAGE
            and paired_camera_fraction >= MIN_PAIRED_CAMERA_COVERAGE
        ),
    }


def _validate_side_record(
    record: Mapping[str, Any],
    *,
    side: str,
    input_row: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    array_index: int,
    contract: Mapping[str, Any],
) -> tuple[bool, bool]:
    if side not in SIDES:
        raise ValueError("invalid candidate track-cache side")
    input_field = "src_video" if side == "source" else "tgt_video"
    media = input_row["source_bindings"]["media"][input_field]
    data_root = Path(str(contract["data_root"]))
    expected_path = _safe_video_path(data_root, str(input_row[input_field]))
    if (
        media.get("relative_path") != input_row[input_field]
        or media.get("sha256") is None
        or media.get("bytes") is None
    ):
        raise ValueError("candidate input media binding differs")
    if (
        expected_path.is_symlink()
        or not expected_path.is_file()
        or expected_path.stat().st_size != media["bytes"]
        or _file_digest(expected_path) != media["sha256"]
    ):
        raise ValueError(f"candidate media bytes changed: {expected_path}")
    if not isinstance(record, Mapping):
        raise ValueError(f"candidate cache row lacks {side}")
    track_valid = bool(arrays[f"{side}_track_valid"][array_index])
    camera_valid = bool(arrays[f"{side}_camera_valid"][array_index])
    crossfit_valid = bool(
        arrays[f"{side}_camera_crossfit_valid"][array_index]
    )
    status = record.get("status")
    expected_status = (
        "camera_ready"
        if camera_valid
        else "track_only"
        if track_valid
        else "failed"
    )
    expected_fields = set(_SIDE_COMMON_FIELDS)
    if record.get("failure_stage") != "decode":
        expected_fields.add("decode")
    if track_valid:
        expected_fields.add("tracker")
    if camera_valid:
        expected_fields.add("camera_crossfit")
    if (
        set(record) != expected_fields
        or status != expected_status
        or record.get("track_valid") is not track_valid
        or record.get("camera_valid") is not camera_valid
        or record.get("resolved_path") != str(expected_path)
        or record.get("video_sha256") != media["sha256"]
    ):
        raise ValueError(f"candidate cache {side} state/media differs")
    if status == "failed":
        if (
            record.get("failure_stage") not in {"decode", "tracking"}
            or type(record.get("failure_reason")) is not str
            or not record["failure_reason"]
            or type(record.get("failure_message")) is not str
            or not record["failure_message"]
        ):
            raise ValueError(f"candidate cache {side} failure differs")
    elif status == "track_only":
        if (
            record.get("failure_stage") != "camera_compensation"
            or type(record.get("failure_reason")) is not str
            or not record["failure_reason"]
            or type(record.get("failure_message")) is not str
            or not record["failure_message"]
        ):
            raise ValueError(f"candidate cache {side} camera failure differs")
    elif any(
        record.get(field) is not None
        for field in ("failure_stage", "failure_reason", "failure_message")
    ):
        raise ValueError(f"candidate cache {side} success has failure data")
    decode = record.get("decode")
    if record.get("failure_stage") == "decode":
        if decode is not None:
            raise ValueError(f"candidate cache {side} decode failure differs")
    else:
        if not isinstance(decode, Mapping):
            raise ValueError(f"candidate cache {side} decode is missing")
        legacy_cache._validate_decode_record(
            decode,
            arrays=arrays,
            side=side,
            array_index=array_index,
            bind_arrays=track_valid,
        )
    if track_valid:
        tracker = record.get("tracker")
        tracker_contract = contract["tracker"]
        expected_provenance = {
            "checkpoint": tracker_contract["checkpoint"],
            "grid_size": tracker_contract["grid_size"],
            "query_frame": 0,
            "backward_tracking": False,
            "device": f"cuda:{int(input_row['_cache_shard_rank'])}",
        }
        if (
            not isinstance(tracker, Mapping)
            or set(tracker) != {"backend", "provenance", "tracks"}
            or tracker.get("backend") != "cotracker"
            or tracker.get("tracks") != tracker_contract["track_count"]
            or tracker.get("provenance") != expected_provenance
        ):
            raise ValueError(
                f"candidate cache {side} tracker provenance differs"
            )
    if camera_valid:
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
        camera = record.get("camera_crossfit")
        if not isinstance(camera, Mapping) or dict(camera) != expected_camera:
            raise ValueError(f"candidate cache {side} crossfit differs")
    return track_valid, camera_valid


def _validate_row_binding(
    row: Mapping[str, Any],
    *,
    input_row: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    array_index: int,
    contract: Mapping[str, Any],
    final: bool,
) -> None:
    expected_fields = set(_ROW_FIELDS)
    if final:
        expected_fields.add("final_array_index")
    if set(row) != expected_fields:
        raise ValueError("candidate cache row fields differ")
    input_index = row.get("input_index")
    if type(input_index) is not int:
        raise ValueError("candidate cache input index is not an integer")
    shard_rank = input_index % FINAL_WORLD_SIZE
    if (
        row.get("schema_version") != ROW_SCHEMA
        or row.get("shard_rank") != shard_rank
        or row.get("world_size") != FINAL_WORLD_SIZE
        or row.get("shard_array_index")
        != input_index // FINAL_WORLD_SIZE
        or (final and row.get("final_array_index") != input_index)
        or row.get("iid") != input_row["iid"]
        or row.get("input_row_sha256") != _object_digest(input_row)
        or row.get("input_artifact_digest")
        != contract["input_binding"]["artifact_digest"]
        or any(row.get(field) is not False for field in SAFETY_FIELDS)
    ):
        raise ValueError("candidate cache row input/safety binding differs")
    if "label" in row or "positive" in row or "r5_pilot_label" in row:
        raise ValueError("candidate cache row materializes a label")
    validation_input = dict(input_row)
    validation_input["_cache_shard_rank"] = shard_rank
    source_flags = _validate_side_record(
        row["source"],
        side="source",
        input_row=validation_input,
        arrays=arrays,
        array_index=array_index,
        contract=contract,
    )
    target_flags = _validate_side_record(
        row["target"],
        side="target",
        input_row=validation_input,
        arrays=arrays,
        array_index=array_index,
        contract=contract,
    )
    if row.get("paired_track_valid") is not (
        source_flags[0] and target_flags[0]
    ):
        raise ValueError("candidate cache paired track flag differs")
    if row.get("paired_camera_valid") is not (
        source_flags[1] and target_flags[1]
    ):
        raise ValueError("candidate cache paired camera flag differs")


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        _canonical_json(row).encode("utf-8") + b"\n" for row in rows
    )


def _load_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or raw != _json_bytes(value):
        raise ValueError(f"{path} is not canonical pretty JSON")
    return value


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("candidate cache manifest lacks terminal LF")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line:
            raise ValueError(
                f"candidate cache manifest line {line_number} is blank"
            )
        value = json.loads(line)
        if (
            not isinstance(value, dict)
            or line.decode("utf-8") != _canonical_json(value)
        ):
            raise ValueError(
                f"candidate cache manifest line {line_number} is not canonical"
            )
        rows.append(value)
    return rows


def _write_regular(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)


def _write_npz(
    path: Path,
    arrays: Mapping[str, np.ndarray],
) -> None:
    if "positive" in arrays:
        raise ValueError("candidate archive cannot contain positive")
    with path.open("xb") as handle:
        np.savez_compressed(
            handle,
            **{
                name: np.asarray(arrays[name])
                for name in sorted(arrays)
            },
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)


def _validate_closure(directory: Path) -> dict[str, dict[str, Any]]:
    unresolved = directory.expanduser()
    if unresolved.is_symlink() or not unresolved.is_dir():
        raise FileNotFoundError(
            f"candidate cache commit is not a real directory: {unresolved}"
        )
    root = unresolved.resolve(strict=True)
    artifact_permissions.assert_sealed_tree(root)
    actual = {entry.name for entry in root.iterdir()}
    if actual != set(OUTPUT_NAMES):
        raise ValueError("candidate cache artifact set differs")
    records: dict[str, dict[str, Any]] = {}
    for name in OUTPUT_NAMES:
        path = root / name
        metadata = path.stat()
        if (
            path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o444
        ):
            raise ValueError(
                f"candidate cache artifact mode/type differs: {path}"
            )
        records[name] = {
            "sha256": _file_digest(path),
            "bytes": int(metadata.st_size),
            "mode_octal": "0444",
        }
    return records


def _summary_payload(
    *,
    rows: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    contract: Mapping[str, Any],
    final: bool,
) -> dict[str, Any]:
    indices = [int(row["input_index"]) for row in rows]
    return {
        "schema_version":
            FINAL_SUMMARY_SCHEMA if final else SHARD_SUMMARY_SCHEMA,
        "status": "complete",
        "rows": len(rows),
        "input_rows": int(contract["input_binding"]["rows"]),
        "track_count": int(contract["tracker"]["track_count"]),
        "contract": dict(contract),
        "contract_sha256": _object_digest(contract),
        "input_indices_sha256": _object_digest(indices),
        "array_sha256": {
            name: _array_digest(np.asarray(value))
            for name, value in sorted(arrays.items())
        },
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
        "label_semantics": dict(LABEL_SEMANTICS),
        **_safety_flags(),
    }


def _done_payload(
    *,
    rows: int,
    input_rows: int,
    contract_sha256: str,
    input_done_sha256: str,
    payload_files: Mapping[str, Mapping[str, Any]],
    final: bool,
) -> dict[str, Any]:
    core = {
        "schema_version": FINAL_DONE_SCHEMA if final else SHARD_DONE_SCHEMA,
        "status": "complete",
        "committed": True,
        "rows": rows,
        "input_rows": input_rows,
        "contract_sha256": contract_sha256,
        "input_done_sha256": input_done_sha256,
        "artifact_closure": list(OUTPUT_NAMES),
        "permission_contract": artifact_permissions.permission_contract(),
        "payload_files": {
            name: dict(payload_files[name]) for name in PAYLOAD_NAMES
        },
        "label_semantics": dict(LABEL_SEMANTICS),
        **_safety_flags(),
    }
    return {
        **core,
        "artifact_digest": _object_digest(core["payload_files"]),
    }


def _commit(
    *,
    directory: Path,
    rows: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    contract: Mapping[str, Any],
    input_commit: _InputCommit,
    final: bool,
) -> dict[str, Any]:
    _validate_contract_semantics(
        contract,
        input_commit=input_commit,
        final=final,
    )
    track_count = int(contract["tracker"]["track_count"])
    _validate_array_contract(
        arrays,
        rows=len(rows),
        track_count=track_count,
    )
    canonical_rows = [dict(row) for row in rows]
    indices = [int(row["input_index"]) for row in canonical_rows]
    if indices != np.asarray(arrays["input_indices"]).tolist():
        raise ValueError("candidate cache manifest/archive indices differ")
    expected_indices = (
        list(range(len(input_commit.rows)))
        if final
        else list(
            range(
                int(contract["rank"]),
                len(input_commit.rows),
                FINAL_WORLD_SIZE,
            )
        )
    )
    if indices != expected_indices:
        raise ValueError("candidate cache modulo coverage differs")
    for array_index, row in enumerate(canonical_rows):
        _validate_row_binding(
            row,
            input_row=input_commit.rows[int(row["input_index"])],
            arrays=arrays,
            array_index=array_index,
            contract=contract,
            final=final,
        )
    target = directory.expanduser()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
    )
    published = False
    try:
        paths = _artifact_paths(stage)
        _write_npz(paths["archive"], arrays)
        _write_regular(paths["manifest"], _jsonl_bytes(canonical_rows))
        summary = _summary_payload(
            rows=canonical_rows,
            arrays=arrays,
            contract=contract,
            final=final,
        )
        _write_regular(paths["summary"], _json_bytes(summary))
        payload_files = {
            name: {
                "sha256": _file_digest(stage / name),
                "bytes": int((stage / name).stat().st_size),
                "mode_octal": "0444",
            }
            for name in PAYLOAD_NAMES
        }
        done = _done_payload(
            rows=len(canonical_rows),
            input_rows=len(input_commit.rows),
            contract_sha256=summary["contract_sha256"],
            input_done_sha256=input_commit.done_sha256,
            payload_files=payload_files,
            final=final,
        )
        _write_regular(paths["done"], _json_bytes(done))
        directory_fd = os.open(stage, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _assert_snapshot(input_commit)
        artifact_permissions.seal_staging_tree(
            stage,
            leave_root_writable=True,
        )
        artifact_permissions.assert_sealed_tree(
            stage,
            allow_writable_root=True,
        )
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        os.rename(stage, target)
        published = True
        artifact_permissions.seal_published_root(target)
        parent_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        _assert_snapshot(input_commit)
        _validate_closure(target)
        return done
    finally:
        if not published and stage.exists():
            artifact_permissions.remove_staging_tree(stage)


def _validate_summary_done(
    *,
    summary: Mapping[str, Any],
    done: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    contract: Mapping[str, Any],
    closure: Mapping[str, Mapping[str, Any]],
    input_commit: _InputCommit,
    final: bool,
) -> None:
    expected_summary = _summary_payload(
        rows=rows,
        arrays=arrays,
        contract=contract,
        final=final,
    )
    if dict(summary) != expected_summary:
        raise ValueError("candidate cache summary differs from recomputation")
    expected_done = _done_payload(
        rows=len(rows),
        input_rows=len(input_commit.rows),
        contract_sha256=expected_summary["contract_sha256"],
        input_done_sha256=input_commit.done_sha256,
        payload_files={
            name: closure[name] for name in PAYLOAD_NAMES
        },
        final=final,
    )
    if dict(done) != expected_done:
        raise ValueError("candidate cache done differs from recomputation")


def validate_commit(
    directory: Path,
    *,
    input_dir: Path,
    expected_input_done_sha256: str,
    expected_contract: Mapping[str, Any] | None = None,
    final: bool = False,
    verify_source_shards: bool | None = None,
) -> dict[str, Any]:
    input_commit = _load_input_commit(
        input_dir,
        expected_done_sha256=expected_input_done_sha256,
    )
    closure = _validate_closure(directory)
    paths = _artifact_paths(directory)
    summary = _load_json(paths["summary"])
    done = _load_json(paths["done"])
    rows = _load_manifest(paths["manifest"])
    with np.load(paths["archive"], allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    contract = summary.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("candidate cache contract is missing")
    _validate_contract_semantics(
        contract,
        input_commit=input_commit,
        final=final,
    )
    if (
        expected_contract is not None
        and dict(contract) != dict(expected_contract)
    ):
        raise ValueError("candidate cache contract differs from expected")
    track_count = int(contract["tracker"]["track_count"])
    _validate_array_contract(
        arrays,
        rows=len(rows),
        track_count=track_count,
    )
    indices = [row.get("input_index") for row in rows]
    if (
        any(type(value) is not int for value in indices)
        or indices != arrays["input_indices"].tolist()
        or len(indices) != len(set(indices))
    ):
        raise ValueError("candidate cache row/archive indices differ")
    expected_indices = (
        list(range(len(input_commit.rows)))
        if final
        else list(
            range(
                int(contract["rank"]),
                len(input_commit.rows),
                FINAL_WORLD_SIZE,
            )
        )
    )
    if indices != expected_indices:
        raise ValueError("candidate cache shard/final coverage differs")
    for array_index, row in enumerate(rows):
        _validate_row_binding(
            row,
            input_row=input_commit.rows[int(row["input_index"])],
            arrays=arrays,
            array_index=array_index,
            contract=contract,
            final=final,
        )
    _validate_summary_done(
        summary=summary,
        done=done,
        rows=rows,
        arrays=arrays,
        contract=contract,
        closure=closure,
        input_commit=input_commit,
        final=final,
    )
    _validate_live_cotracker_source(contract)
    coverage = _operational_coverage(arrays)
    if final and coverage["passed"] is not True:
        raise ValueError("candidate final track-cache coverage failed")
    result = {
        "directory": Path(directory).resolve(strict=True),
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
            final_result=result,
            input_commit=input_commit,
        )
    _assert_snapshot(input_commit)
    return result


def _validate_final_source_shards(
    *,
    final_result: Mapping[str, Any],
    input_commit: _InputCommit,
) -> None:
    final_directory = Path(final_result["directory"])
    output_root = final_directory.parent
    _validate_final_outer_closure(output_root)
    _validate_live_cotracker_source(final_result["contract"])
    shards_root = output_root / "shards"
    source_results = [
        validate_commit(
            _rank_directory(output_root, rank),
            input_dir=input_commit.directory,
            expected_input_done_sha256=input_commit.done_sha256,
            final=False,
            verify_source_shards=False,
        )
        for rank in range(FINAL_WORLD_SIZE)
    ]
    final_contract = final_result["contract"]
    expected_hashes = [
        _file_digest(_rank_directory(output_root, rank) / DONE_NAME)
        for rank in range(FINAL_WORLD_SIZE)
    ]
    if final_contract["source_shard_done_sha256"] != expected_hashes:
        raise ValueError("candidate final/source shard done hashes differ")
    final_base = _base_contract(final_contract)
    for rank, source in enumerate(source_results):
        if _base_contract(source["contract"]) != final_base:
            raise ValueError(
                f"candidate final/source shard {rank} contract differs"
            )
    final_rows = final_result["rows"]
    final_arrays = final_result["arrays"]
    for rank, source in enumerate(source_results):
        indices = np.asarray(source["arrays"]["input_indices"], dtype=np.int64)
        expected = np.arange(
            rank,
            len(input_commit.rows),
            FINAL_WORLD_SIZE,
            dtype=np.int64,
        )
        if not np.array_equal(indices, expected):
            raise ValueError(f"candidate source shard {rank} coverage differs")
        for local_index, raw_index in enumerate(indices):
            input_index = int(raw_index)
            final_row = dict(final_rows[input_index])
            if final_row.pop("final_array_index", None) != input_index:
                raise ValueError("candidate final row index differs")
            if final_row != source["rows"][local_index]:
                raise ValueError("candidate final/source row differs")
            for name in final_arrays:
                if not np.array_equal(
                    final_arrays[name][input_index],
                    source["arrays"][name][local_index],
                ):
                    raise ValueError(
                        f"candidate final/source array differs: {name}"
                    )


def _expected_shard_names() -> set[str]:
    return {
        f"rank-{rank:03d}-of-{FINAL_WORLD_SIZE:03d}"
        for rank in range(FINAL_WORLD_SIZE)
    }


def _validate_live_cotracker_source(
    contract: Mapping[str, Any],
) -> None:
    tracker = contract.get("tracker")
    source = tracker.get("source") if isinstance(tracker, Mapping) else None
    if not isinstance(source, Mapping) or type(source.get("root")) is not str:
        raise ValueError("candidate CoTracker source provenance is missing")
    root = Path(source["root"])
    if legacy_cache.cotracker_source_provenance(root) != dict(source):
        raise ValueError(
            "candidate CoTracker Python source changed before finalization"
        )


def _repair_interrupted_commit_root(directory: Path) -> bool:
    """Finish only the root-mode step of an otherwise sealed commit.

    A process can be killed after the atomic rename and before the private
    ``0700`` staging root is changed to ``0555``.  Resume may repair exactly
    that state, but it never changes payload bytes or descendant modes.
    """

    unresolved = directory.expanduser()
    if unresolved.is_symlink() or not unresolved.is_dir():
        raise FileNotFoundError(unresolved)
    root = unresolved.resolve(strict=True)
    mode = stat.S_IMODE(root.lstat().st_mode)
    if mode == artifact_permissions.DIRECTORY_MODE:
        artifact_permissions.assert_sealed_tree(root)
        return False
    if mode != 0o700:
        raise ValueError(
            f"interrupted commit root has non-repairable mode {mode:04o}"
        )
    artifact_permissions.assert_sealed_tree(
        root,
        allow_writable_root=True,
    )
    artifact_permissions.seal_published_root(root)
    return True


def _quarantine_stale_staging_directories(
    output_root: Path,
) -> tuple[Path, ...]:
    """Move exact private-stage leftovers outside the artifact closure."""

    unresolved = output_root.expanduser()
    if not unresolved.exists():
        return ()
    if unresolved.is_symlink() or not unresolved.is_dir():
        raise ValueError("candidate cache output root is not a real directory")
    root = unresolved.resolve(strict=True)
    parents_and_patterns = (
        (root, re.compile(r"^\.final\..+\.tmp$")),
        (
            root / "shards",
            re.compile(
                rf"^\.rank-[0-9]{{3}}-of-{FINAL_WORLD_SIZE:03d}\..+\.tmp$"
            ),
        ),
    )
    stale: list[tuple[Path, str]] = []
    for parent, pattern in parents_and_patterns:
        if not parent.exists():
            continue
        if parent.is_symlink() or not parent.is_dir():
            raise ValueError("candidate cache stage parent is not real")
        prefix = "root" if parent == root else "shards"
        for entry in sorted(parent.iterdir(), key=lambda item: item.name):
            if pattern.fullmatch(entry.name) is None:
                continue
            if entry.is_symlink() or not entry.is_dir():
                raise ValueError(
                    "candidate cache stale-stage name is not a real directory"
                )
            stale.append((entry, prefix))
    if not stale:
        return ()
    quarantine = Path(
        tempfile.mkdtemp(
            prefix=f".{root.name}.stale-stages.",
            suffix=".quarantine",
            dir=root.parent,
        )
    )
    moved: list[Path] = []
    for source, prefix in stale:
        destination = quarantine / f"{prefix}__{source.name}"
        os.rename(source, destination)
        moved.append(destination)
    quarantine_fd = os.open(quarantine, os.O_RDONLY)
    try:
        os.fsync(quarantine_fd)
    finally:
        os.close(quarantine_fd)
    parent_fd = os.open(root.parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return tuple(moved)


def _validate_open_shard_layout(output_root: Path) -> tuple[Path, Path]:
    unresolved = output_root.expanduser()
    if unresolved.is_symlink() or not unresolved.is_dir():
        raise FileNotFoundError(
            f"candidate cache output root is not a real directory: "
            f"{unresolved}"
        )
    root = unresolved.resolve(strict=True)
    shards = root / "shards"
    if shards.is_symlink() or not shards.is_dir():
        raise FileNotFoundError(shards)
    actual_root = {entry.name for entry in root.iterdir()}
    if actual_root != {"shards"}:
        raise ValueError(
            "candidate cache pre-final output set differs"
        )
    actual_shards = {entry.name for entry in shards.iterdir()}
    if actual_shards != _expected_shard_names():
        raise ValueError("candidate cache source shard set differs")
    return root, shards


def _validate_final_outer_closure(output_root: Path) -> tuple[Path, Path]:
    unresolved = output_root.expanduser()
    if unresolved.is_symlink() or not unresolved.is_dir():
        raise FileNotFoundError(
            f"candidate cache output root is not a real directory: "
            f"{unresolved}"
        )
    root = unresolved.resolve(strict=True)
    shards = root / "shards"
    final = root / FINAL_DIR_NAME
    if (
        shards.is_symlink()
        or not shards.is_dir()
        or final.is_symlink()
        or not final.is_dir()
    ):
        raise ValueError("candidate final cache layout is not real")
    if {entry.name for entry in root.iterdir()} != {
        "shards",
        FINAL_DIR_NAME,
    }:
        raise ValueError("candidate final cache output set differs")
    if {entry.name for entry in shards.iterdir()} != _expected_shard_names():
        raise ValueError("candidate cache source shard set differs")
    artifact_permissions.assert_sealed_tree(root)
    return root, shards


def _candidate_output_row(
    *,
    input_row: Mapping[str, Any],
    input_index: int,
    array_index: int,
    rank: int,
    input_artifact_digest: str,
    source: Mapping[str, Any],
    target: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": ROW_SCHEMA,
        "input_index": input_index,
        "shard_array_index": array_index,
        "shard_rank": rank,
        "world_size": FINAL_WORLD_SIZE,
        "iid": input_row["iid"],
        "input_row_sha256": _object_digest(input_row),
        "input_artifact_digest": input_artifact_digest,
        "source": dict(source),
        "target": dict(target),
        "paired_track_valid": bool(
            source["track_valid"] and target["track_valid"]
        ),
        "paired_camera_valid": bool(
            source["camera_valid"] and target["camera_valid"]
        ),
        **_safety_flags(),
    }


def _build_rank_runtime_contract(
    *,
    input_commit: _InputCommit,
    tracker_checkpoint: Path,
    cotracker_root: Path,
    tracker_grid_size: int,
    rank: int,
    world_size: int,
    local_rank: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    determinism = legacy_cache._configure_determinism(
        seed,
        local_rank=local_rank,
    )
    runtime = legacy_cache.runtime_provenance(
        local_rank=local_rank,
        determinism=determinism,
    )
    source_provenance = legacy_cache.cotracker_source_provenance(
        cotracker_root
    )
    contract = build_cache_contract(
        input_dir=input_commit.directory,
        expected_input_done_sha256=input_commit.done_sha256,
        data_root=input_commit.data_root,
        tracker_checkpoint=tracker_checkpoint,
        cotracker_provenance=source_provenance,
        runtime=runtime,
        tracker_grid_size=tracker_grid_size,
        rank=rank,
        world_size=world_size,
        device=f"cuda:{local_rank}",
        seed=seed,
    )
    return contract, runtime, source_provenance


def runtime_preflight(
    *,
    input_dir: Path,
    expected_input_done_sha256: str,
    data_root: Path,
    output_root: Path,
    tracker_checkpoint: Path,
    cotracker_root: Path,
    source_snapshot: Path,
    tracker_grid_size: int = 16,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Build one real rank-zero GPU contract before starting torchrun.

    The ordinary CPU preflight intentionally remains GPU-free.  This probe is
    run inside the allocated node and exercises determinism, runtime
    provenance, CoTracker source provenance, checkpoint hashing, and the
    complete strict contract exactly once.  A contract incompatibility thus
    fails before eight worker processes are launched.
    """

    input_commit = _load_input_commit(
        input_dir,
        expected_done_sha256=expected_input_done_sha256,
        expected_data_root=data_root,
    )
    checkpoint = tracker_checkpoint.expanduser().resolve(strict=True)
    cotracker = cotracker_root.expanduser().resolve(strict=True)
    snapshot = source_snapshot.expanduser().resolve(strict=True)
    output = legacy_cache.validate_output_root(
        output_root=output_root,
        input_manifest=input_commit.paths[candidate_manifest.MANIFEST_NAME],
        data_root=input_commit.data_root,
        tracker_checkpoint=checkpoint,
        cotracker_root=cotracker,
        source_snapshot=snapshot,
    )
    contract, runtime, source_provenance = _build_rank_runtime_contract(
        input_commit=input_commit,
        tracker_checkpoint=checkpoint,
        cotracker_root=cotracker,
        tracker_grid_size=tracker_grid_size,
        rank=0,
        world_size=FINAL_WORLD_SIZE,
        local_rank=0,
        seed=seed,
    )
    return {
        "schema_version":
            "motive-r7-candidate-temporal-runtime-preflight-v1",
        "status": "ready",
        "rank": 0,
        "world_size": FINAL_WORLD_SIZE,
        "rows": len(input_commit.rows),
        "output_root": str(output),
        "input_done_sha256": input_commit.done_sha256,
        "tracker_grid_size": tracker_grid_size,
        "tracker_checkpoint_sha256":
            contract["tracker"]["checkpoint_sha256"],
        "cotracker_source_bundle_sha256":
            source_provenance["python_source_bundle_sha256"],
        "runtime_sha256": _object_digest(runtime),
        "contract_sha256": _object_digest(contract),
        "implementation_sha256": dict(contract["implementation_sha256"]),
        "runtime": dict(runtime),
        **_safety_flags(),
    }


def extract_rank(
    *,
    input_dir: Path,
    expected_input_done_sha256: str,
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
        or type(rank) is not int
        or type(local_rank) is not int
        or not 0 <= rank < world_size
        or local_rank != rank
    ):
        raise ValueError(
            "single-node candidate cache requires rank=local_rank in [0,8)"
        )
    input_commit = _load_input_commit(
        input_dir,
        expected_done_sha256=expected_input_done_sha256,
        expected_data_root=data_root,
    )
    checkpoint = tracker_checkpoint.expanduser().resolve(strict=True)
    cotracker = cotracker_root.expanduser().resolve(strict=True)
    snapshot = (
        None
        if source_snapshot is None
        else source_snapshot.expanduser().resolve(strict=True)
    )
    output = legacy_cache.validate_output_root(
        output_root=output_root,
        input_manifest=input_commit.paths[candidate_manifest.MANIFEST_NAME],
        data_root=input_commit.data_root,
        tracker_checkpoint=checkpoint,
        cotracker_root=cotracker,
        source_snapshot=snapshot,
    )
    contract, _runtime, _source_provenance = (
        _build_rank_runtime_contract(
            input_commit=input_commit,
            tracker_checkpoint=checkpoint,
            cotracker_root=cotracker,
            tracker_grid_size=tracker_grid_size,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            seed=seed,
        )
    )
    directory = _rank_directory(output, rank)
    if directory.exists() or directory.is_symlink():
        if not resume:
            raise FileExistsError(directory)
        _repair_interrupted_commit_root(directory)
        return validate_commit(
            directory,
            input_dir=input_commit.directory,
            expected_input_done_sha256=input_commit.done_sha256,
            expected_contract=contract,
            final=False,
        )["done"]
    selected = [
        (index, row)
        for index, row in enumerate(input_commit.rows)
        if index % FINAL_WORLD_SIZE == rank
    ]
    track_count = tracker_grid_size**2
    arrays = _empty_arrays(len(selected), track_count=track_count)
    tracker = LazyCoTrackerAdapter(
        checkpoint=checkpoint,
        device=f"cuda:{local_rank}",
        grid_size=tracker_grid_size,
        backward_tracking=False,
    )
    camera_config = TemporalTeacherConfig()
    output_rows: list[dict[str, Any]] = []
    for array_index, (input_index, input_row) in enumerate(selected):
        arrays["input_indices"][array_index] = input_index
        results: dict[str, dict[str, Any]] = {}
        for side, field in (
            ("source", "src_video"),
            ("target", "tgt_video"),
        ):
            path = _safe_video_path(
                input_commit.data_root,
                str(input_row[field]),
            )
            results[side] = legacy_cache._extract_side(
                path=path,
                side=side,
                array_index=array_index,
                arrays=arrays,
                tracker=tracker,
                track_count=track_count,
                camera_config=camera_config,
            )
        output_rows.append(
            _candidate_output_row(
                input_row=input_row,
                input_index=input_index,
                array_index=array_index,
                rank=rank,
                input_artifact_digest=input_commit.artifact_digest,
                source=results["source"],
                target=results["target"],
            )
        )
    done = _commit(
        directory=directory,
        rows=output_rows,
        arrays=arrays,
        contract=contract,
        input_commit=input_commit,
        final=False,
    )
    validate_commit(
        directory,
        input_dir=input_commit.directory,
        expected_input_done_sha256=input_commit.done_sha256,
        expected_contract=contract,
        final=False,
    )
    return done


def _seal_and_validate_final_outer(
    *,
    output_root: Path,
    input_commit: _InputCommit,
    final_contract: Mapping[str, Any],
) -> dict[str, Any]:
    root = output_root.expanduser().resolve(strict=True)
    artifact_permissions.seal_staging_tree(root)
    artifact_permissions.assert_sealed_tree(root)
    output_fd = os.open(root, os.O_RDONLY)
    try:
        os.fsync(output_fd)
    finally:
        os.close(output_fd)
    parent_fd = os.open(root.parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    _assert_snapshot(input_commit)
    return validate_commit(
        root / FINAL_DIR_NAME,
        input_dir=input_commit.directory,
        expected_input_done_sha256=input_commit.done_sha256,
        expected_contract=final_contract,
        final=True,
    )


def finalize_shards(
    *,
    input_dir: Path,
    expected_input_done_sha256: str,
    output_root: Path,
    resume: bool = False,
) -> dict[str, Any]:
    input_commit = _load_input_commit(
        input_dir,
        expected_done_sha256=expected_input_done_sha256,
    )
    _quarantine_stale_staging_directories(output_root)
    final_directory = output_root.expanduser() / FINAL_DIR_NAME
    if final_directory.exists() or final_directory.is_symlink():
        if not resume:
            raise FileExistsError(final_directory)
        _repair_interrupted_commit_root(final_directory)
        provisional = validate_commit(
            final_directory,
            input_dir=input_commit.directory,
            expected_input_done_sha256=input_commit.done_sha256,
            final=True,
            verify_source_shards=False,
        )
        repaired = _seal_and_validate_final_outer(
            output_root=output_root,
            input_commit=input_commit,
            final_contract=provisional["contract"],
        )
        return repaired["done"]
    shard_results = [
        validate_commit(
            _rank_directory(output_root, rank),
            input_dir=input_commit.directory,
            expected_input_done_sha256=input_commit.done_sha256,
            final=False,
            verify_source_shards=False,
        )
        for rank in range(FINAL_WORLD_SIZE)
    ]
    resolved_output_root, _shards_root = _validate_open_shard_layout(
        output_root
    )
    _validate_live_cotracker_source(shard_results[0]["contract"])
    base_contracts = [
        _base_contract(result["contract"]) for result in shard_results
    ]
    if any(value != base_contracts[0] for value in base_contracts[1:]):
        raise ValueError("candidate track-cache shard contracts differ")
    track_count = int(base_contracts[0]["tracker"]["track_count"])
    arrays = _empty_arrays(len(input_commit.rows), track_count=track_count)
    rows_by_index: dict[int, dict[str, Any]] = {}
    for rank, result in enumerate(shard_results):
        indices = np.asarray(result["arrays"]["input_indices"], dtype=np.int64)
        expected = np.arange(
            rank,
            len(input_commit.rows),
            FINAL_WORLD_SIZE,
            dtype=np.int64,
        )
        if not np.array_equal(indices, expected):
            raise ValueError(
                f"candidate track-cache shard {rank} coverage differs"
            )
        for local_index, raw_index in enumerate(indices):
            input_index = int(raw_index)
            if input_index in rows_by_index:
                raise ValueError("candidate cache input index is duplicated")
            row = dict(result["rows"][local_index])
            row["final_array_index"] = input_index
            rows_by_index[input_index] = row
            for name in arrays:
                arrays[name][input_index] = result["arrays"][name][local_index]
    if set(rows_by_index) != set(range(len(input_commit.rows))):
        raise ValueError("candidate final cache coverage is incomplete")
    coverage = _operational_coverage(arrays)
    if coverage["passed"] is not True:
        raise ValueError(
            "refusing candidate final cache: operational coverage failed; "
            + _canonical_json(coverage)
        )
    final_contract = dict(base_contracts[0])
    final_contract.update(
        {
            "rank": "merged",
            "device": "eight-shard-final",
            "merge_world_size": FINAL_WORLD_SIZE,
            "merge_policy": MERGE_POLICY,
            "source_shard_done_sha256": [
                _file_digest(_rank_directory(output_root, rank) / DONE_NAME)
                for rank in range(FINAL_WORLD_SIZE)
            ],
        }
    )
    rows = [
        rows_by_index[index] for index in range(len(input_commit.rows))
    ]
    done = _commit(
        directory=final_directory,
        rows=rows,
        arrays=arrays,
        contract=final_contract,
        input_commit=input_commit,
        final=True,
    )
    # Validate the newly published final commit before making its enclosing
    # output tree immutable.  Source shards were each validated above; the
    # final full validator runs after sealing and checks the complete closure.
    validate_commit(
        final_directory,
        input_dir=input_commit.directory,
        expected_input_done_sha256=input_commit.done_sha256,
        expected_contract=final_contract,
        final=True,
        verify_source_shards=False,
    )
    validated = _seal_and_validate_final_outer(
        output_root=resolved_output_root,
        input_commit=input_commit,
        final_contract=final_contract,
    )
    if validated["done"] != done:
        raise RuntimeError("candidate final done changed during outer sealing")
    return validated["done"]


def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-input-done-sha256",
        required=True,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    _add_input_arguments(preflight_parser)
    preflight_parser.add_argument("--data-root", type=Path, required=True)
    preflight_parser.add_argument("--output-dir", type=Path, required=True)
    preflight_parser.add_argument(
        "--cotracker-checkpoint",
        type=Path,
        required=True,
    )
    preflight_parser.add_argument(
        "--cotracker-root",
        type=Path,
        required=True,
    )
    preflight_parser.add_argument(
        "--source-snapshot",
        type=Path,
        required=True,
    )
    runtime_parser = subparsers.add_parser("runtime-preflight")
    _add_input_arguments(runtime_parser)
    runtime_parser.add_argument("--data-root", type=Path, required=True)
    runtime_parser.add_argument("--output-dir", type=Path, required=True)
    runtime_parser.add_argument(
        "--cotracker-checkpoint",
        type=Path,
        required=True,
    )
    runtime_parser.add_argument(
        "--cotracker-root",
        type=Path,
        required=True,
    )
    runtime_parser.add_argument(
        "--source-snapshot",
        type=Path,
        required=True,
    )
    runtime_parser.add_argument(
        "--tracker-grid-size",
        type=int,
        default=16,
    )
    runtime_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    extract_parser = subparsers.add_parser("extract")
    _add_input_arguments(extract_parser)
    extract_parser.add_argument("--data-root", type=Path, required=True)
    extract_parser.add_argument("--output-dir", type=Path, required=True)
    extract_parser.add_argument(
        "--cotracker-checkpoint",
        type=Path,
        required=True,
    )
    extract_parser.add_argument(
        "--cotracker-root",
        type=Path,
        required=True,
    )
    extract_parser.add_argument(
        "--source-snapshot",
        type=Path,
        required=True,
    )
    extract_parser.add_argument("--tracker-grid-size", type=int, default=16)
    extract_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    extract_parser.add_argument("--rank", type=int)
    extract_parser.add_argument("--world-size", type=int)
    extract_parser.add_argument("--local-rank", type=int)
    extract_parser.add_argument("--resume", action="store_true")
    finalize_parser = subparsers.add_parser("finalize")
    _add_input_arguments(finalize_parser)
    finalize_parser.add_argument("--output-dir", type=Path, required=True)
    finalize_parser.add_argument("--resume", action="store_true")
    validate_parser = subparsers.add_parser("validate")
    _add_input_arguments(validate_parser)
    validate_parser.add_argument("--output-dir", type=Path, required=True)
    validate_parser.add_argument("--final", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "preflight":
        source = _load_input_commit(
            args.input_dir,
            expected_done_sha256=args.expected_input_done_sha256,
            expected_data_root=args.data_root,
        )
        checkpoint = args.cotracker_checkpoint.expanduser().resolve(strict=True)
        output = legacy_cache.validate_output_root(
            output_root=args.output_dir,
            input_manifest=source.paths[candidate_manifest.MANIFEST_NAME],
            data_root=source.data_root,
            tracker_checkpoint=checkpoint,
            cotracker_root=args.cotracker_root,
            source_snapshot=args.source_snapshot,
        )
        provenance = legacy_cache.cotracker_source_provenance(
            args.cotracker_root
        )
        result = {
            "schema_version":
                "motive-r7-candidate-temporal-track-cache-preflight-v1",
            "status": "ready",
            "rows": len(source.rows),
            "input_done_sha256": source.done_sha256,
            "output_root": str(output),
            "tracker_checkpoint_sha256": _file_digest(checkpoint),
            "cotracker_source_bundle_sha256":
                provenance["python_source_bundle_sha256"],
            "label_semantics": dict(LABEL_SEMANTICS),
            **_safety_flags(),
        }
    elif args.command == "runtime-preflight":
        result = runtime_preflight(
            input_dir=args.input_dir,
            expected_input_done_sha256=args.expected_input_done_sha256,
            data_root=args.data_root,
            output_root=args.output_dir,
            tracker_checkpoint=args.cotracker_checkpoint,
            cotracker_root=args.cotracker_root,
            source_snapshot=args.source_snapshot,
            tracker_grid_size=args.tracker_grid_size,
            seed=args.seed,
        )
    elif args.command == "extract":
        rank, world_size, local_rank = resolve_torchrun_coordinates(
            rank=args.rank,
            world_size=args.world_size,
            local_rank=args.local_rank,
        )
        result = extract_rank(
            input_dir=args.input_dir,
            expected_input_done_sha256=args.expected_input_done_sha256,
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
            input_dir=args.input_dir,
            expected_input_done_sha256=args.expected_input_done_sha256,
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
            input_dir=args.input_dir,
            expected_input_done_sha256=args.expected_input_done_sha256,
            final=args.final,
        )["done"]
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ARCHIVE_NAME",
    "CACHE_SCHEMA",
    "DONE_NAME",
    "FINAL_DIR_NAME",
    "FINAL_DONE_SCHEMA",
    "FINAL_SUMMARY_SCHEMA",
    "MANIFEST_NAME",
    "ROW_SCHEMA",
    "SHARD_DONE_SCHEMA",
    "SHARD_SUMMARY_SCHEMA",
    "SUMMARY_NAME",
    "build_cache_contract",
    "extract_rank",
    "finalize_shards",
    "main",
    "runtime_preflight",
    "validate_commit",
]
