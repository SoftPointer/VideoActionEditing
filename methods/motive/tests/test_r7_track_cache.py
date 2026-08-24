from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from motive import r7_track_cache as track_cache
from motive.r7_preflight_extract import (
    DecodedVideo,
    GlobalExtractionError,
    R7_VIDEO_SAMPLING,
    _file_digest,
    _object_digest,
)
from motive.r7_temporal_teacher import (
    TemporalTeacherError,
    TrackObservations,
)
from motive.r7_track_cache import (
    COHORT_ID,
    EXPECTED_COHORT_ROWS,
    FINAL_WORLD_SIZE,
    FORMAL_STATUS,
    R7_TRACK_CACHE_ROW_SCHEMA,
    _commit,
    _empty_arrays,
    _extract_side,
    _rank_directory,
    _validate_array_contract,
    build_cache_contract,
    cotracker_source_provenance,
    finalize_shards,
    validate_commit,
    validate_output_root,
)


SEED = 23


def _input_row(index: int) -> dict[str, object]:
    positive = index % 2 == 0
    return {
        "iid": f"iid-{index:03d}",
        "src_video": f"videos/{index}/source.mp4",
        "tgt_video": f"videos/{index}/target.mp4",
        "input_digest": f"input-{index}",
        "prompt": f"action {index}",
        "r5_pilot_label": {
            "class": "positive" if positive else "negative",
            "negative_type": None if positive else "static",
            "action_signature": f"action-{index % 3}",
        },
    }


def _write_manifest(
    path: Path,
    count: int = EXPECTED_COHORT_ROWS,
) -> list[dict[str, object]]:
    rows = [_input_row(index) for index in range(count)]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            for field in ("src_video", "tgt_video"):
                video = path.parent / str(row[field])
                video.parent.mkdir(parents=True, exist_ok=True)
                video.write_bytes(b"video")
    return rows


def _ensure_cotracker_repo(root: Path) -> Path:
    repository = root / "co-tracker"
    if (repository / ".git").is_dir():
        return repository
    package = repository / "cotracker"
    package.mkdir(parents=True)
    (package / "predictor.py").write_text("VALUE = 1\n")
    (package / "model.py").write_text("MODEL = 2\n")
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@x"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "add", "cotracker"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-qm", "fixture"],
        check=True,
    )
    return repository


def _fake_runtime(seed: int = SEED) -> dict[str, object]:
    return {
        "schema_version": "motive-r7-runtime-v1",
        "python_version": "3.11.9",
        "python_implementation": "CPython",
        "python_executable": "/opt/test/bin/python",
        "platform": "Linux-test",
        "numpy_version": "2.0.0",
        "opencv_version": "4.10.0",
        "torch_version": "2.5.0",
        "torch_hip_version": "6.2",
        "torch_build_config_sha256": _object_digest("test-build"),
        "visible_device_count": FINAL_WORLD_SIZE,
        "device_type": "cuda-hip",
        "device_name": "AMD Instinct MI210",
        "device_capability": [9, 0],
        "device_total_memory": 64 * 1024**3,
        "determinism": {
            "schema_version": "motive-r7-determinism-v1",
            "seed": seed,
            "rank_seed_policy":
                "identical-base-seed-on-all-eight-ranks-v1",
            "python_random_seeded": True,
            "numpy_seeded": True,
            "torch_cpu_seeded": True,
            "torch_all_visible_devices_seeded": True,
            "torch_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "python_hash_seed": str(seed),
        },
    }


def _contract(
    *,
    manifest: Path,
    root: Path,
    checkpoint: Path,
    rank: int,
) -> dict[str, object]:
    return build_cache_contract(
        input_manifest=manifest,
        data_root=root,
        tracker_checkpoint=checkpoint,
        cotracker_provenance=cotracker_source_provenance(
            _ensure_cotracker_repo(root)
        ),
        runtime=_fake_runtime(),
        tracker_grid_size=2,
        rank=rank,
        world_size=FINAL_WORLD_SIZE,
        device=f"cuda:{rank}",
        seed=SEED,
    )


def _valid_arrays(
    input_rows: list[dict[str, object]],
    *,
    track_count: int,
    input_indices: list[int],
) -> dict[str, np.ndarray]:
    count = len(input_indices)
    arrays = _empty_arrays(count, track_count=track_count)
    arrays["input_indices"][:] = input_indices
    arrays["positive"][:] = [
        input_rows[index]["r5_pilot_label"]["class"] == "positive"
        for index in input_indices
    ]
    identity = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    times = np.arange(32, dtype=np.float64) / 25.0
    frame_indices = np.arange(32, dtype=np.int64)
    tracks = np.zeros((32, track_count, 2), dtype=np.float32)
    tracks[..., 0] = np.linspace(0.1, 0.9, track_count)[None]
    tracks[..., 1] = np.linspace(0.2, 0.8, track_count)[None]
    for side in ("source", "target"):
        arrays[f"{side}_track_valid"][:] = True
        arrays[f"{side}_camera_valid"][:] = True
        arrays[f"{side}_camera_crossfit_valid"][:] = True
        arrays[f"{side}_normalized_tracks"][:] = tracks
        arrays[f"{side}_stabilized_tracks"][:] = tracks
        arrays[f"{side}_visibility"][:] = 1.0
        arrays[f"{side}_frame_times"][:] = times
        arrays[f"{side}_source_frame_indices"][:] = frame_indices
        arrays[f"{side}_resized_size"][:] = [100, 200]
        arrays[f"{side}_source_fps"][:] = 25.0
        arrays[f"{side}_transition_affines"][:] = identity
        arrays[f"{side}_cumulative_affines"][:] = identity
        arrays[f"{side}_camera_crossfit_raw_median"][:] = 0.01
        arrays[f"{side}_camera_crossfit_residual_median"][:] = 0.002
        arrays[f"{side}_camera_crossfit_residual_reduction"][:] = 0.8
        arrays[f"{side}_background_residual_reduction"][:] = 0.7
    return arrays


def _camera_record(
    *,
    input_row: dict[str, object],
    input_index: int,
    rank: int,
    root: Path,
    checkpoint: Path,
    side: str,
) -> dict[str, object]:
    field = "src_video" if side == "source" else "tgt_video"
    return {
        "status": "camera_ready",
        "track_valid": True,
        "camera_valid": True,
        "failure_stage": None,
        "failure_reason": None,
        "failure_message": None,
        "resolved_path": str((root / str(input_row[field])).resolve()),
        "video_sha256": _file_digest(
            root / str(input_row[field])
        ),
        "decode": {
            "sampling_version": R7_VIDEO_SAMPLING,
            "decoded_frames": 32,
            "source_frame_indices": list(range(32)),
            "source_fps": 25.0,
            "source_frame_count": 32,
            "source_size": [100, 200],
            "resized_size": [100, 200],
        },
        "tracker": {
            "backend": "cotracker",
            "provenance": {
                "checkpoint": str(checkpoint.resolve()),
                "grid_size": 2,
                "query_frame": 0,
                "backward_tracking": False,
                "device": f"cuda:{rank}",
            },
            "tracks": 4,
        },
        "camera_crossfit": {
            "valid": True,
            "raw_median": float(np.float32(0.01)),
            "residual_median": float(np.float32(0.002)),
            "residual_reduction": float(np.float32(0.8)),
        },
    }


def _output_row(
    input_row: dict[str, object],
    *,
    input_index: int,
    local_index: int,
    rank: int,
    root: Path,
    checkpoint: Path,
) -> dict[str, object]:
    pilot = input_row["r5_pilot_label"]
    positive = pilot["class"] == "positive"
    return {
        "schema_version": R7_TRACK_CACHE_ROW_SCHEMA,
        "input_index": input_index,
        "shard_array_index": local_index,
        "shard_rank": rank,
        "world_size": FINAL_WORLD_SIZE,
        "iid": input_row["iid"],
        "input_row": dict(input_row),
        "input_row_sha256": _object_digest(input_row),
        "input_digest": input_row.get("input_digest"),
        "prompt": input_row.get("prompt"),
        "label_type": pilot["class"],
        "negative_type": pilot.get("negative_type"),
        "positive": positive,
        "action_signature": pilot.get("action_signature"),
        "source": _camera_record(
            input_row=input_row,
            input_index=input_index,
            rank=rank,
            root=root,
            checkpoint=checkpoint,
            side="source",
        ),
        "target": _camera_record(
            input_row=input_row,
            input_index=input_index,
            rank=rank,
            root=root,
            checkpoint=checkpoint,
            side="target",
        ),
        "paired_track_valid": True,
        "paired_camera_valid": True,
    }


def _shard_fixture(
    *,
    input_rows: list[dict[str, object]],
    rank: int,
    root: Path,
    checkpoint: Path,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    selected = list(range(rank, len(input_rows), FINAL_WORLD_SIZE))
    arrays = _valid_arrays(
        input_rows,
        track_count=4,
        input_indices=selected,
    )
    rows = [
        _output_row(
            input_rows[input_index],
            input_index=input_index,
            local_index=local_index,
            rank=rank,
            root=root,
            checkpoint=checkpoint,
        )
        for local_index, input_index in enumerate(selected)
    ]
    return arrays, rows


def _invalidate_side(
    arrays: dict[str, np.ndarray],
    rows: list[dict[str, object]],
    *,
    local_index: int,
    side: str,
) -> None:
    for name, value in arrays.items():
        if name.startswith(f"{side}_"):
            value[local_index] = 0
    record = rows[local_index][side]
    rows[local_index][side] = {
        "status": "failed",
        "track_valid": False,
        "camera_valid": False,
        "failure_stage": "tracking",
        "failure_reason": "per_video_tracking_failure",
        "failure_message": "unit-test per-video tracking failure",
        "resolved_path": record["resolved_path"],
        "video_sha256": record["video_sha256"],
        "decode": record["decode"],
    }
    rows[local_index]["paired_track_valid"] = False
    rows[local_index]["paired_camera_valid"] = False


class TrackCacheArrayTests(unittest.TestCase):
    def test_contract_is_fail_closed_and_hashes_checkpoint_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "input.jsonl"
            checkpoint = root / "tracker.pth"
            _write_manifest(manifest)
            checkpoint.write_bytes(b"tracker")
            contract = _contract(
                manifest=manifest,
                root=root,
                checkpoint=checkpoint,
                rank=0,
            )
            self.assertEqual(contract["tracker"]["track_count"], 4)
            self.assertEqual(
                contract["tracker"]["checkpoint_sha256"],
                hashlib.sha256(b"tracker").hexdigest(),
            )
            self.assertEqual(contract["formal_status"], FORMAL_STATUS)
            self.assertEqual(contract["cohort_id"], COHORT_ID)
            self.assertFalse(contract["generation_authorized"])

    def test_array_validation_geometry_metrics_and_time(self) -> None:
        input_rows = [_input_row(0)]
        arrays = _valid_arrays(
            input_rows,
            track_count=4,
            input_indices=[0],
        )
        _validate_array_contract(arrays, rows=1, track_count=4)
        bad = {name: value.copy() for name, value in arrays.items()}
        bad["target_visibility"][0, 0, 0] = 1.1
        with self.assertRaisesRegex(ValueError, "visibility"):
            _validate_array_contract(bad, rows=1, track_count=4)
        bad = {name: value.copy() for name, value in arrays.items()}
        bad["target_frame_times"][0, 4] += 0.01
        with self.assertRaisesRegex(ValueError, "indices / FPS"):
            _validate_array_contract(bad, rows=1, track_count=4)
        bad = {name: value.copy() for name, value in arrays.items()}
        bad["source_stabilized_tracks"][0, 2, 0, 0] += 0.1
        with self.assertRaisesRegex(ValueError, "coordinate relation"):
            _validate_array_contract(bad, rows=1, track_count=4)
        bad = {name: value.copy() for name, value in arrays.items()}
        bad["source_camera_valid"][0] = False
        bad["source_camera_crossfit_valid"][0] = False
        bad["source_stabilized_tracks"][0] = 0
        bad["source_transition_affines"][0] = 0
        bad["source_cumulative_affines"][0] = 0
        with self.assertRaisesRegex(ValueError, "nonzero when camera"):
            _validate_array_contract(bad, rows=1, track_count=4)

    def test_camera_failure_preserves_raw_tracks_only(self) -> None:
        frames = np.zeros((32, 20, 30, 3), dtype=np.uint8)
        decoded = DecodedVideo(
            frames_rgb=frames,
            frame_times=np.arange(32, dtype=np.float64) / 25.0,
            source_frame_indices=np.arange(32, dtype=np.int64),
            source_fps=25.0,
            source_frame_count=32,
            source_size=(20, 30),
            resized_size=(20, 30),
        )
        observations = TrackObservations.create(
            tracks=np.zeros((32, 4, 2), dtype=np.float32),
            visibility=np.ones((32, 4), dtype=np.float32),
            frame_times=decoded.frame_times,
            frame_size=(20, 30),
            backend="fake",
        )

        class FakeTracker:
            def track(self, frames_rgb, *, frame_times):
                return observations

        arrays = _empty_arrays(1, track_count=4)
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "video.mp4"
            video.write_bytes(b"video")
            with patch(
                "motive.r7_track_cache.decode_video_fixed_frames",
                return_value=decoded,
            ), patch(
                "motive.r7_track_cache.robust_camera_compensation",
                side_effect=TemporalTeacherError("camera_fail", "unit test"),
            ):
                record = _extract_side(
                    path=video,
                    side="target",
                    array_index=0,
                    arrays=arrays,
                    tracker=FakeTracker(),
                    track_count=4,
                    camera_config=__import__(
                        "motive.r7_temporal_teacher",
                        fromlist=["TemporalTeacherConfig"],
                    ).TemporalTeacherConfig(),
                )
        self.assertEqual(record["status"], "track_only")
        self.assertTrue(arrays["target_track_valid"][0])
        self.assertFalse(arrays["target_camera_valid"][0])
        self.assertTrue(np.all(arrays["target_stabilized_tracks"] == 0))
        _validate_array_contract(arrays, rows=1, track_count=4)

    def test_tracker_abi_error_is_global(self) -> None:
        frames = np.zeros((32, 20, 30, 3), dtype=np.uint8)
        decoded = DecodedVideo(
            frames_rgb=frames,
            frame_times=np.arange(32, dtype=np.float64) / 25.0,
            source_frame_indices=np.arange(32, dtype=np.int64),
            source_fps=25.0,
            source_frame_count=32,
            source_size=(20, 30),
            resized_size=(20, 30),
        )

        class BrokenTracker:
            def track(self, frames_rgb, *, frame_times):
                raise TemporalTeacherError(
                    "invalid_tracker_output", "broken ABI"
                )

        arrays = _empty_arrays(1, track_count=4)
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "video.mp4"
            video.write_bytes(b"video")
            with patch(
                "motive.r7_track_cache.decode_video_fixed_frames",
                return_value=decoded,
            ):
                with self.assertRaisesRegex(
                    GlobalExtractionError, "ABI contract"
                ):
                    _extract_side(
                        path=video,
                        side="target",
                        array_index=0,
                        arrays=arrays,
                        tracker=BrokenTracker(),
                        track_count=4,
                        camera_config=__import__(
                            "motive.r7_temporal_teacher",
                            fromlist=["TemporalTeacherConfig"],
                        ).TemporalTeacherConfig(),
                    )


class TrackCacheCommitTests(unittest.TestCase):
    def test_round_trip_archive_tamper_and_input_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "input.jsonl"
            checkpoint = root / "tracker.pth"
            input_rows = _write_manifest(manifest)
            checkpoint.write_bytes(b"tracker")
            arrays, rows = _shard_fixture(
                input_rows=input_rows,
                rank=0,
                root=root,
                checkpoint=checkpoint,
            )
            contract = _contract(
                manifest=manifest,
                root=root,
                checkpoint=checkpoint,
                rank=0,
            )
            output = root / "output"
            _commit(
                directory=output,
                rows=rows,
                arrays=arrays,
                contract=contract,
                input_rows=len(input_rows),
                final=False,
            )
            validated = validate_commit(
                output,
                expected_contract=contract,
                input_manifest=manifest,
            )
            self.assertEqual(validated["done"]["rows"], len(rows))
            with np.load(output / "track_cache.npz") as loaded:
                tampered = {name: loaded[name] for name in loaded.files}
            tampered["source_visibility"][0, 0, 0] = 0.5
            np.savez_compressed(output / "track_cache.npz", **tampered)
            with self.assertRaisesRegex(ValueError, "archive_sha256"):
                validate_commit(output, input_manifest=manifest)

            bad_arrays, bad_rows = _shard_fixture(
                input_rows=input_rows,
                rank=1,
                root=root,
                checkpoint=checkpoint,
            )
            bad_rows[0]["positive"] = not bad_rows[0]["positive"]
            bad_output = root / "bad-output"
            with self.assertRaisesRegex(ValueError, "positive"):
                _commit(
                    directory=bad_output,
                    rows=bad_rows,
                    arrays=bad_arrays,
                    contract=_contract(
                        manifest=manifest,
                        root=root,
                        checkpoint=checkpoint,
                        rank=1,
                    ),
                    input_rows=len(input_rows),
                    final=False,
                )
            self.assertFalse(bad_output.exists())

    def test_side_path_decode_and_contract_tampering_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "input.jsonl"
            checkpoint = root / "tracker.pth"
            input_rows = _write_manifest(manifest)
            checkpoint.write_bytes(b"tracker")
            arrays, rows = _shard_fixture(
                input_rows=input_rows,
                rank=2,
                root=root,
                checkpoint=checkpoint,
            )
            rows[0]["source"]["resolved_path"] = rows[0]["target"][
                "resolved_path"
            ]
            with self.assertRaisesRegex(ValueError, "path differs"):
                _commit(
                    directory=root / "bad-path",
                    rows=rows,
                    arrays=arrays,
                    contract=_contract(
                        manifest=manifest,
                        root=root,
                        checkpoint=checkpoint,
                        rank=2,
                    ),
                    input_rows=len(input_rows),
                    final=False,
                )
            arrays, rows = _shard_fixture(
                input_rows=input_rows,
                rank=2,
                root=root,
                checkpoint=checkpoint,
            )
            rows[0]["source"]["decode"]["source_fps"] = 30.0
            with self.assertRaisesRegex(ValueError, "decode/FPS"):
                _commit(
                    directory=root / "bad-decode",
                    rows=rows,
                    arrays=arrays,
                    contract=_contract(
                        manifest=manifest,
                        root=root,
                        checkpoint=checkpoint,
                        rank=2,
                    ),
                    input_rows=len(input_rows),
                    final=False,
                )
            contract = _contract(
                manifest=manifest,
                root=root,
                checkpoint=checkpoint,
                rank=2,
            )
            contract["generation_authorized"] = True
            arrays, rows = _shard_fixture(
                input_rows=input_rows,
                rank=2,
                root=root,
                checkpoint=checkpoint,
            )
            with self.assertRaisesRegex(ValueError, "fail-closed"):
                _commit(
                    directory=root / "bad-contract",
                    rows=rows,
                    arrays=arrays,
                    contract=contract,
                    input_rows=len(input_rows),
                    final=False,
                )

    def test_fail_closed_summary_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "input.jsonl"
            checkpoint = root / "tracker.pth"
            input_rows = _write_manifest(manifest)
            checkpoint.write_bytes(b"tracker")
            arrays, rows = _shard_fixture(
                input_rows=input_rows,
                rank=0,
                root=root,
                checkpoint=checkpoint,
            )
            output = root / "output"
            _commit(
                directory=output,
                rows=rows,
                arrays=arrays,
                contract=_contract(
                    manifest=manifest,
                    root=root,
                    checkpoint=checkpoint,
                    rank=0,
                ),
                input_rows=len(input_rows),
                final=False,
            )
            summary_path = output / "summary.json"
            done_path = output / "done.json"
            summary = json.loads(summary_path.read_text())
            summary["formal_status"] = "READY"
            summary_path.write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            done = json.loads(done_path.read_text())
            done["summary_sha256"] = _file_digest(summary_path)
            done_path.write_text(
                json.dumps(done, indent=2) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "fail-closed"):
                validate_commit(output, input_manifest=manifest)
            summary["formal_status"] = FORMAL_STATUS
            summary_path.write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            done["summary_sha256"] = _file_digest(summary_path)
            done["formal_status"] = "READY"
            done_path.write_text(
                json.dumps(done, indent=2) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "fail-closed"):
                validate_commit(output, input_manifest=manifest)

    def test_eight_shard_finalize_and_strict_source_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "input.jsonl"
            checkpoint = root / "tracker.pth"
            input_rows = _write_manifest(manifest)
            checkpoint.write_bytes(b"tracker")
            output_root = root / "cache"
            for rank in range(FINAL_WORLD_SIZE):
                arrays, rows = _shard_fixture(
                    input_rows=input_rows,
                    rank=rank,
                    root=root,
                    checkpoint=checkpoint,
                )
                _commit(
                    directory=_rank_directory(
                        output_root, rank, FINAL_WORLD_SIZE
                    ),
                    rows=rows,
                    arrays=arrays,
                    contract=_contract(
                        manifest=manifest,
                        root=root,
                        checkpoint=checkpoint,
                        rank=rank,
                    ),
                    input_rows=len(input_rows),
                    final=False,
                )
            done = finalize_shards(
                input_manifest=manifest,
                output_root=output_root,
            )
            self.assertEqual(done["rows"], len(input_rows))
            final = validate_commit(
                output_root / "final",
                input_manifest=manifest,
                final=True,
            )
            self.assertEqual(
                final["arrays"]["input_indices"].tolist(),
                list(range(len(input_rows))),
            )
            source_done = (
                _rank_directory(output_root, 3, FINAL_WORLD_SIZE)
                / "done.json"
            )
            source_done.rename(source_done.with_suffix(".missing"))
            with self.assertRaises(FileNotFoundError):
                validate_commit(
                    output_root / "final",
                    input_manifest=manifest,
                    final=True,
                )

    def test_failed_operational_coverage_never_commits_final(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "input.jsonl"
            checkpoint = root / "tracker.pth"
            input_rows = _write_manifest(manifest)
            checkpoint.write_bytes(b"tracker")
            output_root = root / "cache"
            for rank in range(FINAL_WORLD_SIZE):
                arrays, rows = _shard_fixture(
                    input_rows=input_rows,
                    rank=rank,
                    root=root,
                    checkpoint=checkpoint,
                )
                if rank == 0:
                    for local_index in range(len(rows)):
                        _invalidate_side(
                            arrays,
                            rows,
                            local_index=local_index,
                            side="source",
                        )
                        _invalidate_side(
                            arrays,
                            rows,
                            local_index=local_index,
                            side="target",
                        )
                _commit(
                    directory=_rank_directory(
                        output_root, rank, FINAL_WORLD_SIZE
                    ),
                    rows=rows,
                    arrays=arrays,
                    contract=_contract(
                        manifest=manifest,
                        root=root,
                        checkpoint=checkpoint,
                        rank=rank,
                    ),
                    input_rows=len(input_rows),
                    final=False,
                )
            with self.assertRaisesRegex(ValueError, "coverage failed"):
                finalize_shards(
                    input_manifest=manifest,
                    output_root=output_root,
                )
            self.assertFalse((output_root / "final").exists())


class TrackCacheProvenanceAndPathTests(unittest.TestCase):
    def test_runtime_versions_are_normalized_to_builtin_strings(self) -> None:
        class VersionText(str):
            pass

        fake_torch = SimpleNamespace(
            __version__=VersionText("2.7.1+rocm6.3"),
            version=SimpleNamespace(hip=VersionText("6.3")),
            __config__=SimpleNamespace(
                show=lambda: VersionText("build config")
            ),
            cuda=SimpleNamespace(
                device_count=lambda: FINAL_WORLD_SIZE,
                get_device_properties=lambda _rank: SimpleNamespace(
                    total_memory=64 * 1024**3
                ),
                get_device_capability=lambda _rank: (9, 0),
                get_device_name=lambda _rank: VersionText(
                    "AMD Instinct MI210"
                ),
            ),
        )
        fake_cv2 = SimpleNamespace(__version__=VersionText("4.10.0"))
        with patch.dict(
            sys.modules,
            {"torch": fake_torch, "cv2": fake_cv2},
        ), patch.object(
            track_cache.np,
            "__version__",
            VersionText("2.0.0"),
        ):
            runtime = track_cache.runtime_provenance(
                local_rank=0,
                determinism={"seed": SEED},
            )
        for field in (
            "numpy_version",
            "opencv_version",
            "torch_version",
            "torch_hip_version",
            "device_name",
        ):
            self.assertIs(type(runtime[field]), str)
        self.assertEqual(runtime["torch_version"], "2.7.1+rocm6.3")
        self.assertEqual(runtime["torch_hip_version"], "6.3")
        self.assertEqual(runtime["device_type"], "cuda-hip")
        fake_torch.version.hip = None
        with patch.dict(
            sys.modules,
            {"torch": fake_torch, "cv2": fake_cv2},
        ):
            cuda_runtime = track_cache.runtime_provenance(
                local_rank=0,
                determinism={"seed": SEED},
            )
        self.assertIsNone(cuda_runtime["torch_hip_version"])
        self.assertEqual(cuda_runtime["device_type"], "cuda")

    def test_cotracker_bundle_requires_clean_tracked_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "co-tracker"
            package = root / "cotracker"
            package.mkdir(parents=True)
            (package / "predictor.py").write_text("VALUE = 1\n")
            (package / "model.py").write_text("MODEL = 2\n")
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(
                ["git", "-C", str(root), "config", "user.email", "test@x"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(root), "config", "user.name", "test"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(root), "add", "cotracker"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(root), "commit", "-qm", "fixture"],
                check=True,
            )
            provenance = cotracker_source_provenance(root)
            self.assertEqual(provenance["python_source_file_count"], 2)
            self.assertTrue(provenance["git_tracked_clean"])
            (package / "model.py").write_text("MODEL = 3\n")
            with self.assertRaisesRegex(ValueError, "modified/deleted"):
                cotracker_source_provenance(root)

    def test_output_path_overlap_and_completed_final_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "input.jsonl"
            checkpoint = root / "tracker.pth"
            data_root = root / "data"
            cotracker = root / "co-tracker"
            snapshot = root / "snapshot"
            data_root.mkdir()
            cotracker.mkdir()
            snapshot.mkdir()
            _write_manifest(manifest)
            checkpoint.write_bytes(b"tracker")
            with self.assertRaisesRegex(ValueError, "overlaps"):
                validate_output_root(
                    output_root=data_root / "cache",
                    input_manifest=manifest,
                    data_root=data_root,
                    tracker_checkpoint=checkpoint,
                    cotracker_root=cotracker,
                    source_snapshot=snapshot,
                )
            output = root / "run" / "cache"
            resolved = validate_output_root(
                output_root=output,
                input_manifest=manifest,
                data_root=data_root,
                tracker_checkpoint=checkpoint,
                cotracker_root=cotracker,
                source_snapshot=snapshot,
            )
            self.assertEqual(resolved, output.resolve())
            (output / "final").mkdir(parents=True)
            (output / "final" / "done.json").write_text("{}\n")
            with self.assertRaisesRegex(FileExistsError, "8-GPU"):
                validate_output_root(
                    output_root=output,
                    input_manifest=manifest,
                    data_root=data_root,
                    tracker_checkpoint=checkpoint,
                    cotracker_root=cotracker,
                    source_snapshot=snapshot,
                )


if __name__ == "__main__":
    unittest.main()
