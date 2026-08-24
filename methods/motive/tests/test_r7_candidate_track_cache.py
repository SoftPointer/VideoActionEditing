from __future__ import annotations

import json
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from motive import r7_candidate_temporal_manifest as candidate_manifest
from motive import r7_candidate_track_cache as cache
from motive.r7_preflight_extract import (
    FINAL_WORLD_SIZE,
    R7_VIDEO_SAMPLING,
    _file_digest,
    _object_digest,
)


SEED = 29
TRACK_COUNT = 16


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _candidate_fixture(
    root: Path,
    *,
    rows: int = 9,
) -> tuple[
    Path,
    Path,
    list[dict[str, object]],
    dict[str, object],
    dict[str, object],
    str,
]:
    input_dir = root / "candidate-input"
    data_root = root / "media"
    input_dir.mkdir()
    data_root.mkdir()
    candidate_rows: list[dict[str, object]] = []
    media_by_path: dict[str, dict[str, object]] = {}
    for index in range(rows):
        bindings: dict[str, object] = {"data_root": str(data_root.resolve())}
        paths: dict[str, str] = {}
        for role in ("src_video", "tgt_video"):
            relative = f"videos/{index:03d}/{role}.mp4"
            path = data_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{role}-{index}".encode("utf-8"))
            record = {
                "relative_path": relative,
                "sha256": _file_digest(path),
                "bytes": path.stat().st_size,
            }
            bindings[role] = record
            media_by_path[relative] = record
            paths[role] = relative
        candidate_rows.append(
            {
                "schema_version": candidate_manifest.ROW_SCHEMA,
                "iid": f"candidate-{index:03d}",
                "input_digest": f"digest-{index:03d}",
                "prompt": f"move object {index}",
                "src_video": paths["src_video"],
                "tgt_video": paths["tgt_video"],
                # Deliberately present upstream: the cache must not copy this.
                "label": {
                    "class": "positive" if index % 2 == 0 else "negative",
                    "action_signature": f"action-{index % 3}",
                },
                "source_bindings": {"media": bindings},
                "formal_evidence": False,
                "formal_split": False,
                "human_labels_asserted": False,
                "training_authorized": False,
                "generation_authorized": False,
            }
        )
    summary: dict[str, object] = {
        "schema_version": candidate_manifest.SUMMARY_SCHEMA,
        "media": {
            "binding_digest": _object_digest(
                {
                    name: media_by_path[name]
                    for name in sorted(media_by_path)
                }
            ),
        },
    }
    done: dict[str, object] = {
        "schema_version": candidate_manifest.DONE_SCHEMA,
        "status": "complete",
        "output_rows": len(candidate_rows),
        "artifact_digest": _object_digest(
            [_object_digest(row) for row in candidate_rows]
        ),
    }
    manifest_path = input_dir / candidate_manifest.MANIFEST_NAME
    manifest_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in candidate_rows
        ),
        encoding="utf-8",
    )
    _write_json(input_dir / candidate_manifest.SUMMARY_NAME, summary)
    _write_json(input_dir / candidate_manifest.DONE_NAME, done)
    return (
        input_dir,
        data_root,
        candidate_rows,
        summary,
        done,
        _file_digest(input_dir / candidate_manifest.DONE_NAME),
    )


def _validator_result(
    *,
    input_dir: Path,
    rows: list[dict[str, object]],
    summary: dict[str, object],
    done: dict[str, object],
) -> dict[str, object]:
    return {
        "directory": input_dir.resolve(),
        "rows": rows,
        "summary": summary,
        "done": done,
    }


def _runtime(seed: int = SEED) -> dict[str, object]:
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
            "python_hash_seed": str(seed),
            "torch_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
        },
    }


def _source_provenance(root: Path) -> dict[str, object]:
    inventory = [
        {
            "path": "cotracker/predictor.py",
            "sha256": "a" * 64,
            "size": 17,
        }
    ]
    return {
        "root": str(root.resolve()),
        "git_toplevel": str(root.resolve()),
        "git_head": "b" * 40,
        "git_tracked_clean": True,
        "python_source_files": inventory,
        "python_source_file_count": len(inventory),
        "python_source_bundle_sha256": _object_digest(inventory),
    }


def _contract(
    *,
    input_dir: Path,
    done_sha256: str,
    data_root: Path,
    checkpoint: Path,
    cotracker_root: Path,
    rank: int,
) -> dict[str, object]:
    return cache.build_cache_contract(
        input_dir=input_dir,
        expected_input_done_sha256=done_sha256,
        data_root=data_root,
        tracker_checkpoint=checkpoint,
        cotracker_provenance=_source_provenance(cotracker_root),
        runtime=_runtime(),
        tracker_grid_size=4,
        rank=rank,
        world_size=FINAL_WORLD_SIZE,
        device=f"cuda:{rank}",
        seed=SEED,
    )


def _valid_arrays(input_indices: list[int]) -> dict[str, np.ndarray]:
    arrays = cache._empty_arrays(
        len(input_indices),
        track_count=TRACK_COUNT,
    )
    arrays["input_indices"][:] = input_indices
    identity = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    frame_times = np.arange(32, dtype=np.float64) / 25.0
    frame_indices = np.arange(32, dtype=np.int64)
    tracks = np.zeros((32, TRACK_COUNT, 2), dtype=np.float32)
    tracks[..., 0] = np.linspace(0.1, 0.9, TRACK_COUNT)[None]
    tracks[..., 1] = np.linspace(0.2, 0.8, TRACK_COUNT)[None]
    for side in cache.SIDES:
        arrays[f"{side}_track_valid"][:] = True
        arrays[f"{side}_camera_valid"][:] = True
        arrays[f"{side}_camera_crossfit_valid"][:] = True
        arrays[f"{side}_normalized_tracks"][:] = tracks
        arrays[f"{side}_stabilized_tracks"][:] = tracks
        arrays[f"{side}_visibility"][:] = 1.0
        arrays[f"{side}_frame_times"][:] = frame_times
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
    rank: int,
    data_root: Path,
    checkpoint: Path,
    side: str,
) -> dict[str, object]:
    field = "src_video" if side == "source" else "tgt_video"
    media = input_row["source_bindings"]["media"][field]
    return {
        "status": "camera_ready",
        "track_valid": True,
        "camera_valid": True,
        "failure_stage": None,
        "failure_reason": None,
        "failure_message": None,
        "resolved_path": str((data_root / input_row[field]).resolve()),
        "video_sha256": media["sha256"],
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
                "grid_size": 4,
                "query_frame": 0,
                "backward_tracking": False,
                "device": f"cuda:{rank}",
            },
            "tracks": TRACK_COUNT,
        },
        "camera_crossfit": {
            "valid": True,
            "raw_median": float(np.float32(0.01)),
            "residual_median": float(np.float32(0.002)),
            "residual_reduction": float(np.float32(0.8)),
        },
    }


def _shard_fixture(
    *,
    input_rows: list[dict[str, object]],
    rank: int,
    data_root: Path,
    checkpoint: Path,
    artifact_digest: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    selected = list(range(rank, len(input_rows), FINAL_WORLD_SIZE))
    arrays = _valid_arrays(selected)
    rows = []
    for local_index, input_index in enumerate(selected):
        input_row = input_rows[input_index]
        rows.append(
            cache._candidate_output_row(
                input_row=input_row,
                input_index=input_index,
                array_index=local_index,
                rank=rank,
                input_artifact_digest=artifact_digest,
                source=_camera_record(
                    input_row=input_row,
                    rank=rank,
                    data_root=data_root,
                    checkpoint=checkpoint,
                    side="source",
                ),
                target=_camera_record(
                    input_row=input_row,
                    rank=rank,
                    data_root=data_root,
                    checkpoint=checkpoint,
                    side="target",
                ),
            )
        )
    return arrays, rows


class CandidateTrackCacheTests(unittest.TestCase):
    def test_candidate_array_contract_has_no_positive_label(self) -> None:
        arrays = _valid_arrays([0])
        self.assertNotIn("positive", arrays)
        cache._validate_array_contract(
            arrays,
            rows=1,
            track_count=TRACK_COUNT,
        )
        polluted = dict(arrays)
        polluted["positive"] = np.ones(1, dtype=np.bool_)
        with self.assertRaisesRegex(ValueError, "arrays differ|positive"):
            cache._validate_array_contract(
                polluted,
                rows=1,
                track_count=TRACK_COUNT,
            )

    def test_runtime_contract_reports_exact_non_native_string_field(
        self,
    ) -> None:
        class VersionText(str):
            pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                input_dir,
                data_root,
                rows,
                summary,
                done,
                done_sha,
            ) = _candidate_fixture(root)
            checkpoint = root / "tracker.pth"
            checkpoint.write_bytes(b"checkpoint")
            cotracker_root = root / "cotracker"
            cotracker_root.mkdir()
            runtime = _runtime()
            runtime["torch_version"] = VersionText("2.7.1+rocm6.3")
            result = _validator_result(
                input_dir=input_dir,
                rows=rows,
                summary=summary,
                done=done,
            )
            with patch.object(
                cache.candidate_manifest,
                "validate_candidate_temporal_manifest",
                return_value=result,
            ), self.assertRaisesRegex(
                ValueError,
                r"runtime\.torch_version.*type=VersionText",
            ):
                cache.build_cache_contract(
                    input_dir=input_dir,
                    expected_input_done_sha256=done_sha,
                    data_root=data_root,
                    tracker_checkpoint=checkpoint,
                    cotracker_provenance=_source_provenance(
                        cotracker_root
                    ),
                    runtime=runtime,
                    tracker_grid_size=4,
                    rank=0,
                    world_size=FINAL_WORLD_SIZE,
                    device="cuda:0",
                    seed=SEED,
                )

    def test_runtime_preflight_builds_real_rank_zero_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                input_dir,
                data_root,
                rows,
                summary,
                done,
                done_sha,
            ) = _candidate_fixture(root)
            checkpoint = root / "tracker.pth"
            checkpoint.write_bytes(b"checkpoint")
            cotracker_root = root / "cotracker"
            cotracker_root.mkdir()
            source_snapshot = root / "snapshot"
            source_snapshot.mkdir()
            output_root = root / "cache"
            runtime = _runtime()
            result = _validator_result(
                input_dir=input_dir,
                rows=rows,
                summary=summary,
                done=done,
            )
            with patch.object(
                cache.candidate_manifest,
                "validate_candidate_temporal_manifest",
                return_value=result,
            ), patch.object(
                cache.legacy_cache,
                "validate_output_root",
                return_value=output_root.resolve(),
            ), patch.object(
                cache.legacy_cache,
                "_configure_determinism",
                return_value=runtime["determinism"],
            ) as configure, patch.object(
                cache.legacy_cache,
                "runtime_provenance",
                return_value=runtime,
            ) as provenance, patch.object(
                cache.legacy_cache,
                "cotracker_source_provenance",
                return_value=_source_provenance(cotracker_root),
            ):
                receipt = cache.runtime_preflight(
                    input_dir=input_dir,
                    expected_input_done_sha256=done_sha,
                    data_root=data_root,
                    output_root=output_root,
                    tracker_checkpoint=checkpoint,
                    cotracker_root=cotracker_root,
                    source_snapshot=source_snapshot,
                    tracker_grid_size=4,
                    seed=SEED,
                )
            configure.assert_called_once_with(SEED, local_rank=0)
            provenance.assert_called_once_with(
                local_rank=0,
                determinism=runtime["determinism"],
            )
            self.assertEqual(receipt["status"], "ready")
            self.assertEqual(receipt["rank"], 0)
            self.assertEqual(receipt["world_size"], FINAL_WORLD_SIZE)
            self.assertEqual(receipt["rows"], len(rows))
            self.assertEqual(receipt["input_done_sha256"], done_sha)
            self.assertRegex(receipt["runtime_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(receipt["contract_sha256"], r"^[0-9a-f]{64}$")
            for field in cache.SAFETY_FIELDS:
                self.assertIs(receipt[field], False)

    def test_gpu_launcher_preflights_runtime_before_torchrun(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "auh_r7_candidate_track_cache_packed.sbatch"
        )
        subprocess.run(["bash", "-n", str(script)], check=True)
        text = script.read_text(encoding="utf-8")
        self.assertLess(
            text.index("export PYTHONHASHSEED"),
            text.index("runtime-preflight"),
        )
        self.assertLess(
            text.index("runtime-preflight"),
            text.index("torch.distributed.run"),
        )
        runtime_block = text[
            text.index("runtime-preflight"):
            text.index("echo \"[r7-candidate-cache] host")
        ]
        self.assertIn(
            "--tracker-grid-size \"${tracker_grid_size}\"",
            runtime_block,
        )
        self.assertIn("--seed \"${seed}\"", runtime_block)

    def test_external_done_anchor_and_contract_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                input_dir,
                data_root,
                rows,
                summary,
                done,
                done_sha,
            ) = _candidate_fixture(root)
            checkpoint = root / "tracker.pth"
            checkpoint.write_bytes(b"checkpoint")
            cotracker_root = root / "cotracker"
            cotracker_root.mkdir()
            result = _validator_result(
                input_dir=input_dir,
                rows=rows,
                summary=summary,
                done=done,
            )
            with patch.object(
                cache.candidate_manifest,
                "validate_candidate_temporal_manifest",
                return_value=result,
            ):
                with self.assertRaisesRegex(ValueError, "external done SHA"):
                    _contract(
                        input_dir=input_dir,
                        done_sha256="0" * 64,
                        data_root=data_root,
                        checkpoint=checkpoint,
                        cotracker_root=cotracker_root,
                        rank=0,
                    )
                contract = _contract(
                    input_dir=input_dir,
                    done_sha256=done_sha,
                    data_root=data_root,
                    checkpoint=checkpoint,
                    cotracker_root=cotracker_root,
                    rank=0,
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "at least 16 tracks",
                ):
                    cache.build_cache_contract(
                        input_dir=input_dir,
                        expected_input_done_sha256=done_sha,
                        data_root=data_root,
                        tracker_checkpoint=checkpoint,
                        cotracker_provenance=_source_provenance(
                            cotracker_root
                        ),
                        runtime=_runtime(),
                        tracker_grid_size=3,
                        rank=0,
                        world_size=FINAL_WORLD_SIZE,
                        device="cuda:0",
                        seed=SEED,
                    )
            self.assertEqual(contract["world_size"], FINAL_WORLD_SIZE)
            self.assertEqual(
                contract["input_binding"]["done_sha256"], done_sha
            )
            for field in cache.SAFETY_FIELDS:
                self.assertIs(contract[field], False)
            self.assertFalse(
                contract["label_semantics"]["input_labels_copied_to_cache"]
            )

    def test_create_only_commit_is_sealed_and_label_neutral(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                input_dir,
                data_root,
                input_rows,
                summary,
                input_done,
                done_sha,
            ) = _candidate_fixture(root)
            checkpoint = root / "tracker.pth"
            checkpoint.write_bytes(b"checkpoint")
            cotracker_root = root / "cotracker"
            cotracker_root.mkdir()
            validator = _validator_result(
                input_dir=input_dir,
                rows=input_rows,
                summary=summary,
                done=input_done,
            )
            with patch.object(
                cache.candidate_manifest,
                "validate_candidate_temporal_manifest",
                return_value=validator,
            ), patch.object(
                cache.legacy_cache,
                "cotracker_source_provenance",
                return_value=_source_provenance(cotracker_root),
            ):
                contract = _contract(
                    input_dir=input_dir,
                    done_sha256=done_sha,
                    data_root=data_root,
                    checkpoint=checkpoint,
                    cotracker_root=cotracker_root,
                    rank=0,
                )
                arrays, rows = _shard_fixture(
                    input_rows=input_rows,
                    rank=0,
                    data_root=data_root,
                    checkpoint=checkpoint,
                    artifact_digest=input_done["artifact_digest"],
                )
                output = root / "cache" / "rank-000"
                input_commit = cache._load_input_commit(
                    input_dir,
                    expected_done_sha256=done_sha,
                )
                cache._commit(
                    directory=output,
                    rows=rows,
                    arrays=arrays,
                    contract=contract,
                    input_commit=input_commit,
                    final=False,
                )
                validated = cache.validate_commit(
                    output,
                    input_dir=input_dir,
                    expected_input_done_sha256=done_sha,
                    expected_contract=contract,
                )
                with self.assertRaises(FileExistsError):
                    cache._commit(
                        directory=output,
                        rows=rows,
                        arrays=arrays,
                        contract=contract,
                        input_commit=input_commit,
                        final=False,
                    )
            self.assertEqual(
                stat.S_IMODE(output.stat().st_mode),
                0o555,
            )
            for name in cache.OUTPUT_NAMES:
                self.assertEqual(
                    stat.S_IMODE((output / name).stat().st_mode),
                    0o444,
                )
            with np.load(output / cache.ARCHIVE_NAME) as loaded:
                self.assertNotIn("positive", loaded.files)
            manifest_bytes = (output / cache.MANIFEST_NAME).read_bytes()
            self.assertNotIn(b'"label"', manifest_bytes)
            self.assertNotIn(b'"positive"', manifest_bytes)
            manifest_rows = validated["rows"]
            self.assertTrue(manifest_rows)
            for artifact in (
                validated["summary"],
                validated["done"],
            ):
                for field in cache.SAFETY_FIELDS:
                    self.assertIs(artifact[field], False)
            self.assertEqual(
                validated["done"]["permission_contract"],
                cache.artifact_permissions.permission_contract(),
            )
            for row in manifest_rows:
                self.assertNotIn("label", row)
                self.assertNotIn("positive", row)
                self.assertNotIn("input_row", row)
                self.assertNotIn("prompt", row)
                for field in cache.SAFETY_FIELDS:
                    self.assertIs(row[field], False)

    def test_validation_rehashes_referenced_media(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                input_dir,
                data_root,
                input_rows,
                summary,
                input_done,
                done_sha,
            ) = _candidate_fixture(root)
            checkpoint = root / "tracker.pth"
            checkpoint.write_bytes(b"checkpoint")
            cotracker_root = root / "cotracker"
            cotracker_root.mkdir()
            validator = _validator_result(
                input_dir=input_dir,
                rows=input_rows,
                summary=summary,
                done=input_done,
            )
            with patch.object(
                cache.candidate_manifest,
                "validate_candidate_temporal_manifest",
                return_value=validator,
            ), patch.object(
                cache.legacy_cache,
                "cotracker_source_provenance",
                return_value=_source_provenance(cotracker_root),
            ):
                contract = _contract(
                    input_dir=input_dir,
                    done_sha256=done_sha,
                    data_root=data_root,
                    checkpoint=checkpoint,
                    cotracker_root=cotracker_root,
                    rank=1,
                )
                arrays, rows = _shard_fixture(
                    input_rows=input_rows,
                    rank=1,
                    data_root=data_root,
                    checkpoint=checkpoint,
                    artifact_digest=input_done["artifact_digest"],
                )
                output = root / "cache"
                cache._commit(
                    directory=output,
                    rows=rows,
                    arrays=arrays,
                    contract=contract,
                    input_commit=cache._load_input_commit(
                        input_dir,
                        expected_done_sha256=done_sha,
                    ),
                    final=False,
                )
                changed = data_root / input_rows[1]["src_video"]
                original = changed.read_bytes()
                changed.write_bytes(b"X" * len(original))
                self.assertEqual(changed.stat().st_size, len(original))
                with self.assertRaisesRegex(
                    ValueError, "media bytes changed"
                ):
                    cache.validate_commit(
                        output,
                        input_dir=input_dir,
                        expected_input_done_sha256=done_sha,
                    )

    def test_exact_eight_shard_merge_and_strict_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                input_dir,
                data_root,
                input_rows,
                summary,
                input_done,
                done_sha,
            ) = _candidate_fixture(root, rows=9)
            checkpoint = root / "tracker.pth"
            checkpoint.write_bytes(b"checkpoint")
            cotracker_root = root / "cotracker"
            cotracker_root.mkdir()
            output_root = root / "cache"
            validator = _validator_result(
                input_dir=input_dir,
                rows=input_rows,
                summary=summary,
                done=input_done,
            )
            with patch.object(
                cache.candidate_manifest,
                "validate_candidate_temporal_manifest",
                return_value=validator,
            ), patch.object(
                cache.legacy_cache,
                "cotracker_source_provenance",
                return_value=_source_provenance(cotracker_root),
            ):
                input_commit = cache._load_input_commit(
                    input_dir,
                    expected_done_sha256=done_sha,
                )
                for rank in range(FINAL_WORLD_SIZE):
                    contract = _contract(
                        input_dir=input_dir,
                        done_sha256=done_sha,
                        data_root=data_root,
                        checkpoint=checkpoint,
                        cotracker_root=cotracker_root,
                        rank=rank,
                    )
                    arrays, rows = _shard_fixture(
                        input_rows=input_rows,
                        rank=rank,
                        data_root=data_root,
                        checkpoint=checkpoint,
                        artifact_digest=input_done["artifact_digest"],
                    )
                    cache._commit(
                        directory=cache._rank_directory(
                            output_root, rank
                        ),
                        rows=rows,
                        arrays=arrays,
                        contract=contract,
                        input_commit=input_commit,
                        final=False,
                    )
                stale_final = output_root / ".final.interrupted.tmp"
                stale_rank = (
                    output_root
                    / "shards"
                    / ".rank-000-of-008.interrupted.tmp"
                )
                stale_final.mkdir()
                stale_rank.mkdir()
                (stale_final / "partial").write_text("partial")
                (stale_rank / "partial").write_text("partial")
                done = cache.finalize_shards(
                    input_dir=input_dir,
                    expected_input_done_sha256=done_sha,
                    output_root=output_root,
                )
                quarantines = list(
                    root.glob(".cache.stale-stages.*.quarantine")
                )
                self.assertEqual(len(quarantines), 1)
                self.assertEqual(
                    sorted(path.name for path in quarantines[0].iterdir()),
                    [
                        "root__.final.interrupted.tmp",
                        "shards__.rank-000-of-008.interrupted.tmp",
                    ],
                )
                # Model an interruption after final publication but before
                # the enclosing output/shards directories were sealed.
                output_root.chmod(0o755)
                (output_root / "shards").chmod(0o755)
                resumed = cache.finalize_shards(
                    input_dir=input_dir,
                    expected_input_done_sha256=done_sha,
                    output_root=output_root,
                    resume=True,
                )
                final = cache.validate_commit(
                    output_root / cache.FINAL_DIR_NAME,
                    input_dir=input_dir,
                    expected_input_done_sha256=done_sha,
                    final=True,
                )
                self.assertEqual(done, resumed)
                self.assertEqual(
                    final["arrays"]["input_indices"].tolist(),
                    list(range(len(input_rows))),
                )
                with np.load(
                    output_root
                    / cache.FINAL_DIR_NAME
                    / cache.ARCHIVE_NAME
                ) as loaded:
                    self.assertNotIn("positive", loaded.files)
                for directory in (
                    output_root,
                    output_root / "shards",
                    output_root / cache.FINAL_DIR_NAME,
                    *(
                        cache._rank_directory(output_root, rank)
                        for rank in range(FINAL_WORLD_SIZE)
                    ),
                ):
                    self.assertEqual(
                        stat.S_IMODE(directory.stat().st_mode),
                        0o555,
                    )
                changed_source = _source_provenance(cotracker_root)
                changed_source["git_head"] = "c" * 40
                with patch.object(
                    cache.legacy_cache,
                    "cotracker_source_provenance",
                    return_value=changed_source,
                ):
                    with self.assertRaisesRegex(
                        ValueError, "Python source changed"
                    ):
                        cache.validate_commit(
                            output_root / cache.FINAL_DIR_NAME,
                            input_dir=input_dir,
                            expected_input_done_sha256=done_sha,
                            final=True,
                        )
                shard = cache._rank_directory(output_root, 7)
                # Deliberately break the sealed outer closure first, then
                # remove one source shard; either fact must invalidate final.
                (output_root / "shards").chmod(0o755)
                shard.chmod(0o700)
                shard.rename(shard.with_name(shard.name + ".missing"))
                with self.assertRaisesRegex(
                    ValueError, "source shard set"
                ):
                    cache.validate_commit(
                        output_root / cache.FINAL_DIR_NAME,
                        input_dir=input_dir,
                        expected_input_done_sha256=done_sha,
                        final=True,
                    )

    def test_failed_coverage_never_publishes_final(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                input_dir,
                data_root,
                input_rows,
                summary,
                input_done,
                done_sha,
            ) = _candidate_fixture(root, rows=9)
            checkpoint = root / "tracker.pth"
            checkpoint.write_bytes(b"checkpoint")
            cotracker_root = root / "cotracker"
            cotracker_root.mkdir()
            output_root = root / "cache"
            validator = _validator_result(
                input_dir=input_dir,
                rows=input_rows,
                summary=summary,
                done=input_done,
            )
            with patch.object(
                cache.candidate_manifest,
                "validate_candidate_temporal_manifest",
                return_value=validator,
            ), patch.object(
                cache.legacy_cache,
                "cotracker_source_provenance",
                return_value=_source_provenance(cotracker_root),
            ):
                input_commit = cache._load_input_commit(
                    input_dir,
                    expected_done_sha256=done_sha,
                )
                for rank in range(FINAL_WORLD_SIZE):
                    contract = _contract(
                        input_dir=input_dir,
                        done_sha256=done_sha,
                        data_root=data_root,
                        checkpoint=checkpoint,
                        cotracker_root=cotracker_root,
                        rank=rank,
                    )
                    arrays, rows = _shard_fixture(
                        input_rows=input_rows,
                        rank=rank,
                        data_root=data_root,
                        checkpoint=checkpoint,
                        artifact_digest=input_done["artifact_digest"],
                    )
                    for side in cache.SIDES:
                        for name, value in arrays.items():
                            if name.startswith(f"{side}_"):
                                value[:] = 0
                        for row in rows:
                            old = row[side]
                            row[side] = {
                                "status": "failed",
                                "track_valid": False,
                                "camera_valid": False,
                                "failure_stage": "tracking",
                                "failure_reason": "unit_test_failure",
                                "failure_message": "unit test",
                                "resolved_path": old["resolved_path"],
                                "video_sha256": old["video_sha256"],
                                "decode": old["decode"],
                            }
                            row["paired_track_valid"] = False
                            row["paired_camera_valid"] = False
                    cache._commit(
                        directory=cache._rank_directory(
                            output_root, rank
                        ),
                        rows=rows,
                        arrays=arrays,
                        contract=contract,
                        input_commit=input_commit,
                        final=False,
                    )
                with self.assertRaisesRegex(
                    ValueError, "operational coverage failed"
                ):
                    cache.finalize_shards(
                        input_dir=input_dir,
                        expected_input_done_sha256=done_sha,
                        output_root=output_root,
                    )
            self.assertFalse(
                (output_root / cache.FINAL_DIR_NAME).exists()
            )


if __name__ == "__main__":
    unittest.main()
