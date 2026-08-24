from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from motive.r7_coherent_actor import CoherentActorConfig
from motive.r7_p1_diagnostic import (
    R7_P1_DIAGNOSTIC_DONE_SCHEMA,
    R7_P1_DIAGNOSTIC_ROW_SCHEMA,
    R7_P1_DIAGNOSTIC_SUMMARY_SCHEMA,
    DiagnosticGateConfig,
    DownstreamAuditConfig,
)
from motive.r7_preflight_extract import (
    _canonical_json,
    _file_digest,
    _object_digest,
)
from motive.r7_track_cache import (
    R7_TRACK_CACHE_FINAL_DONE_SCHEMA,
    R7_TRACK_CACHE_FINAL_SUMMARY_SCHEMA,
)


SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "r7_p1_review_bundle.py"
)
SPEC = importlib.util.spec_from_file_location(
    "r7_p1_review_bundle",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_source_video(
    path: Path,
    *,
    frames: int = 32,
    size: tuple[int, int] = (64, 48),
    fps: float = 16.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        size,
    )
    if not writer.isOpened():
        raise RuntimeError("test MJPG VideoWriter could not be opened")
    try:
        width, height = size
        for index in range(frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[..., 0] = min(255, index * 5)
            frame[..., 1] = 30
            cv2.rectangle(
                frame,
                (index % max(1, width - 8), height - 10),
                (index % max(1, width - 8) + 6, height - 4),
                (20, 180, 240),
                -1,
            )
            writer.write(frame)
    finally:
        writer.release()
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError("test source video was not written")


def _audit_metrics() -> dict[str, float]:
    return {
        "actor_mask_iou": 0.5,
        "event_window_iou": 0.8,
        "trajectory_rmse": 0.02,
        "shared_actor_track_fraction": 0.5,
        "per_track_trajectory_rmse": 0.005,
        "energy_cosine": 0.95,
        "shape_profile_cosine": 0.9,
        "event_duration_relative_error": 0.03,
    }


def _event(start: int, stop: int, fps: float = 16.0) -> dict[str, object]:
    return {
        "transition_start": start,
        "transition_stop": stop - 1,
        "frame_start": start,
        "frame_stop": stop,
        "start_time": start / fps,
        "end_time": (stop - 1) / fps,
        "duration": (stop - 1 - start) / fps,
        "normalized_start": start / 31.0,
        "normalized_end": (stop - 1) / 31.0,
        "captured_energy_fraction": 0.9,
    }


def _synthetic_verified(root: Path) -> tuple[dict[str, object], Path]:
    data_root = root / "data"
    data_root.mkdir()
    video = data_root / "sample.avi"
    _write_source_video(video)
    source_sha = _file_digest(video)
    frame_indices = np.arange(32, dtype=np.int64)
    frame_times = frame_indices.astype(np.float64) / 16.0
    tracks = np.zeros((32, 4, 2), dtype=np.float32)
    tracks[..., 1] = 0.75
    tracks[:, 0, 0] = 0.10
    tracks[:, 1, 0] = 0.30
    tracks[:, 2, 0] = 0.60
    tracks[:, 3, 0] = np.linspace(0.75, 0.88, 32)
    visibility = np.ones((32, 4), dtype=np.float32)
    decode = {
        "sampling_version": "synthetic-cache-sampling",
        "decoded_frames": 32,
        "source_frame_indices": frame_indices.tolist(),
        "source_fps": 16.0,
        "source_frame_count": 32,
        "source_size": [48, 64],
        "resized_size": [48, 64],
    }
    side_cache = {
        "status": "camera_ready",
        "track_valid": True,
        "camera_valid": True,
        "failure_stage": None,
        "failure_reason": None,
        "failure_message": None,
        "resolved_path": str(video.resolve()),
        "video_sha256": source_sha,
        "decode": decode,
    }
    cache_row = {
        "input_index": 0,
        "iid": "iid-synthetic-001",
        "input_row": {
            "iid": "iid-synthetic-001",
            "src_video": "sample.avi",
            "tgt_video": "sample.avi",
        },
        "source": dict(side_cache),
        "target": dict(side_cache),
    }
    base_mask = [False, True, False, True]
    perturbed_mask = [False, False, True, True]
    base_side = {
        "diagnostic_ready": True,
        "selector_ready": True,
        "event_ready": True,
        "failure_stage": None,
        "failure_reason": None,
        "failure_detail": None,
        "score": 0.125,
        "actor_track_mask": base_mask,
        "event_window": _event(5, 24),
    }
    perturbed = {
        "diagnostic_ready": True,
        "selector_ready": True,
        "event_ready": True,
        "failure_stage": None,
        "failure_reason": None,
        "failure_detail": None,
        "score": 0.11,
        "actor_track_mask": perturbed_mask,
        "event_window": _event(7, 25),
    }
    audit = {
        "eligible": True,
        "performed": True,
        "comparison_available": True,
        "ready_consistent": True,
        "joint_pass": False,
        "failure_reason": "joint_threshold_failed",
        "metrics": _audit_metrics(),
        "perturbed": perturbed,
    }
    diagnostic_row = {
        "schema_version": R7_P1_DIAGNOSTIC_ROW_SCHEMA,
        "input_index": 0,
        "iid": "iid-synthetic-001",
        "positive": True,
        "label_type": "positive",
        "negative_type": None,
        "action_signature": "synthetic move",
        "source_camera_valid": True,
        "target_camera_valid": True,
        "source": dict(base_side),
        "target": dict(base_side),
        "target_audit": audit,
    }
    arrays: dict[str, np.ndarray] = {}
    for side in ("source", "target"):
        arrays[f"{side}_track_valid"] = np.asarray([True])
        arrays[f"{side}_normalized_tracks"] = tracks[None]
        arrays[f"{side}_visibility"] = visibility[None]
        arrays[f"{side}_source_frame_indices"] = frame_indices[None]
        arrays[f"{side}_frame_times"] = frame_times[None]
        arrays[f"{side}_resized_size"] = np.asarray([[48, 64]], dtype=np.int32)
    cache = {
        "rows": [cache_row],
        "arrays": arrays,
        "contract": {"tracker": {"track_count": 4}},
    }
    cache_final = root / "cache" / "final"
    diagnostic_directory = root / "diagnostic"
    cache_final.mkdir(parents=True)
    diagnostic_directory.mkdir()
    contract = {
        "independent_audit": {
            "config": asdict(DownstreamAuditConfig())
        }
    }
    provenance = {
        "schema_version": review.REVIEW_BUNDLE_SCHEMA,
        "review_script": str(SCRIPT.resolve()),
        "review_script_sha256": _file_digest(SCRIPT),
        "input_manifest": str(root / "mock-input.jsonl"),
        "input_manifest_sha256": "a" * 64,
        "data_root": str(data_root.resolve()),
        "cache": {
            "final_directory": str(cache_final.resolve()),
            "done_sha256": "b" * 64,
            "summary_sha256": "c" * 64,
            "manifest_sha256": "d" * 64,
            "archive_sha256": "e" * 64,
            "contract_sha256": "f" * 64,
            "strict_final_and_eight_source_shards_revalidated": True,
        },
        "diagnostic": {
            "directory": str(diagnostic_directory.resolve()),
            "done_sha256": "1" * 64,
            "summary_sha256": "2" * 64,
            "rows_sha256": "3" * 64,
            "contract_sha256": "4" * 64,
            "rows_and_recomputed_summary_revalidated": True,
        },
    }
    verified = {
        "cache": cache,
        "diagnostic_rows": [diagnostic_row],
        "diagnostic_summary": {"contract": contract},
        "data_root": data_root.resolve(),
        "provenance": provenance,
    }
    return verified, video


def _artifact_envelopes(root: Path) -> dict[str, object]:
    data_root = root / "data"
    data_root.mkdir()
    input_manifest = root / "input.jsonl"
    input_manifest.write_text("{}\n", encoding="utf-8")
    cache_final = root / "cache" / "final"
    cache_final.mkdir(parents=True)
    diagnostic = root / "diagnostic"
    diagnostic.mkdir()
    cache_manifest = cache_final / "manifest.jsonl"
    cache_archive = cache_final / "track_cache.npz"
    cache_manifest.write_text("{}\n", encoding="utf-8")
    cache_archive.write_bytes(b"synthetic archive")
    cache_contract = {
        "input_manifest": str(input_manifest.resolve()),
        "data_root": str(data_root.resolve()),
        "tracker": {"track_count": 4},
    }
    cache_summary = {
        "schema_version": R7_TRACK_CACHE_FINAL_SUMMARY_SCHEMA,
        "rows": 1,
        "contract": cache_contract,
        "contract_sha256": _object_digest(cache_contract),
    }
    _write_json(cache_final / "summary.json", cache_summary)
    cache_done = {
        "schema_version": R7_TRACK_CACHE_FINAL_DONE_SCHEMA,
        "committed": True,
        "rows": 1,
        "archive_sha256": _file_digest(cache_archive),
        "manifest_sha256": _file_digest(cache_manifest),
        "summary_sha256": _file_digest(cache_final / "summary.json"),
        "contract_sha256": _object_digest(cache_contract),
    }
    _write_json(cache_final / "done.json", cache_done)

    diagnostic_contract = {
        "input_manifest": str(input_manifest.resolve()),
        "seed": 20260727,
        "cache": {"final_directory": str(cache_final.resolve())},
        "selector": {"config": asdict(CoherentActorConfig())},
        "independent_audit": {
            "config": asdict(DownstreamAuditConfig())
        },
        "diagnostic_gate": {
            "config": asdict(DiagnosticGateConfig())
        },
    }
    rows = [
        {
            "schema_version": R7_P1_DIAGNOSTIC_ROW_SCHEMA,
            "input_index": 0,
            "iid": "iid-envelope",
        }
    ]
    rows_path = diagnostic / "rows.jsonl"
    rows_path.write_text(
        "".join(_canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    diagnostic_summary = {
        "schema_version": R7_P1_DIAGNOSTIC_SUMMARY_SCHEMA,
        "rows": 1,
        "contract": diagnostic_contract,
        "contract_sha256": _object_digest(diagnostic_contract),
    }
    _write_json(diagnostic / "summary.json", diagnostic_summary)
    diagnostic_done = {
        "schema_version": R7_P1_DIAGNOSTIC_DONE_SCHEMA,
        "committed": True,
        "rows": 1,
        "rows_sha256": _file_digest(rows_path),
        "summary_sha256": _file_digest(diagnostic / "summary.json"),
        "contract_sha256": _object_digest(diagnostic_contract),
    }
    _write_json(diagnostic / "done.json", diagnostic_done)
    cache = {
        "contract": cache_contract,
        "rows": [],
        "arrays": {},
    }
    return {
        "data_root": data_root,
        "input_manifest": input_manifest,
        "cache_final": cache_final,
        "diagnostic": diagnostic,
        "cache": cache,
        "cache_done": cache_done,
        "cache_summary": cache_summary,
        "diagnostic_contract": diagnostic_contract,
        "diagnostic_done": diagnostic_done,
        "diagnostic_summary": diagnostic_summary,
        "rows": rows,
    }


class ReviewTrackMappingTests(unittest.TestCase):
    def test_normalized_coordinate_inversion_and_mask_index_layers(self) -> None:
        pixels, clipped = review._normalized_to_pixel(
            np.asarray(
                [[0.25, 0.50], [1.0, 1.0], [-0.10, 1.20]],
                dtype=np.float32,
            ),
            width=200,
            height=100,
        )
        np.testing.assert_array_equal(
            pixels,
            np.asarray([[50, 50], [199, 99], [0, 99]]),
        )
        self.assertEqual(clipped, 2)

        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        tracks = np.asarray(
            [[[0.10, 0.70], [0.30, 0.70], [0.60, 0.70]]],
            dtype=np.float32,
        )
        visibility = np.ones((1, 3), dtype=np.float32)
        review._draw_track_layers(
            frame,
            tracks,
            visibility,
            frame_index=0,
            base_mask=np.asarray([False, True, False]),
            perturbed_mask=np.asarray([False, False, True]),
        )
        background = frame[70, 20].astype(int)
        base = frame[70, 60].astype(int)
        perturbed = frame[70, 120].astype(int)
        self.assertGreater(int(background.sum()), 0)
        self.assertLessEqual(int(background.max() - background.min()), 2)
        self.assertGreater(base[1], base[0])
        self.assertGreater(base[1], base[2])
        self.assertGreater(perturbed[0], perturbed[1])
        self.assertGreater(perturbed[2], perturbed[1])

    def test_exact_cached_indices_and_tracking_size_are_used(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "source.avi"
            _write_source_video(
                video,
                frames=8,
                size=(64, 48),
                fps=12.0,
            )
            indices = np.asarray([0, 2, 5, 7], dtype=np.int64)
            frames, probe = review._decode_cached_frames(
                video,
                decode={
                    "source_frame_indices": indices.tolist(),
                    "source_fps": 12.0,
                    "source_frame_count": 8,
                    "source_size": [48, 64],
                },
                frame_indices=indices,
                resized_size=(24, 32),
            )
            self.assertEqual(len(frames), 4)
            self.assertTrue(
                all(frame.shape == (24, 32, 3) for frame in frames)
            )
            means = [float(frame[..., 0].mean()) for frame in frames]
            self.assertEqual(means, sorted(means))
            self.assertTrue(probe["exact_seek_position_verified"])


class ReviewInputValidationTests(unittest.TestCase):
    def test_loader_reuses_both_strict_core_validators(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _artifact_envelopes(Path(temporary))
            validated = {
                "done": fixture["diagnostic_done"],
                "summary": fixture["diagnostic_summary"],
                "rows": fixture["rows"],
            }
            with mock.patch.object(
                review,
                "load_final_cache",
                return_value=fixture["cache"],
            ) as load_cache, mock.patch.object(
                review,
                "build_diagnostic_contract",
                return_value=fixture["diagnostic_contract"],
            ) as build_contract, mock.patch.object(
                review,
                "validate_output_commit",
                return_value=validated,
            ) as validate_diagnostic:
                result = review.load_verified_inputs(
                    cache_final_directory=fixture["cache_final"],
                    diagnostic_directory=fixture["diagnostic"],
                    data_root=fixture["data_root"],
                )
            load_cache.assert_called_once_with(
                input_manifest=fixture["input_manifest"].resolve(),
                cache_root=fixture["cache_final"].parent.resolve(),
            )
            build_contract.assert_called_once()
            validate_diagnostic.assert_called_once()
            self.assertTrue(
                result["provenance"]["cache"][
                    "strict_final_and_eight_source_shards_revalidated"
                ]
            )
            self.assertTrue(
                result["provenance"]["diagnostic"][
                    "rows_and_recomputed_summary_revalidated"
                ]
            )

    def test_tampered_diagnostic_rows_fail_before_core_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _artifact_envelopes(Path(temporary))
            rows_path = fixture["diagnostic"] / "rows.jsonl"
            rows_path.write_bytes(rows_path.read_bytes() + b" ")
            with mock.patch.object(
                review,
                "load_final_cache",
                side_effect=AssertionError("core cache validation was reached"),
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "rows byte digest",
                ):
                    review.load_verified_inputs(
                        cache_final_directory=fixture["cache_final"],
                        diagnostic_directory=fixture["diagnostic"],
                        data_root=fixture["data_root"],
                    )


class ReviewBundleSyntheticTests(unittest.TestCase):
    def test_mock_one_row_bundle_has_32_frame_video_and_hash_provenance(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            verified, _ = _synthetic_verified(root)
            iid_list = root / "iids.txt"
            iid_list.write_text("iid-synthetic-001\n", encoding="utf-8")
            output = root / "review-output"
            with mock.patch.object(
                review,
                "load_verified_inputs",
                return_value=verified,
            ):
                summary = review.build_review_bundle(
                    cache_final_directory=root / "cache" / "final",
                    diagnostic_directory=root / "diagnostic",
                    data_root=root / "data",
                    iid_list=iid_list,
                    output_directory=output,
                )
            self.assertTrue(summary["committed"])
            self.assertEqual(summary["review_items"], 1)
            manifest_path = output / "review_manifest.jsonl"
            rows = review._read_canonical_jsonl(manifest_path)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["tracks"]["base_track_indices"], [1, 3])
            self.assertEqual(
                row["tracks"]["perturbed_track_indices"],
                [2, 3],
            )
            self.assertEqual(row["tracks"]["overlap_track_indices"], [3])
            self.assertEqual(
                row["diagnostic"]["target_audit"]["failed_axes"],
                ["mask", "traj"],
            )
            video = output / row["output_video"]["relative_path"]
            self.assertEqual(_file_digest(video), row["output_video"]["sha256"])
            self.assertEqual(row["output_video"]["frame_count"], 32)
            self.assertEqual(row["output_video"]["frame_size"], [48, 64])
            self.assertEqual(
                summary["review_manifest_sha256"],
                _file_digest(manifest_path),
            )
            self.assertFalse(
                summary["coordinate_contract"][
                    "camera_stabilized_tracks_drawn"
                ]
            )

    def test_include_source_emits_target_then_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            verified, _ = _synthetic_verified(root)
            iid_list = root / "iids.json"
            iid_list.write_text(
                '["iid-synthetic-001"]\n',
                encoding="utf-8",
            )
            output = root / "with-source"
            with mock.patch.object(
                review,
                "load_verified_inputs",
                return_value=verified,
            ):
                summary = review.build_review_bundle(
                    cache_final_directory=root / "cache" / "final",
                    diagnostic_directory=root / "diagnostic",
                    data_root=root / "data",
                    iid_list=iid_list,
                    output_directory=output,
                    include_source=True,
                )
            rows = review._read_canonical_jsonl(
                output / "review_manifest.jsonl"
            )
            self.assertEqual([row["side"] for row in rows], ["target", "source"])
            self.assertEqual(summary["review_items"], 2)
            self.assertIsNone(rows[1]["diagnostic"]["target_audit"])

    def test_existing_output_is_rejected_before_input_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "already-there"
            output.mkdir()
            with mock.patch.object(
                review,
                "load_verified_inputs",
                side_effect=AssertionError("input loader was reached"),
            ):
                with self.assertRaisesRegex(
                    FileExistsError,
                    "refusing to overwrite",
                ):
                    review.build_review_bundle(
                        cache_final_directory=root / "cache" / "final",
                        diagnostic_directory=root / "diagnostic",
                        data_root=root / "data",
                        iid_list=root / "iids.txt",
                        output_directory=output,
                    )

    def test_selected_source_video_tamper_fails_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            verified, video = _synthetic_verified(root)
            with video.open("ab") as handle:
                handle.write(b"tampered")
            iid_list = root / "iids.txt"
            iid_list.write_text("iid-synthetic-001\n", encoding="utf-8")
            output = root / "tampered-output"
            with mock.patch.object(
                review,
                "load_verified_inputs",
                return_value=verified,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "source video bytes changed",
                ):
                    review.build_review_bundle(
                        cache_final_directory=root / "cache" / "final",
                        diagnostic_directory=root / "diagnostic",
                        data_root=root / "data",
                        iid_list=iid_list,
                        output_directory=output,
                    )
            self.assertFalse(output.exists())

    def test_render_failure_leaves_no_visible_or_staging_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            verified, _ = _synthetic_verified(root)
            iid_list = root / "iids.txt"
            iid_list.write_text("iid-synthetic-001\n", encoding="utf-8")
            output = root / "failed-output"

            def fail_after_partial(path: Path, **_: object) -> object:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"partial")
                raise RuntimeError("synthetic render failure")

            with mock.patch.object(
                review,
                "load_verified_inputs",
                return_value=verified,
            ), mock.patch.object(
                review,
                "_write_review_video",
                side_effect=fail_after_partial,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "synthetic render failure",
                ):
                    review.build_review_bundle(
                        cache_final_directory=root / "cache" / "final",
                        diagnostic_directory=root / "diagnostic",
                        data_root=root / "data",
                        iid_list=iid_list,
                        output_directory=output,
                    )
            self.assertFalse(output.exists())
            self.assertEqual(
                list(root.glob(".failed-output.staging-*")),
                [],
            )
            self.assertFalse(
                (root / ".failed-output.review-build.lock").exists()
            )


if __name__ == "__main__":
    unittest.main()
