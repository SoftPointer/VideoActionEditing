from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from motive.r7_preflight_extract import (
    FINAL_WORLD_SIZE,
    PerVideoError,
    R7_ROW_SCHEMA,
    _commit_shard,
    _empty_arrays,
    _file_digest,
    _object_digest,
    build_extraction_contract,
    compute_p0_gate,
    difference_hash,
    dino_frame_offsets,
    finalize_shards,
    rank_directory,
    resized_dimensions,
    resolve_torchrun_coordinates,
    sampled_frame_times,
    uniform_sample_indices,
    validate_final,
    validate_shard,
)
from motive.r7_temporal_teacher import TemporalTeacherConfig


def _input_row(index: int, *, positive: bool = True) -> dict[str, object]:
    return {
        "iid": f"iid-{index:03d}",
        "src_video": f"videos/iid-{index:03d}/source.mp4",
        "tgt_video": f"videos/iid-{index:03d}/edited.mp4",
        "prompt": f"action {index}",
        "input_digest": f"digest-{index}",
        "r5_pilot_label": {
            "class": "positive" if positive else "negative",
            "action_signature": f"action-{index % 4}",
        },
    }


def _write_input_manifest(path: Path, count: int) -> list[dict[str, object]]:
    rows = [_input_row(index) for index in range(count)]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return rows


def _usable_arrays(count: int) -> dict[str, np.ndarray]:
    arrays = _empty_arrays(count)
    arrays["input_indices"][:] = np.arange(count)
    arrays["positive"][:] = True
    for side in ("source", "target"):
        arrays[f"{side}_usable"][:] = True
        arrays[f"{side}_base_valid"][:] = True
        arrays[f"{side}_dino_valid"][:] = True
        arrays[f"{side}_audit_available"][:] = True
        arrays[f"{side}_audit_pass"][:] = True
        arrays[f"{side}_camera_crossfit_valid"][:] = True
        arrays[f"{side}_teacher_embedding"][:, 0] = 1.0
        arrays[f"{side}_dino_cls"][:, :, 0] = 1.0
        arrays[f"{side}_stability_event_iou"][:] = 0.75
        arrays[f"{side}_stability_embedding_cosine"][:] = 0.90
        arrays[f"{side}_audit_event_iou"][:] = 0.75
        arrays[f"{side}_audit_embedding_cosine"][:] = 0.90
        arrays[f"{side}_audit_duration_relative_error"][:] = 0.05
        arrays[f"{side}_audit_embedding_norm_relative_error"][:] = 0.05
        arrays[f"{side}_audit_trajectory_rmse"][:] = 0.005
        arrays[f"{side}_camera_crossfit_raw_median"][:] = 0.003
        arrays[f"{side}_camera_crossfit_residual_median"][:] = 0.001
        arrays[f"{side}_camera_crossfit_residual_reduction"][:] = 0.40
        arrays[f"{side}_background_residual_reduction"][:] = 0.40
    return arrays


def _gate_fixture(
    positive_count: int = 100,
    *,
    include_instruction_mismatch: int = 0,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    negative_types = ["static"] * 40 + ["endpoint_only"] * 2
    negative_types += ["instruction_mismatch"] * include_instruction_mismatch
    count = positive_count + len(negative_types)
    arrays = _usable_arrays(count)
    rows: list[dict[str, object]] = [
        {"iid": f"positive-{index}", "positive": True}
        for index in range(positive_count)
    ]
    for offset, negative_type in enumerate(negative_types):
        index = positive_count + offset
        arrays["positive"][index] = False
        for side in ("source", "target"):
            arrays[f"{side}_usable"][index] = False
            arrays[f"{side}_base_valid"][index] = False
            arrays[f"{side}_audit_available"][index] = False
            arrays[f"{side}_audit_pass"][index] = False
            arrays[f"{side}_camera_crossfit_valid"][index] = False
        rows.append(
            {
                "iid": f"negative-{offset}",
                "positive": False,
                "negative_type": negative_type,
                "action_signature": f"negative:{negative_type}",
            }
        )
    return arrays, rows


def _output_row(
    input_row: dict[str, object],
    index: int,
    rank: int,
) -> dict[str, object]:
    result = {
        "status": "usable",
        "usable": True,
        "failure_reason": None,
        "resolved_path": f"/data/{index}.mp4",
        "video_sha256": None,
    }
    return {
        "schema_version": R7_ROW_SCHEMA,
        "input_index": index,
        "shard_array_index": 0,
        "shard_rank": rank,
        "world_size": FINAL_WORLD_SIZE,
        "iid": input_row["iid"],
        "input_row_sha256": _object_digest(input_row),
        "positive": True,
        "source": dict(result),
        "target": dict(result),
        "paired_usable": True,
    }


class SamplingHelperTests(unittest.TestCase):
    def test_fixed_sampling_resize_and_dino_offsets(self) -> None:
        indices = uniform_sample_indices(81, 32)
        self.assertEqual(indices.shape, (32,))
        self.assertEqual((int(indices[0]), int(indices[-1])), (0, 80))
        self.assertEqual(resized_dimensions(960, 704), (384, 282))
        self.assertEqual(resized_dimensions(200, 300), (200, 300))
        offsets = dino_frame_offsets()
        self.assertEqual(offsets.shape, (6,))
        self.assertEqual(len(np.unique(offsets)), 6)
        self.assertEqual((int(offsets[0]), int(offsets[-1])), (0, 31))

    def test_difference_hash_is_deterministic_and_dependency_free(self) -> None:
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        frame[:, 15:] = 255
        first = difference_hash(frame)
        second = difference_hash(frame.copy())
        self.assertEqual(first, second)
        self.assertEqual(len(first), 16)

    def test_torchrun_environment_resolution(self) -> None:
        environment = {"RANK": "3", "WORLD_SIZE": "8", "LOCAL_RANK": "1"}
        with patch.dict(os.environ, environment, clear=True):
            self.assertEqual(resolve_torchrun_coordinates(), (3, 8, 1))
        with self.assertRaisesRegex(ValueError, "rank"):
            resolve_torchrun_coordinates(rank=8, world_size=8)

    def test_sampled_times_use_real_indices_and_reject_duplicates(self) -> None:
        indices = np.asarray([0, 3, 8, 12], dtype=np.int64)
        np.testing.assert_allclose(
            sampled_frame_times(indices, 24.0),
            indices.astype(np.float64) / 24.0,
        )
        with self.assertRaisesRegex(
            PerVideoError,
            "duplicate_sampled_frames",
        ):
            sampled_frame_times(
                np.asarray([0, 1, 1, 2], dtype=np.int64),
                24.0,
            )

    def test_extraction_contract_disables_backward_tracking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "input.jsonl"
            checkpoint = root / "tracker.pth"
            manifest.write_text("{}\n", encoding="utf-8")
            checkpoint.write_bytes(b"checkpoint")
            contract = build_extraction_contract(
                input_manifest=manifest,
                data_root=root,
                rank=0,
                world_size=1,
                device="cuda:0",
                tracker_checkpoint=checkpoint,
                tracker_checkpoint_sha256=_file_digest(checkpoint),
                tracker_grid_size=10,
                dino_provenance={"model": "unit-test"},
                seed=17,
                teacher_config=TemporalTeacherConfig(),
            )
            self.assertFalse(contract["tracker"]["backward_tracking"])


class GateTests(unittest.TestCase):
    def test_gate_pass_is_still_formally_insufficient(self) -> None:
        arrays, rows = _gate_fixture()
        gate = compute_p0_gate(rows, arrays)
        self.assertTrue(gate["diagnostic_gate_passed"])
        self.assertEqual(gate["diagnostic_status"], "DIAGNOSTIC_FEATURE_READY")
        self.assertEqual(gate["formal_status"], "INSUFFICIENT")
        self.assertFalse(gate["production_decision"])
        self.assertFalse(gate["generation_authorized"])
        self.assertEqual(gate["counts"]["negative_no_action_rows"], 42)

    def test_gate_requires_paired_count_and_all_medians(self) -> None:
        arrays, rows = _gate_fixture(99)
        arrays["source_usable"][:20] = False
        arrays["target_audit_event_iou"][:99] = 0.69
        gate = compute_p0_gate(rows, arrays)
        self.assertFalse(gate["diagnostic_gate_passed"])
        self.assertFalse(
            gate["criteria"]["paired_usable_positive_events"]["passed"]
        )
        self.assertFalse(
            gate["criteria"][
                "median_independent_audit_event_window_iou"
            ]["passed"]
        )

    def test_low_audit_rows_do_not_disappear_after_screening(self) -> None:
        arrays, rows = _gate_fixture()
        arrays["target_usable"][:50] = False
        arrays["target_audit_available"][:50] = False
        arrays["target_audit_pass"][:50] = False
        # Deliberately leave high stored metrics behind.  Availability must
        # still force these 50 failed audits to zero similarity/unit error.
        gate = compute_p0_gate(rows, arrays)
        self.assertEqual(
            gate["counts"]["positive_target_audit_eligible"],
            100,
        )
        self.assertEqual(
            gate["counts"]["positive_target_audit_failed"],
            50,
        )
        self.assertAlmostEqual(
            gate["criteria"][
                "median_independent_audit_event_window_iou"
            ]["value"],
            0.375,
        )
        self.assertFalse(
            gate["criteria"][
                "median_independent_audit_event_window_iou"
            ]["passed"]
        )

    def test_static_camera_cannot_satisfy_crossfit_sample_gate(self) -> None:
        arrays, rows = _gate_fixture()
        arrays["target_camera_crossfit_raw_median"][:100] = 0.0
        arrays[
            "target_camera_crossfit_residual_reduction"
        ][:100] = 1.0
        gate = compute_p0_gate(rows, arrays)
        self.assertEqual(
            gate["counts"][
                "positive_target_camera_crossfit_motion_eligible"
            ],
            0,
        )
        self.assertFalse(
            gate["criteria"][
                "camera_crossfit_motion_eligible_samples"
            ]["passed"]
        )

    def test_no_action_false_events_exclude_instruction_mismatch(self) -> None:
        arrays, rows = _gate_fixture(include_instruction_mismatch=40)
        # Nine false events among static/endpoint-only exceed 20%; every
        # instruction-mismatch row may contain real motion and is excluded.
        arrays["target_usable"][100:109] = True
        arrays["target_usable"][142:] = True
        arrays["target_base_valid"][100:109] = True
        arrays["target_base_valid"][142:] = True
        gate = compute_p0_gate(rows, arrays)
        self.assertEqual(gate["counts"]["negative_no_action_rows"], 42)
        self.assertEqual(
            gate["counts"]["negative_instruction_mismatch_excluded"],
            40,
        )
        self.assertEqual(
            gate["counts"]["negative_no_action_false_events"],
            9,
        )
        self.assertFalse(
            gate["criteria"][
                "no_action_negative_false_event_fraction"
            ]["passed"]
        )


class CommitTests(unittest.TestCase):
    def test_shard_commit_resume_validation_and_tamper_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "input.jsonl"
            input_rows = _write_input_manifest(manifest, 8)
            directory = rank_directory(root / "run", 0, FINAL_WORLD_SIZE)
            arrays = _usable_arrays(1)
            arrays["input_indices"][0] = 0
            contract = {
                "rank": 0,
                "world_size": FINAL_WORLD_SIZE,
                "input_manifest_sha256": _file_digest(manifest),
            }
            _commit_shard(
                directory=directory,
                rows=[_output_row(input_rows[0], 0, 0)],
                arrays=arrays,
                contract=contract,
                input_rows=8,
            )
            result = validate_shard(
                directory,
                expected_contract=contract,
                input_manifest=manifest,
            )
            self.assertEqual(result["summary"]["rows"], 1)
            with (directory / "manifest.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                validate_shard(directory)

    def test_finalize_merges_exactly_eight_shards_in_input_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "input.jsonl"
            input_rows = _write_input_manifest(manifest, FINAL_WORLD_SIZE)
            run = root / "run"
            for rank in range(FINAL_WORLD_SIZE):
                arrays = _usable_arrays(1)
                arrays["input_indices"][0] = rank
                contract = {
                    "rank": rank,
                    "world_size": FINAL_WORLD_SIZE,
                    "input_manifest_sha256": _file_digest(manifest),
                }
                _commit_shard(
                    directory=rank_directory(run, rank, FINAL_WORLD_SIZE),
                    rows=[_output_row(input_rows[rank], rank, rank)],
                    arrays=arrays,
                    contract=contract,
                    input_rows=FINAL_WORLD_SIZE,
                )
            done = finalize_shards(
                input_manifest=manifest,
                output_root=run,
            )
            self.assertEqual(done["rows"], FINAL_WORLD_SIZE)
            self.assertEqual(done["formal_status"], "INSUFFICIENT")
            final = validate_final(run / "final", input_manifest=manifest)
            self.assertEqual(
                final["arrays"]["input_indices"].tolist(),
                list(range(FINAL_WORLD_SIZE)),
            )
            self.assertFalse(final["summary"]["generation_authorized"])
            resumed = finalize_shards(
                input_manifest=manifest,
                output_root=run,
                resume=True,
            )
            self.assertEqual(resumed, done)


if __name__ == "__main__":
    unittest.main()
