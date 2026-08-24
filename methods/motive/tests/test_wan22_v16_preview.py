from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from motive.goku_full_motion_qwen_v16 import (
    MOTION_EVIDENCE_SCHEMA,
    PASSED_SCHEMA,
    SOURCE_CAMERA_SCHEMA,
    SOURCE_CENSUS_SCHEMA,
    SOURCE_SUBJECT_SCHEMA,
    TARGET_CAMERA_SCHEMA,
    TARGET_COVERAGE_SCHEMA,
    TARGET_PLAN_SCHEMA,
    TARGET_SUBJECT_SCHEMA,
    compile_instruction,
    object_sha256,
)
from motive import wan22_i2v_batch as batch


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_jsonl(path: Path, row: dict) -> None:
    path.write_bytes(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _v16_passed_row() -> dict:
    iid = "v16-preview-0001"
    evidence = {
        "schema_version": MOTION_EVIDENCE_SCHEMA,
        "start_frame": 0,
        "end_frame": 80,
        "description": "the person's arms move upward across ordered frames",
    }
    census = {
        "schema_version": SOURCE_CENSUS_SCHEMA,
        "iid": iid,
        "dynamic_subjects": [
            {
                "schema_version": SOURCE_SUBJECT_SCHEMA,
                "subject_id": "subject_01",
                "entity_type": "person",
                "stable_reference": "the person in blue at frame center",
                "i0_bbox_xyxy_1000": [200, 100, 800, 980],
                "i0_state": "standing with both hands lowered",
                "source_action_signature": "raise_both_arms",
                "source_motion": "raises both arms from the waist overhead",
                "motion_evidence": [evidence],
                "dynamic": True,
            }
        ],
        "camera": {
            "schema_version": SOURCE_CAMERA_SCHEMA,
            "motion_class": "locked_off",
            "source_motion": "the camera remains completely locked off",
            "motion_evidence": [
                {
                    **evidence,
                    "description": "the background stays fixed in every ordered frame",
                }
            ],
        },
        "all_dynamic_subjects_enumerated": True,
        "crowd_or_unresolved_motion": False,
        "confidence": "high",
    }
    plan = {
        "schema_version": TARGET_PLAN_SCHEMA,
        "iid": iid,
        "dynamic_subject_targets": [
            {
                "schema_version": TARGET_SUBJECT_SCHEMA,
                "subject_id": "subject_01",
                "target_action_signature": "crouch_and_touch_floor",
                "target_motion": (
                    "immediately bends into a crouch and touches the floor "
                    "with both hands"
                ),
                "substantive_change": True,
            }
        ],
        "camera_target": {
            "schema_version": TARGET_CAMERA_SCHEMA,
            "relation": "preserve_static",
            "motion_class": "locked_off",
            "target_motion": "the camera remains completely locked off",
        },
        "coverage": {
            "schema_version": TARGET_COVERAGE_SCHEMA,
            "dynamic_subject_ids": ["subject_01"],
            "camera_covered": True,
        },
        "confidence": "high",
    }
    compiled = compile_instruction(census, plan)
    return {
        "schema_version": PASSED_SCHEMA,
        "iid": iid,
        "group_id": "group-v16-preview-0001",
        "family": "people",
        "source_video": "media/source.mp4",
        "resolved_source_video": "/frozen/media/source.mp4",
        "anchor_image": "media/anchor.png",
        "resolved_anchor_image": "/frozen/media/anchor.png",
        "source_video_sha256": "1" * 64,
        "anchor_sha256": "2" * 64,
        "strict_temporal_geometry": {
            "frame_count": 81,
            "fps": "25/1",
            "timeline_span_seconds": 3.2,
            "width": 1280,
            "height": 720,
        },
        "edit_instruction": compiled["instruction"],
        "edit_instruction_sha256": compiled["instruction_sha256"],
        "source_census": census,
        "target_plan": plan,
        "compiled_instruction": compiled,
        "qwen_record_digest": "3" * 64,
        "action_change_substantive": True,
        "all_dynamic_subjects_covered": True,
        "camera_covered": True,
        "human_review_status": "pending",
        "generation_authorized": False,
        "production_eligible": False,
    }


def _v6_row() -> dict:
    instruction = "Make every moving subject wave while keeping the camera locked."
    return {
        "schema_version": batch.FULL_MOTION_GENERATION_SCHEMA,
        "iid": "v6-preview-0001",
        "group_id": "group-v6-preview-0001",
        "family": "motion_editing",
        "source_video": "/frozen/source.mp4",
        "resolved_source_video": "/frozen/source.mp4",
        "source_video_sha256": "4" * 64,
        "anchor_image": "/frozen/anchor.png",
        "resolved_anchor_image": "/frozen/anchor.png",
        "anchor_sha256": "5" * 64,
        "edit_instruction": instruction,
        "edit_instruction_sha256": _sha(instruction.encode("utf-8")),
        "qwen_evidence": {
            "result_digest": "6" * 64,
            "provenance_digest": "7" * 64,
        },
        "motion_spec": {"schema_version": "fixture"},
        "motion_spec_sha256": "8" * 64,
        "action_change_substantive": "yes",
        "manifest_role": "review_proposal",
        "human_review_status": "pending",
        "generation_authorized": False,
        "production_eligible": False,
        "approval": None,
    }


class Wan22QwenV16PreviewTests(unittest.TestCase):
    def _load(self, row: dict) -> dict:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        manifest = Path(directory.name) / "preview.jsonl"
        _write_jsonl(manifest, row)
        return batch.load_non_production_preview_manifest(
            manifest,
            allow_pending_review=False,
            max_samples=None,
        )

    def test_valid_v16_row_is_deeply_loaded_and_runtime_stamped(self) -> None:
        raw = _v16_passed_row()
        loaded = self._load(raw)
        row = loaded["selected_rows"][0]
        self.assertEqual(row["_row_digest"], object_sha256(raw))
        self.assertEqual(
            row["_authorization_mode"],
            batch.QWEN_V16_NON_PRODUCTION_PREVIEW_AUTHORIZATION_MODE,
        )
        self.assertTrue(
            batch._is_non_production_preview_authorization(
                row["_authorization_mode"]
            )
        )
        self.assertEqual(row["manifest_role"], "pending_review")
        self.assertEqual(row["action_change_substantive"], "yes")
        self.assertFalse(row["production_eligible"])
        self.assertFalse(row["generation_authorized"])
        self.assertIsNone(row["approval"])
        row["_input_media"] = {
            "source_video_path": "/canonical/source.mp4",
            "anchor_path": "/canonical/anchor.png",
        }
        bindings = batch._non_production_preview_bindings(
            row, manifest_sha256="9" * 64
        )
        self.assertEqual(bindings["iid"], raw["iid"])
        self.assertEqual(bindings["source_census"], raw["source_census"])
        self.assertEqual(bindings["target_plan"], raw["target_plan"])
        self.assertEqual(
            bindings["compiled_instruction"], raw["compiled_instruction"]
        )
        self.assertEqual(bindings["qwen_record_digest"], "3" * 64)
        self.assertTrue(bindings["all_dynamic_subjects_covered"])
        self.assertTrue(bindings["camera_covered"])

    def test_v16_instruction_digest_and_coverage_tampering_fail_closed(self) -> None:
        cases = {}
        instruction = _v16_passed_row()
        instruction["edit_instruction"] += " Tampered."
        cases["instruction"] = instruction
        digest = _v16_passed_row()
        digest["qwen_record_digest"] = "not-a-sha256"
        cases["digest"] = digest
        coverage = _v16_passed_row()
        coverage["all_dynamic_subjects_covered"] = False
        cases["coverage"] = coverage
        for name, row in cases.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    batch.Wan22BatchError, "deep validation failed"
                ):
                    self._load(row)

    def test_v16_row_is_rejected_by_production_structure_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "v16.jsonl"
            _write_jsonl(manifest, _v16_passed_row())
            with self.assertRaisesRegex(
                batch.Wan22BatchError, "schema_version"
            ):
                batch.validate_generation_manifest_structure(
                    manifest,
                    allow_pending_review=False,
                    max_samples=None,
                )
            with self.assertRaisesRegex(
                batch.Wan22BatchError, "signed generation release"
            ):
                batch.load_generation_manifest(
                    manifest,
                    allow_pending_review=False,
                    max_samples=None,
                )

    def test_v6_preview_mode_and_bindings_are_unchanged(self) -> None:
        raw = _v6_row()
        with mock.patch(
            "motive.goku_full_motion_finalize.validate_generation_row",
            return_value=copy.deepcopy(raw),
        ) as validator:
            loaded = self._load(raw)
        validator.assert_called_once_with(raw)
        row = loaded["selected_rows"][0]
        self.assertEqual(
            row["_authorization_mode"],
            batch.NON_PRODUCTION_PREVIEW_AUTHORIZATION_MODE,
        )
        bindings = batch._non_production_preview_bindings(
            row, manifest_sha256="a" * 64
        )
        self.assertEqual(
            set(bindings),
            {
                "manifest_sha256",
                "manifest_row_digest",
                "edit_instruction_sha256",
                "qwen_result_digest",
                "qwen_provenance_digest",
            },
        )
        self.assertEqual(bindings["qwen_result_digest"], "6" * 64)
        self.assertEqual(bindings["qwen_provenance_digest"], "7" * 64)


if __name__ == "__main__":
    unittest.main()
