from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_mev840_candidate_action_observer_batch_v1 as batch  # noqa: E402


BASE_SPEC = ROOT / "assets" / "mev840_target_frozen_sam2_action_observer_spec_v1.json"
FORMAL6_PENDING = (
    ROOT / "assets" / "mev840_candidate_action_observer_formal6_pending_v1.json"
)
FORMAL6_IDS = [
    "p0_s2027",
    "p0_s2028",
    "p1_s2027",
    "p1_s2028",
    "p2_s2027",
    "p2_s2028",
]


def manifest() -> dict:
    return {
        "schema_version": batch.SCHEMA,
        "case_id": "MEV840",
        "output_root": "/fresh/output",
        "extractor": {"path": "/extractor.py", "sha256": "a" * 64},
        "oracle_program": {"path": "/oracle.py", "sha256": "b" * 64},
        "base_target_spec": {"path": "/base.json", "sha256": "c" * 64},
        "target_action_oracle": {
            "path": "/target-action.json",
            "sha256": "d" * 64,
            "representation_digest": "e" * 64,
        },
        "source_initial_reference": {
            "path": "/source-reference-656x368.mp4",
            "sha256": "f" * 64,
            "frame_count": 81,
            "fps": 25.0,
            "width": 656,
            "height": 368,
            "derivation": {
                "original_path": "/source-exact81.mp4",
                "original_sha256": "2" * 64,
                "original_frame_count": 81,
                "original_fps": 25.0,
                "original_width": 1280,
                "original_height": 720,
                "target_width": 656,
                "target_height": 368,
                "algorithm": "ffmpeg_scale_bicubic_libx264_preset_veryslow_crf1_yuv420p_r25",
                "ffmpeg_path": "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/mev840_candidate_action_observer_v1_20260822_control/ffmpeg_4.4.2_authority",
                "ffmpeg_sha256": "36d94a605d612e4090d1b8aec889d0c0801c6eafb1593c90f5c0dfd2e2966a45",
            },
        },
        "candidates": [
            {
                "candidate_id": "p1_seed_101",
                "path": "/candidate.mp4",
                "sha256": "1" * 64,
                "frame_count": 81,
                "fps": 25.0,
                "width": 656,
                "height": 368,
            }
        ],
        "authority": {
            "post_generation_only": True,
            "all_candidate_media_closed_before_observer_start": True,
            "generator_process_reads_manifest": False,
            "generator_process_reads_target_action": False,
            "generator_process_reads_real_target_media": False,
            "observer_calls_generator": False,
            "training_authorized": False,
            "optimizer_updates": 0,
            "failed_candidate_policy": "unassigned_reject",
            "appearance_quality_gate_external_required": True,
            "appearance_quality_gate_passed": None,
        },
    }


class CandidateActionObserverBatchTests(unittest.TestCase):
    def test_manifest_is_strictly_post_generation(self):
        value = manifest()
        batch._validate_manifest(value)
        self.assertFalse(value["authority"]["generator_process_reads_target_action"])
        self.assertFalse(value["authority"]["observer_calls_generator"])
        self.assertTrue(value["authority"]["appearance_quality_gate_external_required"])
        self.assertIsNone(value["authority"]["appearance_quality_gate_passed"])

    def test_shared_frame0_boxes_are_scaled_without_export_to_action_abi(self):
        base = json.loads(BASE_SPEC.read_text(encoding="utf-8"))
        value = manifest()
        candidate = value["candidates"][0]
        result = batch._scaled_candidate_spec(
            base, candidate, value["source_initial_reference"]
        )
        self.assertEqual(result["video"]["role"], "generated_candidate")
        self.assertEqual((result["video"]["width"], result["video"]["height"]), (656, 368))
        source_box = base["roles"][1]["box_xyxy"]
        scaled_box = result["roles"][1]["box_xyxy"]
        self.assertAlmostEqual(scaled_box[0], source_box[0] * 656 / 960)
        self.assertAlmostEqual(scaled_box[1], source_box[1] * 368 / 540)
        self.assertEqual(
            result["source_initial_reference"]["derivation"]["original_width"], 1280
        )
        self.assertEqual(
            result["source_initial_reference"]["derivation"]["target_width"], 656
        )
        self.assertFalse(result["claim_limits"]["absolute_coordinates_exported_to_representation"])

    def test_duplicate_candidate_ids_fail_closed(self):
        value = manifest()
        value["candidates"].append(dict(value["candidates"][0]))
        with self.assertRaises(batch.CandidateActionObserverBatchError):
            batch._validate_manifest(value)

    def test_formal6_pending_interface_is_complete_but_not_runnable(self):
        value = json.loads(FORMAL6_PENDING.read_text(encoding="utf-8"))
        self.assertEqual(
            [row["candidate_id"] for row in value["candidates"]], FORMAL6_IDS
        )
        self.assertEqual(
            value["formal6_interface"]["required_candidate_ids_in_order"],
            FORMAL6_IDS,
        )
        self.assertTrue(
            value["formal6_interface"]["external_gate_contract"][
                "single_bottle_gate_external_required"
            ]
        )
        self.assertIsNone(
            value["formal6_interface"]["external_gate_contract"][
                "single_bottle_gate_passed"
            ]
        )
        self.assertFalse(
            value["formal6_interface"]["external_gate_contract"][
                "selection_authorized"
            ]
        )
        self.assertTrue(all(row["path"] is None for row in value["candidates"]))
        self.assertTrue(all(row["sha256"] is None for row in value["candidates"]))
        with self.assertRaises(batch.CandidateActionObserverBatchError):
            batch._validate_manifest(value)

    def test_formal6_requires_all_slots_before_manifest_can_validate(self):
        value = json.loads(FORMAL6_PENDING.read_text(encoding="utf-8"))
        value["authority"]["all_candidate_media_closed_before_observer_start"] = True
        for index, row in enumerate(value["candidates"]):
            row["path"] = f"/sealed/candidate-{index}.mp4"
            row["sha256"] = f"{index + 1:x}" * 64
        batch._validate_manifest(value)
        value["candidates"][4]["path"] = None
        # The schema check remains deliberately lightweight; the single-file
        # byte pin is enforced by _regular_exact during preflight, before the
        # fresh output directory is created.
        with self.assertRaises(batch.CandidateActionObserverBatchError):
            batch._regular_exact(
                value["candidates"][4]["path"],
                value["candidates"][4]["sha256"],
                "p2_s2027",
            )


if __name__ == "__main__":
    unittest.main()
