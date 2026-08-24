from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import checkpoint_visual_collapse_gate_v2 as gate


def source_video(case_index: int = 0) -> np.ndarray:
    height, width = 32, 40
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    xx /= width - 1
    yy /= height - 1
    frames = []
    for frame_index in range(81):
        phase = frame_index / 80.0
        shift = 0.015 * np.sin(2.0 * np.pi * phase + case_index * 0.2)
        red = np.clip(0.16 + 0.55 * xx + shift, 0.0, 1.0)
        green = np.clip(0.17 + 0.58 * yy + 0.025 * phase, 0.0, 1.0)
        blue = np.clip(0.22 + 0.20 * xx + 0.18 * yy - shift, 0.0, 1.0)
        frames.append(np.stack((red, green, blue), axis=-1))
    return np.stack(frames).astype(np.float32)


def frozen_base_video(source: np.ndarray, case_index: int = 0) -> np.ndarray:
    result = source.copy()
    for frame_index in range(81):
        phase = frame_index / 80.0
        result[frame_index] = np.roll(
            result[frame_index], (frame_index + case_index) // 12, axis=1
        )
        result[frame_index, :, :, 1] = np.clip(
            result[frame_index, :, :, 1] + 0.008 * np.sin(phase * np.pi),
            0.0,
            1.0,
        )
    return result


def collapsed_video(source: np.ndarray, case_index: int = 0) -> np.ndarray:
    random = np.random.default_rng(1000 + case_index)
    result = source.copy()
    yy, xx = np.mgrid[0 : source.shape[1], 0 : source.shape[2]].astype(np.float32)
    pattern = 0.08 * np.sin(xx * 1.7 + yy * 0.9)[..., None]
    for frame_index in range(1, 81):
        colour = random.uniform(0.08, 0.92, size=(1, 1, 3)).astype(np.float32)
        noise = random.normal(0.0, 0.08, size=source.shape[1:]).astype(np.float32)
        result[frame_index] = np.clip(colour + pattern + noise, 0.0, 1.0)
    return result


def synthetic_calibration() -> dict:
    examples = []
    for case_index in range(3):
        source = source_video(case_index)
        base = frozen_base_video(source, case_index)
        collapsed = collapsed_video(source, case_index)
        examples.append(
            gate.CalibrationExample(
                case_id=f"synthetic{case_index}",
                source=source,
                frozen_base=base,
                collapsed=collapsed,
            )
        )
    return gate.calibrate_examples(examples, cohort_name="synthetic-unit-cohort")


class CheckpointVisualCollapseGateV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.calibration = synthetic_calibration()

    def test_calibration_is_strict_self_checked_and_json_safe(self) -> None:
        calibration = self.calibration
        self.assertEqual(calibration["case_count"], 3)
        self.assertTrue(calibration["self_check"]["passed"])
        self.assertEqual(calibration["self_check"]["clean_actual_pass_count"], 3)
        self.assertEqual(calibration["self_check"]["collapsed_actual_fail_count"], 3)
        for metric in calibration["required_separating_metrics"]:
            self.assertTrue(calibration["thresholds"][metric]["fully_separating"])
            self.assertGreater(
                calibration["thresholds"][metric]["strict_separation_margin"], 0.0
            )
        json.dumps(calibration, allow_nan=False)

    def test_clean_base_passes_and_all_frames_are_reported(self) -> None:
        source = source_video()
        base = frozen_base_video(source)
        report = gate.evaluate_visual_collapse(
            source,
            base,
            frozen_base_frames=base,
            calibration=self.calibration,
            metadata={"sample_id": "clean", "checkpoint_step": 1},
        )

        self.assertTrue(report["passed"])
        self.assertFalse(report["collapsed"])
        self.assertEqual(report["failure_codes"], [])
        self.assertEqual(len(report["features"]["frame_metrics"]), 81)
        self.assertEqual(len(report["features"]["transition_metrics"]), 80)
        self.assertTrue(report["features"]["all_frames_evaluated"])
        self.assertEqual(
            [row["frame_index"] for row in report["features"]["frame_metrics"]],
            list(range(81)),
        )
        json.dumps(report, allow_nan=False)

    def test_dynamic_noise_collapse_fails_with_multiple_evidence_families(self) -> None:
        source = source_video()
        base = frozen_base_video(source)
        collapsed = collapsed_video(source)
        report = gate.evaluate_visual_collapse(
            source,
            collapsed,
            frozen_base_frames=base,
            calibration=self.calibration,
            metadata={"sample_id": "bad", "checkpoint_step": 64},
        )

        self.assertFalse(report["passed"])
        self.assertTrue(report["collapsed"])
        self.assertIn("calibrated_visual_collapse", report["failure_codes"])
        families = report["decision"]["evidence_families"]
        self.assertTrue(families["temporal_incoherence"]["triggered"])
        self.assertTrue(families["temporal_color_incoherence"]["triggered"])
        self.assertTrue(families["source_structure_divergence"]["triggered"])
        self.assertTrue(families["frozen_base_relative_divergence"]["triggered"])

    def test_source_only_gate_still_catches_calibrated_collapse(self) -> None:
        source = source_video(1)
        report = gate.evaluate_visual_collapse(
            source,
            collapsed_video(source, 1),
            calibration=self.calibration,
            metadata={"sample_id": "source-only", "checkpoint_step": 8},
        )
        self.assertFalse(report["passed"])
        self.assertTrue(report["collapsed"])
        self.assertFalse(report["features"]["frozen_base_supplied"])

    def test_coherent_appearance_shift_is_not_called_collapse_by_structure_alone(self) -> None:
        source = source_video()
        base = frozen_base_video(source)
        coherent = np.clip(base * 0.65 + 0.23, 0.0, 1.0)
        report = gate.evaluate_visual_collapse(
            source,
            coherent,
            frozen_base_frames=base,
            calibration=self.calibration,
        )
        self.assertTrue(report["passed"])
        self.assertTrue(
            report["decision"]["evidence_families"][
                "source_structure_divergence"
            ]["triggered"]
        )
        self.assertFalse(
            report["decision"]["evidence_families"]["temporal_incoherence"][
                "triggered"
            ]
        )

    def test_blackout_and_freeze_are_hard_failures(self) -> None:
        source = source_video()
        base = frozen_base_video(source)
        blackout = base.copy()
        blackout[1:] = 0.0
        blackout_report = gate.evaluate_visual_collapse(
            source,
            blackout,
            frozen_base_frames=base,
            calibration=self.calibration,
        )
        self.assertIn("trajectory_blackout", blackout_report["failure_codes"])

        frozen = np.repeat(base[0:1], 81, axis=0)
        frozen_report = gate.evaluate_visual_collapse(
            source,
            frozen,
            frozen_base_frames=base,
            calibration=self.calibration,
        )
        self.assertIn("trajectory_freeze", frozen_report["failure_codes"])

    def test_incomplete_input_and_tampered_calibration_fail_closed(self) -> None:
        source = source_video()
        base = frozen_base_video(source)
        incomplete = gate.evaluate_visual_collapse(
            source,
            base[:-1],
            frozen_base_frames=base,
            calibration=self.calibration,
        )
        self.assertEqual(incomplete["status"], "error")
        self.assertFalse(incomplete["passed"])
        self.assertEqual(
            incomplete["failure_codes"],
            ["input_or_calibration_contract_violation"],
        )

        tampered = copy.deepcopy(self.calibration)
        tampered["thresholds"]["source_l1_mean"]["threshold"] += 0.01
        invalid = gate.evaluate_visual_collapse(
            source,
            base,
            frozen_base_frames=base,
            calibration=tampered,
        )
        self.assertEqual(invalid["status"], "error")
        self.assertIn("fingerprint", invalid["error"])


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg absent")
class Heldout8CalibrationIntegrationTest(unittest.TestCase):
    def test_existing_heldout8_base_and_v16r3_are_strictly_separated(self) -> None:
        corpus = (
            REPOSITORY
            / "artifacts"
            / "v16r3_s644_heldout8_site_20260824"
            / "public"
            / "assets"
            / "media"
        )
        if not corpus.is_dir():
            self.skipTest("Heldout8 visual calibration corpus is absent")
        calibration = gate.calibrate_video_corpus(corpus, collapsed_suffix="v16r3")

        self.assertEqual(calibration["case_count"], 8)
        self.assertEqual(
            calibration["case_ids"], [f"case{index:02d}" for index in range(8)]
        )
        self.assertTrue(calibration["self_check"]["passed"])
        self.assertEqual(calibration["self_check"]["clean_actual_pass_count"], 8)
        self.assertEqual(calibration["self_check"]["collapsed_actual_fail_count"], 8)
        self.assertGreater(
            calibration["thresholds"]["temporal_frame_l1_median"][
                "strict_separation_margin"
            ],
            0.0,
        )
        self.assertGreater(
            calibration["thresholds"]["source_l1_mean"][
                "strict_separation_margin"
            ],
            0.0,
        )
        for row in calibration["examples"]:
            self.assertEqual(row["identities"]["source"]["decoded_frame_count"], 81)
            self.assertEqual(
                row["identities"]["frozen_base"]["decoded_frame_count"], 81
            )
            self.assertEqual(
                row["identities"]["collapsed"]["decoded_frame_count"], 81
            )


if __name__ == "__main__":
    unittest.main()
