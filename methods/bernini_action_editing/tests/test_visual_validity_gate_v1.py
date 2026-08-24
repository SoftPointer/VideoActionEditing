from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import visual_validity_gate_v1 as gate


def clean_moving_video() -> np.ndarray:
    """Smooth, non-frozen exact81 RGB fixture with modest natural-like motion."""

    height, width = 32, 40
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    xx /= width - 1
    yy /= height - 1
    frames = []
    for index in range(81):
        phase = index / 80.0
        red = np.clip(0.18 + 0.55 * xx + 0.08 * np.sin(phase * np.pi), 0, 1)
        green = np.clip(0.15 + 0.60 * yy + 0.10 * phase, 0, 1)
        blue = np.clip(0.22 + 0.28 * xx + 0.20 * yy - 0.08 * phase, 0, 1)
        frames.append(np.stack((red, green, blue), axis=-1))
    return np.stack(frames).astype(np.float32)


class VisualValidityGateTest(unittest.TestCase):
    def test_clean_full_trajectory_passes_and_is_json_safe(self) -> None:
        video = clean_moving_video()
        report = gate.evaluate_visual_validity(video, reference_frames=video.copy())

        self.assertTrue(report["passed"])
        self.assertTrue(report["publishable"])
        self.assertFalse(report["phase0_only"])
        self.assertEqual(report["failure_codes"], [])
        trajectory = report["trajectory"]
        self.assertEqual(trajectory["evaluated_frame_count"], 81)
        self.assertEqual(trajectory["post_onset_frame_count"], 80)
        self.assertEqual(trajectory["evaluated_transition_count"], 80)
        self.assertEqual(len(trajectory["frame_metrics"]), 81)
        self.assertEqual(len(trajectory["temporal_segments"]), 4)
        self.assertAlmostEqual(
            trajectory["reference_metrics"]["full_trajectory_global_ssim_mean"],
            1.0,
            places=7,
        )
        json.dumps(report, allow_nan=False)

    def test_normal_phase0_followed_by_black_frames_fails_closed(self) -> None:
        video = clean_moving_video()
        video[1:] = 0.0
        report = gate.evaluate_visual_validity(video)

        self.assertFalse(report["passed"])
        self.assertFalse(report["publishable"])
        self.assertIn("post_onset_blackout", report["failure_codes"])
        self.assertIn("post_onset_median_luma_collapse", report["failure_codes"])
        self.assertEqual(report["trajectory"]["black_frame_indices"], list(range(1, 81)))
        self.assertGreater(report["trajectory"]["phase0_to_frame1_l1"], 0.1)

    def test_normal_phase0_followed_by_changing_colour_noise_fails(self) -> None:
        video = clean_moving_video()
        random = np.random.default_rng(20260818)
        video[1:] = random.random(video[1:].shape, dtype=np.float32)
        report = gate.evaluate_visual_validity(video)

        self.assertFalse(report["publishable"])
        self.assertIn("post_onset_spatial_color_noise", report["failure_codes"])
        self.assertIn("post_onset_temporal_incoherence", report["failure_codes"])
        self.assertGreater(
            len(report["trajectory"]["spatial_color_noise_frame_indices"]), 70
        )

    def test_static_colour_noise_is_caught_even_without_later_temporal_noise(self) -> None:
        video = clean_moving_video()
        random = np.random.default_rng(7)
        noisy = random.random(video.shape[1:], dtype=np.float32)
        video[1:] = noisy
        report = gate.evaluate_visual_validity(video)

        self.assertFalse(report["publishable"])
        self.assertIn("post_onset_spatial_color_noise", report["failure_codes"])
        self.assertIn("post_onset_temporal_freeze", report["failure_codes"])
        self.assertNotIn("post_onset_temporal_incoherence", report["failure_codes"])

    def test_phase0_match_cannot_hide_later_reference_collapse(self) -> None:
        reference = clean_moving_video()
        candidate = reference.copy()
        candidate[1:] = 0.0
        report = gate.evaluate_visual_validity(
            candidate, reference_frames=reference
        )
        reference_metrics = report["trajectory"]["reference_metrics"]

        self.assertAlmostEqual(reference_metrics["phase0_global_ssim"], 1.0, places=7)
        self.assertLess(reference_metrics["post_onset_global_ssim_mean"], 0.1)
        self.assertLess(
            reference_metrics["full_trajectory_global_ssim_mean"], 0.2
        )
        self.assertFalse(report["publishable"])
        self.assertTrue(
            all(segment["frame_count"] > 0 for segment in report["trajectory"]["temporal_segments"])
        )

    def test_incomplete_nonfinite_and_out_of_range_decodes_fail_contract(self) -> None:
        fixtures = []
        incomplete = clean_moving_video()[:-1]
        fixtures.append(incomplete)
        nonfinite = clean_moving_video()
        nonfinite[40, 0, 0, 0] = np.nan
        fixtures.append(nonfinite)
        out_of_range = clean_moving_video()
        out_of_range[60, 0, 0, 1] = 1.5
        fixtures.append(out_of_range)

        for fixture in fixtures:
            with self.subTest(shape=fixture.shape):
                report = gate.evaluate_visual_validity(fixture)
                self.assertFalse(report["input_contract_passed"])
                self.assertFalse(report["publishable"])
                self.assertEqual(
                    report["failure_codes"], ["input_contract_violation"]
                )

    def test_require_gate_raises_instead_of_publishing_failure(self) -> None:
        video = clean_moving_video()
        video[1:] = 0.0
        with self.assertRaisesRegex(
            gate.VisualValidityError, "post_onset_blackout"
        ):
            gate.require_visual_validity(video)

    def test_uint8_input_is_supported(self) -> None:
        video = np.rint(clean_moving_video() * 255.0).astype(np.uint8)
        report = gate.require_visual_validity(video)
        self.assertTrue(report["publishable"])


if __name__ == "__main__":
    unittest.main()
