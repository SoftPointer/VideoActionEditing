from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import checkpoint_visual_quality_gate_v3 as gate
import replay_checkpoint_visual_quality_gate_v3 as replay_gate


def synthetic_clean_high() -> np.ndarray:
    height, width = 288, 384
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    checker = ((xx // 12 + yy // 12) % 2).astype(np.float32)
    frames = np.empty((81, height, width, 3), dtype=np.uint8)
    for index in range(81):
        phase = index / 80.0
        shift = index * 2
        texture = np.roll(checker, shift, axis=1)
        wave = np.sin((xx + shift) * 0.055) * np.cos((yy - shift) * 0.043)
        red = np.clip(0.20 + 0.42 * xx / width + 0.13 * texture + 0.06 * wave, 0, 1)
        green = np.clip(0.18 + 0.45 * yy / height + 0.10 * texture - 0.05 * wave, 0, 1)
        blue = np.clip(0.24 + 0.20 * xx / width + 0.18 * yy / height + 0.04 * wave, 0, 1)
        frame = np.stack((red, green, blue), axis=-1)
        # A sharp moving object ensures the base is not a near-duplicate video.
        x0 = (17 + 5 * index) % (width - 48)
        y0 = 80 + int(25 * np.sin(phase * 2 * np.pi))
        frame[y0 : y0 + 36, x0 : x0 + 48] = (0.92, 0.08, 0.18)
        frames[index] = np.rint(frame * 255.0).astype(np.uint8)
    return frames


def scale_maps(high: np.ndarray) -> dict[str, np.ndarray]:
    accumulator = high[:, 0::2, 0::2].astype(np.uint16)
    accumulator += high[:, 1::2, 0::2]
    accumulator += high[:, 0::2, 1::2]
    accumulator += high[:, 1::2, 1::2]
    low = ((accumulator + 2) // 4).astype(np.uint8)
    return {"192x144": low, "384x288": high}


def box_blur_rgb(video: np.ndarray, window: int = 15) -> np.ndarray:
    result = np.empty_like(video)
    for channel in range(3):
        normalised = video[..., channel].astype(np.float32) / 255.0
        blurred = gate._box_mean(normalised, window)
        result[..., channel] = np.rint(np.clip(blurred, 0, 1) * 255).astype(np.uint8)
    return result


class CheckpointVisualQualityGateV3SyntheticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        clean = synthetic_clean_high()
        clean_maps = scale_maps(clean)
        cls.clean_report = gate.evaluate_visual_quality(
            clean_maps,
            clean_maps,
            frozen_base_frames_by_scale=clean_maps,
            metadata={"sample_id": "synthetic-clean"},
        )

        blurred = box_blur_rgb(clean)
        cls.blur_report = gate.evaluate_visual_quality(
            clean_maps,
            scale_maps(blurred),
            frozen_base_frames_by_scale=clean_maps,
            metadata={"sample_id": "synthetic-blur"},
        )
        del blurred

        random = np.random.default_rng(20260824)
        noisy = np.clip(
            clean.astype(np.float32) / 255.0
            + random.normal(0.0, 0.20, size=clean.shape).astype(np.float32),
            0.0,
            1.0,
        )
        noisy = np.rint(noisy * 255).astype(np.uint8)
        cls.noise_report = gate.evaluate_visual_quality(
            clean_maps,
            scale_maps(noisy),
            frozen_base_frames_by_scale=clean_maps,
            metadata={"sample_id": "synthetic-noise"},
        )
        del noisy

        partial = clean.copy()
        partial[20:37] = random.integers(
            0, 256, size=partial[20:37].shape, dtype=np.uint8
        )
        cls.partial_report = gate.evaluate_visual_quality(
            clean_maps,
            scale_maps(partial),
            frozen_base_frames_by_scale=clean_maps,
            metadata={"sample_id": "synthetic-partial-corruption"},
        )
        del partial

        frozen = np.repeat(clean[0:1], 81, axis=0)
        cls.freeze_report = gate.evaluate_visual_quality(
            clean_maps,
            scale_maps(frozen),
            frozen_base_frames_by_scale=clean_maps,
            metadata={"sample_id": "synthetic-freeze"},
        )

    def test_clean_passes_with_new_independent_schema(self) -> None:
        report = self.clean_report
        self.assertEqual(report["schema_version"], gate.SCHEMA_VERSION)
        self.assertNotIn("collapse-gate-v2", report["schema_version"])
        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["passed"])
        self.assertFalse(report["hard_artifact_failure"])
        self.assertFalse(report["unresolved"])
        self.assertEqual(set(report["features"]["scales"]), {"192x144", "384x288"})
        for scale in report["features"]["scales"].values():
            self.assertEqual(len(scale["frame_metrics"]), 81)
            self.assertEqual(len(scale["transition_metrics"]), 80)
            self.assertTrue(scale["windowed_ssim"]["implemented_as_local_map"])
            self.assertEqual(scale["windowed_ssim"]["window_width"], 11)
        json.dumps(report, allow_nan=False)

    def test_synthetic_blur_hard_fails_blur_family(self) -> None:
        report = self.blur_report
        self.assertEqual(report["status"], "fail")
        self.assertTrue(
            report["decision"]["evidence_families"]["BLUR"]["triggered"]
        )

    def test_synthetic_noise_hard_fails_noise_family(self) -> None:
        report = self.noise_report
        self.assertEqual(report["status"], "fail")
        self.assertTrue(
            report["decision"]["evidence_families"]["NOISE"]["triggered"]
        )

    def test_partial_corruption_is_not_hidden_by_clean_median(self) -> None:
        report = self.partial_report
        self.assertEqual(report["status"], "fail")
        self.assertTrue(report["hard_artifact_failure"])
        self.assertGreaterEqual(len(report["failure_codes"]), 1)

    def test_synthetic_freeze_hard_fails_freeze_family(self) -> None:
        report = self.freeze_report
        self.assertEqual(report["status"], "fail")
        self.assertTrue(
            report["decision"]["evidence_families"]["FREEZE"]["triggered"]
        )

    def test_ssim_is_windowed_not_a_global_alias(self) -> None:
        clean = np.zeros((1, 32, 40), dtype=np.float32)
        clean[:, :, :20] = 0.15
        clean[:, :, 20:] = 0.85
        swapped = clean[:, :, ::-1].copy()
        local = gate._windowed_ssim_per_frame(clean, swapped)[0]
        global_value = gate._global_ssim_per_frame(clean, swapped)[0]
        self.assertGreater(abs(float(local) - float(global_value)), 0.05)

    def test_missing_second_scale_fails_closed(self) -> None:
        high = synthetic_clean_high()
        incomplete = {"384x288": high}
        report = gate.evaluate_visual_quality(
            incomplete,
            incomplete,
            frozen_base_frames_by_scale=incomplete,
        )
        self.assertEqual(report["status"], "error")
        self.assertFalse(report["passed"])
        self.assertTrue(report["fail_closed"])


class LabelledReplayContractTest(unittest.TestCase):
    def test_replay_inventory_is_exactly_seven_pass_thirteen_fail(self) -> None:
        public = (
            REPOSITORY
            / "artifacts/v16r3_s644_heldout8_site_20260824/public"
        )
        if not public.is_dir():
            self.skipTest("local labelled replay corpus is absent")
        rows = replay_gate._rows(public)
        self.assertEqual(len(rows), 20)
        self.assertEqual(sum(row["expected"] == "PASS" for row in rows), 7)
        self.assertEqual(sum(row["expected"] == "FAIL" for row in rows), 13)

    @unittest.skipUnless(
        os.environ.get("RUN_QUALITY_GATE_V3_REPLAY") == "1"
        and shutil.which("ffmpeg")
        and shutil.which("ffprobe"),
        "set RUN_QUALITY_GATE_V3_REPLAY=1 for the full labelled replay",
    )
    def test_full_labelled_replay_is_fail_closed_separating(self) -> None:
        public = (
            REPOSITORY
            / "artifacts/v16r3_s644_heldout8_site_20260824/public"
        )
        report = replay_gate.replay(public)
        self.assertEqual(
            report["validation_status"],
            "labelled_replay_not_independent_validation",
        )
        self.assertTrue(report["all_labels_replayed_correctly"])
        self.assertEqual(
            report["confusion_matrix"],
            {
                "expected_PASS_predicted_PASS": 7,
                "expected_PASS_predicted_FAIL": 0,
                "expected_FAIL_predicted_PASS": 0,
                "expected_FAIL_predicted_FAIL": 13,
            },
        )


if __name__ == "__main__":
    unittest.main()
