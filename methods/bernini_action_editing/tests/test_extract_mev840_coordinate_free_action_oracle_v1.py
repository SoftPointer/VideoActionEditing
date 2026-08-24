from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import extract_mev840_coordinate_free_action_oracle_v1 as extractor  # noqa: E402
import mev840_coordinate_free_action_oracle_v1 as oracle  # noqa: E402


SPEC = ROOT / "assets" / "mev840_target_frozen_sam2_action_observer_spec_v1.json"


def numpy_gap(left: np.ndarray, right: np.ndarray) -> float:
    if np.logical_and(left, right).any():
        return 0.0
    left_points = np.argwhere(left)
    right_points = np.argwhere(right)
    return float(
        np.sqrt(
            np.square(left_points[:, None] - right_points[None, :]).sum(axis=2)
        ).min()
    )


def synthetic_masks() -> dict[str, np.ndarray]:
    masks = {
        name: np.zeros((81, 48, 80), dtype=np.bool_)
        for name in extractor.ROLE_NAMES
    }
    for frame in range(81):
        phase = frame / 4.0
        x = 64 if phase >= 18 else min(23 + int(round(2.0 * phase)), 64)
        masks["human_agent"][frame, 12:44, 5:26] = True
        masks["moving_object"][frame, 25:31, x : x + 4] = True
        masks["recipient"][frame, 20:45, 68:79] = True
        head_width = max(5, 12 - int(phase // 3))
        masks["head"][frame, 4:16, 10 : 10 + head_width] = True
    return masks


class FrozenSAM2ActionObserverTests(unittest.TestCase):
    def test_target_spec_is_static_and_selection_only(self):
        spec = json.loads(SPEC.read_text(encoding="utf-8"))
        extractor._validate_spec(spec)
        self.assertEqual(spec["video"]["role"], "real_target_oracle")
        self.assertFalse(spec["claim_limits"]["generator_read_authorized"])
        self.assertFalse(spec["claim_limits"]["raw_masks_exported"])

    def test_synthetic_masks_reduce_to_sanitized_21_phase_abi(self):
        with mock.patch.object(extractor, "_mask_gap", side_effect=numpy_gap):
            representation, matrix = extractor._relations_from_masks(synthetic_masks())
        oracle.validate_representation(representation)
        self.assertEqual(matrix.shape, (21, len(oracle.CHANNELS)))
        self.assertEqual(matrix.dtype, np.float32)
        encoded = json.dumps(representation, sort_keys=True).lower()
        for forbidden in (
            "path",
            "rgb",
            "mask",
            "bbox",
            "_xy",
            "feature",
            "embedding",
            "latent",
            "gaussian",
            "sha256",
        ):
            self.assertNotIn(forbidden, encoded)
        self.assertIsNotNone(representation["events"]["recipient_contact_start"])
        self.assertIsNotNone(representation["events"]["release"])

    def test_frame0_correspondence_rejects_an_unmatched_scene(self):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        same = extractor._frame0_correspondence(frame, frame.copy())
        self.assertTrue(same["gate_passed"])
        with self.assertRaises(extractor.FrozenSAM2ActionObserverError):
            extractor._frame0_correspondence(frame, np.full_like(frame, 255))

    def test_contact_latch_and_terminal_only_singleton_release(self):
        agent = [True] * 20 + [False]
        gaps = [1.0] * 16 + [0.039, 0.0, 0.044, 0.078, 0.102]
        recipient, contact_end, recipient_start, release = extractor._infer_contact_events(
            agent, gaps
        )
        self.assertEqual(recipient_start, 16)
        self.assertEqual(recipient, [False] * 16 + [True] * 5)
        self.assertEqual(contact_end, 20)
        self.assertEqual(release, 20)

        # The same singleton away from the terminal phase is not accepted.
        early = [True] * 21
        early[18] = False
        _, early_end, _, early_release = extractor._infer_contact_events(early, gaps)
        self.assertIsNone(early_end)
        self.assertIsNone(early_release)


if __name__ == "__main__":
    unittest.main()
