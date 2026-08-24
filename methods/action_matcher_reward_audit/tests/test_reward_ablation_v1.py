from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_ROOT = METHOD_ROOT.parent / "semantic_moments_reward_audit"
for root in (METHOD_ROOT, SEMANTIC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from run_reward_ablation_v1 import (  # noqa: E402
    _composite_selection,
    appearance_set_similarity,
    camera_agreement,
    fixed_grid_similarity,
    preservation_pool,
    quality_score,
)


class RewardAblationV1Test(unittest.TestCase):
    def test_identity_proxies_reach_one(self) -> None:
        generator = torch.Generator().manual_seed(7)
        sequence = torch.randn(8, 12, generator=generator)
        dense = torch.randn(8, 16, 12, generator=generator)
        dense = torch.nn.functional.normalize(dense, dim=-1)
        self.assertAlmostEqual(appearance_set_similarity(sequence, sequence), 1.0, places=6)
        self.assertAlmostEqual(fixed_grid_similarity(dense, dense), 1.0, places=6)

    def test_raw_video_diagnostics_are_finite(self) -> None:
        generator = torch.Generator().manual_seed(11)
        frames = torch.rand(8, 3, 96, 96, generator=generator)
        self.assertAlmostEqual(camera_agreement(frames, frames), 1.0, places=6)
        score, detail = quality_score(frames, frames)
        self.assertTrue(0.0 <= score <= 1.0)
        self.assertTrue(all(math.isfinite(value) for value in detail.values()))

    def test_preservation_pool_uses_non_compensating_minimum(self) -> None:
        generator = torch.Generator().manual_seed(17)
        source = {
            "frame_sequence": torch.nn.functional.normalize(
                torch.randn(8, 12, generator=generator), dim=-1
            ),
            "dense_sequence": torch.nn.functional.normalize(
                torch.randn(8, 16, 12, generator=generator), dim=-1
            ),
            "raw_small": torch.rand(8, 3, 96, 96, generator=generator),
        }
        candidates = []
        for index, scale in enumerate((0.0, 0.3, 1.0)):
            candidates.append(
                {
                    "candidate_id": f"c{index}",
                    "feature": {
                        "frame_sequence": torch.nn.functional.normalize(
                            source["frame_sequence"] + scale * torch.randn(
                                source["frame_sequence"].shape, generator=generator
                            ),
                            dim=-1,
                        ),
                        "dense_sequence": torch.nn.functional.normalize(
                            source["dense_sequence"] + scale * torch.randn(
                                source["dense_sequence"].shape, generator=generator
                            ),
                            dim=-1,
                        ),
                        "raw_small": (
                            source["raw_small"] + scale * torch.rand(
                                source["raw_small"].shape, generator=generator
                            )
                        ).clamp(0, 1),
                    },
                }
            )
        result = preservation_pool(source, candidates)
        self.assertEqual(result["diagnostic_top_candidate_id"], "c0")
        self.assertFalse(result["absolute_thresholds_calibrated"])
        for row in result["rows"]:
            self.assertAlmostEqual(
                row["preservation_score"], min(row["pool_percentiles"].values())
            )

    def test_composite_abstains_on_small_action_gap(self) -> None:
        action = {
            "candidates": [
                {"candidate_id": "a", "eligible": True, "event_score": 0.8},
                {"candidate_id": "b", "eligible": True, "event_score": 0.7},
            ]
        }
        preservation = {
            "rows": [
                {"candidate_id": "a", "preservation_score": 1.0},
                {"candidate_id": "b", "preservation_score": 1.0},
            ]
        }
        result = _composite_selection(action, preservation)
        self.assertTrue(result["abstain_required"])
        self.assertIsNone(result["selected_candidate_id"])
        self.assertIn("top_two_gap_too_small", result["abstain_reasons"])


if __name__ == "__main__":
    unittest.main()
