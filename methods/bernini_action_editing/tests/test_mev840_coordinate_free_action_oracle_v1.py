from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mev840_coordinate_free_action_oracle_v1 as oracle  # noqa: E402


ASSET = ROOT / "assets" / "mev840_target_action_oracle_sparse_v1.json"


class CoordinateFreeOracleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target = json.loads(ASSET.read_text(encoding="ascii"))

    def test_sparse_target_validates_and_self_scores_exactly(self):
        oracle.validate_representation(self.target)
        score = oracle.score_representations(self.target, self.target)
        self.assertEqual(score["scores"]["action"], 1.0)
        self.assertTrue(score["decision"]["action_gate_passed"])
        self.assertTrue(score["decision"]["appearance_quality_gate_external_required"])
        self.assertIsNone(score["decision"]["appearance_quality_gate_passed"])
        self.assertFalse(score["decision"]["selection_authorized"])
        self.assertFalse(score["authority"]["generator_read_authorized"])

    def test_target_leakage_fields_are_rejected_even_when_false(self):
        for key in ("rgb", "mask_sha256", "feature_tensor", "bbox_xyxy", "latent"):
            value = copy.deepcopy(self.target)
            value[key] = False
            with self.assertRaises(oracle.CoordinateFreeActionOracleError):
                oracle.validate_representation(value)

    def test_failed_place_candidate_scores_below_exact_target(self):
        candidate = copy.deepcopy(self.target)
        for row in candidate["phase_relations"]:
            row["object_motion_progress"] = 0.0
            row["object_incremental_motion"] = 0.0
            row["object_recipient_gap_ratio"] = 1.0
            row["object_recipient_contact"] = 0.0
            row["agent_object_contact"] = 1.0
        candidate["events"]["recipient_contact_start"] = None
        candidate["events"]["release"] = None
        candidate["representation_digest"] = oracle.object_sha256(
            oracle.representation_payload(candidate)
        )
        score = oracle.score_representations(self.target, candidate)
        self.assertLess(score["scores"]["action"], 0.78)
        self.assertFalse(score["decision"]["action_gate_passed"])


if __name__ == "__main__":
    unittest.main()
