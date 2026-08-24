from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "tools" / "build_mev840_native_target_action_matrix_review_v1.py"
SPEC = importlib.util.spec_from_file_location("mev840_native_target_review_builder", BUILDER_PATH)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class NativeTargetActionMatrixReviewTests(unittest.TestCase):
    def test_metrics_keep_prompt_and_selection_claims_fail_closed(self):
        metrics = builder.build_metrics()
        self.assertEqual([row["candidate_id"] for row in metrics["candidates"]], builder.IDS)
        self.assertEqual(metrics["terminology"]["P0"], "base-action baseline; not null")
        self.assertIn("event-order", metrics["terminology"]["P1"])
        self.assertIn("same-bottle", metrics["terminology"]["P2"])
        self.assertFalse(metrics["decision"]["selection_authorized"])
        self.assertFalse(metrics["decision"]["scientific_claim_authorized"])
        self.assertEqual(metrics["design"]["pairing_scope"], "within_each_seed")
        self.assertFalse(metrics["design"]["cross_seed_source_condition_bit_exact"])
        self.assertIsNone(metrics["external_gates"]["appearance_quality_gate_passed"])
        self.assertIsNone(metrics["external_gates"]["single_bottle_gate_passed"])

    def test_five_scores_fail_and_one_observer_result_is_unassigned(self):
        metrics = builder.build_metrics()
        assigned = [row for row in metrics["candidates"] if row["action_score"] is not None]
        unassigned = [row for row in metrics["candidates"] if row["action_score"] is None]
        self.assertEqual(len(assigned), 5)
        self.assertTrue(all(row["action_score"] < builder.ACTION_THRESHOLD for row in assigned))
        self.assertEqual([row["candidate_id"] for row in unassigned], ["p1_s2028"])
        self.assertEqual(unassigned[0]["observer_reason"], "role_mask_became_empty")
        self.assertIsNone(unassigned[0]["action_gate_passed"])

    def test_generated_page_inventory_and_claims_validate(self):
        metrics = json.loads((builder.ROOT / "metrics.json").read_text(encoding="utf-8"))
        manifest = json.loads((builder.ROOT / "manifest.json").read_text(encoding="utf-8"))
        builder.validate_outputs(metrics, manifest)
        html = (builder.ROOT / "index.html").read_text(encoding="utf-8")
        self.assertEqual(html.count("<video "), 11)
        self.assertEqual(manifest["missing_reference_count"], 0)
        self.assertEqual(manifest["video_count"], 11)


if __name__ == "__main__":
    unittest.main()
