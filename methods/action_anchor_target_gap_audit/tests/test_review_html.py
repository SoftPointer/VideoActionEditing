from __future__ import annotations

import unittest

from methods.action_anchor_target_gap_audit import audit
from methods.action_anchor_target_gap_audit import build_review_html


class ReviewHtmlTests(unittest.TestCase):
    def setUp(self):
        self.samples = []
        self.qwen_rows = []
        self.sm_rows = []
        for ordinal in range(16):
            prefix = f"{ordinal:012x}"
            pair_id = prefix + "a" * 52
            self.samples.append({
                "pair_id": pair_id,
                "pair_prefix": prefix,
                "instruction": f"Perform action {ordinal}.",
                "source_action_caption": "The subject waits.",
                "target_action_caption": f"The subject performs action {ordinal}.",
                "source": {"path": f"/protected/{prefix}/source.mp4"},
                "real_target": {"path": f"/protected/{prefix}/target.mp4"},
                "generation": {
                    "normalized_source": {"path": f"/outside/{prefix}/source.mp4"},
                    "anchor": {"path": f"/outside/{prefix}/anchor.mp4"},
                    "frozen_base": {"path": f"/outside/{prefix}/base.mp4"},
                },
            })
            scores = {
                "action_semantics": 2.0,
                "temporal_order": 2.0,
                "action_completion": 2.0,
                "reference_motion_match": 2.0,
                "gate_score": 2.0,
                "evidence": [["Visible evidence pass 1."], ["Visible evidence pass 2."]],
            }
            self.qwen_rows.append({
                "pair_id": pair_id,
                "pair_prefix": prefix,
                "winner": "frozen_base" if ordinal % 2 else "anchor",
                "reason": "noncompensatory_gate_consistent_across_slot_swap",
                "gate_pass_winners": ["frozen_base", "frozen_base"],
                "direct_pairwise_winners": ["frozen_base", "frozen_base"],
                "role_scores": {"anchor": scores, "frozen_base": scores},
                "passes": [],
            })
            self.sm_rows.append({
                "pair_id": pair_id,
                "pair_prefix": prefix,
                "similarities": {
                    "m3": {"anchor_minus_frozen_base": -0.1},
                    "m23": {"anchor_minus_frozen_base": -0.01},
                },
            })
        self.manifest = {
            "schema_version": audit.MANIFEST_SCHEMA,
            "manifest_digest": "1" * 64,
            "samples": self.samples,
        }
        self.qwen = {
            "schema_version": audit.QWEN_SUMMARY_SCHEMA,
            "pairs": self.qwen_rows,
        }
        self.sm = {
            "schema_version": audit.SM_SUMMARY_SCHEMA,
            "manifest_digest": "1" * 64,
            "pairs": self.sm_rows,
        }

    def test_render_contains_four_synchronized_videos_per_case(self):
        cases = build_review_html.build_cases(self.manifest, self.qwen, self.sm)
        rendered = build_review_html.render_html(cases)
        self.assertEqual(rendered.count('class="case-card"'), 16)
        self.assertEqual(rendered.count("<video muted"), 64)
        self.assertIn("function seekGroup", rendered)
        self.assertIn("function tick", rendered)
        self.assertIn("−1 帧", rendered)
        self.assertNotIn("/protected/", rendered)
        self.assertNotIn("/outside/", rendered)

    def test_rejects_mismatched_pair_sets(self):
        self.sm["pairs"] = self.sm["pairs"][:-1]
        with self.assertRaisesRegex(ValueError, "pair sets differ"):
            build_review_html.build_cases(self.manifest, self.qwen, self.sm)


if __name__ == "__main__":
    unittest.main()
