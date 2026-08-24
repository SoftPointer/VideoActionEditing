from __future__ import annotations

import unittest

from methods.action_anchor_target_gap_audit.build_source_residue_review_html import render_html


class SourceResidueReviewHtmlTest(unittest.TestCase):
    def test_render_keeps_old_pages_and_synchronized_controls(self) -> None:
        case = {
            "pair_prefix": "abc", "ordinal": 0, "target_action": "turn", "manual_winner": "anchor",
            "qwen_winner": "anchor", "agrees": True, "human_note": "note", "pass_winners": ["anchor", "anchor"],
            "instruction": "edit", "source_caption": "source", "target_caption": "target",
            "source_aware_gates": {r: [4, 4] if r in {"anchor", "frozen_base"} else [4] for r in ("anchor", "frozen_base", "target_forward", "source_noop", "target_reverse", "target_shuffle")},
            "target_only_gates": {r: [4, 4] if r in {"anchor", "frozen_base"} else [4] for r in ("anchor", "frozen_base", "target_forward", "source_noop", "target_reverse", "target_shuffle")},
            "coverage": {},
            "residue_results": {r: ["no", "no"] if r in {"anchor", "frozen_base"} else ["no"] for r in ("anchor", "frozen_base", "target_forward", "source_noop", "target_reverse", "target_shuffle")},
            "residue_contract": {"id": "source_behavior", "description": "harmful behavior"},
            "qwen_roles": {"anchor": [], "frozen_base": []},
            "vjepa_scores": {m: {r: 0.1 for r in ("anchor", "frozen_base", "target_forward", "source_noop", "target_reverse", "target_shuffle")} for m in ("ordered_residual", "global_mean")},
            "videoprism_scores": {r: 0.1 for r in ("anchor", "frozen_base", "target_forward", "source_noop", "target_reverse", "target_shuffle")},
        }
        admission = {"admitted_for_candidate_voting": True, "counts": {"forward_over_source_noop": 16, "forward_over_target_reverse": 16, "forward_over_target_shuffle": 16}}
        qwen = {"manual_agreement_count": 1, "pair_count": 1, "manual_agreement_rate": 1.0, "winner_counts": {"anchor": 1}, "control_calibration": {"target_forward_strict_pass_count": 1, "source_noop_strict_pass_count": 0, "reverse_below_forward_count": 1, "shuffle_below_forward_count": 1, "source_noop_residue_yes_count": 1, "target_forward_residue_no_count": 1}}
        vjepa = {"metrics": {"ordered_residual": {"admission": admission}, "global_mean": {"admission": admission}}}
        videoprism = {"metrics": {"text_margin": {"admission": admission}}}
        rendered = render_html([case], qwen, vjepa, videoprism, "../media")
        self.assertIn("20260819_anchor_gap16_review/index.html", rendered)
        self.assertIn("20260819_anchor_gap16_review_v2/index.html", rendered)
        self.assertEqual(rendered.count("<video "), 4)
        self.assertIn("data-act=\"play\"", rendered)
        self.assertIn("source-action residue", rendered)


if __name__ == "__main__":
    unittest.main()
