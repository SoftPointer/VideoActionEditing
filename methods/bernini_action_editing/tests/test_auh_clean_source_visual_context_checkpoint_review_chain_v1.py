#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
CHAIN = (
    METHOD_ROOT
    / "scripts/auh_decode_clean_source_visual_context_checkpoint_review_holder_v1.sh"
)
COMBINED = (
    METHOD_ROOT
    / "scripts/auh_train_then_review_clean_source_visual_context_holder_v1.sh"
)


class CheckpointReviewChainTests(unittest.TestCase):
    def test_chain_is_sequential_world4_and_retains_parent(self) -> None:
        text = CHAIN.read_text(encoding="utf-8")
        self.assertIn("readonly steps=(0 20 40 60 80)", text)
        self.assertIn('for ordinal in "${!steps[@]}"', text)
        self.assertIn("--nproc_per_node=4", text)
        self.assertIn("--gres=gpu:mi210:4", text)
        self.assertIn("--checkpoint-step", text)
        self.assertIn("controller.DECODE_REVIEW_COMPLETE", text)
        self.assertIn("parent_retained=true", text)
        self.assertNotIn("scancel", text)
        self.assertNotIn("optimizer.step", text)

    def test_chain_requires_training_handoff_and_builds_html_last(self) -> None:
        text = CHAIN.read_text(encoding="utf-8")
        training_gate = text.index("controller.TRAINING_COMPLETE")
        decode = text.index('readonly steps=(0 20 40 60 80)')
        html = text.index('"${python_bin}" -B "${html_builder}"')
        complete = text.index("controller.DECODE_REVIEW_COMPLETE")
        self.assertLess(training_gate, decode)
        self.assertLess(decode, html)
        self.assertLess(html, complete)
        self.assertIn("expected_runtime_closure", text)
        self.assertIn("expected_controller_sha", text)
        self.assertIn("--expected-training-receipt-sha256", text)
        self.assertIn("--expected-review-manifest-sha256", text)

    def test_combined_wrapper_does_not_modify_or_release_training_holder(self) -> None:
        text = COMBINED.read_text(encoding="utf-8")
        self.assertIn("auh_train_clean_source_visual_context_stage_b_holder_v1.sh", text)
        self.assertIn("controller.TRAINING_COMPLETE", text)
        self.assertIn("auh_decode_clean_source_visual_context_checkpoint_review_holder_v1.sh", text)
        self.assertNotIn("scancel", text)
        self.assertNotIn("sed -i", text)


if __name__ == "__main__":
    unittest.main()
