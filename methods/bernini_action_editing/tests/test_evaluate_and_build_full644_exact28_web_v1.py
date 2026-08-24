from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluate_and_build_full644_exact28_web_v1 as evaluator


def clean_moving_video() -> np.ndarray:
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


class Exact28FullTrajectoryWebGateTest(unittest.TestCase):
    def test_legacy_machine_completion_is_revoked_by_closed_tombstone(self) -> None:
        legacy_root = REPO_ROOT / "md/action_editing/20260818_full644_exact28_eval"
        metrics_path = legacy_root / "metrics.json"
        html_path = legacy_root / "index.html"
        notice_path = legacy_root / "INVALIDATED_BY_EXACT160_BOX.md"
        completion_path = legacy_root / "WEB_COMPLETE"
        invalid_marker_path = legacy_root / "WEB_INVALID"
        receipt_path = legacy_root / "web_build_receipt.json"

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertFalse(completion_path.exists())
        completion = invalid_marker_path.read_text(encoding="utf-8")

        self.assertTrue(metrics["invalidated"])
        self.assertFalse(metrics["promotion_authorized"])
        self.assertFalse(metrics["serving_authorized"])
        self.assertEqual(metrics["formal_box_training_rows"], 0)
        self.assertNotIn("training_coverage_rows", metrics)
        self.assertIn("status=INVALIDATED_DO_NOT_PROMOTE", completion)
        self.assertNotIn("training_coverage=644", completion)

        self.assertEqual(receipt["status"], "INVALIDATED_DO_NOT_PROMOTE")
        self.assertTrue(receipt["legacy_completion_authority_revoked"])
        self.assertEqual(receipt["legacy_completion_marker_absent"], "WEB_COMPLETE")
        self.assertFalse(receipt["promotion_authorized"])
        self.assertFalse(receipt["serving_authorized"])
        for path_key, sha_key, path in (
            ("invalidated_html_path", "invalidated_html_sha256", html_path),
            ("invalidated_metrics_path", "invalidated_metrics_sha256", metrics_path),
            ("invalidation_notice_path", "invalidation_notice_sha256", notice_path),
        ):
            self.assertEqual(receipt[path_key], path.name)
            self.assertEqual(
                receipt[sha_key], hashlib.sha256(path.read_bytes()).hexdigest()
            )
        projection_digest = receipt.pop("invalidation_projection_digest")
        canonical = json.dumps(
            receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        self.assertEqual(projection_digest, hashlib.sha256(canonical).hexdigest())

    def test_perfect_phase0_cannot_authorize_collapsed_trajectory(self) -> None:
        source = clean_moving_video()
        candidate = source.copy()
        candidate[1:] = 0.0

        evaluated = evaluator.evaluate_candidate(source, source, candidate)

        self.assertAlmostEqual(
            evaluated["metrics"]["source_phase0_global_ssim"], 1.0, places=7
        )
        self.assertFalse(evaluated["visual_validity_passed"])
        self.assertEqual(evaluated["visual_validity_status"], "fail")
        self.assertIn("post_onset_blackout", evaluated["failure_codes"])
        self.assertEqual(
            evaluated["visual_validity"]["trajectory"]["evaluated_frame_count"],
            81,
        )
        self.assertFalse(evaluated["visual_validity"]["phase0_only"])

    def test_one_failed_variant_blocks_promotion_and_remains_in_html(self) -> None:
        source = clean_moving_video()
        passed = evaluator.evaluate_candidate(source, source, source.copy())
        collapsed = source.copy()
        collapsed[1:] = 0.0
        failed = evaluator.evaluate_candidate(source, source, collapsed)
        result = {
            "variants": {
                "frozen": passed,
                "seed20260820": failed,
                "seed20260821": passed,
                "seed20260822": passed,
            }
        }

        summary = evaluator.promotion_summary([result], expected_rows=1)

        self.assertFalse(summary["promotion_authorized"])
        self.assertFalse(summary["phase0_metrics_can_authorize_promotion"])
        self.assertEqual(
            summary["variants"]["seed20260820"]["full81_failed_rows"], 1
        )
        self.assertEqual(
            summary["variants"]["seed20260820"]["failure_code_row_counts"][
                "post_onset_blackout"
            ],
            1,
        )

        cell = evaluator._render_variant_cell(
            "seed20260820", "row-1", "seed20260820.mp4", failed
        )
        self.assertIn('class="cell validity-fail"', cell)
        self.assertIn("FULL81 VISUAL VALIDITY: FAIL", cell)
        self.assertIn("post_onset_blackout", cell)
        self.assertIn("<video", cell)
        self.assertIn('src="media/row-1/seed20260820.mp4"', cell)
        self.assertNotIn("phase0 SSIM", cell)

    def test_media_gate_pass_cannot_promote_legacy_objective(self) -> None:
        source = clean_moving_video()
        passed = evaluator.evaluate_candidate(source, source, source.copy())
        result = {"variants": {variant: passed for variant in evaluator.VARIANTS}}

        summary = evaluator.promotion_summary([result], expected_rows=1)

        self.assertTrue(summary["evaluation_panel_valid"])
        self.assertTrue(summary["trained_candidates_full81_valid"])
        self.assertTrue(summary["visual_media_gate_all_pass"])
        self.assertFalse(summary["legacy_objective_formal_box_compliant"])
        self.assertFalse(summary["promotion_authorized"])
        self.assertIn(
            "legacy_self_generated_anchor_objective_is_not_exact160_target_grounded",
            summary["scientific_promotion_blockers"],
        )
        self.assertTrue(
            all(
                value["full81_valid_rows"] == 1
                for value in summary["variants"].values()
            )
        )


if __name__ == "__main__":
    unittest.main()
