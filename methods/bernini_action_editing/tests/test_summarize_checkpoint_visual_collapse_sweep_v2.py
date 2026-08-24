from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import checkpoint_visual_collapse_gate_v2 as gate
import summarize_checkpoint_visual_collapse_sweep_v2 as summary


FINGERPRINT = "a" * 64


def report(
    step: int,
    sample_id: str,
    *,
    passed: bool,
    collapsed: bool,
    fingerprint: str = FINGERPRINT,
) -> dict:
    return {
        "schema_version": gate.SCHEMA_VERSION,
        "status": "pass" if passed else "fail",
        "passed": passed,
        "publishable": passed,
        "collapsed": collapsed,
        "failure_codes": [] if passed else ["calibrated_visual_collapse"],
        "metadata": {
            "checkpoint_step": step,
            "sample_id": sample_id,
        },
        "calibration": {"calibration_fingerprint": fingerprint},
    }


class SweepSummaryV2Test(unittest.TestCase):
    def test_frontier_is_computed_without_manual_review(self) -> None:
        reports = []
        for sample_id in ("case00", "case01"):
            reports.append(report(1, sample_id, passed=True, collapsed=False))
        reports.append(report(64, "case00", passed=False, collapsed=True))
        reports.append(report(64, "case01", passed=True, collapsed=False))
        for sample_id in ("case00", "case01"):
            reports.append(report(644, sample_id, passed=False, collapsed=True))

        result = summary.summarize_reports(
            reports, expected_sample_ids=("case00", "case01")
        )

        self.assertFalse(result["passed"])
        self.assertTrue(result["all_checkpoints_complete"])
        self.assertTrue(result["calibration_consistent"])
        self.assertEqual(result["frontier"]["fully_passing_checkpoint_steps"], [1])
        self.assertEqual(result["frontier"]["last_fully_passing_checkpoint_step"], 1)
        self.assertEqual(result["frontier"]["first_any_collapse_checkpoint_step"], 64)
        self.assertEqual(result["frontier"]["first_all_collapsed_checkpoint_step"], 644)
        self.assertEqual(result["failure_code_counts"]["calibrated_visual_collapse"], 3)

    def test_all_complete_clean_checkpoints_pass_summary(self) -> None:
        result = summary.summarize_reports(
            [
                report(step, sample_id, passed=True, collapsed=False)
                for step in (1, 4)
                for sample_id in ("case00", "case01")
            ],
            expected_sample_ids=("case00", "case01"),
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["incomplete_checkpoint_steps"], [])

    def test_missing_case_fails_closed(self) -> None:
        result = summary.summarize_reports(
            [report(1, "case00", passed=True, collapsed=False)],
            expected_sample_ids=("case00", "case01"),
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["incomplete_checkpoint_steps"], [1])
        self.assertEqual(result["checkpoints"][0]["missing_sample_ids"], ["case01"])

    def test_mixed_calibration_fingerprints_fail_closed(self) -> None:
        result = summary.summarize_reports(
            [
                report(1, "case00", passed=True, collapsed=False),
                report(
                    1,
                    "case01",
                    passed=True,
                    collapsed=False,
                    fingerprint="b" * 64,
                ),
            ],
            expected_sample_ids=("case00", "case01"),
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["calibration_consistent"])

    def test_duplicate_checkpoint_sample_is_rejected(self) -> None:
        duplicate = report(1, "case00", passed=True, collapsed=False)
        with self.assertRaisesRegex(summary.SweepSummaryError, "duplicate"):
            summary.summarize_reports([duplicate, dict(duplicate)])


if __name__ == "__main__":
    unittest.main()
