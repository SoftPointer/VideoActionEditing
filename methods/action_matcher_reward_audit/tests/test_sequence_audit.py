import importlib.util
from pathlib import Path
import unittest

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "run_sequence_audit.py"
SPEC = importlib.util.spec_from_file_location("run_sequence_audit", MODULE_PATH)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class SequenceAuditTest(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(7)
        self.sequence = torch.randn(32, 24, generator=generator)
        self.sequence += torch.linspace(-2.0, 2.0, 32).unsqueeze(1) * torch.randn(
            1, 24, generator=generator
        )

    def test_variants_are_deterministic(self):
        left = audit.controlled_variants("case", self.sequence)
        right = audit.controlled_variants("case", self.sequence)
        self.assertEqual(set(left), set(right))
        for key in left:
            self.assertTrue(torch.equal(left[key], right[key]))

    def test_order_metrics_prefer_forward_over_reverse(self):
        for name in (
            "frame_diagonal_centered",
            "dtw_centered",
            "dtw_derivative",
            "otam_style_centered",
            "endpoint",
        ):
            metric = audit.METRICS[name]
            forward = metric(self.sequence, self.sequence)
            reverse = metric(self.sequence, torch.flip(self.sequence, dims=(0,)))
            self.assertGreater(forward, reverse, name)

    def test_threshold_is_fit_only(self):
        rows = [
            {"split": "fit", "label": 1, "score": 0.9, "variant": "p"},
            {"split": "fit", "label": 0, "score": 0.1, "variant": "n"},
            {"split": "heldout", "label": 1, "score": 0.8, "variant": "p"},
            {"split": "heldout", "label": 0, "score": 0.2, "variant": "n"},
        ]
        summary = audit.classification_summary(rows)
        self.assertEqual(summary["fit"]["accuracy"], 1.0)
        self.assertEqual(summary["heldout"]["accuracy"], 1.0)
        self.assertGreater(summary["threshold_from_fit"], 0.1)
        self.assertLess(summary["threshold_from_fit"], 0.9)

    def test_calibrated_transfer_can_report_neither(self):
        metric = next(iter(audit.METRICS))
        controlled = {metric: {"threshold_from_fit": 0.5}}
        simmotion = {
            metric: {
                "dataset_designated_positive_over_negative": {
                    "rows": [{"positive_score": 0.2, "negative_score": 0.1}]
                }
            }
        }
        project = {
            metric: {
                "rows": [
                    {
                        "negative_branch": "reverse",
                        "positive_score": 0.2,
                        "negative_score": 0.1,
                    },
                    {
                        "negative_branch": "noop",
                        "positive_score": 0.2,
                        "negative_score": 0.1,
                    },
                ]
            }
        }
        original = audit.METRICS
        audit.METRICS = {metric: original[metric]}
        try:
            result = audit.calibrated_transfer(controlled, simmotion, project)
        finally:
            audit.METRICS = original
        self.assertEqual(
            result[metric]["simmotion_pair_outcomes"]["neither_accepted"], 1
        )


if __name__ == "__main__":
    unittest.main()
