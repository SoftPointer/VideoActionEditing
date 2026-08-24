from __future__ import annotations

from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import diagnose_saic_source8_hidden_quotient_confirmation_v1 as diagnostic


class Source8HiddenConfirmationTest(unittest.TestCase):
    def test_representation_is_exact_prior_winner(self) -> None:
        import torch

        torch.manual_seed(7)
        tensor = torch.randn(1, 21, 16, 1536, dtype=torch.float32)
        expected = diagnostic.prior_diagnostic.temporal_representations(tensor)[
            diagnostic.REPRESENTATION_NAME
        ]
        observed = diagnostic.preregistered_representation(tensor)
        self.assertTrue(torch.equal(expected, observed))
        self.assertEqual(tuple(observed.shape), (210,))

    def test_fit_direction_consumes_two_sources_and_four_contrasts(self) -> None:
        import torch

        sources = [
            {
                "iid": f"fit-{index}",
                "features": {
                    "forward": torch.tensor([1.0, 0.0, 0.0]),
                    "noop": torch.tensor([0.0, 1.0, 0.0]),
                    "reverse": torch.tensor([0.0, 0.0, 1.0]),
                },
            }
            for index in range(2)
        ]
        direction, rows = diagnostic.fit_direction(
            sources, positive_branch="forward"
        )
        self.assertEqual(tuple(direction.shape), (3,))
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["positive"] for row in rows))

    def test_swapped_controls_preserve_action_but_fail_pseudo_positives(self) -> None:
        import torch

        fit = [
            {
                "iid": f"fit-{index}",
                "features": {
                    "forward": torch.tensor([1.0, 0.0, 0.0]),
                    "noop": torch.tensor([0.0, 1.0, 0.0]),
                    "reverse": torch.tensor([0.0, 0.0, 1.0]),
                },
            }
            for index in range(2)
        ]
        confirmation = [
            {
                "iid": f"confirmation-{index}",
                "features": {
                    "forward": torch.tensor([1.0, 0.0, 0.0]),
                    "noop": torch.tensor([0.0, 0.0, 1.0]),
                    "reverse": torch.tensor([0.0, 1.0, 0.0]),
                },
            }
            for index in range(2)
        ]
        results = {
            branch: diagnostic.evaluate_positive_branch(
                fit_sources=fit,
                confirmation_sources=confirmation,
                positive_branch=branch,
            )
            for branch in diagnostic.BRANCH_ORDER
        }
        self.assertTrue(results["forward"]["confirmation_all_positive"])
        self.assertFalse(results["noop"]["confirmation_all_positive"])
        self.assertFalse(results["reverse"]["confirmation_all_positive"])

    def test_wrong_fit_source_count_fails_closed(self) -> None:
        import torch

        source = {
            "iid": "fit-only",
            "features": {
                branch: torch.ones(3) for branch in diagnostic.BRANCH_ORDER
            },
        }
        with self.assertRaises(diagnostic.Source8HiddenConfirmationError):
            diagnostic.fit_direction([source], positive_branch="forward")


if __name__ == "__main__":
    unittest.main()
