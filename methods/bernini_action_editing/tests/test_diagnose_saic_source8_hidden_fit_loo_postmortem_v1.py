from __future__ import annotations

from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import diagnose_saic_source8_hidden_fit_loo_postmortem_v1 as diagnostic


try:
    import torch
except ModuleNotFoundError:
    torch = None


class Source8FitLOOPostmortemTest(unittest.TestCase):
    def test_candidate_registry_is_fixed_and_unique(self) -> None:
        self.assertEqual(len(diagnostic.SINGLE_VIEWS), 11)
        self.assertEqual(len(diagnostic.CANDIDATES), 24)
        ids = [diagnostic.candidate_id(row) for row in diagnostic.CANDIDATES]
        self.assertEqual(len(ids), len(set(ids)))

    @unittest.skipIf(torch is None, "torch is unavailable")
    def test_composed_feature_normalizes_components(self) -> None:
        views = {
            "a": torch.tensor([3.0, 4.0]),
            "b": torch.tensor([0.0, 0.0, 2.0]),
        }
        result = diagnostic.compose_feature(views, ("a", "b"))
        self.assertEqual(tuple(result.shape), (5,))
        self.assertAlmostEqual(float(torch.linalg.vector_norm(result)), 1.0, places=6)

    def test_selection_key_prefers_count_then_minimum(self) -> None:
        weaker_count = {
            "positive_count": 7,
            "minimum_margin": 1.0,
            "mean_margin": 1.0,
            "candidate_id": "a",
        }
        stronger_count = {
            "positive_count": 8,
            "minimum_margin": 0.01,
            "mean_margin": 0.01,
            "candidate_id": "b",
        }
        stronger_minimum = {
            **stronger_count,
            "minimum_margin": 0.02,
            "candidate_id": "c",
        }
        self.assertEqual(
            sorted(
                (weaker_count, stronger_count, stronger_minimum),
                key=diagnostic._selection_key,
            )[0],
            stronger_minimum,
        )


if __name__ == "__main__":
    unittest.main()
