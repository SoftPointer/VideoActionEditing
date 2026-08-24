from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "methods/semantic_moments_reward_audit/run_audit.py"
SPEC = importlib.util.spec_from_file_location("semantic_moments_reward_audit", SOURCE)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


class SemanticMomentsRewardAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(20260815)
        self.tokens = torch.randn(32, 16, 24, generator=generator)

    def test_all_moments_are_temporal_permutation_invariant(self) -> None:
        baseline = audit.temporal_components(self.tokens)
        for indices in (
            torch.arange(31, -1, -1),
            torch.randperm(32, generator=torch.Generator().manual_seed(9)),
        ):
            changed = audit.temporal_components(self.tokens[indices])
            self.assertTrue(torch.allclose(baseline, changed, atol=2e-6, rtol=2e-6))
            for weights in audit.DEFAULT_WEIGHTS.values():
                left = audit.compose_moments(baseline, weights)
                right = audit.compose_moments(changed, weights)
                self.assertAlmostEqual(audit.cosine(left, right), 1.0, places=6)

    def test_truncation_changes_distribution(self) -> None:
        baseline = audit.temporal_components(self.tokens)
        first_half = torch.cat([self.tokens[:16], self.tokens[:16]], dim=0)
        changed = audit.temporal_components(first_half)
        self.assertFalse(torch.allclose(baseline, changed, atol=1e-5, rtol=1e-5))

    def test_order_diagnostics_distinguish_exact_reverse(self) -> None:
        sequence = self.tokens.mean(dim=1)
        self.assertGreater(audit.order_margin(sequence, sequence), 0.0)
        self.assertLess(audit.order_margin(sequence, torch.flip(sequence, dims=(0,))), 0.0)
        self.assertGreater(audit.endpoint_score(sequence, sequence), 0.99)
        self.assertLess(audit.endpoint_score(sequence, torch.flip(sequence, dims=(0,))), -0.99)

    def test_binary_preference_summary(self) -> None:
        summary = audit.binary_preference_summary([0.2, -0.1, 0.3, 0.0])
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["wins"], 2)
        self.assertEqual(summary["ties"], 1)
        self.assertEqual(summary["accuracy"], 0.5)

    def test_simmotion_retrieval_includes_other_triplets(self) -> None:
        def record(example_id: str, role: str, values: tuple[float, float]):
            vector = torch.tensor(values, dtype=torch.float32)
            vector = audit.unit(vector, dim=0)
            return {
                "item_id": f"simmotion:{example_id}:{role}",
                "metadata": {"example_id": example_id, "role": role},
                "components": torch.stack([vector, vector, vector]),
                "frame_sequence": torch.stack([vector, vector, vector]),
            }

        records = [
            record("example_1", "ref", (1.0, 0.0)),
            record("example_1", "positive", (0.99, 0.10)),
            record("example_1", "negative", (-1.0, 0.0)),
            record("example_2", "ref", (0.0, 1.0)),
            record("example_2", "positive", (0.10, 0.99)),
            record("example_2", "negative", (0.0, -1.0)),
        ]
        result = audit.simmotion_analysis(records)
        retrieval = result["representations"]["m123"][
            "within_dataset_recall_at_1"
        ]
        self.assertEqual(retrieval["count"], 2)
        self.assertEqual(retrieval["candidate_count_per_query"], 5)
        self.assertEqual(retrieval["other_triplet_video_count"], 3)
        self.assertEqual(retrieval["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
