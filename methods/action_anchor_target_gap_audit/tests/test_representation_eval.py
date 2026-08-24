import unittest

try:
    import numpy as np
except ModuleNotFoundError as error:  # Minimal local test environments may omit numerical extras.
    raise unittest.SkipTest("numpy is required for representation tests") from error

from methods.action_anchor_target_gap_audit.representation_eval import (
    admission_counts,
    candidate_winner,
    ordered_residual_descriptor,
)


class RepresentationEvaluationTest(unittest.TestCase):
    def test_ordered_descriptor_changes_under_reversal_and_shuffle(self):
        time = np.arange(8, dtype=np.float32)
        tokens = np.stack([time, time ** 2, np.sin(time)], axis=1)
        forward = ordered_residual_descriptor(tokens)
        reverse = ordered_residual_descriptor(tokens[::-1])
        shuffle = ordered_residual_descriptor(tokens[[0, 1, 6, 7, 2, 3, 4, 5]])
        self.assertLess(float(np.dot(forward, reverse)), 0.5)
        self.assertLess(float(np.dot(forward, shuffle)), 0.95)

    def test_admission_requires_every_control_at_twelve_of_sixteen(self):
        pairs = []
        for index in range(16):
            pairs.append({
                "scores": {"metric": {
                    "target_forward": 1.0,
                    "target_reverse": 0.0 if index < 12 else 1.0,
                    "target_shuffle": 0.0 if index < 12 else 1.0,
                    "source_noop": 0.0 if index < 12 else 1.0,
                }}
            })
        self.assertTrue(admission_counts(pairs, "metric", 0.005)["admitted_for_candidate_voting"])
        pairs[11]["scores"]["metric"]["target_reverse"] = 1.0
        self.assertFalse(admission_counts(pairs, "metric", 0.005)["admitted_for_candidate_voting"])

    def test_candidate_tie_has_epsilon_deadband(self):
        self.assertEqual(candidate_winner(0.010, 0.006, 0.005), "tie")
        self.assertEqual(candidate_winner(0.012, 0.006, 0.005), "anchor")
        self.assertEqual(candidate_winner(0.006, 0.012, 0.005), "frozen_base")


if __name__ == "__main__":
    unittest.main()
