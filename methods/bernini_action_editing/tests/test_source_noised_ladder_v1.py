from __future__ import annotations

from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import source_noised_ladder_v1 as ladder  # noqa: E402


class SourceNoisedLadderStaticTests(unittest.TestCase):
    def test_contract_refuses_inversion_claim(self) -> None:
        valid = {}
        receipt = ladder.SourceLadderContract(**valid).receipt()
        self.assertIs(receipt["inversion_claimed"], False)
        self.assertIs(receipt["same_epsilon_as_matched_edit_required"], True)
        self.assertIs(receipt["same_epsilon_as_matched_edit_verified"], False)
        self.assertIs(receipt["clean_source_route_verified"], False)
        self.assertIs(receipt["matched_edit_query_sigma_binding_verified"], False)
        self.assertIs(receipt["runtime_integration_verified"], False)
        self.assertNotIn("clean_source_condition_remains_available", receipt)
        self.assertNotIn("source_state_is_same_sigma_coordinate", receipt)
        unsigned = dict(receipt)
        digest = unsigned.pop("receipt_digest")
        self.assertEqual(ladder.object_sha256(unsigned), digest)
        for mutation in (
            {**valid, "inversion_claimed": True},
            {**valid, "exact_roundtrip_claimed": True},
            {**valid, "same_epsilon_as_matched_edit_required": False},
            {**valid, "schedule_indices": (16, 29, 35)},
            {**valid, "sigmas": tuple(reversed(ladder.DEFAULT_SIGMAS))},
        ):
            with self.subTest(mutation=mutation):
                with self.assertRaises(ladder.SourceNoisedLadderError):
                    ladder.SourceLadderContract(**mutation)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is required")
class SourceNoisedLadderTensorTests(unittest.TestCase):
    def test_endpoints_and_midpoint(self) -> None:
        source = torch.tensor([1.0, 3.0], dtype=torch.float32)
        epsilon = torch.tensor([-1.0, 1.0], dtype=torch.float32)
        self.assertTrue(torch.equal(ladder.shared_noise_source_state(source, epsilon, 0), source))
        self.assertTrue(torch.equal(ladder.shared_noise_source_state(source, epsilon, 1), epsilon))
        self.assertTrue(
            torch.equal(
                ladder.shared_noise_source_state(source, epsilon, 0.5),
                torch.tensor([0.0, 2.0], dtype=torch.float32),
            )
        )

    def test_invalid_tensor_inputs_fail_closed(self) -> None:
        source = torch.ones(2, dtype=torch.float32)
        epsilon = torch.zeros(2, dtype=torch.float32)
        invalid_pairs = (
            (source.to(torch.int64), epsilon.to(torch.int64)),
            (source, epsilon.reshape(1, 2)),
            (source, torch.tensor([float("nan"), 0.0])),
            (source.requires_grad_(True), epsilon),
        )
        for left, right in invalid_pairs:
            with self.subTest(left=left.dtype, right_shape=tuple(right.shape)):
                with self.assertRaises(ladder.SourceNoisedLadderError):
                    ladder.shared_noise_source_state(left, right, 0.5)


if __name__ == "__main__":
    unittest.main()
