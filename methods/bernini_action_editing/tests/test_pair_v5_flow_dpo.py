from __future__ import annotations

import inspect
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch  # noqa: E402
except ImportError:
    torch = None

if torch is not None:
    import pair_v5_flow_dpo as dpo  # noqa: E402
else:  # pragma: no cover
    dpo = None


def _fixture(batch: int = 2):
    chosen = torch.zeros(batch, 16, 21, 2, 2, dtype=torch.float32)
    rejected = torch.ones_like(chosen)
    epsilon = torch.full_like(chosen, 3.0)
    sigma = torch.tensor(0.5, dtype=torch.float32)
    chosen_target = epsilon - chosen
    rejected_target = epsilon - rejected
    # The student prefers chosen more strongly than the frozen reference.
    chosen_leaf = (chosen_target + 0.1).clone().requires_grad_(True)
    rejected_leaf = (rejected_target + 0.5).clone().requires_grad_(True)
    student_chosen = chosen_leaf * 1.0
    student_rejected = rejected_leaf * 1.0
    student_chosen.retain_grad()
    student_rejected.retain_grad()
    reference_chosen = (chosen_target + 0.2).clone()
    reference_rejected = (rejected_target + 0.3).clone()
    return (
        chosen,
        rejected,
        epsilon,
        sigma,
        student_chosen,
        student_rejected,
        reference_chosen,
        reference_rejected,
    )


@unittest.skipIf(torch is None, "torch is unavailable")
class PairV5FlowDPOTests(unittest.TestCase):
    def test_exact_shared_noise_states_and_reference_corrected_sign(self) -> None:
        values = _fixture()
        result = dpo.reference_corrected_flow_dpo(*values, beta=2.0)
        chosen, rejected, epsilon, sigma = values[:4]
        self.assertTrue(
            torch.equal(result.chosen_x_sigma, (1.0 - sigma) * chosen + sigma * epsilon)
        )
        self.assertTrue(
            torch.equal(result.rejected_x_sigma, (1.0 - sigma) * rejected + sigma * epsilon)
        )
        self.assertTrue(torch.equal(result.chosen_velocity_target, epsilon - chosen))
        self.assertTrue(
            torch.equal(result.rejected_velocity_target, epsilon - rejected)
        )
        # student gap=.25-.01=.24; reference gap=.09-.04=.05
        self.assertTrue(torch.allclose(result.student_gap, torch.full((2,), 0.24)))
        self.assertTrue(torch.allclose(result.reference_gap, torch.full((2,), 0.05)))
        self.assertTrue(torch.allclose(result.advantage, torch.full((2,), 0.19)))
        self.assertTrue(result.loss.requires_grad)
        result.loss.backward()
        self.assertIsNotNone(values[4].grad)
        self.assertIsNotNone(values[5].grad)

    def test_reference_correction_is_not_detached_from_student(self) -> None:
        values = list(_fixture(batch=1))
        result = dpo.reference_corrected_flow_dpo(*values, beta=1.0)
        first = float(result.loss.detach().item())
        values[4] = (values[4].detach() + 0.8).requires_grad_(True) * 1.0
        second = float(
            dpo.reference_corrected_flow_dpo(*values, beta=1.0).loss.detach().item()
        )
        self.assertGreater(second, first)

    def test_weighted_batch_reduction(self) -> None:
        values = list(_fixture())
        values[4] = values[4].clone()
        values[5] = values[5].clone()
        values[5][1] = values[7][1] - 0.2
        weights = torch.tensor([1.0, 3.0], dtype=torch.float32)
        result = dpo.reference_corrected_flow_dpo(
            *values, beta=1.0, sample_weight=weights
        )
        expected = (result.per_sample_loss * weights).sum() / weights.sum()
        self.assertTrue(torch.equal(result.loss, expected))

    def test_fail_closed_on_geometry_alias_and_reference_grad(self) -> None:
        values = list(_fixture(batch=1))
        with self.assertRaisesRegex(dpo.PairV5FlowDPOError, "tensor-identical"):
            dpo.reference_corrected_flow_dpo(
                values[0], values[0].clone(), *values[2:], beta=1.0
            )
        bad = list(values)
        bad[0] = torch.zeros(1, 16, 20, 2, 2)
        with self.assertRaisesRegex(dpo.PairV5FlowDPOError, "exact81"):
            dpo.reference_corrected_flow_dpo(*bad, beta=1.0)
        bad = list(values)
        bad[6] = bad[6].requires_grad_(True)
        with self.assertRaisesRegex(dpo.PairV5FlowDPOError, "frozen-reference"):
            dpo.reference_corrected_flow_dpo(*bad, beta=1.0)

    def test_fail_closed_on_nonshared_noise_api_and_detached_student(self) -> None:
        parameters = set(
            inspect.signature(dpo.reference_corrected_flow_dpo).parameters
        )
        self.assertIn("epsilon", parameters)
        self.assertNotIn("chosen_epsilon", parameters)
        self.assertNotIn("rejected_epsilon", parameters)
        self.assertTrue(parameters.isdisjoint(dpo.FORBIDDEN_EXTERNAL_INPUT_NAMES))
        values = list(_fixture(batch=1))
        values[4] = values[4].detach()
        with self.assertRaisesRegex(dpo.PairV5FlowDPOError, "connected to the student"):
            dpo.reference_corrected_flow_dpo(*values, beta=1.0)

    def test_receipt_binds_no_proposal_or_privileged_visual_input(self) -> None:
        receipt = dict(dpo.contract_receipt())
        digest = receipt.pop("digest")
        self.assertEqual(digest, dpo._object_sha256(receipt))
        self.assertFalse(receipt["proposal_visual_data_consumed"])
        self.assertFalse(receipt["paired_target_consumed"])
        self.assertEqual(
            receipt["candidate_origin"],
            "native_rv2v_source_first_deployment_path_only",
        )


if __name__ == "__main__":
    unittest.main()
