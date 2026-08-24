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
    import saic_rf_preference_objective_v1 as objective  # noqa: E402
else:  # pragma: no cover
    objective = None


def _fixture():
    chosen = torch.zeros(2, 16, 21, 2, 2, dtype=torch.float32)
    rejected = torch.ones_like(chosen)
    epsilon = torch.full_like(chosen, 3.0)
    sigma = torch.tensor([0.25, 0.75], dtype=torch.float32)
    chosen_target = epsilon - chosen
    rejected_target = epsilon - rejected
    chosen_leaf = (chosen_target + 0.1).clone().requires_grad_(True)
    rejected_leaf = (rejected_target + 0.5).clone().requires_grad_(True)
    student_chosen = chosen_leaf * 1.0
    student_rejected = rejected_leaf * 1.0
    student_chosen.retain_grad()
    student_rejected.retain_grad()
    reference_chosen = (chosen_target + 0.2).clone()
    reference_rejected = (rejected_target + 0.3).clone()
    return [
        chosen,
        rejected,
        epsilon,
        sigma,
        student_chosen,
        student_rejected,
        reference_chosen,
        reference_rejected,
    ]


@unittest.skipIf(torch is None, "torch unavailable")
class SAICRFPreferenceObjectiveTests(unittest.TestCase):
    def _run(self, values=None, **kwargs):
        return objective.reference_relative_rf_preference(
            *(values or _fixture()),
            exact40_index=kwargs.get("exact40_index", 20),
            arm_ids=kwargs.get("arm_ids", ("dog", "human")),
        )

    def test_shared_state_reference_relative_beta5_and_chosen_side(self) -> None:
        values = _fixture()
        result = self._run(values)
        chosen, rejected, epsilon, sigma = values[:4]
        view = sigma.reshape(2, 1, 1, 1, 1)
        self.assertTrue(
            torch.equal(result.chosen_x_sigma, (1.0 - view) * chosen + view * epsilon)
        )
        self.assertTrue(
            torch.equal(result.rejected_x_sigma, (1.0 - view) * rejected + view * epsilon)
        )
        self.assertTrue(torch.allclose(result.student_gap, torch.full((2,), 0.24)))
        self.assertTrue(torch.allclose(result.reference_gap, torch.full((2,), 0.05)))
        self.assertTrue(
            torch.allclose(result.reference_relative_advantage, torch.full((2,), 0.19))
        )
        expected_preference = torch.nn.functional.softplus(torch.full((2,), -0.95))
        self.assertTrue(torch.allclose(result.preference_term, expected_preference))
        self.assertTrue(torch.allclose(result.chosen_side_term, torch.full((2,), 0.01)))
        self.assertTrue(
            torch.allclose(result.per_sample_loss, expected_preference + 0.01)
        )
        self.assertEqual(result.loss.dtype, torch.float32)
        result.loss.backward()
        self.assertIsNotNone(values[4].grad)
        self.assertIsNotNone(values[5].grad)

    def test_chosen_side_term_keeps_direct_chosen_gradient(self) -> None:
        values = _fixture()
        # Make pairwise preference very easy; the chosen FM term still has a
        # direct nonzero derivative toward its own codec-reencoded endpoint.
        values[5] = (values[5].detach() + 20.0).requires_grad_(True) * 1.0
        result = self._run(values)
        gradient = torch.autograd.grad(result.chosen_side_term.mean(), values[4])[0]
        self.assertGreater(float(gradient.abs().sum().item()), 0.0)

    def test_one_epsilon_one_sigma_api_and_two_arm_gate(self) -> None:
        parameters = set(inspect.signature(objective.reference_relative_rf_preference).parameters)
        self.assertIn("epsilon", parameters)
        self.assertIn("sigma", parameters)
        self.assertNotIn("chosen_epsilon", parameters)
        self.assertNotIn("rejected_epsilon", parameters)
        self.assertTrue(parameters.isdisjoint(objective.FORBIDDEN_VISUAL_INPUT_NAMES))
        with self.assertRaisesRegex(objective.SAICRFPreferenceError, "dog and human"):
            self._run(arm_ids=("dog", "dog"))

    def test_fp32_finite_and_frozen_reference_contract(self) -> None:
        values = _fixture()
        values[0] = values[0].to(torch.bfloat16)
        with self.assertRaisesRegex(objective.SAICRFPreferenceError, "FP32"):
            self._run(values)
        values = _fixture()
        values[6] = values[6].requires_grad_(True)
        with self.assertRaisesRegex(objective.SAICRFPreferenceError, "detached"):
            self._run(values)
        values = _fixture()
        values[7][0, 0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(objective.SAICRFPreferenceError, "finite FP32"):
            self._run(values)

    def test_exact_registered_update_indices_only(self) -> None:
        self.assertEqual(
            objective.EXACT40_UPDATE_INDICES,
            (4, 12, 20, 28, 33, 34, 35, 37),
        )
        for index in objective.EXACT40_UPDATE_INDICES:
            self.assertEqual(self._run(exact40_index=index).exact40_index, index)
        for index in (0, 38, 39):
            with self.assertRaises(objective.SAICRFPreferenceError):
                self._run(exact40_index=index)

    def test_contract_pins_beta_five_and_no_t2v_endpoint(self) -> None:
        receipt = dict(objective.contract_receipt())
        digest = receipt.pop("digest")
        self.assertEqual(digest, objective._sha(receipt))
        self.assertEqual(receipt["beta"], 5.0)
        self.assertEqual(receipt["chosen_side_weight"], 1.0)
        self.assertFalse(receipt["pure_t2v_visual_data_consumed"])
        self.assertFalse(receipt["weighted_reward_compensation_used"])


if __name__ == "__main__":
    unittest.main()
