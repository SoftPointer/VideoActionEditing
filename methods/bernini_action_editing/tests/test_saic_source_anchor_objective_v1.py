from __future__ import annotations

from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch

    import saic_source_anchor_objective_v1 as anchor

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    anchor = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def _latent(offset: float = 0.0):
    value = torch.arange(1 * 16 * 21 * 2 * 2, dtype=torch.float32)
    return value.reshape(1, 16, 21, 2, 2).div(1000.0).add(offset).contiguous()


@unittest.skipUnless(_TORCH_AVAILABLE, "torch is required")
class SourceAnchorObjectiveTests(unittest.TestCase):
    def test_scramble_is_deterministic_bijective_and_does_not_mutate_source(self) -> None:
        source = _latent()
        before = source.clone()
        first, order_a = anchor.scramble_source_condition(source, seed=809)
        second, order_b = anchor.scramble_source_condition(source, seed=809)
        self.assertEqual(order_a, order_b)
        self.assertEqual(set(order_a), set(range(21)))
        self.assertNotEqual(order_a, tuple(range(21)))
        self.assertNotEqual(order_a, tuple(reversed(range(21))))
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, source))
        self.assertTrue(torch.equal(source, before))

    def test_flow_state_and_target_are_source_anchored(self) -> None:
        source = _latent()
        noise = _latent(1.0)
        state, target = anchor.build_source_flow_state(source, noise, sigma=0.25)
        self.assertTrue(torch.equal(state, 0.75 * source + 0.25 * noise))
        self.assertTrue(torch.equal(target, noise - source))
        self.assertFalse(state.requires_grad)
        self.assertFalse(target.requires_grad)

    def test_objective_rewards_correct_source_and_backpropagates_both_leaves(self) -> None:
        target = _latent()
        correct = (target + 0.01).detach().requires_grad_(True)
        wrong = (target + 0.20).detach().requires_grad_(True)
        objective = anchor.build_source_anchor_objective(
            correct_source_prediction=correct,
            wrong_source_prediction=wrong,
            source_flow_target=target,
            wrong_source_margin=0.01,
        )
        self.assertGreater(float(objective.wrong_source_advantage.item()), 0.0)
        self.assertEqual(float(objective.ranking_hinge.item()), 0.0)
        objective.loss.backward()
        self.assertIsNotNone(correct.grad)
        # The already-satisfied ranking branch is inactive, so a zero gradient
        # for the wrong-source leaf is expected and is not a missing VJP.
        self.assertIsNotNone(wrong.grad)

    def test_ranking_hinge_prevents_source_ignoring(self) -> None:
        target = _latent()
        correct = (target + 0.10).detach().requires_grad_(True)
        wrong = (target + 0.10).detach().requires_grad_(True)
        objective = anchor.build_source_anchor_objective(
            correct_source_prediction=correct,
            wrong_source_prediction=wrong,
            source_flow_target=target,
            wrong_source_margin=0.05,
        )
        self.assertAlmostEqual(float(objective.wrong_source_advantage.item()), 0.0)
        self.assertAlmostEqual(float(objective.ranking_hinge.item()), 0.05, places=6)
        objective.loss.backward()
        self.assertGreater(float(correct.grad.abs().sum().item()), 0.0)
        self.assertGreater(float(wrong.grad.abs().sum().item()), 0.0)

    def test_wrong_shapes_detached_predictions_and_wrong_sigma_fail(self) -> None:
        target = _latent()
        with self.assertRaisesRegex(anchor.SAICSourceAnchorObjectiveError, "output cotangent"):
            anchor.build_source_anchor_objective(
                correct_source_prediction=target,
                wrong_source_prediction=target,
                source_flow_target=target,
            )
        with self.assertRaisesRegex(anchor.SAICSourceAnchorObjectiveError, "35..39"):
            anchor.validate_active_sigma_index(34)
        self.assertEqual(anchor.validate_active_sigma_index(39), 39)


if __name__ == "__main__":
    unittest.main()
