from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from action.flow import (  # noqa: E402
    DiffSynthWanTrainingScheduler,
    full_target_flow_loss,
    prepare_full_target_flow,
    reconstruct_x0,
    shifted_rectified_flow_sigma,
)


class ActionFlowTest(unittest.TestCase):
    def test_complete_target_formula_and_reconstruction(self) -> None:
        target = torch.arange(32, dtype=torch.float32).reshape(2, 1, 2, 2, 4)
        noise = torch.full_like(target, 10.0)
        sigma = torch.tensor([0.0, 0.75])
        result = prepare_full_target_flow(target, sigma, noise=noise)
        self.assertIs(result.target_x0, target)
        self.assertIs(result.noise, noise)
        self.assertTrue(torch.equal(result.x_t[0], target[0]))
        self.assertTrue(
            torch.allclose(result.x_t[1], 0.25 * target[1] + 0.75 * noise[1])
        )
        self.assertTrue(torch.equal(result.target_velocity, noise - target))
        self.assertTrue(
            torch.allclose(
                reconstruct_x0(result.x_t, result.target_velocity, sigma),
                target,
                atol=1e-6,
                rtol=1e-6,
            )
        )

        parameters = set(inspect.signature(prepare_full_target_flow).parameters)
        self.assertEqual(parameters, {"target_x0", "sigma", "noise", "generator"})

    def test_loss_reduces_over_all_elements_and_supports_batch_weights(self) -> None:
        target_velocity = torch.zeros(2, 1, 1, 1, 2)
        prediction = torch.tensor([[[[[1.0, 1.0]]]], [[[[2.0, 2.0]]]]])
        self.assertAlmostEqual(
            float(full_target_flow_loss(prediction, target_velocity)), 2.5
        )
        weighted = full_target_flow_loss(
            prediction, target_velocity, sample_weight=torch.tensor([3.0, 1.0])
        )
        self.assertAlmostEqual(float(weighted), 3.5)
        scalar_weighted = full_target_flow_loss(
            prediction, target_velocity, sample_weight=torch.tensor(2.0)
        )
        self.assertAlmostEqual(float(scalar_weighted), 5.0)

        with self.assertRaisesRegex(ValueError, "positive sum"):
            full_target_flow_loss(
                prediction, target_velocity, sample_weight=torch.zeros(2)
            )

    def test_shifted_schedule_is_fp32_discrete_and_batch_shared(self) -> None:
        scheduler = DiffSynthWanTrainingScheduler(
            shift=5.0, num_train_timesteps=1000
        )
        self.assertEqual(scheduler.sigmas.dtype, torch.float32)
        self.assertEqual(scheduler.sigmas.shape, (1000,))
        self.assertEqual(float(scheduler.sigmas[0]), 1.0)
        expected_second = 5.0 * 0.999 / (1.0 + 4.0 * 0.999)
        self.assertAlmostEqual(float(scheduler.sigmas[1]), expected_second, places=6)
        self.assertAlmostEqual(float(scheduler.flow_weights.mean()), 1.0, places=5)

        generator_a = torch.Generator().manual_seed(123)
        generator_b = torch.Generator().manual_seed(123)
        first = scheduler.sample(3, generator=generator_a)
        second = scheduler.sample(3, generator=generator_b)
        self.assertEqual(first.timestep_id, second.timestep_id)
        self.assertTrue(torch.equal(first.sigma, first.sigma[:1].expand(3)))
        self.assertTrue(torch.equal(first.sigma, second.sigma))
        self.assertEqual(first.flow_weight.ndim, 0)

    def test_invalid_domains_fail_closed(self) -> None:
        target = torch.zeros(1, 1, 1, 1, 1)
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            prepare_full_target_flow(target, 1.1)
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            prepare_full_target_flow(target, 0.5, noise=torch.zeros(2, 1, 1, 1, 1))
        with self.assertRaisesRegex(ValueError, "\\[0, 1\\]"):
            shifted_rectified_flow_sigma(torch.tensor([-0.1]))
        with self.assertRaisesRegex(ValueError, "at least 2"):
            DiffSynthWanTrainingScheduler(num_train_timesteps=1)


if __name__ == "__main__":
    unittest.main()
