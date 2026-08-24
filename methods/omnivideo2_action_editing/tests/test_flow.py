import unittest

import torch

from pact.flow import (
    flow_noisy_latent,
    reconstruct_x0,
    shared_noise_local_latent_splice,
    velocity_target,
)


class RectifiedFlowTests(unittest.TestCase):
    def test_noising_velocity_and_x0_reconstruction(self) -> None:
        x0 = torch.randn(2, 3, 2, 2, 2)
        noise = torch.randn_like(x0)
        sigma = torch.tensor([0.0, 0.75])
        x_t = flow_noisy_latent(x0, noise, sigma)
        self.assertTrue(torch.equal(x_t[0], x0[0]))
        expected_second = 0.25 * x0[1] + 0.75 * noise[1]
        self.assertTrue(torch.allclose(x_t[1], expected_second))
        reconstructed = reconstruct_x0(x_t, velocity_target(x0, noise), sigma)
        self.assertTrue(torch.allclose(reconstructed, x0, atol=1e-6, rtol=1e-6))

    def test_shared_noise_splice_selects_local_branch(self) -> None:
        source = torch.zeros(1, 2, 2, 2, 2)
        target = torch.ones_like(source)
        noise = torch.full_like(source, 2.0)
        mask = torch.zeros(1, 1, 2, 2, 2)
        mask[..., :, 1] = 1.0
        result = shared_noise_local_latent_splice(
            source, target, mask, 0.5, noise=noise
        )
        self.assertTrue(torch.equal(result.source_x_t, torch.ones_like(source)))
        self.assertTrue(torch.equal(result.target_x_t, torch.full_like(source, 1.5)))
        expected = result.source_x_t * (1.0 - mask) + result.target_x_t * mask
        self.assertTrue(torch.equal(result.x_t, expected))
        expected_x0 = source * (1.0 - mask) + target * mask
        self.assertTrue(torch.equal(result.local_x0, expected_x0))
        self.assertTrue(
            torch.equal(result.target_velocity, result.noise - result.local_x0)
        )
        self.assertTrue(
            torch.allclose(
                reconstruct_x0(result.x_t, result.target_velocity, 0.5),
                result.local_x0,
            )
        )
        self.assertIs(result.noise, noise)

    def test_splice_gradient_reaches_both_clean_latents(self) -> None:
        source = torch.randn(1, 1, 2, 2, 2, requires_grad=True)
        target = torch.randn(1, 1, 2, 2, 2, requires_grad=True)
        mask = torch.zeros(1, 1, 2, 2, 2)
        mask[..., 0] = 1.0
        result = shared_noise_local_latent_splice(
            source, target, mask, 0.25, noise=torch.zeros_like(source)
        )
        result.x_t.sum().backward()
        self.assertGreater(float(source.grad.abs().sum()), 0.0)
        self.assertGreater(float(target.grad.abs().sum()), 0.0)

    def test_invalid_sigma_and_shapes_fail(self) -> None:
        x0 = torch.zeros(1, 1, 2, 2, 2)
        with self.assertRaises(ValueError):
            flow_noisy_latent(x0, torch.zeros(2, 1, 2, 2, 2), 0.5)
        with self.assertRaises(ValueError):
            flow_noisy_latent(x0, torch.zeros_like(x0), 1.1)
        with self.assertRaises(ValueError):
            flow_noisy_latent(x0, torch.zeros_like(x0), torch.ones(3))


if __name__ == "__main__":
    unittest.main()
