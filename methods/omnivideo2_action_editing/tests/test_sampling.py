import unittest

import torch

from pact.flow import flow_noisy_latent
from pact.sampling import (
    anchored_euler_flow_step,
    euler_flow_step,
    sample_anchored_flow,
    validate_inference_sigmas,
    wan_rational_shifted_sigmas,
)


class SamplingTests(unittest.TestCase):
    def test_wan_rational_shifted_schedule_includes_terminal_zero(self) -> None:
        sigmas = wan_rational_shifted_sigmas(4, shift=2.0, dtype=torch.float64)
        base = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0], dtype=torch.float64)
        expected = 2.0 * base / (1.0 + base)
        self.assertTrue(torch.equal(sigmas[[0, -1]], torch.tensor([1.0, 0.0], dtype=torch.float64)))
        self.assertTrue(torch.allclose(sigmas, expected, atol=0.0, rtol=0.0))
        self.assertEqual(sigmas.numel(), 5)
        self.assertTrue(bool((sigmas[1:] < sigmas[:-1]).all()))

    def test_schedule_validation_is_strict(self) -> None:
        for invalid in (
            torch.tensor([1.0]),
            torch.tensor([0.9, 0.0]),
            torch.tensor([1.0, 0.5, 0.5, 0.0]),
            torch.tensor([1.0, 0.4, 0.6, 0.0]),
            torch.tensor([1.0, 0.5, 0.1]),
            torch.tensor([1.0, float("nan"), 0.0]),
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    validate_inference_sigmas(invalid)
        with self.assertRaises(TypeError):
            validate_inference_sigmas(torch.tensor([1, 0]))
        with self.assertRaises(ValueError):
            wan_rational_shifted_sigmas(0, shift=5.0)
        with self.assertRaises(ValueError):
            wan_rational_shifted_sigmas(True, shift=5.0)
        with self.assertRaises(ValueError):
            wan_rational_shifted_sigmas(4, shift=0.0)

    def test_euler_uses_descending_sigma_delta(self) -> None:
        current = torch.full((1, 1, 1, 1, 2), 3.0)
        velocity = torch.tensor([[[[[2.0, -4.0]]]]])
        result = euler_flow_step(
            current,
            velocity,
            sigma_current=0.8,
            sigma_next=0.3,
        )
        self.assertTrue(torch.equal(result, current - 0.5 * velocity))

    def test_anchor_is_constructed_at_sigma_next_and_supports_soft_strength(self) -> None:
        current = torch.zeros(1, 1, 1, 1, 2)
        velocity = torch.full_like(current, 2.0)
        source = torch.tensor([[[[[2.0, 4.0]]]]])
        initial_noise = torch.tensor([[[[[10.0, 20.0]]]]])
        soft_mask = torch.tensor([[[[[0.25, 1.0]]]]])
        step = anchored_euler_flow_step(
            current,
            velocity,
            source,
            initial_noise,
            soft_mask,
            sigma_current=0.8,
            sigma_next=0.3,
            anchor_strength=0.4,
        )
        expected_source_next = 0.7 * source + 0.3 * initial_noise
        expected_euler = current - 0.5 * velocity
        anchor_weight = (1.0 - soft_mask) * 0.4
        expected = expected_euler + anchor_weight * (
            expected_source_next - expected_euler
        )
        self.assertTrue(torch.allclose(step.source_x_next, expected_source_next))
        self.assertTrue(torch.equal(step.euler_x_next, expected_euler))
        self.assertTrue(torch.allclose(step.x_next, expected))
        source_at_current = 0.2 * source + 0.8 * initial_noise
        self.assertFalse(torch.allclose(step.source_x_next, source_at_current))

    def test_local_endpoint_oracle_tracks_path_and_preserves_source_exactly(self) -> None:
        source = torch.tensor(
            [[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]]
        )
        target = source + 20.0
        initial_noise = torch.tensor(
            [[[[[-3.0, -2.0], [-1.0, 0.0]], [[1.0, 2.0], [3.0, 4.0]]]]]
        )
        edit_mask = torch.tensor(
            [[[[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]]]]
        )
        sigmas = wan_rational_shifted_sigmas(5, shift=5.0)
        local_endpoint = source * (1.0 - edit_mask) + target * edit_mask
        observed_states: list[tuple[torch.Tensor, torch.Tensor]] = []

        def endpoint_oracle(current_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            observed_states.append((current_x.detach().clone(), sigma.detach().clone()))
            # Deliberately proposes the target trajectory everywhere.  The
            # post-step anchor must replace the keep region with source flow.
            return initial_noise - target

        result = sample_anchored_flow(
            initial_noise,
            source,
            edit_mask,
            sigmas,
            endpoint_oracle,
        )

        for state, sigma in observed_states:
            expected = flow_noisy_latent(local_endpoint, initial_noise, sigma)
            self.assertTrue(torch.allclose(state, expected, atol=2e-6, rtol=2e-6))
        self.assertTrue(torch.allclose(result, local_endpoint, atol=2e-6, rtol=2e-6))
        keep = edit_mask.expand_as(source) == 0
        self.assertTrue(torch.equal(result[keep], source[keep]))

    def test_wrong_shapes_and_sigma_order_are_rejected(self) -> None:
        latent = torch.zeros(1, 2, 2, 2, 2)
        mask = torch.zeros(1, 1, 2, 2, 2)
        with self.assertRaises(ValueError):
            euler_flow_step(
                latent,
                torch.zeros(1, 1, 2, 2, 2),
                sigma_current=0.8,
                sigma_next=0.2,
            )
        with self.assertRaises(ValueError):
            euler_flow_step(
                latent,
                torch.zeros_like(latent),
                sigma_current=0.2,
                sigma_next=0.8,
            )
        with self.assertRaises(ValueError):
            euler_flow_step(
                latent,
                torch.zeros_like(latent),
                sigma_current=0.2,
                sigma_next=0.2,
            )
        with self.assertRaises(ValueError):
            sample_anchored_flow(
                latent,
                torch.zeros_like(latent),
                torch.zeros(1, 2, 2, 2, 2),
                torch.tensor([1.0, 0.0]),
                lambda x, sigma: torch.zeros_like(x),
            )
        with self.assertRaises(ValueError):
            sample_anchored_flow(
                latent,
                torch.zeros(2, 2, 2, 2, 2),
                mask,
                torch.tensor([1.0, 0.0]),
                lambda x, sigma: torch.zeros_like(x),
            )


if __name__ == "__main__":
    unittest.main()
