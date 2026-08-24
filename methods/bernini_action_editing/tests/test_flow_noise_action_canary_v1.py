from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import flow_noise_action_canary_v1 as method  # noqa: E402


class FlowNoiseActionCanaryTests(unittest.TestCase):
    def test_zero_flow_keeps_temporal_correlation_and_moments(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(7)
        baseline = torch.randn(1, 4, 21, 8, 10, generator=generator)
        flow = torch.zeros(20, 2, 8, 10)
        validity = torch.ones(20, 1, 8, 10)
        result = method.build_flow_transported_noise(
            baseline, flow, validity, degradation=0.0
        )
        self.assertTrue(torch.equal(result.initial_noise[:, :, 0], baseline[:, :, 0]))
        for phase in range(1, 21):
            self.assertTrue(
                torch.allclose(
                    result.initial_noise[:, :, phase].mean(dim=(-2, -1)),
                    baseline[:, :, phase].mean(dim=(-2, -1)),
                    atol=1e-5,
                )
            )
            self.assertTrue(
                torch.allclose(
                    result.initial_noise[:, :, phase].std(
                        dim=(-2, -1), unbiased=False
                    ),
                    baseline[:, :, phase].std(dim=(-2, -1), unbiased=False),
                    atol=1e-5,
                )
            )
        propagated_delta = torch.mean(
            (result.initial_noise[:, :, 1] - result.initial_noise[:, :, 0]) ** 2
        )
        iid_delta = torch.mean((baseline[:, :, 1] - baseline[:, :, 0]) ** 2)
        self.assertLess(float(propagated_delta), float(iid_delta))

    def test_invalid_cells_use_native_fresh_slice(self) -> None:
        baseline = torch.randn(1, 2, 21, 6, 7)
        flow = torch.zeros(20, 2, 6, 7)
        validity = torch.zeros(20, 1, 6, 7)
        result = method.build_flow_transported_noise(
            baseline, flow, validity, degradation=0.0
        )
        self.assertTrue(torch.allclose(result.initial_noise, baseline, atol=1e-5))

    def test_nonzero_flow_changes_noise(self) -> None:
        baseline = torch.randn(1, 2, 21, 6, 7)
        flow = torch.zeros(20, 2, 6, 7)
        flow[:, 0] = 1.0
        validity = torch.ones(20, 1, 6, 7)
        result = method.build_flow_transported_noise(
            baseline, flow, validity, degradation=0.3
        )
        self.assertFalse(torch.equal(result.initial_noise, baseline))
        self.assertEqual(result.receipt["degradation"], 0.3)


if __name__ == "__main__":
    unittest.main()
