from __future__ import annotations

import math
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
    import dual_conditional_ratio_core as dcr  # noqa: E402
else:  # pragma: no cover - dependency-light environments
    dcr = None


@unittest.skipIf(torch is None, "torch is unavailable")
class RectifiedFlowAndEnergyTests(unittest.TestCase):
    def test_rectified_flow_state_uses_clean_zero_noise_one_convention(self) -> None:
        clean = torch.tensor(
            [[0.0, 2.0], [4.0, 6.0]], dtype=torch.float32
        )
        noise = torch.tensor(
            [[2.0, 4.0], [8.0, 10.0]], dtype=torch.float32
        )
        state = dcr.rectified_flow_state(
            clean, noise, torch.tensor([0.0, 0.25], dtype=torch.float32)
        )
        self.assertTrue(torch.equal(state.x_sigma[0], clean[0]))
        self.assertTrue(
            torch.allclose(
                state.x_sigma[1], torch.tensor([5.0, 7.0], dtype=torch.float32)
            )
        )
        self.assertTrue(torch.equal(state.true_velocity, noise - clean))

        with self.assertRaisesRegex(dcr.DualConditionalRatioError, r"\[0, 1\]"):
            dcr.rectified_flow_state(clean, noise, 1.01)

    def test_masked_mse_normalizes_after_channel_broadcast(self) -> None:
        prediction = torch.tensor(
            [
                [[1.0, 9.0], [3.0, 9.0]],
                [[9.0, 2.0], [9.0, 4.0]],
            ],
            dtype=torch.float32,
        )
        target = torch.zeros_like(prediction)
        mask = torch.tensor(
            [[[1.0, 0.0]], [[0.0, 1.0]]], dtype=torch.float32
        )
        energy = dcr.masked_per_sample_mse(prediction, target, mask)
        self.assertTrue(
            torch.allclose(energy, torch.tensor([5.0, 10.0], dtype=torch.float32))
        )

        zero_mass = mask.clone()
        zero_mass[1] = 0.0
        with self.assertRaisesRegex(
            dcr.DualConditionalRatioError, "positive mask mass"
        ):
            dcr.masked_per_sample_mse(prediction, target, zero_mass)

    def test_masked_mse_and_ratio_remain_differentiable(self) -> None:
        prediction = torch.ones(2, 3, dtype=torch.float32, requires_grad=True)
        target = torch.zeros_like(prediction)
        preferred = dcr.masked_per_sample_mse(prediction, target)
        contrast = torch.full_like(preferred, 4.0)
        proxy = dcr.denoising_error_log_ratio_proxy(preferred, contrast)
        proxy.sum().backward()
        self.assertTrue(torch.isfinite(prediction.grad).all().item())
        self.assertGreater(prediction.grad.abs().sum().item(), 0.0)

    def test_mse_allows_mixed_float_dtypes_but_rf_state_does_not(self) -> None:
        prediction = torch.tensor(
            [[1.0, 3.0]], dtype=torch.float16, requires_grad=True
        )
        target = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        energy = dcr.masked_per_sample_mse(prediction, target)
        self.assertEqual(energy.dtype, torch.float64)
        self.assertTrue(
            torch.allclose(energy, torch.tensor([2.5], dtype=torch.float64))
        )
        energy.sum().backward()
        self.assertIsNotNone(prediction.grad)

        low_precision_target = target.to(dtype=torch.bfloat16)
        fp32_energy = dcr.masked_per_sample_mse(
            prediction.detach(), low_precision_target
        )
        self.assertEqual(fp32_energy.dtype, torch.float32)

        with self.assertRaisesRegex(
            dcr.DualConditionalRatioError, "noise dtype differs"
        ):
            dcr.rectified_flow_state(
                torch.zeros(1, 2, dtype=torch.float32),
                torch.zeros(1, 2, dtype=torch.float64),
                0.5,
            )


@unittest.skipIf(torch is None, "torch is unavailable")
class DualConditionalProxyTests(unittest.TestCase):
    def test_action_and_source_proxies_have_preferred_lower_error_sign(self) -> None:
        target = torch.zeros(2, 1, dtype=torch.float32)
        action = torch.ones_like(target)
        noop = torch.full_like(target, 2.0)
        action_proxy = dcr.action_t2v_cond_vs_noop_proxy(action, noop, target)
        expected_action = torch.full(
            (2,), math.log(4.0), dtype=torch.float32
        )
        self.assertTrue(torch.allclose(action_proxy, expected_action, atol=1.0e-6))

        correct_source = torch.full_like(target, 0.5)
        wrong_source = torch.full_like(target, 1.5)
        source_proxy = dcr.source_correct_vs_wrong_proxy(
            correct_source, wrong_source, target
        )
        expected_source = torch.full(
            (2,), math.log(9.0), dtype=torch.float32
        )
        self.assertTrue(torch.allclose(source_proxy, expected_source, atol=1.0e-6))

        reversed_action_proxy = dcr.action_t2v_cond_vs_noop_proxy(
            noop, action, target
        )
        self.assertTrue((reversed_action_proxy < 0.0).all().item())

    def test_equal_zero_errors_are_a_finite_neutral_proxy(self) -> None:
        zero = torch.zeros(3, dtype=torch.float32)
        proxy = dcr.denoising_error_log_ratio_proxy(zero, zero)
        self.assertTrue(torch.equal(proxy, zero))
        self.assertTrue(torch.isfinite(proxy).all().item())

    def test_multi_sigma_weighted_mean_and_gradient(self) -> None:
        values = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        weights = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float32)
        result = dcr.aggregate_multi_sigma(values, weights)
        self.assertTrue(
            torch.allclose(result, torch.tensor([3.0, 4.0], dtype=torch.float32))
        )
        result.sum().backward()
        self.assertTrue(torch.isfinite(values.grad).all().item())
        self.assertTrue(
            torch.allclose(
                values.grad[:, 0],
                torch.tensor([0.25, 0.5, 0.25], dtype=torch.float32),
            )
        )

        with self.assertRaisesRegex(
            dcr.DualConditionalRatioError, "positive total mass"
        ):
            dcr.aggregate_multi_sigma(values.detach(), torch.zeros(3))

    def test_multi_sigma_primary_proxy_aggregates_errors_before_ratio(self) -> None:
        preferred = torch.tensor(
            [[1.0, 2.0], [100.0, 8.0]], dtype=torch.float32
        )
        contrast = torch.tensor(
            [[4.0, 8.0], [100.0, 2.0]], dtype=torch.float64
        )
        weights = torch.tensor([1.0, 1.0], dtype=torch.float32)
        result = dcr.multi_sigma_denoising_error_ratio_proxy(
            preferred, contrast, weights
        )

        expected_preferred = torch.tensor([50.5, 5.0], dtype=torch.float64)
        expected_contrast = torch.tensor([52.0, 5.0], dtype=torch.float64)
        expected_proxy = torch.log(expected_contrast) - torch.log(
            expected_preferred
        )
        self.assertTrue(torch.allclose(result.preferred_error, expected_preferred))
        self.assertTrue(torch.allclose(result.contrast_error, expected_contrast))
        self.assertTrue(torch.allclose(result.proxy, expected_proxy, atol=1.0e-7))
        self.assertEqual(tuple(result.per_sigma_proxy.shape), (2, 2))
        self.assertTrue(
            torch.allclose(
                result.normalized_sigma_weights,
                torch.tensor([0.5, 0.5], dtype=torch.float64),
            )
        )

        mean_of_log_ratios = dcr.aggregate_multi_sigma(
            result.per_sigma_proxy, weights
        )
        self.assertFalse(
            torch.allclose(result.proxy, mean_of_log_ratios, atol=1.0e-4)
        )

    def test_multi_sigma_rejects_candidate_specific_weights(self) -> None:
        errors = torch.ones(2, 3, 4, dtype=torch.float32)
        candidate_specific = torch.ones(2, 3, 4, dtype=torch.float32)
        with self.assertRaisesRegex(
            dcr.DualConditionalRatioError, r"shared shape \[S\]"
        ):
            dcr.multi_sigma_denoising_error_ratio_proxy(
                errors, errors, candidate_specific
            )
        with self.assertRaisesRegex(
            dcr.DualConditionalRatioError, r"shared shape \[S\]"
        ):
            dcr.aggregate_multi_sigma(errors, candidate_specific)


@unittest.skipIf(torch is None, "torch is unavailable")
class LexicographicSelectionTests(unittest.TestCase):
    def test_multi_axis_ands_axes_then_ranks_worst_calibrated_margins(self) -> None:
        action = torch.tensor(
            [
                [
                    [0.90, 0.70],
                    [0.85, 0.90],
                    [1.00, 0.90],
                    [1.20, 0.59],
                ],
                [
                    [0.90, 0.70],
                    [0.70, 0.50],
                    [0.60, 0.40],
                    [0.50, 0.30],
                ],
            ],
            dtype=torch.float32,
        )
        preservation = torch.tensor(
            [
                [
                    [0.80, 0.60],
                    [0.90, 0.55],
                    [0.80, 0.60],
                    [1.00, 1.00],
                ],
                [
                    [0.60, 0.40],
                    [0.90, 0.80],
                    [0.90, 0.80],
                    [0.90, 0.80],
                ],
            ],
            dtype=torch.float64,
        )
        result = dcr.multi_axis_lexicographic_candidate_selection(
            action,
            preservation,
            action_thresholds=torch.tensor([0.80, 0.60], dtype=torch.float32),
            preservation_thresholds=torch.tensor(
                [0.70, 0.50], dtype=torch.float64
            ),
        )

        # Candidates 0 and 2 tie on their worst calibrated preservation
        # margin (0.1).  Candidate 2 has the larger worst action margin.
        self.assertEqual(result.index.tolist(), [2, -1])
        self.assertEqual(result.has_feasible.tolist(), [True, False])
        self.assertEqual(tuple(result.action_axis_pass.shape), (2, 4, 2))
        self.assertEqual(tuple(result.preservation_axis_pass.shape), (2, 4, 2))
        self.assertFalse(result.action_axis_pass[0, 3, 1].item())
        self.assertFalse(result.action_pass[0, 3].item())
        self.assertFalse(result.preservation_pass[1, 0].item())
        self.assertTrue(result.action_pass[1, 0].item())
        self.assertTrue(result.preservation_pass[1, 1].item())
        self.assertFalse(result.joint_pass[1].any().item())

        self.assertAlmostEqual(
            result.worst_preservation_calibrated_margin[0, 2].item(),
            0.1,
            places=6,
        )
        self.assertAlmostEqual(
            result.worst_action_calibrated_margin[0, 2].item(),
            0.2,
            places=6,
        )

    def test_multi_axis_threshold_cannot_vary_by_candidate(self) -> None:
        action = torch.ones(2, 3, 2, dtype=torch.float32)
        preservation = torch.ones(2, 3, 1, dtype=torch.float32)
        with self.assertRaisesRegex(
            dcr.DualConditionalRatioError, "cannot vary along the candidate"
        ):
            dcr.multi_axis_lexicographic_candidate_selection(
                action,
                preservation,
                action_thresholds=torch.zeros(2, 3, 2),
                preservation_thresholds=torch.zeros(1),
            )

    def test_joint_gate_is_fail_closed_for_unilateral_winners(self) -> None:
        action = torch.tensor(
            [[0.95, 0.85, 0.30], [0.90, 0.20, 0.10]], dtype=torch.float32
        )
        preservation = torch.tensor(
            [[0.82, 0.95, 1.00], [0.50, 0.95, 0.90]], dtype=torch.float32
        )
        result = dcr.lexicographic_candidate_selection(
            action,
            preservation,
            action_threshold=0.8,
            preservation_threshold=0.8,
        )
        # Batch 0 has two joint-pass candidates.  Once action passes, higher
        # preservation wins even though its action proxy is lower.
        self.assertEqual(result.index.tolist(), [1, -1])
        self.assertEqual(result.has_feasible.tolist(), [True, False])
        # Batch 1 has an action-only winner and preservation-only winners.  No
        # candidate satisfies the intersection, so selection fails closed.
        self.assertEqual(result.joint_pass[1].tolist(), [False, False, False])
        self.assertTrue(result.action_pass[1, 0].item())
        self.assertTrue(result.preservation_pass[1, 1].item())

    def test_action_then_tie_breaker_and_lowest_index_are_deterministic(self) -> None:
        action = torch.tensor([0.9, 0.95, 0.95], dtype=torch.float32)
        preservation = torch.tensor([0.9, 0.9, 0.9], dtype=torch.float32)
        tie = torch.tensor([9.0, 1.0, 2.0], dtype=torch.float32)
        result = dcr.lexicographic_candidate_selection(
            action,
            preservation,
            action_threshold=0.8,
            preservation_threshold=0.8,
            tie_breaker=tie,
        )
        self.assertEqual(result.index.item(), 2)

        exact_tie = dcr.lexicographic_candidate_selection(
            action[1:],
            preservation[1:],
            action_threshold=0.8,
            preservation_threshold=0.8,
        )
        self.assertEqual(exact_tie.index.item(), 0)

    def test_thresholds_are_hard_and_inclusive(self) -> None:
        result = dcr.lexicographic_candidate_selection(
            torch.tensor([0.8], dtype=torch.float32),
            torch.tensor([0.7], dtype=torch.float32),
            action_threshold=0.8,
            preservation_threshold=0.7,
        )
        self.assertTrue(result.has_feasible.item())
        self.assertEqual(result.index.item(), 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
