from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import source_kv_route_objective as objective


try:
    import torch
except ImportError:  # pragma: no cover - local lightweight environment
    torch = None


class StaticContractTests(unittest.TestCase):
    def test_contract_is_source_instruction_only_and_unclipped(self) -> None:
        contract = objective.objective_contract()
        self.assertEqual(contract["frames"], 81)
        self.assertEqual(contract["latent_phases"], 21)
        self.assertEqual(
            tuple(contract["forward_branch_order"]),
            objective.FORWARD_BRANCH_ORDER,
        )
        self.assertEqual(contract["inference_conditions"], ["source_video", "action_instruction"])
        self.assertIn("paired_target_video", contract["forbidden_inference_conditions"])
        self.assertFalse(contract["carrier_is_decoded_output"])
        self.assertFalse(contract["hard_radius_projection"])
        self.assertEqual(contract["target_clipping_fraction"], 0.0)
        self.assertEqual(contract["target_energy_retention"], 1.0)
        self.assertEqual(
            contract["sigma_weighting"], "L_inner/max(sigma,sigma_floor)"
        )
        self.assertIn("detached_frozen_rms", contract["high_band_normalization"])
        self.assertRegex(contract["contract_digest"], r"^[0-9a-f]{64}$")

    def test_loss_config_rejects_disabled_boundary_or_negative_weight(self) -> None:
        with self.assertRaises(objective.SourceKVRouteObjectiveError):
            objective.SourceKVRouteLossConfig(raw_phase0_weight=0.0).validate()
        with self.assertRaises(objective.SourceKVRouteObjectiveError):
            objective.SourceKVRouteLossConfig(pair_high_weight=-0.1).validate()
        with self.assertRaises(objective.SourceKVRouteObjectiveError):
            objective.SourceKVRouteLossConfig(amplitude_floor=float("nan")).validate()
        with self.assertRaises(objective.SourceKVRouteObjectiveError):
            objective.SourceKVRouteLossConfig(sigma_floor=0.0).validate()


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class TensorObjectiveTests(unittest.TestCase):
    def _exact_fields(self):
        shape = (1, 21, 3, 4)
        source = torch.linspace(-0.4, 0.4, steps=1 * 21 * 3 * 4).reshape(shape).float()
        phase = torch.arange(21, dtype=torch.float32).reshape(1, 21, 1, 1)
        motion = 0.01 * phase.expand(shape)
        target = (source + motion).detach()
        # Construct the branch fields from the actual float32 executable
        # teacher.  Reusing the mathematical ``motion`` would make
        # ``(source + motion) - source`` differ by ULPs and create a false
        # high-band error in this bit-equality control.
        executable_motion = (target - source).detach()
        frozen_noop = torch.zeros(shape, dtype=torch.float32)
        frozen_action = executable_motion.clone()
        adapted_noop = torch.nn.Parameter(torch.zeros(shape, dtype=torch.float32))
        adapted_action = torch.nn.Parameter(executable_motion.clone())
        fields = objective.RouteCleanFields(
            frozen_noop=frozen_noop,
            frozen_action=frozen_action,
            adapted_noop=adapted_noop,
            adapted_action=adapted_action,
            source_clean=source.detach(),
            target_clean=target,
        )
        return fields, adapted_noop, adapted_action

    def test_exact_teacher_has_zero_loss_and_both_adapted_paths_receive_grad(self) -> None:
        fields, adapted_noop, adapted_action = self._exact_fields()
        result = objective.compute_source_kv_route_objective(fields, sigma=0.5)
        self.assertAlmostEqual(float(result.total.detach()), 0.0, places=6)
        result.total.backward()
        self.assertIsNotNone(adapted_noop.grad)
        self.assertIsNotNone(adapted_action.grad)
        self.assertTrue(bool(torch.isfinite(adapted_noop.grad).all()))
        self.assertTrue(bool(torch.isfinite(adapted_action.grad).all()))

    def test_pair_loss_has_nonzero_gradient_to_both_independent_adapted_branches(self) -> None:
        fields, adapted_noop, adapted_action = self._exact_fields()
        with torch.no_grad():
            adapted_action[:, 5:12].add_(0.07)
        result = objective.compute_source_kv_route_objective(fields, sigma=0.5)
        noop_grad, action_grad = torch.autograd.grad(
            result.pair_low, (adapted_noop, adapted_action), retain_graph=True
        )
        self.assertGreater(float(noop_grad.abs().sum()), 0.0)
        self.assertGreater(float(action_grad.abs().sum()), 0.0)
        self.assertTrue(bool(torch.isfinite(noop_grad).all()))
        self.assertTrue(bool(torch.isfinite(action_grad).all()))

    def test_raw_phase_zero_object_injection_is_penalized(self) -> None:
        fields, _, _ = self._exact_fields()
        injection = torch.zeros_like(fields.adapted_action)
        injection[:, 0] = 0.25
        adapted_noop = torch.nn.Parameter(fields.adapted_noop.detach().clone())
        adapted_action = torch.nn.Parameter(
            fields.adapted_action.detach().clone() + injection
        )
        changed = objective.RouteCleanFields(
            frozen_noop=fields.frozen_noop,
            frozen_action=fields.frozen_action,
            adapted_noop=adapted_noop,
            adapted_action=adapted_action,
            source_clean=fields.source_clean,
            target_clean=fields.target_clean,
        )
        result = objective.compute_source_kv_route_objective(changed, sigma=0.5)
        self.assertGreater(float(result.raw_phase0.detach()), 0.0)
        self.assertGreater(float(result.total.detach()), 0.0)
        noop_grad, action_grad = torch.autograd.grad(
            result.raw_phase0, (adapted_noop, adapted_action)
        )
        self.assertGreater(float(noop_grad[:, :1].abs().sum()), 0.0)
        self.assertGreater(float(action_grad[:, :1].abs().sum()), 0.0)
        self.assertEqual(float(noop_grad[:, 1:].abs().sum()), 0.0)
        self.assertEqual(float(action_grad[:, 1:].abs().sum()), 0.0)

    def test_prior_gate_zero_is_exact_and_has_finite_zero_gradients(self) -> None:
        fields, adapted_noop, adapted_action = self._exact_fields()
        anti_aligned = objective.RouteCleanFields(
            frozen_noop=fields.frozen_noop,
            frozen_action=-fields.frozen_action,
            adapted_noop=adapted_noop,
            adapted_action=adapted_action,
            source_clean=fields.source_clean,
            target_clean=fields.target_clean,
        )
        result = objective.compute_source_kv_route_objective(
            anti_aligned, sigma=0.5
        )
        self.assertEqual(float(result.prior_direction.detach()), 0.0)
        self.assertEqual(
            float(result.diagnostics.prior_direction_weight[:, 1:].max()), 0.0
        )
        noop_grad, action_grad = torch.autograd.grad(
            result.prior_direction, (adapted_noop, adapted_action)
        )
        self.assertTrue(bool(torch.isfinite(noop_grad).all()))
        self.assertTrue(bool(torch.isfinite(action_grad).all()))
        self.assertEqual(float(noop_grad.abs().sum()), 0.0)
        self.assertEqual(float(action_grad.abs().sum()), 0.0)

    def test_all_zero_motion_has_finite_loss_and_backward(self) -> None:
        shape = (1, 21, 3, 4)
        zero = torch.zeros(shape, dtype=torch.float32)
        adapted_noop = torch.nn.Parameter(zero.clone())
        adapted_action = torch.nn.Parameter(zero.clone())
        fields = objective.RouteCleanFields(
            frozen_noop=zero,
            frozen_action=zero.clone(),
            adapted_noop=adapted_noop,
            adapted_action=adapted_action,
            source_clean=zero.clone(),
            target_clean=zero.clone(),
        )
        result = objective.compute_source_kv_route_objective(fields, sigma=0.0)
        self.assertTrue(bool(torch.isfinite(result.total)))
        result.total.backward()
        self.assertTrue(bool(torch.isfinite(adapted_noop.grad).all()))
        self.assertTrue(bool(torch.isfinite(adapted_action.grad).all()))

    def test_disabled_prior_cannot_pollute_zero_prediction_backward(self) -> None:
        fields, _, _ = self._exact_fields()
        adapted_noop = torch.nn.Parameter(torch.zeros_like(fields.adapted_noop))
        adapted_action = torch.nn.Parameter(torch.zeros_like(fields.adapted_action))
        zero_prediction = objective.RouteCleanFields(
            frozen_noop=fields.frozen_noop,
            frozen_action=fields.frozen_action,
            adapted_noop=adapted_noop,
            adapted_action=adapted_action,
            source_clean=fields.source_clean,
            target_clean=fields.target_clean,
        )
        config = objective.SourceKVRouteLossConfig(prior_direction_weight=0.0)
        result = objective.compute_source_kv_route_objective(
            zero_prediction, sigma=0.5, config=config
        )
        result.total.backward()
        self.assertTrue(bool(torch.isfinite(adapted_noop.grad).all()))
        self.assertTrue(bool(torch.isfinite(adapted_action.grad).all()))

    def test_high_band_uses_frozen_scale_for_smooth_teacher(self) -> None:
        fields, _, _ = self._exact_fields()
        phase = torch.arange(21, dtype=torch.float32).reshape(1, 21, 1, 1)
        oscillation = 0.2 * (1.0 - 2.0 * (phase.remainder(2.0)))
        frozen_action = oscillation.expand_as(fields.frozen_action).clone()
        adapted_noop = torch.nn.Parameter(fields.adapted_noop.detach().clone())
        adapted_action = torch.nn.Parameter(frozen_action.clone())
        high_prior = objective.RouteCleanFields(
            frozen_noop=fields.frozen_noop,
            frozen_action=frozen_action,
            adapted_noop=adapted_noop,
            adapted_action=adapted_action,
            source_clean=fields.source_clean,
            target_clean=fields.target_clean,
        )
        result = objective.compute_source_kv_route_objective(
            high_prior, sigma=0.5
        )
        self.assertTrue(bool(torch.isfinite(result.pair_high)))
        self.assertLess(float(result.pair_high.detach()), 2.0)

    def test_teacher_is_never_clipped_and_energy_retention_is_one(self) -> None:
        fields, _, _ = self._exact_fields()
        phase = torch.arange(21, dtype=torch.float32).reshape(1, 21, 1, 1)
        large_motion = 1.0e6 * phase.expand_as(fields.target_clean)
        target_clean = fields.source_clean + large_motion
        large_teacher = objective.RouteCleanFields(
            frozen_noop=fields.frozen_noop,
            frozen_action=fields.frozen_action,
            adapted_noop=fields.adapted_noop,
            adapted_action=fields.adapted_action,
            source_clean=fields.source_clean,
            target_clean=target_clean.detach(),
        )
        result = objective.compute_source_kv_route_objective(
            large_teacher, sigma=0.5
        )
        metrics = objective.detached_objective_metrics(result)
        self.assertEqual(metrics["target_clipped_fraction"], 0.0)
        self.assertEqual(metrics["target_energy_retention"], 1.0)
        raw_teacher = target_clean - fields.source_clean
        expected = raw_teacher - raw_teacher[:, :1]
        expected = expected.clone()
        expected[:, :1] = 0.0
        self.assertTrue(
            torch.equal(result.diagnostics.target_quotient, expected.detach())
        )
        self.assertGreater(float(result.diagnostics.target_quotient.abs().max()), 1.0e6)
        reconstructed = (
            result.diagnostics.target_low_increments
            + result.diagnostics.target_high_increments
        )
        expected_increments = torch.cat(
            (
                torch.zeros_like(expected[:, :1]),
                expected[:, 1:] - expected[:, :-1],
            ),
            dim=1,
        )
        torch.testing.assert_close(
            reconstructed, expected_increments, rtol=0.0, atol=1.0e-6
        )

    def test_sigma_floor_and_inverse_weight_are_exact(self) -> None:
        fields, adapted_noop, adapted_action = self._exact_fields()
        with torch.no_grad():
            adapted_action[:, 4:10].add_(0.03)
        floor = objective.SourceKVRouteLossConfig().sigma_floor
        at_half = objective.compute_source_kv_route_objective(fields, sigma=0.5)
        below_floor = objective.compute_source_kv_route_objective(
            fields, sigma=floor / 10.0
        )
        expected_ratio = 0.5 / floor
        self.assertAlmostEqual(
            float((below_floor.total / at_half.total).detach()),
            expected_ratio,
            places=5,
        )
        metrics = objective.detached_objective_metrics(below_floor)
        self.assertAlmostEqual(metrics["sigma"], floor / 10.0, places=7)
        self.assertAlmostEqual(metrics["inverse_sigma_weight"], 1.0 / floor, places=5)

    def test_invalid_or_graph_bearing_sigma_is_rejected(self) -> None:
        fields, _, _ = self._exact_fields()
        for sigma in (-0.1, float("nan"), True):
            with self.subTest(sigma=sigma):
                with self.assertRaises(objective.SourceKVRouteObjectiveError):
                    objective.compute_source_kv_route_objective(fields, sigma=sigma)
        with self.assertRaises(objective.SourceKVRouteObjectiveError):
            objective.compute_source_kv_route_objective(
                fields, sigma=torch.tensor(0.5, requires_grad=True)
            )

    def test_frozen_or_label_graph_is_rejected(self) -> None:
        fields, _, _ = self._exact_fields()
        invalid = objective.RouteCleanFields(
            frozen_noop=fields.frozen_noop.requires_grad_(),
            frozen_action=fields.frozen_action,
            adapted_noop=fields.adapted_noop,
            adapted_action=fields.adapted_action,
            source_clean=fields.source_clean,
            target_clean=fields.target_clean,
        )
        with self.assertRaises(objective.SourceKVRouteObjectiveError):
            objective.compute_source_kv_route_objective(invalid, sigma=0.5)

    def test_missing_adapted_graph_is_rejected(self) -> None:
        fields, _, _ = self._exact_fields()
        invalid = objective.RouteCleanFields(
            frozen_noop=fields.frozen_noop,
            frozen_action=fields.frozen_action,
            adapted_noop=fields.adapted_noop.detach(),
            adapted_action=fields.adapted_action,
            source_clean=fields.source_clean,
            target_clean=fields.target_clean,
        )
        with self.assertRaises(objective.SourceKVRouteObjectiveError):
            objective.compute_source_kv_route_objective(invalid, sigma=0.5)

    def test_wrong_phase_count_or_nonfinite_input_is_rejected(self) -> None:
        fields, _, _ = self._exact_fields()
        invalid = objective.RouteCleanFields(
            **{
                name: getattr(fields, name)[:, :20]
                for name in objective.RouteCleanFields.__dataclass_fields__
            }
        )
        with self.assertRaises(objective.SourceKVRouteObjectiveError):
            objective.compute_source_kv_route_objective(invalid, sigma=0.5)


if __name__ == "__main__":
    unittest.main()
