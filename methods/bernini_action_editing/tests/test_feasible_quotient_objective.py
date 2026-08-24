#!/usr/bin/env python3

from pathlib import Path
import sys
import unittest

try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover - environment dependent
    HAS_TORCH = False


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import feasible_quotient_objective as objective  # noqa: E402
import gauge_anchored_commutator as gauge  # noqa: E402


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for v8 objective tests")
class FeasibleQuotientObjectiveTests(unittest.TestCase):
    def _fields(self, *, target_scale: float = 1.0):
        torch.manual_seed(19)
        shape = (2, 21, 3, 4)
        frozen_noop = torch.randn(shape, dtype=torch.float32)
        frozen_action = frozen_noop + 0.2 * torch.randn(shape)
        adapted_noop = (
            frozen_noop + 0.03 * torch.randn(shape)
        ).detach().requires_grad_(True)
        adapted_action = (
            frozen_action + 0.05 * torch.randn(shape)
        ).detach().requires_grad_(True)
        source = torch.randn(shape, dtype=torch.float32)
        target = source + target_scale * 0.3 * torch.randn(shape)
        return objective.FiveBranchCleanFields(
            frozen_editor_noop=frozen_noop,
            frozen_editor_action=frozen_action,
            adapted_editor_noop=adapted_noop,
            adapted_editor_action=adapted_action,
            source_clean=source,
            target_clean=target,
        )

    def test_target_is_canonicalized_strictly_inside_deployment_radius(self):
        result = objective.compute_feasible_quotient_objective(
            self._fields(target_scale=9.0), step_index=0
        )
        diagnostics = result.diagnostics
        target_rms = gauge.phase_rms(
            diagnostics.canonical_target_increments
        )
        self.assertTrue(
            torch.all(
                target_rms[:, 1:]
                <= 0.95 * diagnostics.source_only_radius[:, 1:] + 2.0e-6
            )
        )
        self.assertTrue(
            torch.equal(
                diagnostics.canonical_target_increments[:, 0],
                torch.zeros_like(
                    diagnostics.canonical_target_increments[:, 0]
                ),
            )
        )

    def test_motion_losses_follow_full_deployed_action_minus_noop_gradient(self):
        fields = self._fields()
        result = objective.compute_feasible_quotient_objective(
            fields, step_index=0
        )
        motion_only = (
            result.canonical
            + result.executed
            + result.margin
            + result.temporal_jitter
        )
        action_gradient, noop_gradient = torch.autograd.grad(
            motion_only,
            (fields.adapted_editor_action, fields.adapted_editor_noop),
            allow_unused=True,
            retain_graph=True,
        )
        self.assertIsNotNone(action_gradient)
        self.assertGreater(float(action_gradient.abs().sum()), 0.0)
        self.assertIsNotNone(noop_gradient)
        self.assertGreater(float(noop_gradient.abs().sum()), 0.0)
        self.assertTrue(
            torch.allclose(action_gradient, -noop_gradient, atol=2.0e-6)
        )
        result.total.backward()
        self.assertIsNotNone(fields.adapted_editor_noop.grad)
        self.assertGreater(
            float(fields.adapted_editor_noop.grad.abs().sum()), 0.0
        )

    def test_rho_zero_is_noop_only_update(self):
        fields = self._fields()
        result = objective.compute_feasible_quotient_objective(
            fields, step_index=31
        )
        self.assertEqual(result.rho, 0.0)
        self.assertEqual(float(result.canonical.detach()), 0.0)
        self.assertEqual(float(result.executed.detach()), 0.0)
        self.assertEqual(float(result.margin.detach()), 0.0)
        self.assertEqual(float(result.temporal_jitter.detach()), 0.0)
        result.total.backward()
        action_gradient = fields.adapted_editor_action.grad
        self.assertTrue(
            action_gradient is None
            or bool(torch.equal(action_gradient, torch.zeros_like(action_gradient)))
        )
        self.assertGreater(
            float(fields.adapted_editor_noop.grad.abs().sum()), 0.0
        )

    def test_radius_is_target_independent(self):
        fields_a = self._fields(target_scale=0.1)
        fields_b = objective.FiveBranchCleanFields(
            frozen_editor_noop=fields_a.frozen_editor_noop,
            frozen_editor_action=fields_a.frozen_editor_action,
            adapted_editor_noop=fields_a.adapted_editor_noop,
            adapted_editor_action=fields_a.adapted_editor_action,
            source_clean=fields_a.source_clean,
            target_clean=fields_a.source_clean + 100.0,
        )
        result_a = objective.compute_feasible_quotient_objective(
            fields_a, step_index=0
        )
        result_b = objective.compute_feasible_quotient_objective(
            fields_b, step_index=0
        )
        self.assertTrue(
            torch.equal(
                result_a.diagnostics.source_only_radius,
                result_b.diagnostics.source_only_radius,
            )
        )

    def test_target_teacher_is_same_query_noop_section_relative(self):
        fields = self._fields()
        result = objective.compute_feasible_quotient_objective(
            fields, step_index=0
        )
        expected = objective._causal_gauge(
            fields.target_clean - fields.frozen_editor_noop
        )
        self.assertTrue(
            torch.equal(result.diagnostics.target_quotient, expected)
        )

        # Changing S while holding the deployed N0 section and T fixed must not
        # change the teacher.  This is the beta=1 double-count regression test.
        changed_source = objective.FiveBranchCleanFields(
            frozen_editor_noop=fields.frozen_editor_noop,
            frozen_editor_action=fields.frozen_editor_action,
            adapted_editor_noop=fields.adapted_editor_noop,
            adapted_editor_action=fields.adapted_editor_action,
            source_clean=fields.source_clean + 100.0,
            target_clean=fields.target_clean,
        )
        changed = objective.compute_feasible_quotient_objective(
            changed_source, step_index=0
        )
        self.assertTrue(
            torch.equal(
                result.diagnostics.target_quotient,
                changed.diagnostics.target_quotient,
            )
        )
        self.assertTrue(
            torch.equal(
                result.diagnostics.target_execution,
                changed.diagnostics.target_execution,
            )
        )

    def test_training_projection_matches_deployment_primitive(self):
        fields = self._fields()
        result = objective.compute_feasible_quotient_objective(
            fields, step_index=0
        )
        deployed = gauge.project_complete_action_quotient(
            frozen_action_field=fields.frozen_editor_action,
            frozen_noop_field=fields.frozen_editor_noop,
            adapted_action_field=fields.adapted_editor_action,
            adapted_noop_field=fields.adapted_editor_noop,
        )
        diagnostics = result.diagnostics
        self.assertTrue(
            torch.equal(
                diagnostics.source_only_radius,
                deployed.diagnostics.radius,
            )
        )
        self.assertTrue(
            torch.equal(
                diagnostics.predicted_smoothed_increments,
                deployed.diagnostics.adapted_smoothed_increments,
            )
        )
        self.assertTrue(
            torch.equal(
                diagnostics.executed_predicted_increments,
                deployed.diagnostics.bounded_increments,
            )
        )

    def test_executed_loss_is_exact_lifted_deployment_field_error(self):
        fields = self._fields()
        config = objective.FeasibleQuotientLossConfig()
        result = objective.compute_feasible_quotient_objective(
            fields, step_index=0, config=config
        )
        diagnostics = result.diagnostics
        predicted = gauge.execute_feasible_quotient_transport(
            fields.frozen_editor_noop.detach(),
            gauge.integrate_phase_increments(
                diagnostics.executed_predicted_increments
            ),
            step_index=0,
        ).executed_clean_field
        target = gauge.execute_feasible_quotient_transport(
            fields.frozen_editor_noop.detach(),
            gauge.integrate_phase_increments(
                diagnostics.canonical_target_increments
            ),
            step_index=0,
        ).executed_clean_field
        self.assertTrue(torch.equal(diagnostics.predicted_execution, predicted))
        self.assertTrue(torch.equal(diagnostics.target_execution, target))
        increment_error = (
            diagnostics.executed_predicted_increments
            - diagnostics.canonical_target_increments
        )
        accumulated_error = gauge.integrate_phase_increments(increment_error)
        self.assertTrue(
            torch.allclose(
                predicted - target,
                accumulated_error,
                rtol=0.0,
                atol=2.0e-6,
            )
        )
        manual = objective._phase_charbonnier_per_sample(
            (predicted - target)[:, 1:],
            diagnostics.execution_normalization_radius,
            epsilon=config.charbonnier_epsilon,
        ).mean()
        self.assertTrue(torch.allclose(result.executed, manual))

    def test_receipt_diagnostics_are_finite_and_explicit(self):
        result = objective.compute_feasible_quotient_objective(
            self._fields(), step_index=0
        )
        receipt = objective.detached_receipt_diagnostics(result)
        cosine = result.diagnostics.target_frozen_prior_cosine_per_phase
        self.assertTrue(torch.all(cosine >= -1.0))
        self.assertTrue(torch.all(cosine <= 1.0))
        self.assertTrue(receipt["target_inside_deployment_radius"])
        self.assertTrue(receipt["motion_noop_full_gradient"])
        self.assertGreater(receipt["source_only_radius_mean_active"], 0.0)
        self.assertGreaterEqual(
            receipt["radius_floor_dominated_fraction_active"], 0.0
        )
        self.assertGreaterEqual(
            receipt["target_clipped_fraction_active"], 0.0
        )
        self.assertGreaterEqual(
            receipt["target_required_radius_multiplier_p50"], 0.0
        )
        self.assertGreaterEqual(
            receipt["target_required_radius_multiplier_p90"],
            receipt["target_required_radius_multiplier_p50"],
        )
        self.assertGreaterEqual(
            receipt["target_required_radius_multiplier_max"],
            receipt["target_required_radius_multiplier_p90"],
        )
        self.assertGreaterEqual(receipt["frozen_noop_to_source_rms"], 0.0)
        self.assertGreaterEqual(receipt["frozen_noop_to_target_rms"], 0.0)
        self.assertGreaterEqual(
            receipt["frozen_noop_target_over_source_error_ratio"], 0.0
        )
        self.assertGreaterEqual(
            receipt["target_frozen_prior_cosine_mean_active"], -1.0
        )
        self.assertLessEqual(
            receipt["target_frozen_prior_cosine_mean_active"], 1.0
        )
        self.assertGreaterEqual(
            receipt["target_frozen_prior_cosine_p10_active"], -1.0
        )
        self.assertLessEqual(
            receipt["target_frozen_prior_cosine_p10_active"], 1.0
        )
        self.assertGreaterEqual(
            receipt["target_frozen_prior_positive_cosine_fraction_active"],
            0.0,
        )
        self.assertLessEqual(
            receipt["target_frozen_prior_positive_cosine_fraction_active"],
            1.0,
        )
        self.assertGreaterEqual(
            receipt["target_high_frequency_fraction_mean_active"], 0.0
        )

    def test_invalid_graph_contract_fails_closed(self):
        fields = self._fields()
        invalid = objective.FiveBranchCleanFields(
            frozen_editor_noop=fields.frozen_editor_noop,
            frozen_editor_action=fields.frozen_editor_action,
            adapted_editor_noop=fields.adapted_editor_noop.detach(),
            adapted_editor_action=fields.adapted_editor_action,
            source_clean=fields.source_clean,
            target_clean=fields.target_clean,
        )
        with self.assertRaises(objective.FeasibleQuotientObjectiveError):
            objective.compute_feasible_quotient_objective(invalid, step_index=0)


class FeasibleQuotientPureContractTests(unittest.TestCase):
    def test_contract_exposes_no_inference_oracle(self):
        contract = objective.immutable_objective_contract()
        self.assertEqual(contract["forwards_per_candidate"], 5)
        self.assertEqual(contract["generator_forwards"], 0)
        self.assertFalse(contract["paired_target_used_as_model_condition"])
        self.assertFalse(contract["first_frame_anchor"])
        self.assertEqual(contract["adapted_quotient"], "Q0(Atheta-Ntheta)")
        self.assertIn("JA_minus_JN", contract["adapted_quotient_gradient"])
        self.assertEqual(
            contract["target_quotient"],
            "Q0(target_clean-stopgrad(frozen_noop_section))",
        )
        self.assertIn("beta1", contract["target_section_reference"])
        self.assertIn("source_only_radius", contract)


if __name__ == "__main__":
    unittest.main()
