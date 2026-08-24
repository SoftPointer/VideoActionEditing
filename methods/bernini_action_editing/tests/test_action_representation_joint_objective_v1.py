from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest

try:
    import torch
except ImportError as error:  # pragma: no cover
    raise unittest.SkipTest("torch unavailable") from error


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import action_representation_joint_objective_v1 as objective  # noqa: E402


def _field(seed: int, *, grad: bool) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    value = torch.randn((1, 5, 6, 8), generator=generator)
    value[:, 0] = 0.0
    return value.requires_grad_(grad)


class JointObjectiveTest(unittest.TestCase):
    def _inputs(self) -> objective.JointObjectiveInputs:
        teacher = _field(1, grad=False)
        correct = (teacher.detach().clone() + 0.05 * _field(2, grad=False)).requires_grad_()
        controls = {
            name: (0.10 * _field(10 + index, grad=False)).requires_grad_()
            for index, name in enumerate(objective.REQUIRED_CONTROLS)
        }
        controls["zero_or_noop"] = torch.zeros_like(teacher)
        route_off = torch.zeros_like(teacher, requires_grad=True)
        frozen = torch.zeros_like(teacher)
        mask = torch.zeros((1, 5, 6, 1), dtype=torch.bool)
        # Real action ABI: phase0 is the unchanged source state; motion starts
        # at phase1 and remains represented through the terminal phase.
        mask[:, 1:, :3] = True
        return objective.JointObjectiveInputs(
            student_correct=correct,
            student_controls=controls,
            detached_teacher_correct=teacher,
            student_route_off=route_off,
            detached_frozen_route_off=frozen,
            action_activity=mask,
        )

    def test_contract_forbids_target_content_and_single_reward(self):
        contract = objective.objective_contract()
        self.assertFalse(contract["target_rgb_accepted"])
        self.assertFalse(contract["target_vae_or_clean_latent_accepted"])
        self.assertFalse(contract["target_absolute_hidden_or_value_accepted"])
        self.assertFalse(contract["single_weighted_reward_exposed"])
        self.assertEqual(contract["required_controls"], list(objective.REQUIRED_CONTROLS))

    def test_objectives_stay_separate_and_cover_every_control(self):
        result = objective.compute_joint_objectives(self._inputs())
        self.assertTrue(result.action.requires_grad)
        self.assertTrue(result.preservation.requires_grad)
        self.assertEqual(
            set(result.diagnostics["independent_control_margins"]),
            set(objective.REQUIRED_CONTROLS),
        )
        self.assertEqual(
            set(result.preservation_components),
            {
                "exact_zero_route_replay",
                "phase0_source_tether",
                "outside_action_tube_tether",
            },
        )

    def test_missing_control_fails_closed(self):
        inputs = self._inputs()
        controls = dict(inputs.student_controls)
        controls.pop("incomplete")
        with self.assertRaises(objective.ActionRepresentationObjectiveError):
            objective.compute_joint_objectives(
                objective.JointObjectiveInputs(
                    student_correct=inputs.student_correct,
                    student_controls=controls,
                    detached_teacher_correct=inputs.detached_teacher_correct,
                    student_route_off=inputs.student_route_off,
                    detached_frozen_route_off=inputs.detached_frozen_route_off,
                    action_activity=inputs.action_activity,
                )
            )

    def test_real_activity_mask_requires_inactive_phase0_onset_and_terminal(self):
        valid = self._inputs()
        self.assertFalse(bool(valid.action_activity[:, 0].any().item()))
        objective.compute_joint_objectives(valid)

        phase0_active = valid.action_activity.clone()
        phase0_active[:, 0, 0] = True
        with self.assertRaisesRegex(
            objective.ActionRepresentationObjectiveError,
            "phase0 must be entirely inactive",
        ):
            objective.compute_joint_objectives(
                replace(valid, action_activity=phase0_active)
            )

        no_onset = valid.action_activity.clone()
        no_onset[:, 1] = False
        with self.assertRaisesRegex(
            objective.ActionRepresentationObjectiveError,
            "phase1 must cover the onset transition",
        ):
            objective.compute_joint_objectives(
                replace(valid, action_activity=no_onset)
            )

        no_terminal = valid.action_activity.clone()
        no_terminal[:, -1] = False
        with self.assertRaisesRegex(
            objective.ActionRepresentationObjectiveError,
            "cover the terminal phase",
        ):
            objective.compute_joint_objectives(
                replace(valid, action_activity=no_terminal)
            )

    def test_projection_removes_action_component_opposing_preservation(self):
        action = (torch.tensor([-2.0, 4.0]),)
        preservation = (torch.tensor([1.0, 0.0]),)
        result = objective.project_action_against_preservation_gradients(
            action, preservation, maximum_gradient_norm=100.0
        )
        projected_action = result.gradients[0] - preservation[0]
        self.assertTrue(result.diagnostics["conflict_projected"])
        self.assertAlmostEqual(
            float(torch.dot(projected_action, preservation[0]).item()),
            0.0,
            places=6,
        )

    def test_pcgrad_assigns_gradients_only_when_noop_trust_holds(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, -0.5]))
        action = parameter.square().sum()
        preservation = (parameter.sum() * 0.25).square()
        result = objective.JointObjectiveResult(
            action=action,
            preservation=preservation,
            action_components={},
            preservation_components={
                "exact_zero_route_replay": parameter.new_zeros(()),
            },
            diagnostics={},
        )
        metrics = objective.backward_with_preservation_pcgrad(result, (parameter,))
        self.assertIsNotNone(parameter.grad)
        self.assertIn("action_preservation_dot", metrics)
        bad = objective.JointObjectiveResult(
            action=parameter.square().sum(),
            preservation=parameter.square().mean(),
            action_components={},
            preservation_components={
                "exact_zero_route_replay": parameter.new_tensor(1.0e-3),
            },
            diagnostics={},
        )
        with self.assertRaises(objective.ActionRepresentationObjectiveError):
            objective.backward_with_preservation_pcgrad(bad, (parameter,))

    def test_zero_preservation_gradient_uses_bounded_action_only_failsafe(self):
        parameter = torch.nn.Parameter(torch.tensor([0.25, -0.75]))
        action = parameter.sum()
        zero_preservation = (parameter * 0.0).sum()
        result = objective.JointObjectiveResult(
            action=action,
            preservation=zero_preservation,
            action_components={},
            preservation_components={
                "exact_zero_route_replay": parameter.new_zeros(()),
            },
            diagnostics={},
        )
        metrics = objective.backward_with_preservation_pcgrad(
            result, (parameter,)
        )
        self.assertTrue(metrics["zero_preservation_gradient_fallback"])
        self.assertFalse(metrics["preservation_gradient_nonzero"])
        self.assertEqual(
            metrics["gradient_combination_mode"],
            "initial_zero_preservation_action_only_failsafe",
        )
        self.assertFalse(metrics["fallback_establishes_tp_preservation"])
        self.assertTrue(metrics["preservation_scalar_within_noop_trust"])
        self.assertIsNotNone(parameter.grad)
        self.assertGreater(int(torch.count_nonzero(parameter.grad).item()), 0)

        outside_trust = objective.JointObjectiveResult(
            action=parameter.sum(),
            preservation=(parameter * 0.0).sum() + 1.0e-4,
            action_components={},
            preservation_components={
                "exact_zero_route_replay": parameter.new_zeros(()),
            },
            diagnostics={},
        )
        with self.assertRaisesRegex(
            objective.ActionRepresentationObjectiveError,
            "outside the preservation trust boundary",
        ):
            objective.backward_with_preservation_pcgrad(
                outside_trust, (parameter,)
            )

    def test_exact_zero_student_has_finite_nonzero_action_gradient(self):
        inputs = self._inputs()
        exact_zero = torch.zeros_like(
            inputs.student_correct, requires_grad=True
        )
        zero_controls = {
            name: torch.zeros_like(exact_zero)
            for name in objective.REQUIRED_CONTROLS
        }
        zero_inputs = replace(
            inputs,
            student_correct=exact_zero,
            student_controls=zero_controls,
        )
        result = objective.compute_joint_objectives(zero_inputs)
        self.assertTrue(bool(torch.isfinite(result.action).item()))
        self.assertEqual(
            result.diagnostics["rms_stabilization"],
            "mean_square_clamp_floor_squared_before_sqrt",
        )
        self.assertEqual(
            float(result.diagnostics["student_action_rms_raw"].item()),
            0.0,
        )
        self.assertAlmostEqual(
            float(result.diagnostics["student_action_rms"].item()),
            objective.JointObjectiveConfig().normalization_floor,
            places=12,
        )
        gradient = torch.autograd.grad(result.action, exact_zero)[0]
        self.assertTrue(bool(torch.isfinite(gradient).all().item()))
        self.assertGreater(int(torch.count_nonzero(gradient).item()), 0)


if __name__ == "__main__":
    unittest.main()
