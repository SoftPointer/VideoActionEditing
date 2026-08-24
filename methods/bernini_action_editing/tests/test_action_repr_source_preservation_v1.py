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

import action_repr_source_preservation_v1 as preservation  # noqa: E402


class _ToyRoute(torch.nn.Module):
    """Two genuinely disjoint branches with zero-initialized outputs."""

    def __init__(self, *, feature_width: int) -> None:
        super().__init__()
        self.motion_gain = torch.nn.Parameter(torch.zeros(()))
        self.source_copy = torch.nn.Parameter(torch.zeros(feature_width))

    def fields(
        self, activity: torch.Tensor, motion_direction: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        motion = self.motion_gain * motion_direction.view(1, 1, 1, -1)
        motion = motion.expand(*activity.shape[:3], -1)
        motion = torch.where(activity, motion, torch.zeros_like(motion))
        source_copy = self.source_copy.view(1, 1, 1, -1).expand_as(motion)
        return motion, source_copy


def _authorities() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260824)
    frozen = torch.randn((1, 4, 5, 4), generator=generator)
    prefix = torch.randn((1, 3, 4), generator=generator)
    action = torch.zeros_like(frozen)
    action[:, 1:, :3, 0] = 1.0
    activity = torch.zeros((1, 4, 5, 1), dtype=torch.bool)
    activity[:, 1:, :3] = True
    return frozen, prefix, action, activity


def _inputs(
    model: _ToyRoute,
    *,
    motion_direction: torch.Tensor | None = None,
) -> preservation.SourcePreservationInputs:
    frozen, prefix, action, activity = _authorities()
    if motion_direction is None:
        # e0 is the authorized action direction; e1 models normal adapter spill.
        motion_direction = torch.tensor([1.0, 0.25, 0.0, 0.0])
    motion, source_copy = model.fields(activity, motion_direction)
    return preservation.SourcePreservationInputs(
        detached_frozen_route_hidden=frozen,
        detached_source_prefix=prefix,
        detached_action_representation=action,
        student_motion_middle_delta=motion,
        student_source_copy_delta=source_copy,
        action_activity=activity,
    )


class SourcePreservationTest(unittest.TestCase):
    def test_contract_forbids_target_content_and_declares_gradient_owners(self):
        contract = preservation.preservation_contract()
        self.assertFalse(contract["target_rgb_accepted"])
        self.assertFalse(contract["target_vae_or_clean_latent_accepted"])
        self.assertFalse(contract["target_absolute_hidden_or_value_accepted"])
        self.assertEqual(contract["motion_middle_gradient_owner"], "action_arm_only")
        self.assertEqual(
            contract["source_copy_gradient_owner"], "preservation_arm_only"
        )
        self.assertTrue(contract["step0_zero_preservation_allowed"])
        self.assertTrue(
            contract["after_first_action_update_nonzero_source_copy_gradient_required"]
        )

    def test_owned_views_have_exact_disjoint_gradients(self):
        model = _ToyRoute(feature_width=4)
        inputs = _inputs(model)
        views = preservation.build_owned_route_views(inputs)

        action_loss = views.action.square().mean()
        action_motion, action_copy = torch.autograd.grad(
            action_loss,
            (model.motion_gain, model.source_copy),
            retain_graph=True,
            allow_unused=True,
        )
        self.assertIsNotNone(action_motion)
        self.assertGreater(abs(float(action_motion.item())), 0.0)
        self.assertIsNone(action_copy)

        preservation_loss = views.preservation.square().mean()
        preserve_motion, preserve_copy = torch.autograd.grad(
            preservation_loss,
            (model.motion_gain, model.source_copy),
            allow_unused=True,
        )
        self.assertIsNone(preserve_motion)
        self.assertIsNotNone(preserve_copy)

    def test_step0_can_be_exact_zero_without_claiming_tp(self):
        model = _ToyRoute(feature_width=4)
        result = preservation.compute_source_preservation(_inputs(model))
        self.assertEqual(float(result.loss.item()), 0.0)
        self.assertTrue(all(float(value.item()) == 0.0 for value in result.components.values()))
        metrics = preservation.backward_source_copy_only(
            result,
            source_copy_parameters=(model.source_copy,),
            motion_middle_parameters=(model.motion_gain,),
            expectation="initial_step0",
        )
        self.assertFalse(metrics["source_copy_gradient_nonzero"])
        self.assertFalse(metrics["establishes_tp_preservation"])
        self.assertEqual(int(torch.count_nonzero(model.source_copy.grad).item()), 0)

    def test_first_action_step_creates_nonzero_source_copy_signal_only(self):
        model = _ToyRoute(feature_width=4)
        inputs0 = _inputs(model)
        # The action arm sees source-copy as detached.  A scalar motion gate
        # inevitably carries a small e1 spill while learning the e0 action.
        action_delta = (
            preservation.build_owned_route_views(inputs0).action
            - inputs0.detached_frozen_route_hidden
        )
        desired = inputs0.detached_action_representation
        action_loss = (action_delta - desired).square().masked_select(
            inputs0.action_activity.expand_as(action_delta)
        ).mean()
        motion_gradient, copy_gradient = torch.autograd.grad(
            action_loss,
            (model.motion_gain, model.source_copy),
            allow_unused=True,
        )
        self.assertIsNotNone(motion_gradient)
        self.assertIsNone(copy_gradient)
        with torch.no_grad():
            model.motion_gain.add_(-0.5 * motion_gradient)

        # Preserve the already-computed action gradient as a sentinel: the
        # preservation backward must neither clear nor modify it.
        model.motion_gain.grad = torch.tensor(7.0)
        result1 = preservation.compute_source_preservation(_inputs(model))
        self.assertGreater(float(result1.loss.item()), 0.0)
        metrics = preservation.backward_source_copy_only(
            result1,
            source_copy_parameters=(model.source_copy,),
            motion_middle_parameters=(model.motion_gain,),
            expectation="after_first_action_update",
        )
        self.assertTrue(metrics["source_copy_gradient_nonzero"])
        self.assertTrue(metrics["establishes_tp_preservation"])
        self.assertEqual(float(model.motion_gain.grad.item()), 7.0)
        self.assertGreater(int(torch.count_nonzero(model.source_copy.grad).item()), 0)
        # e0 is exempt as the action tangent; the corrective gradient is in e1.
        self.assertAlmostEqual(float(model.source_copy.grad[0].item()), 0.0, places=7)
        self.assertGreater(abs(float(model.source_copy.grad[1].item())), 0.0)

    def test_first_update_gate_rejects_action_exactly_inside_tangent(self):
        model = _ToyRoute(feature_width=4)
        with torch.no_grad():
            model.motion_gain.fill_(0.25)
        result = preservation.compute_source_preservation(
            _inputs(model, motion_direction=torch.tensor([1.0, 0.0, 0.0, 0.0]))
        )
        with self.assertRaisesRegex(
            preservation.SourcePreservationError,
            "no nontrivial preservation signal",
        ):
            preservation.backward_source_copy_only(
                result,
                source_copy_parameters=(model.source_copy,),
                motion_middle_parameters=(model.motion_gain,),
                expectation="after_first_action_update",
            )

    def test_phase0_and_outside_terms_are_not_structurally_zero(self):
        model = _ToyRoute(feature_width=4)
        valid = _inputs(model)
        route_drift = torch.nn.Parameter(torch.tensor(0.20))
        pattern = torch.zeros_like(valid.student_motion_middle_delta)
        pattern[:, 0, :, 2] = 1.0
        pattern[:, 1:, 3:, 3] = -0.5
        disturbed_motion = valid.student_motion_middle_delta + route_drift * pattern
        result = preservation.compute_source_preservation(
            replace(valid, student_motion_middle_delta=disturbed_motion)
        )
        self.assertGreater(
            float(result.components["phase0_source_tether"].item()), 0.0
        )
        self.assertGreater(
            float(result.components["outside_route_source_tether"].item()), 0.0
        )
        metrics = preservation.backward_source_copy_only(
            result,
            source_copy_parameters=(model.source_copy,),
            motion_middle_parameters=(model.motion_gain, route_drift),
            expectation="after_first_action_update",
        )
        self.assertTrue(metrics["source_copy_gradient_nonzero"])
        self.assertIsNone(route_drift.grad)

    def test_information_firewall_and_phase0_fail_closed(self):
        model = _ToyRoute(feature_width=4)
        valid = _inputs(model)
        leaking_source = valid.detached_frozen_route_hidden.clone().requires_grad_()
        with self.assertRaisesRegex(
            preservation.SourcePreservationError,
            "detached_frozen_route_hidden requires_grad differs",
        ):
            preservation.compute_source_preservation(
                replace(valid, detached_frozen_route_hidden=leaking_source)
            )

        active_phase0 = valid.action_activity.clone()
        active_phase0[:, 0, 0] = True
        with self.assertRaisesRegex(
            preservation.SourcePreservationError,
            "phase0 must be entirely inactive",
        ):
            preservation.compute_source_preservation(
                replace(valid, action_activity=active_phase0)
            )

        nonzero_phase0_action = valid.detached_action_representation.clone()
        nonzero_phase0_action[:, 0, 0, 0] = 1.0
        with self.assertRaisesRegex(
            preservation.SourcePreservationError,
            "representation phase0 must be exact zero",
        ):
            preservation.compute_source_preservation(
                replace(
                    valid,
                    detached_action_representation=nonzero_phase0_action,
                )
            )

    def test_parameter_overlap_and_cross_branch_leak_fail_closed(self):
        model = _ToyRoute(feature_width=4)
        with torch.no_grad():
            model.motion_gain.fill_(0.25)
        result = preservation.compute_source_preservation(_inputs(model))
        with self.assertRaisesRegex(
            preservation.SourcePreservationError,
            "closures overlap",
        ):
            preservation.backward_source_copy_only(
                result,
                source_copy_parameters=(model.source_copy,),
                motion_middle_parameters=(model.source_copy,),
                expectation="steady_state",
            )

        # Malformed source-copy output depends on a motion-owned parameter;
        # the autograd ownership audit catches the leak even though the normal
        # motion-delta input itself is detached by the objective.
        valid = _inputs(model)
        leaked_copy = valid.student_source_copy_delta + model.motion_gain
        leaked = preservation.compute_source_preservation(
            replace(valid, student_source_copy_delta=leaked_copy)
        )
        with self.assertRaisesRegex(
            preservation.SourcePreservationError,
            "leaked gradient into motion/middle",
        ):
            preservation.backward_source_copy_only(
                leaked,
                source_copy_parameters=(model.source_copy,),
                motion_middle_parameters=(model.motion_gain,),
                expectation="steady_state",
            )


if __name__ == "__main__":
    unittest.main()
