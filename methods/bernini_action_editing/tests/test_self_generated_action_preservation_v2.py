import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import self_generated_action_preservation_v2 as preservation


def require_torch():
    try:
        import torch
    except ModuleNotFoundError as error:
        raise unittest.SkipTest("torch is unavailable") from error
    return torch


class RegistryTests(unittest.TestCase):
    def test_registry_is_matched_eight_arm_canary(self):
        self.assertEqual(len(preservation.ARM_NAMES), 8)
        self.assertAlmostEqual(preservation.arm_spec("v2_func010_all").functional_weight, 0.10)
        self.assertAlmostEqual(preservation.arm_spec("v2_func025_all").functional_weight, 0.25)
        self.assertAlmostEqual(preservation.arm_spec("v2_func050_all").functional_weight, 0.50)
        self.assertAlmostEqual(preservation.arm_spec("v2_noop020_all").noop_weight, 0.20)
        self.assertEqual(
            preservation.arm_spec("v2_func025_cross_qo").route_scope,
            "cross_attn2_qo",
        )


class ObjectiveTests(unittest.TestCase):
    def test_onset_envelope_is_exact_weighted_first_three_phases(self):
        torch = require_torch()
        source = torch.zeros(1, 16, 21, 4, 4)
        predicted = source.clone().requires_grad_(True)
        with torch.no_grad():
            predicted[:, :, 0].fill_(1.0)
            predicted[:, :, 1].fill_(2.0)
            predicted[:, :, 2].fill_(3.0)
            predicted[:, :, 3].fill_(100.0)
        loss = preservation.onset_envelope_loss(
            predicted_clean=predicted, source_clean=source
        )
        self.assertAlmostEqual(loss.item(), 1.0 + 0.5 * 4.0 + 0.25 * 9.0)
        loss.backward()
        self.assertIsNotNone(predicted.grad)
        self.assertGreater(predicted.grad[:, :, :3].abs().sum().item(), 0)
        self.assertEqual(predicted.grad[:, :, 3:].abs().sum().item(), 0)

    def test_functional_loss_exempts_teacher_action_direction(self):
        torch = require_torch()
        teacher = torch.zeros(1, 21, 32)
        teacher[0, 0, 0] = 1.0
        frozen = torch.zeros_like(teacher)
        student = (7.0 * teacher).clone().requires_grad_(True)
        loss = preservation.functional_non_regression_loss(
            student_action_code=student,
            frozen_action_code=frozen,
            teacher_action_unit=teacher,
        )
        self.assertAlmostEqual(loss.item(), 0.0, places=12)
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertAlmostEqual(student.grad.abs().sum().item(), 0.0, places=12)

    def test_functional_loss_penalizes_only_orthogonal_drift(self):
        torch = require_torch()
        teacher = torch.zeros(1, 21, 32)
        teacher[0, 0, 0] = 1.0
        frozen = torch.zeros_like(teacher)
        student = torch.zeros_like(teacher, requires_grad=True)
        with torch.no_grad():
            student[0, 0, 0] = 11.0
            student[0, 0, 1] = 3.0
        loss = preservation.functional_non_regression_loss(
            student_action_code=student,
            frozen_action_code=frozen,
            teacher_action_unit=teacher,
        )
        self.assertAlmostEqual(loss.item(), 9.0 / (21 * 32), places=8)
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertAlmostEqual(student.grad[0, 0, 0].item(), 0.0, places=12)
        self.assertGreater(student.grad[0, 0, 1].abs().item(), 0)

    def test_functional_authorities_must_be_detached_and_finite(self):
        torch = require_torch()
        shape = (1, 21, 32)
        student = torch.zeros(shape, requires_grad=True)
        frozen = torch.zeros(shape, requires_grad=True)
        teacher = torch.ones(shape)
        with self.assertRaisesRegex(preservation.ActionPreservationV2Error, "detached"):
            preservation.functional_non_regression_loss(
                student_action_code=student,
                frozen_action_code=frozen,
                teacher_action_unit=teacher,
            )
        frozen = frozen.detach()
        teacher[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(preservation.ActionPreservationV2Error, "non-finite"):
            preservation.functional_non_regression_loss(
                student_action_code=student,
                frozen_action_code=frozen,
                teacher_action_unit=teacher,
            )

    def test_temporal_dc_detects_static_drift_but_not_zero_mean_motion(self):
        torch = require_torch()
        frozen = torch.zeros(1, 16, 21, 2, 2)
        static = frozen.clone().requires_grad_(True)
        with torch.no_grad():
            static[:, :, 3:].fill_(2.0)
        loss = preservation.temporal_dc_non_regression_loss(
            student_velocity=static,
            frozen_velocity=frozen,
            sigma=0.5,
        )
        self.assertAlmostEqual(loss.item(), 1.0, places=7)
        loss.backward()
        self.assertEqual(static.grad[:, :, :3].abs().sum().item(), 0.0)
        self.assertGreater(static.grad[:, :, 3:].abs().sum().item(), 0.0)

        alternating = frozen.clone()
        alternating[:, :, 3::2].fill_(1.0)
        alternating[:, :, 4::2].fill_(-1.0)
        # There are 18 post-onset phases, so this drift has exact zero DC.
        loss = preservation.temporal_dc_non_regression_loss(
            student_velocity=alternating,
            frozen_velocity=frozen,
            sigma=0.75,
        )
        self.assertAlmostEqual(loss.item(), 0.0, places=12)

    def test_temporal_dc_rejects_attached_or_malformed_authority(self):
        torch = require_torch()
        student = torch.zeros(1, 16, 21, 2, 2, requires_grad=True)
        frozen = torch.zeros_like(student, requires_grad=True)
        with self.assertRaisesRegex(preservation.ActionPreservationV2Error, "detached"):
            preservation.temporal_dc_non_regression_loss(
                student_velocity=student,
                frozen_velocity=frozen,
                sigma=0.5,
            )
        with self.assertRaisesRegex(preservation.ActionPreservationV2Error, "sigma"):
            preservation.temporal_dc_non_regression_loss(
                student_velocity=student,
                frozen_velocity=frozen.detach(),
                sigma=float("nan"),
            )


def projection_names():
    return [
        f"diff_dec.transformer.blocks.{block}.attn{attention}.{projection}"
        for block in range(2)
        for attention in (1, 2)
        for projection in ("to_q", "to_k", "to_v", "to_out.0")
    ]


class ScopeTests(unittest.TestCase):
    def test_projection_scope_is_exact_and_does_not_claim_temporal_only(self):
        names = projection_names()
        self.assertEqual(
            preservation.select_projection_scope(names, scope="all_attention"),
            sorted(names),
        )
        selected = preservation.select_projection_scope(names, scope="cross_attn2_qo")
        expected = sorted(
            name
            for name in names
            if ".attn2.to_q" in name or ".attn2.to_out.0" in name
        )
        self.assertEqual(selected, expected)
        self.assertEqual(len(selected), 4)
        with self.assertRaisesRegex(preservation.ActionPreservationV2Error, "unsupported"):
            preservation.select_projection_scope(names, scope="temporal_only")

    def test_projection_scope_rejects_duplicate_or_unregistered_names(self):
        names = projection_names()
        with self.assertRaisesRegex(preservation.ActionPreservationV2Error, "unique"):
            preservation.select_projection_scope(
                names + [names[0]], scope="all_attention"
            )
        with self.assertRaisesRegex(preservation.ActionPreservationV2Error, "invalid"):
            preservation.select_projection_scope(
                ["transformer.foo"], scope="all_attention"
            )


if __name__ == "__main__":
    unittest.main()
