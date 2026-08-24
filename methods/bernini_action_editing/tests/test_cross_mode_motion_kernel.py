from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import cross_mode_motion_kernel as cmkd


class CrossModeMotionKernelPureContractTests(unittest.TestCase):
    def test_fixed_method_and_auditable_result_contract(self) -> None:
        self.assertEqual(
            cmkd.METHOD_NAME,
            "cross-mode-motion-kernel-distillation-v7",
        )
        self.assertEqual(cmkd.EXPECTED_PHASES, 21)
        self.assertEqual(cmkd.ACTIVE_INCREMENTS, 20)
        config = cmkd.CrossModeMotionKernelConfig()
        config.validate()
        self.assertEqual(
            {item.name for item in fields(cmkd.MotionKernelStatistics)},
            {
                "phase_increments",
                "active_increment_mask",
                "temporal_self_kernel",
                "cross_phase_relational_kernel",
                "centered_cross_phase_relational_kernel",
                "centered_relational_frobenius",
                "off_diagonal_relational_rms",
                "phase_rms_envelope",
                "temporal_frequency_power",
                "normalized_temporal_frequency_power",
                "total_increment_rms",
            },
        )

    def test_configuration_fails_closed(self) -> None:
        invalid = (
            {"min_centered_kernel_alignment": -1.01},
            {"min_centered_kernel_alignment": 1.01},
            {"min_off_diagonal_relational_rms": 0.0},
            {"min_off_diagonal_relational_rms": 1.01},
            {"min_envelope_cosine": -0.01},
            {"min_envelope_cosine": 1.01},
            {"max_envelope_relative_error": -0.01},
            {"min_teacher_target_energy_ratio": 0.0},
            {"max_teacher_target_energy_ratio": 0.0},
            {
                "min_teacher_target_energy_ratio": 2.0,
                "max_teacher_target_energy_ratio": 1.0,
            },
            {"min_motion_rms": 0.0},
            {"target_direction_weight": -1.0},
            {"epsilon": 0.0},
            {"epsilon": float("nan")},
            {"epsilon": True},
            {
                "target_direction_weight": 0.0,
                "teacher_kernel_weight": 0.0,
                "amplitude_envelope_weight": 0.0,
                "temporal_jitter_weight": 0.0,
            },
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(cmkd.CrossModeMotionKernelError):
                    cmkd.CrossModeMotionKernelConfig(**values).validate()

    def test_torch_is_lazy(self) -> None:
        tree = ast.parse(Path(cmkd.__file__).read_text(encoding="utf-8"))
        eager = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                eager.extend(alias.name for alias in node.names if alias.name == "torch")
            elif isinstance(node, ast.ImportFrom) and node.module == "torch":
                eager.append(node.module)
        self.assertEqual(eager, [])


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is unavailable")
class CrossModeMotionKernelTensorTests(unittest.TestCase):
    @staticmethod
    def _from_increments(increments):
        zero = torch.zeros_like(increments[:, :1])
        return torch.cat((zero, torch.cumsum(increments, dim=1)), dim=1)

    @classmethod
    def _motion(cls, *, batch=2, spatial=5, channels=4):
        generator = torch.Generator().manual_seed(9147)
        innovations = torch.randn(
            batch,
            20,
            spatial,
            channels,
            generator=generator,
            dtype=torch.float32,
        )
        # Correlated, nonstationary increments make temporal order identifiable
        # rather than producing a near-identity Gram matrix.
        increments = torch.empty_like(innovations)
        state = torch.zeros(batch, spatial, channels, dtype=torch.float32)
        envelope = torch.linspace(0.25, 1.75, 20, dtype=torch.float32)
        envelope = envelope * (
            1.0 + 0.20 * torch.sin(torch.arange(20, dtype=torch.float32) * 0.73)
        )
        for phase in range(20):
            state = 0.72 * state + innovations[:, phase]
            increments[:, phase] = envelope[phase] * state
        return cls._from_increments(increments)

    @staticmethod
    def _orthogonal(batch, width, *, seed):
        generator = torch.Generator().manual_seed(seed)
        matrices = torch.randn(
            batch, width, width, generator=generator, dtype=torch.float32
        )
        orthogonal, _ = torch.linalg.qr(matrices)
        return orthogonal

    @classmethod
    def _separable_transform(cls, value, *, spatial_seed, channel_seed):
        batch, _, spatial, channels = value.shape
        spatial_q = cls._orthogonal(batch, spatial, seed=spatial_seed)
        channel_q = cls._orthogonal(batch, channels, seed=channel_seed)
        transformed = torch.einsum("bij,btjd->btid", spatial_q, value)
        transformed = torch.einsum("btid,bdk->btik", transformed, channel_q)
        # Matrix multiplication of exact zeros remains exact, but make this
        # contract explicit so the test fixture cannot hide a backend change.
        transformed = torch.cat(
            (torch.zeros_like(transformed[:, :1]), transformed[:, 1:]), dim=1
        )
        return transformed

    def test_statistics_shapes_phase_zero_dtype_and_finiteness(self) -> None:
        direction = self._motion(batch=2, spatial=6, channels=3)
        statistics = cmkd.motion_kernel_statistics(direction)
        self.assertEqual(statistics.phase_increments.shape, (2, 21, 6, 3))
        self.assertEqual(statistics.active_increment_mask.shape, (2, 20))
        self.assertEqual(statistics.temporal_self_kernel.shape, (2, 20, 20))
        self.assertEqual(
            statistics.cross_phase_relational_kernel.shape, (2, 20, 20)
        )
        self.assertEqual(
            statistics.centered_cross_phase_relational_kernel.shape,
            (2, 20, 20),
        )
        self.assertEqual(statistics.off_diagonal_relational_rms.shape, (2,))
        self.assertEqual(statistics.phase_rms_envelope.shape, (2, 21))
        self.assertEqual(statistics.temporal_frequency_power.shape, (2, 11))
        self.assertEqual(
            statistics.normalized_temporal_frequency_power.shape, (2, 11)
        )
        zero_increment = torch.zeros_like(statistics.phase_increments[:, 0])
        zero_envelope = torch.zeros_like(statistics.phase_rms_envelope[:, 0])
        self.assertTrue(torch.equal(statistics.phase_increments[:, 0], zero_increment))
        self.assertTrue(torch.equal(statistics.phase_rms_envelope[:, 0], zero_envelope))
        for value in (
            statistics.phase_increments,
            statistics.temporal_self_kernel,
            statistics.cross_phase_relational_kernel,
            statistics.centered_cross_phase_relational_kernel,
            statistics.centered_relational_frobenius,
            statistics.off_diagonal_relational_rms,
            statistics.phase_rms_envelope,
            statistics.temporal_frequency_power,
            statistics.normalized_temporal_frequency_power,
            statistics.total_increment_rms,
        ):
            self.assertEqual(value.dtype, torch.float32)
            self.assertTrue(bool(torch.isfinite(value).all()))

    def test_exact_zero_increment_has_zero_kernel_row_and_no_nan(self) -> None:
        direction = self._motion(batch=1)
        increments = direction[:, 1:] - direction[:, :-1]
        increments[:, 7] = 0.0
        direction = self._from_increments(increments)
        statistics = cmkd.motion_kernel_statistics(direction)
        self.assertFalse(bool(statistics.active_increment_mask[0, 7]))
        self.assertTrue(
            torch.equal(
                statistics.temporal_self_kernel[0, 7],
                torch.zeros_like(statistics.temporal_self_kernel[0, 7]),
            )
        )
        self.assertTrue(
            torch.equal(
                statistics.temporal_self_kernel[0, :, 7],
                torch.zeros_like(statistics.temporal_self_kernel[0, :, 7]),
            )
        )
        self.assertTrue(
            torch.equal(
                torch.diagonal(statistics.cross_phase_relational_kernel[0]),
                torch.zeros(20),
            )
        )

    def test_independent_spatial_and_channel_orthogonal_invariance(self) -> None:
        direction = self._motion(batch=2, spatial=5, channels=4)
        transformed = self._separable_transform(
            direction, spatial_seed=11, channel_seed=29
        )
        original = cmkd.motion_kernel_statistics(direction)
        rotated = cmkd.motion_kernel_statistics(transformed)
        for left, right in (
            (original.temporal_self_kernel, rotated.temporal_self_kernel),
            (
                original.cross_phase_relational_kernel,
                rotated.cross_phase_relational_kernel,
            ),
            (
                original.centered_cross_phase_relational_kernel,
                rotated.centered_cross_phase_relational_kernel,
            ),
            (
                original.off_diagonal_relational_rms,
                rotated.off_diagonal_relational_rms,
            ),
            (original.phase_rms_envelope, rotated.phase_rms_envelope),
            (original.temporal_frequency_power, rotated.temporal_frequency_power),
            (original.total_increment_rms, rotated.total_increment_rms),
        ):
            self.assertTrue(torch.allclose(left, right, atol=2e-5, rtol=2e-5))

    def test_coordinate_permutation_and_sign_invariance(self) -> None:
        direction = self._motion(batch=2, spatial=5, channels=4)
        spatial_permutation = torch.tensor([3, 0, 4, 1, 2])
        channel_permutation = torch.tensor([2, 0, 3, 1])
        permuted = direction[:, :, spatial_permutation][:, :, :, channel_permutation]
        signs = torch.tensor(
            [
                [[1, -1, 1, -1], [-1, -1, 1, 1], [1, 1, -1, -1],
                 [-1, 1, -1, 1], [1, -1, -1, 1]],
                [[-1, 1, 1, -1], [1, -1, 1, -1], [-1, 1, -1, 1],
                 [1, 1, -1, -1], [-1, -1, 1, 1]],
            ],
            dtype=torch.float32,
        )
        permuted = permuted * signs[:, None]
        original = cmkd.motion_kernel_statistics(direction)
        changed = cmkd.motion_kernel_statistics(permuted)
        self.assertTrue(
            torch.equal(original.temporal_self_kernel, changed.temporal_self_kernel)
            or torch.allclose(
                original.temporal_self_kernel,
                changed.temporal_self_kernel,
                atol=2e-6,
                rtol=2e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                original.phase_rms_envelope,
                changed.phase_rms_envelope,
                atol=2e-6,
                rtol=2e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                original.cross_phase_relational_kernel,
                changed.cross_phase_relational_kernel,
                atol=2e-6,
                rtol=2e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                original.temporal_frequency_power,
                changed.temporal_frequency_power,
                atol=2e-5,
                rtol=2e-5,
            )
        )

    def test_identity_diagonal_cannot_admit_independent_high_dimensional_noise(self) -> None:
        def centered(value):
            return (
                value
                - value.mean(dim=-1, keepdim=True)
                - value.mean(dim=-2, keepdim=True)
                + value.mean(dim=(-2, -1), keepdim=True)
            )

        for weak_correlation in (0.0, 0.02):
            with self.subTest(weak_correlation=weak_correlation):
                generator_a = torch.Generator().manual_seed(701)
                generator_b = torch.Generator().manual_seed(809)
                teacher_increments = torch.randn(
                    1, 20, 64, 64, generator=generator_a, dtype=torch.float32
                )
                target_increments = torch.randn(
                    1, 20, 64, 64, generator=generator_b, dtype=torch.float32
                )
                if weak_correlation:
                    teacher_increments[:, 1:] += (
                        weak_correlation * teacher_increments[:, :-1].clone()
                    )
                    target_increments[:, 1:] += (
                        weak_correlation * target_increments[:, :-1].clone()
                    )
                teacher = self._from_increments(teacher_increments)
                target = self._from_increments(target_increments)
                diagnostics = cmkd.evaluate_teacher_target_eligibility(
                    teacher, target
                )

                # This reproduces the rejected v6-style failure: centering the
                # raw normalized Gram leaves the common identity diagonal, so
                # unrelated high-dimensional motion appears almost identical.
                raw_teacher = centered(diagnostics.teacher.temporal_self_kernel)
                raw_target = centered(diagnostics.target.temporal_self_kernel)
                old_diagonal_dominated_alignment = torch.nn.functional.cosine_similarity(
                    raw_teacher.flatten(start_dim=1),
                    raw_target.flatten(start_dim=1),
                    dim=1,
                )
                self.assertGreater(float(old_diagonal_dominated_alignment), 0.90)

                self.assertFalse(bool(diagnostics.eligible.item()))
                self.assertFalse(
                    bool(diagnostics.off_diagonal_relational_pass.item())
                )
                self.assertLess(
                    float(diagnostics.teacher.off_diagonal_relational_rms),
                    cmkd.CrossModeMotionKernelConfig().min_off_diagonal_relational_rms,
                )
                self.assertLess(
                    float(diagnostics.target.off_diagonal_relational_rms),
                    cmkd.CrossModeMotionKernelConfig().min_off_diagonal_relational_rms,
                )

    def test_eligibility_accepts_independently_coordinated_videos(self) -> None:
        base = self._motion(batch=2, spatial=5, channels=4)
        teacher = self._separable_transform(
            base, spatial_seed=101, channel_seed=103
        )
        target = self._separable_transform(
            base, spatial_seed=211, channel_seed=223
        )
        diagnostics = cmkd.evaluate_teacher_target_eligibility(teacher, target)
        self.assertTrue(bool(diagnostics.eligible.all()))
        self.assertTrue(
            torch.allclose(
                diagnostics.centered_kernel_alignment,
                torch.ones(2),
                atol=2e-5,
                rtol=2e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                diagnostics.envelope_cosine,
                torch.ones(2),
                atol=2e-5,
                rtol=2e-5,
            )
        )
        # Different per-video coordinates need not have useful raw cosine;
        # their acceptance is based on self-statistics instead.
        raw_cosine = torch.nn.functional.cosine_similarity(
            teacher[:, 1:].flatten(start_dim=1),
            target[:, 1:].flatten(start_dim=1),
            dim=1,
        )
        self.assertTrue(bool((raw_cosine.abs() < 0.75).all()))

    def test_wrong_temporal_order_is_rejected(self) -> None:
        target = self._motion(batch=2)
        increments = target[:, 1:] - target[:, :-1]
        wrong_order = torch.tensor(
            [0, 11, 2, 17, 4, 15, 6, 13, 8, 19, 10, 1, 12, 3, 14, 5, 16, 7, 18, 9]
        )
        teacher = self._from_increments(increments[:, wrong_order])
        diagnostics = cmkd.evaluate_teacher_target_eligibility(teacher, target)
        self.assertTrue(bool((~diagnostics.eligible).all()))
        self.assertTrue(bool((~diagnostics.kernel_alignment_pass).all()))
        self.assertTrue(
            bool(
                (
                    diagnostics.centered_kernel_alignment
                    < cmkd.CrossModeMotionKernelConfig().min_centered_kernel_alignment
                ).all()
            )
        )

    def test_zero_motion_is_finite_but_ineligible_and_loss_rejects(self) -> None:
        zero = torch.zeros(2, 21, 4, 3, dtype=torch.float32)
        diagnostics = cmkd.evaluate_teacher_target_eligibility(zero, zero)
        self.assertTrue(bool(torch.isfinite(diagnostics.centered_kernel_alignment).all()))
        self.assertTrue(bool(torch.isfinite(diagnostics.envelope_relative_error).all()))
        self.assertTrue(bool((~diagnostics.eligible).all()))
        with self.assertRaisesRegex(cmkd.CrossModeMotionKernelError, "rejected"):
            cmkd.cmkd_student_loss(zero, zero, zero)

    def test_student_loss_is_finite_differentiable_and_fully_audited(self) -> None:
        target = self._motion(batch=2)
        teacher = self._separable_transform(
            target, spatial_seed=307, channel_seed=311
        )
        generator = torch.Generator().manual_seed(401)
        perturbation = 0.04 * torch.randn(
            target.shape, generator=generator, dtype=torch.float32
        )
        perturbation[:, 0] = 0.0
        raw_student = (target + perturbation).detach().requires_grad_(True)
        # Re-projecting makes the optimized input obey the exact Q0 boundary.
        student = raw_student - raw_student[:, :1]
        result = cmkd.cmkd_student_loss(student, target, teacher)
        self.assertEqual(result.loss.ndim, 0)
        self.assertTrue(bool(torch.isfinite(result.loss)))
        diagnostics = result.diagnostics
        self.assertEqual(diagnostics.target_direction_loss.shape, (2,))
        self.assertEqual(diagnostics.teacher_kernel_loss.shape, (2,))
        self.assertEqual(diagnostics.amplitude_envelope_loss.shape, (2,))
        self.assertEqual(diagnostics.temporal_jitter_loss.shape, (2,))
        self.assertEqual(diagnostics.weighted_total_loss.shape, (2,))
        self.assertEqual(int(diagnostics.eligible_sample_count), 2)
        self.assertTrue(bool(diagnostics.eligibility.eligible.all()))
        result.loss.backward()
        self.assertIsNotNone(raw_student.grad)
        self.assertTrue(bool(torch.isfinite(raw_student.grad).all()))
        self.assertGreater(float(raw_student.grad.abs().sum()), 0.0)

    def test_teacher_coordinate_change_does_not_change_student_loss(self) -> None:
        target = self._motion(batch=2)
        teacher_a = self._separable_transform(
            target, spatial_seed=503, channel_seed=509
        )
        teacher_b = self._separable_transform(
            target, spatial_seed=601, channel_seed=607
        )
        student = target.clone()
        student[:, 8:13] = 0.94 * student[:, 8:13]
        result_a = cmkd.cmkd_student_loss(student, target, teacher_a)
        result_b = cmkd.cmkd_student_loss(student, target, teacher_b)
        self.assertTrue(
            torch.allclose(result_a.loss, result_b.loss, atol=2e-5, rtol=2e-5)
        )
        self.assertTrue(
            torch.allclose(
                result_a.diagnostics.teacher_kernel_loss,
                result_b.diagnostics.teacher_kernel_loss,
                atol=2e-5,
                rtol=2e-5,
            )
        )

    def test_residual_temporal_jitter_penalizes_alternation(self) -> None:
        target = self._motion(batch=1)
        teacher = target.clone()
        smooth = target.clone()
        smooth[:, 1:] += 0.02
        alternating = target.clone()
        signs = torch.where(
            torch.arange(20) % 2 == 0,
            torch.tensor(1.0),
            torch.tensor(-1.0),
        ).reshape(1, 20, 1, 1)
        alternating[:, 1:] += 0.02 * signs
        smooth_result = cmkd.cmkd_student_loss(smooth, target, teacher)
        alternating_result = cmkd.cmkd_student_loss(alternating, target, teacher)
        self.assertGreater(
            float(alternating_result.diagnostics.temporal_jitter_loss),
            float(smooth_result.diagnostics.temporal_jitter_loss),
        )

    def test_shape_dtype_phase_zero_and_finiteness_fail_closed(self) -> None:
        valid = self._motion(batch=1, spatial=4, channels=3)
        invalid = (
            torch.zeros(1, 20, 4, 3, dtype=torch.float32),
            torch.zeros(1, 21, 12, dtype=torch.float32),
            valid.double(),
            valid.to(torch.bfloat16),
        )
        for value in invalid:
            with self.subTest(shape=tuple(value.shape), dtype=value.dtype):
                with self.assertRaises(cmkd.CrossModeMotionKernelError):
                    cmkd.motion_kernel_statistics(value)

        nonzero_phase_zero = valid.clone()
        nonzero_phase_zero[:, 0, 0, 0] = 1.0
        with self.assertRaisesRegex(cmkd.CrossModeMotionKernelError, "phase zero"):
            cmkd.motion_kernel_statistics(nonzero_phase_zero)
        nonfinite = valid.clone()
        nonfinite[:, 9, 2, 1] = float("nan")
        with self.assertRaisesRegex(cmkd.CrossModeMotionKernelError, "non-finite"):
            cmkd.motion_kernel_statistics(nonfinite)
        with self.assertRaisesRegex(cmkd.CrossModeMotionKernelError, "shapes differ"):
            cmkd.evaluate_teacher_target_eligibility(
                valid, self._motion(batch=2, spatial=4, channels=3)
            )


if __name__ == "__main__":
    unittest.main()
