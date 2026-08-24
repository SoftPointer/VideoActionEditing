from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import self_imagined_relational_motion as relational  # noqa: E402


def _asymmetric_teacher(*, batch: int = 1, sketches: int = 3, width: int = 7):
    generator = torch.Generator(device="cpu").manual_seed(20260809017)
    increments = torch.randn(
        batch, 21, sketches, width, generator=generator, dtype=torch.float32
    )
    phase_scale = torch.linspace(0.2, 1.7, 21, dtype=torch.float32).view(
        1, 21, 1, 1
    )
    return (torch.cumsum(increments * phase_scale, dim=1) / 5.0).contiguous()


def _affine_basis(height: int, width: int) -> torch.Tensor:
    x = torch.linspace(-1.0, 1.0, width, dtype=torch.float64)
    y = torch.linspace(-1.0, 1.0, height, dtype=torch.float64)
    x_grid = x.unsqueeze(0).expand(height, width)
    y_grid = y.unsqueeze(1).expand(height, width)
    raw = torch.stack([torch.ones_like(x_grid), x_grid, y_grid]).reshape(
        3, height * width
    )
    return raw / torch.linalg.vector_norm(raw, dim=1, keepdim=True)


class FrozenRelationalMotionScorerTests(unittest.TestCase):
    def test_reverse_and_fixed_phase_shuffle_strictly_lower_score(self) -> None:
        teacher = _asymmetric_teacher()
        scorer = relational.FrozenRelationalMotionScorer(teacher)
        same = scorer.forward_sketched_residual(
            teacher.clone(), require_input_grad=False
        ).score
        reverse = scorer.forward_sketched_residual(
            teacher.flip(1), require_input_grad=False
        ).score
        permutation = torch.tensor(
            [(8 * index) % 21 for index in range(21)], dtype=torch.int64
        )
        shuffled = scorer.forward_sketched_residual(
            teacher.index_select(1, permutation), require_input_grad=False
        ).score
        self.assertAlmostEqual(float(same.item()), 0.0, places=6)
        self.assertLess(float(reverse.item()), float(same.item()) - 0.05)
        self.assertLess(float(shuffled.item()), float(same.item()) - 0.05)

    def test_global_channel_and_sketch_orthogonal_bases_leave_gram_invariant(self) -> None:
        teacher = _asymmetric_teacher(batch=2, sketches=4, width=8)
        q_channel, _ = torch.linalg.qr(
            torch.randn(8, 8, generator=torch.Generator().manual_seed(19))
        )
        q_sketch, _ = torch.linalg.qr(
            torch.randn(4, 4, generator=torch.Generator().manual_seed(23))
        )
        transformed = torch.einsum(
            "ij,btjd,de->btie", q_sketch, teacher, q_channel
        ).contiguous()
        teacher_gram = relational.normalized_temporal_gram(teacher)
        transformed_gram = relational.normalized_temporal_gram(transformed)
        self.assertTrue(
            torch.allclose(
                teacher_gram, transformed_gram, rtol=2.0e-6, atol=2.0e-6
            )
        )
        # Only the relational channel is basis-free.  The full scorer also
        # owns a signed Bernini-basis channel so R and -R cannot alias.

    def test_temporal_dc_offset_does_not_change_relational_score(self) -> None:
        teacher = _asymmetric_teacher(batch=2)
        offset = torch.randn(2, 1, 3, 7, dtype=torch.float32)
        scorer = relational.FrozenRelationalMotionScorer(teacher)
        result = scorer.forward_sketched_residual(
            teacher + offset, require_input_grad=False
        )
        self.assertLessEqual(float(result.normalized_frobenius_mismatch.max()), 2e-6)

    def test_teacher_is_stopped_current_is_differentiable_and_no_parameters_exist(self) -> None:
        teacher = _asymmetric_teacher().requires_grad_(True)
        scorer = relational.FrozenRelationalMotionScorer(teacher)
        self.assertEqual(list(scorer.parameters()), [])
        self.assertFalse(scorer.teacher_temporal_gram.requires_grad)
        leaf = torch.randn(
            teacher.shape,
            generator=torch.Generator().manual_seed(31),
            dtype=torch.float32,
            requires_grad=True,
        )
        current = teacher.detach() + 0.03 * leaf
        result = scorer.forward_sketched_residual(current)
        self.assertEqual(result.score.ndim, 0)
        result.score.backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNotNone(leaf.grad)
        assert leaf.grad is not None
        self.assertTrue(bool(torch.isfinite(leaf.grad).all().item()))
        self.assertGreater(float(torch.linalg.vector_norm(leaf.grad)), 0.0)

    def test_live_vjp_contract_rejects_detached_current_and_nonfinite_input(self) -> None:
        teacher = _asymmetric_teacher()
        scorer = relational.FrozenRelationalMotionScorer(teacher)
        with self.assertRaisesRegex(relational.RelationalMotionError, "require gradients"):
            scorer.forward_sketched_residual(teacher.clone())
        detached_leaf = teacher.clone().requires_grad_(True)
        with self.assertRaisesRegex(relational.RelationalMotionError, "live Bernini graph"):
            scorer.forward_sketched_residual(detached_leaf)
        invalid = teacher.clone()
        invalid[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(relational.RelationalMotionError, "NaN or infinity"):
            scorer.forward_sketched_residual(
                invalid.requires_grad_(True)
            )

    def test_degenerate_temporal_residual_fails_closed(self) -> None:
        constant = torch.ones(1, 21, 2, 4, dtype=torch.float32)
        with self.assertRaisesRegex(relational.RelationalMotionError, "temporal energy"):
            relational.FrozenRelationalMotionScorer(constant)

        dynamic = _asymmetric_teacher(sketches=1)
        zero_signed = torch.cat((dynamic, -dynamic), dim=2)
        with self.assertRaisesRegex(relational.RelationalMotionError, "signed temporal"):
            relational.FrozenRelationalMotionScorer(zero_signed)

    def test_signed_channel_rejects_action_noop_global_sign_swap(self) -> None:
        teacher = _asymmetric_teacher()
        scorer = relational.FrozenRelationalMotionScorer(teacher)
        leaf = (-teacher).clone().requires_grad_(True)
        current = leaf + torch.zeros_like(leaf)
        result = scorer.forward_sketched_residual(current)
        self.assertAlmostEqual(float(result.score.item()), -0.5, places=5)
        self.assertAlmostEqual(float(result.signed_feature_loss.item()), 1.0, places=5)
        self.assertAlmostEqual(
            float(result.meaningful_mismatch.item()), math.sqrt(0.5), places=5
        )
        result.score.backward()
        self.assertIsNotNone(leaf.grad)
        assert leaf.grad is not None
        self.assertGreater(float(torch.linalg.vector_norm(leaf.grad)), 1.0e-6)
        ascent = torch.dot(leaf.grad.reshape(-1), (teacher - current).reshape(-1))
        self.assertGreater(float(ascent), 0.0)

    def test_squared_objective_fails_closed_below_meaningful_mismatch(self) -> None:
        teacher = _asymmetric_teacher()
        scorer = relational.FrozenRelationalMotionScorer(teacher)
        perturbation = torch.randn(
            teacher.shape,
            generator=torch.Generator().manual_seed(37),
            dtype=torch.float32,
        )
        leaf = (teacher + 1.0e-7 * perturbation).clone().requires_grad_(True)
        with self.assertRaisesRegex(
            relational.RelationalMotionError, "meaningful-mismatch floor"
        ):
            scorer.forward_sketched_residual(leaf + torch.zeros_like(leaf))

    def test_squared_objective_gradient_decays_toward_teacher(self) -> None:
        teacher = _asymmetric_teacher()
        scorer = relational.FrozenRelationalMotionScorer(teacher)
        perturbation = torch.randn(
            teacher.shape,
            generator=torch.Generator().manual_seed(43),
            dtype=torch.float32,
        )
        norms = []
        mismatches = []
        for epsilon in (0.1, 0.03, 0.01):
            leaf = (teacher + epsilon * perturbation).clone().requires_grad_(True)
            result = scorer.forward_sketched_residual(
                leaf + torch.zeros_like(leaf)
            )
            result.score.backward()
            assert leaf.grad is not None
            norms.append(float(torch.linalg.vector_norm(leaf.grad)))
            mismatches.append(float(result.objective_mismatch.item()))
        self.assertGreater(norms[0], norms[1])
        self.assertGreater(norms[1], norms[2])
        self.assertGreater(mismatches[0], mismatches[1])
        self.assertGreater(mismatches[1], mismatches[2])


class SourceSafeCotangentTests(unittest.TestCase):
    def test_projection_enforces_phase_temporal_affine_and_rms_constraints(self) -> None:
        generator = torch.Generator().manual_seed(41)
        raw = torch.randn(1, 16, 21, 6, 7, generator=generator)
        projected = relational.project_source_safe_cotangent(raw)
        self.assertEqual(tuple(projected.shape), tuple(raw.shape))
        self.assertEqual(projected.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(projected).all().item()))
        self.assertGreater(float(torch.linalg.vector_norm(projected)), 0.0)
        self.assertTrue(
            torch.equal(projected[:, :, 0], torch.zeros_like(projected[:, :, 0]))
        )
        temporal_sum = projected.to(torch.float64).sum(dim=2)
        self.assertLess(float(temporal_sum.abs().max()), 2.0e-6)

        flat = projected.to(torch.float64).reshape(-1, 6 * 7)
        affine_coefficients = torch.matmul(flat, _affine_basis(6, 7).T)
        self.assertLess(float(affine_coefficients.abs().max()), 2.0e-6)
        survival = (
            projected.to(torch.float64).square().mean().sqrt()
            / raw.to(torch.float64).square().mean().sqrt()
        )
        self.assertGreaterEqual(
            float(survival), relational.MIN_PROJECTION_SURVIVAL_RATIO
        )

    def test_projection_keeps_autograd_and_rejects_fully_removed_direction(self) -> None:
        raw = torch.randn(1, 16, 21, 4, 5, dtype=torch.float32).requires_grad_(True)
        projected = relational.project_source_safe_cotangent(raw)
        projected.square().sum().backward()
        self.assertIsNotNone(raw.grad)
        assert raw.grad is not None
        self.assertTrue(bool(torch.isfinite(raw.grad).all().item()))

        affine_only = torch.ones(1, 16, 21, 4, 5, dtype=torch.float32)
        with self.assertRaisesRegex(relational.RelationalMotionError, "entire cotangent"):
            relational.project_source_safe_cotangent(affine_only)

        nearly_removed = torch.ones(1, 16, 21, 4, 5, dtype=torch.float32)
        nearly_removed += 1.0e-6 * torch.randn_like(nearly_removed)
        with self.assertRaisesRegex(relational.RelationalMotionError, "did not survive"):
            relational.project_source_safe_cotangent(nearly_removed)

    def test_symmetric_interventions_have_fixed_dose_and_no_arm_selection(self) -> None:
        generator = torch.Generator().manual_seed(53)
        clean = torch.randn(1, 16, 21, 5, 6, generator=generator)
        q = torch.randn(1, 16, 21, 5, 6, generator=generator)
        result = relational.symmetric_latent_interventions(
            clean, q, dose_rms=0.03
        )
        self.assertTrue(
            torch.allclose(result.plus - clean, result.delta, rtol=0.0, atol=2.0e-7)
        )
        self.assertTrue(
            torch.allclose(clean - result.minus, result.delta, rtol=0.0, atol=2.0e-7)
        )
        midpoint = (result.plus.to(torch.float64) + result.minus.to(torch.float64)) / 2
        self.assertTrue(
            torch.allclose(midpoint, clean.to(torch.float64), rtol=0.0, atol=1.0e-7)
        )
        dose = result.delta.to(torch.float64).square().mean().sqrt()
        self.assertTrue(math.isclose(float(dose), 0.03, rel_tol=1e-6, abs_tol=1e-8))
        self.assertTrue(
            torch.equal(
                result.delta[:, :, 0], torch.zeros_like(result.delta[:, :, 0])
            )
        )
        self.assertLess(
            float(result.delta.to(torch.float64).sum(dim=2).abs().max()), 3.0e-6
        )

    def test_shape_dtype_and_scalar_contracts_fail_closed(self) -> None:
        good = torch.randn(1, 16, 21, 4, 4, dtype=torch.float32)
        with self.assertRaisesRegex(relational.RelationalMotionError, "FP32"):
            relational.project_source_safe_cotangent(good.to(torch.float64))
        with self.assertRaisesRegex(relational.RelationalMotionError, "positive scalar"):
            relational.project_source_safe_cotangent(
                good, minimum_survival_ratio=0.0
            )
        with self.assertRaisesRegex(relational.RelationalMotionError, "identical shape"):
            relational.symmetric_latent_interventions(
                good, torch.randn(1, 16, 21, 4, 5), dose_rms=0.01
            )


if __name__ == "__main__":
    unittest.main()
