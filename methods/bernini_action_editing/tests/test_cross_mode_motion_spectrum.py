from __future__ import annotations

import ast
from dataclasses import fields
import inspect
import math
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import cross_mode_motion_spectrum as cmsg


class CrossModeMotionSpectrumPureContractTests(unittest.TestCase):
    def test_fixed_method_and_default_contract(self) -> None:
        self.assertEqual(
            cmsg.METHOD_NAME,
            "cross-mode-motion-spectrum-guidance-v6",
        )
        self.assertEqual(cmsg.EXPECTED_PHASES, 21)
        self.assertEqual(cmsg.NUM_DENOISING_STEPS, 40)
        config = cmsg.CrossModeMotionSpectrumConfig()
        config.validate()
        self.assertEqual(config.generator_lambda, 0.5)
        self.assertEqual(config.alignment_threshold, 0.1)
        self.assertEqual(config.max_plan_delta_ratio, 0.5)

    def test_release_schedule_is_exact_at_every_boundary(self) -> None:
        rho = cmsg.release_rho_schedule()
        self.assertEqual(len(rho), 40)
        self.assertEqual(rho[:20], (1.0,) * 20)
        self.assertEqual(rho[20], 1.0)
        self.assertEqual(rho[31], 0.0)
        self.assertEqual(rho[32:], (0.0,) * 8)
        self.assertTrue(
            all(rho[index] >= rho[index + 1] for index in range(20, 31))
        )
        self.assertGreater(rho[21], 0.0)
        self.assertLess(rho[30], 1.0)
        for step in range(20, 32):
            with self.subTest(taper_step=step):
                expected = 0.5 * (
                    1.0 + math.cos(math.pi * float(step - 20) / 11.0)
                )
                self.assertAlmostEqual(rho[step], expected)
        for invalid in (-1, 40, True, 1.5, "2"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(cmsg.CrossModeMotionSpectrumError):
                    cmsg.release_rho(invalid)

    def test_configuration_rejects_invalid_bounds(self) -> None:
        invalid = (
            {"generator_lambda": -0.01},
            {"generator_lambda": 1.01},
            {"generator_lambda": True},
            {"alignment_threshold": -1.01},
            {"alignment_threshold": 1.01},
            {"max_plan_delta_ratio": -0.01},
            {"max_plan_delta_ratio": float("inf")},
            {"epsilon": 0.0},
            {"epsilon": float("nan")},
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(cmsg.CrossModeMotionSpectrumError):
                    cmsg.CrossModeMotionSpectrumConfig(**values).validate()

    def test_torch_is_lazily_imported(self) -> None:
        tree = ast.parse(Path(cmsg.__file__).read_text(encoding="utf-8"))
        eager_torch_imports = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                eager_torch_imports.extend(
                    alias.name for alias in node.names if alias.name == "torch"
                )
            elif isinstance(node, ast.ImportFrom) and node.module == "torch":
                eager_torch_imports.append(node.module)
        self.assertEqual(eager_torch_imports, [])
        self.assertEqual(cmsg.release_rho(0), 1.0)
        self.assertEqual(cmsg.release_rho(39), 0.0)

    def test_api_requires_four_clean_fields_and_exposes_spatial_geometry(self) -> None:
        self.assertEqual(
            list(inspect.signature(cmsg.build_cmsg_plan).parameters),
            [
                "editor_action_field",
                "editor_noop_field",
                "generator_action_field",
                "generator_uncond_field",
                "spatial_hw",
                "config",
            ],
        )


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is unavailable")
class CrossModeMotionSpectrumTensorTests(unittest.TestCase):
    @staticmethod
    def _direction(
        *,
        spatial_tokens: int = 4,
        channels: int = 3,
        scale: float = 1.0,
    ):
        phase = torch.arange(21, dtype=torch.float32).reshape(1, 21, 1, 1)
        pattern = torch.arange(
            1,
            spatial_tokens * channels + 1,
            dtype=torch.float32,
        ).reshape(1, 1, spatial_tokens, channels)
        return scale * phase * pattern / float(spatial_tokens * channels)

    def _fields(self, *, generator_scale: float = 2.0):
        direction = self._direction()
        editor_noop = torch.full_like(direction, 8.0)
        editor_action = editor_noop + direction
        generator_uncond = torch.full_like(direction, -4.0)
        generator_action = generator_uncond + generator_scale * direction
        return editor_action, editor_noop, generator_action, generator_uncond

    def test_q0_and_causal_accumulation_have_exact_zero_phase(self) -> None:
        fields_ = self._fields()
        result = cmsg.build_cmsg_plan(*fields_, spatial_hw=(2, 2))
        zero = torch.zeros_like(result.plan[:, 0])
        self.assertTrue(torch.equal(result.official_editor[:, 0], zero))
        self.assertTrue(torch.equal(result.plan[:, 0], zero))
        self.assertEqual(result.plan.shape, (1, 21, 4, 3))
        self.assertEqual(result.plan.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(result.plan).all()))

    def test_rectangular_geometry_is_explicit_and_strict(self) -> None:
        direction = self._direction(spatial_tokens=6)
        zero = torch.zeros_like(direction)
        with self.assertRaisesRegex(
            cmsg.CrossModeMotionSpectrumError, "spatial_hw is required"
        ):
            cmsg.build_cmsg_plan(direction, zero, direction, zero)
        result = cmsg.build_cmsg_plan(
            direction,
            zero,
            direction,
            zero,
            spatial_hw=(2, 3),
        )
        self.assertEqual(result.plan.shape, direction.shape)
        invalid_geometry = ((2, 2), (0, 6), (True, 6), [2, 3], (1, 2, 3))
        for geometry in invalid_geometry:
            with self.subTest(geometry=geometry):
                with self.assertRaises(cmsg.CrossModeMotionSpectrumError):
                    cmsg.build_cmsg_plan(
                        direction,
                        zero,
                        direction,
                        zero,
                        spatial_hw=geometry,
                    )

    def test_square_geometry_can_be_recovered_exactly(self) -> None:
        fields_ = self._fields()
        inferred = cmsg.build_cmsg_plan(*fields_)
        explicit = cmsg.build_cmsg_plan(*fields_, spatial_hw=(2, 2))
        self.assertTrue(torch.equal(inferred.plan, explicit.plan))
        self.assertTrue(
            torch.equal(
                inferred.diagnostics.increment_gain,
                explicit.diagnostics.increment_gain,
            )
        )

    def test_shape_dtype_device_and_finite_contracts_fail_closed(self) -> None:
        fields_ = list(self._fields())
        invalid_sets = []
        phases_20 = torch.zeros(1, 20, 4, 3, dtype=torch.float32)
        invalid_sets.append((phases_20,) * 4)
        rank_3 = torch.zeros(1, 21, 12, dtype=torch.float32)
        invalid_sets.append((rank_3,) * 4)
        wrong_shape = list(fields_)
        wrong_shape[3] = torch.zeros(2, 21, 4, 3, dtype=torch.float32)
        invalid_sets.append(tuple(wrong_shape))
        float64 = tuple(field.double() for field in fields_)
        invalid_sets.append(float64)
        nonfinite = [field.clone() for field in fields_]
        nonfinite[2][0, 7, 1, 1] = float("nan")
        invalid_sets.append(tuple(nonfinite))
        for invalid in invalid_sets:
            with self.subTest(shapes=[tuple(value.shape) for value in invalid]):
                with self.assertRaises(cmsg.CrossModeMotionSpectrumError):
                    cmsg.build_cmsg_plan(*invalid, spatial_hw=(2, 2))

    def test_generator_energy_is_clipped_to_editor_interval(self) -> None:
        fields_ = self._fields(generator_scale=4.0)
        result = cmsg.build_cmsg_plan(*fields_, spatial_hw=(2, 2))
        diagnostics = result.diagnostics
        active = slice(1, None)
        for editor_energy, clipped_energy in (
            (
                diagnostics.editor_spatial_increment_energy,
                diagnostics.clipped_generator_spatial_increment_energy,
            ),
            (
                diagnostics.editor_channel_increment_energy,
                diagnostics.clipped_generator_channel_increment_energy,
            ),
        ):
            self.assertTrue(
                bool((clipped_energy[:, active] >= editor_energy[:, active]).all())
            )
            self.assertTrue(
                bool(
                    (
                        clipped_energy[:, active]
                        <= 3.0 * editor_energy[:, active]
                    ).all()
                )
            )
            self.assertTrue(
                torch.allclose(
                    clipped_energy[:, active],
                    3.0 * editor_energy[:, active],
                    atol=1e-6,
                    rtol=1e-6,
                )
            )
        upper_gain = 1.0 + 0.5 * (math.sqrt(3.0) - 1.0)
        self.assertTrue(bool(diagnostics.alignment_gate[:, 1:].all()))
        self.assertLessEqual(
            float(diagnostics.increment_gain[:, 1:].max()), upper_gain + 1e-5
        )

    def test_alignment_gate_rejects_opposite_generator_motion(self) -> None:
        editor_action, editor_noop, _, generator_uncond = self._fields()
        direction = editor_action - editor_noop
        generator_action = generator_uncond - 2.0 * direction
        result = cmsg.build_cmsg_plan(
            editor_action,
            editor_noop,
            generator_action,
            generator_uncond,
            spatial_hw=(2, 2),
        )
        self.assertTrue(bool((result.diagnostics.mean_alignment[:, 1:] < 0.1).all()))
        self.assertFalse(bool(result.diagnostics.alignment_gate[:, 1:].any()))
        self.assertTrue(
            torch.equal(
                result.diagnostics.increment_gain,
                torch.ones_like(result.diagnostics.increment_gain),
            )
        )
        self.assertTrue(torch.equal(result.plan, result.official_editor))

    def test_both_spectra_must_align_not_merely_their_mean(self) -> None:
        # These pre-filter matrices invert the replicate-padded 2x2 low-pass.
        # After the 3x3 LP the editor spatial rows are [M,1,1,1] and the
        # generator rows are 2*[M,-1,-1,-1], independently in every channel.
        # Thus spatial cosine is nearly +1, while per-row channel cosine is
        # (1-1-1-1)/4 = -0.5.  Their mean exceeds 0.1, so the former mean-gate
        # incorrectly admitted this explicitly channel-anti-aligned oracle.
        magnitude = 100.0
        editor_prefilter = torch.tensor(
            [
                4.0 * magnitude - 3.0,
                -2.0 * magnitude + 3.0,
                -2.0 * magnitude + 3.0,
                magnitude,
            ],
            dtype=torch.float32,
        ).reshape(1, 1, 4, 1)
        generator_prefilter = torch.tensor(
            [
                4.0 * magnitude + 3.0,
                -2.0 * magnitude - 3.0,
                -2.0 * magnitude - 3.0,
                magnitude,
            ],
            dtype=torch.float32,
        ).reshape(1, 1, 4, 1)
        channel_scale = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float32
        ).reshape(1, 1, 1, 3)
        phase = torch.arange(21, dtype=torch.float32).reshape(1, 21, 1, 1)
        editor_direction = phase * editor_prefilter * channel_scale
        generator_direction = 2.0 * phase * generator_prefilter * channel_scale
        zero = torch.zeros_like(editor_direction)
        result = cmsg.build_cmsg_plan(
            editor_direction,
            zero,
            generator_direction,
            zero,
            spatial_hw=(2, 2),
        )
        diagnostics = result.diagnostics
        self.assertTrue(
            bool((diagnostics.spatial_increment_cosine[:, 1:] > 0.99).all())
        )
        self.assertTrue(
            bool((diagnostics.channel_increment_cosine[:, 1:] < -0.49).all())
        )
        self.assertTrue(bool((diagnostics.mean_alignment[:, 1:] > 0.1).all()))
        self.assertFalse(bool(diagnostics.alignment_gate[:, 1:].any()))
        self.assertTrue(torch.equal(result.plan, result.official_editor))

    def test_zero_marginal_checkerboard_motion_keeps_nonzero_energy(self) -> None:
        # Every spatial row and every channel column sums to zero.  Signed
        # marginalization before squaring therefore reported exactly zero for
        # this strong motion even though its low-pass response is nonzero.
        checkerboard = torch.tensor(
            [[1.0, -1.0], [-1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]],
            dtype=torch.float32,
        ).reshape(1, 1, 4, 2)
        self.assertTrue(
            torch.equal(
                checkerboard.sum(dim=2),
                torch.zeros_like(checkerboard.sum(dim=2)),
            )
        )
        self.assertTrue(
            torch.equal(
                checkerboard.sum(dim=3),
                torch.zeros_like(checkerboard.sum(dim=3)),
            )
        )
        phase = torch.arange(21, dtype=torch.float32).reshape(1, 21, 1, 1)
        editor_direction = phase * checkerboard
        generator_direction = 2.0 * editor_direction
        zero = torch.zeros_like(editor_direction)
        result = cmsg.build_cmsg_plan(
            editor_direction,
            zero,
            generator_direction,
            zero,
            spatial_hw=(2, 2),
        )
        diagnostics = result.diagnostics
        self.assertTrue(
            bool(
                (
                    diagnostics.editor_spatial_increment_energy[:, 1:] > 0.0
                ).all()
            )
        )
        self.assertTrue(
            bool(
                (
                    diagnostics.editor_channel_increment_energy[:, 1:] > 0.0
                ).all()
            )
        )
        self.assertTrue(bool(diagnostics.alignment_gate[:, 1:].all()))
        self.assertTrue(
            bool((diagnostics.increment_gain[:, 1:] > 1.0).all())
        )
        self.assertFalse(torch.equal(result.plan, result.official_editor))

    def test_generator_appearance_and_values_cannot_be_directly_injected(self) -> None:
        direction = self._direction()
        editor_noop = torch.full_like(direction, 2.0)
        editor_action = editor_noop + direction
        generator_motion = 2.0 * direction
        generator_uncond_a = torch.zeros_like(direction)
        generator_action_a = generator_motion
        # Integral, exactly representable common appearance cancels from the
        # generator action/unconditional difference.
        common_appearance = torch.full_like(direction, 16.0)
        generator_uncond_b = common_appearance
        generator_action_b = common_appearance + generator_motion
        first = cmsg.build_cmsg_plan(
            editor_action,
            editor_noop,
            generator_action_a,
            generator_uncond_a,
            spatial_hw=(2, 2),
        )
        second = cmsg.build_cmsg_plan(
            editor_action,
            editor_noop,
            generator_action_b,
            generator_uncond_b,
            spatial_hw=(2, 2),
        )
        self.assertTrue(torch.equal(first.plan, second.plan))
        self.assertTrue(
            torch.equal(
                first.diagnostics.increment_gain,
                second.diagnostics.increment_gain,
            )
        )

        editor_increment = torch.cat(
            (
                torch.zeros_like(first.official_editor[:, :1]),
                first.official_editor[:, 1:] - first.official_editor[:, :-1],
            ),
            dim=1,
        )
        plan_increment = torch.cat(
            (
                torch.zeros_like(first.plan[:, :1]),
                first.plan[:, 1:] - first.plan[:, :-1],
            ),
            dim=1,
        )
        expected = editor_increment * first.diagnostics.increment_gain[
            ..., None, None
        ]
        self.assertTrue(torch.allclose(plan_increment, expected, atol=2e-5))
        # A generator has no route to populate a value where the editor
        # increment is exactly zero.
        self.assertTrue(
            torch.equal(
                plan_increment[editor_increment == 0.0],
                torch.zeros_like(plan_increment[editor_increment == 0.0]),
            )
        )

    def test_lambda_zero_returns_the_exact_official_tensor_object(self) -> None:
        result = cmsg.build_cmsg_plan(
            *self._fields(),
            spatial_hw=(2, 2),
            config=cmsg.CrossModeMotionSpectrumConfig(generator_lambda=0.0),
        )
        self.assertIs(result.plan, result.official_editor)

    def test_diagnostics_are_complete_finite_and_auditable(self) -> None:
        result = cmsg.build_cmsg_plan(*self._fields(), spatial_hw=(2, 2))
        self.assertEqual(len(fields(cmsg.MotionSpectrumDiagnostics)), 17)
        profile_shapes = {
            "editor_spatial_energy_profile": (1, 21, 4),
            "generator_spatial_energy_profile": (1, 21, 4),
            "editor_channel_energy_profile": (1, 21, 3),
            "generator_channel_energy_profile": (1, 21, 3),
        }
        for descriptor in fields(cmsg.MotionSpectrumDiagnostics):
            value = getattr(result.diagnostics, descriptor.name)
            self.assertEqual(
                value.shape,
                profile_shapes.get(descriptor.name, (1, 21)),
                descriptor.name,
            )
            if value.dtype == torch.bool:
                continue
            self.assertEqual(value.dtype, torch.float32, descriptor.name)
            self.assertTrue(bool(torch.isfinite(value).all()), descriptor.name)

    def test_plan_and_execution_gradients_are_finite(self) -> None:
        direction = self._direction()
        editor_action = direction.clone().requires_grad_(True)
        editor_noop = torch.zeros_like(direction, requires_grad=True)
        generator_action = (1.2 * direction).clone().requires_grad_(True)
        generator_uncond = torch.zeros_like(direction, requires_grad=True)
        result = cmsg.build_cmsg_plan(
            editor_action,
            editor_noop,
            generator_action,
            generator_uncond,
            spatial_hw=(2, 2),
        )
        executed = cmsg.execute_cmsg_plan(
            result.official_editor,
            result.plan,
            step_index=0,
        )
        executed.executed_field.square().mean().backward()
        for tensor in (
            editor_action,
            editor_noop,
            generator_action,
            generator_uncond,
        ):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(bool(torch.isfinite(tensor.grad).all()))
        self.assertGreater(float(generator_action.grad.abs().sum()), 0.0)

    def test_execution_bounds_delta_and_uses_exact_formula(self) -> None:
        official = cmsg.q0(self._direction())
        plan = official + 10.0 * official
        config = cmsg.CrossModeMotionSpectrumConfig(max_plan_delta_ratio=0.25)
        result = cmsg.execute_cmsg_plan(
            official,
            plan,
            step_index=0,
            config=config,
        )
        self.assertEqual(result.rho, 1.0)
        self.assertTrue(
            bool(
                (
                    result.bound.raw_plan_delta_rms
                    >= result.bound.delta_rms_cap
                )[:, 1:].all()
            )
        )
        bounded_rms = torch.linalg.vector_norm(
            result.bound.bounded_plan_delta, dim=(2, 3)
        ) / math.sqrt(12)
        self.assertTrue(
            bool(
                (
                    bounded_rms[:, 1:]
                    <= result.bound.delta_rms_cap[:, 1:] + 1e-6
                ).all()
            )
        )
        self.assertTrue(
            torch.equal(
                result.executed_field,
                official + result.bound.bounded_plan_delta,
            )
        )

    def test_zero_release_returns_exact_official_object(self) -> None:
        plan_result = cmsg.build_cmsg_plan(*self._fields(), spatial_hw=(2, 2))
        for step in (31, 32, 35, 39):
            with self.subTest(step=step):
                result = cmsg.execute_cmsg_plan(
                    plan_result.official_editor,
                    plan_result.plan,
                    step_index=step,
                )
                self.assertEqual(result.rho, 0.0)
                self.assertIs(result.executed_field, plan_result.official_editor)

    def test_zero_release_has_no_generator_gradient_path(self) -> None:
        direction = self._direction()
        editor_action = direction.clone().requires_grad_(True)
        editor_noop = torch.zeros_like(direction, requires_grad=True)
        generator_action = (1.5 * direction).clone().requires_grad_(True)
        generator_uncond = torch.zeros_like(direction, requires_grad=True)
        plan_result = cmsg.build_cmsg_plan(
            editor_action,
            editor_noop,
            generator_action,
            generator_uncond,
            spatial_hw=(2, 2),
        )
        execution = cmsg.execute_cmsg_plan(
            plan_result.official_editor,
            plan_result.plan,
            step_index=39,
        )
        self.assertEqual(execution.rho, 0.0)
        self.assertIs(execution.executed_field, plan_result.official_editor)
        execution.executed_field.sum().backward()
        self.assertIsNotNone(editor_action.grad)
        self.assertIsNotNone(editor_noop.grad)
        self.assertIsNone(generator_action.grad)
        self.assertIsNone(generator_uncond.grad)

    def test_execution_rejects_noncausal_boundaries(self) -> None:
        official = cmsg.q0(self._direction())
        noncausal_official = official.clone()
        noncausal_official[:, 0] = 1.0
        with self.assertRaisesRegex(
            cmsg.CrossModeMotionSpectrumError, "official_editor"
        ):
            cmsg.execute_cmsg_plan(noncausal_official, official, step_index=0)
        noncausal_plan = official.clone()
        noncausal_plan[:, 0] = 1.0
        with self.assertRaisesRegex(cmsg.CrossModeMotionSpectrumError, "plan"):
            cmsg.execute_cmsg_plan(official, noncausal_plan, step_index=0)


if __name__ == "__main__":
    unittest.main()
