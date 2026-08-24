from __future__ import annotations

import ast
import inspect
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import cross_mode_motion_spectrum as v6_spectrum
import motion_commutator as commutator


class MotionCommutatorPureContractTests(unittest.TestCase):
    def test_method_defaults_and_schedule_are_pinned(self) -> None:
        self.assertEqual(
            commutator.METHOD_NAME,
            "counterfactual-motion-commutator-v7",
        )
        self.assertEqual(commutator.EXPECTED_PHASES, 21)
        self.assertEqual(commutator.NUM_DENOISING_STEPS, 40)
        config = commutator.MotionCommutatorConfig()
        config.validate()
        self.assertEqual(config.max_correction_increment_ratio, 0.25)
        self.assertEqual(config.correction_increment_rms_floor, 1.0e-3)
        self.assertFalse(config.temporal_smoothing)
        self.assertEqual(
            commutator.release_rho_schedule(),
            v6_spectrum.release_rho_schedule(),
        )
        self.assertEqual(commutator.release_rho(0), 1.0)
        self.assertEqual(commutator.release_rho(31), 0.0)
        self.assertEqual(commutator.release_rho(39), 0.0)

    def test_invalid_configuration_and_schedule_fail_closed(self) -> None:
        invalid_configs = (
            {"max_correction_increment_ratio": -0.01},
            {"max_correction_increment_ratio": float("inf")},
            {"max_correction_increment_ratio": True},
            {"correction_increment_rms_floor": -0.01},
            {"correction_increment_rms_floor": float("nan")},
            {"temporal_smoothing": 1},
            {"epsilon": 0.0},
            {"epsilon": float("nan")},
        )
        for values in invalid_configs:
            with self.subTest(values=values):
                with self.assertRaises(commutator.MotionCommutatorError):
                    commutator.MotionCommutatorConfig(**values).validate()
        for step in (-1, 40, True, 1.0, "3"):
            with self.subTest(step=step):
                with self.assertRaises(commutator.MotionCommutatorError):
                    commutator.release_rho(step)

    def test_tensor_library_is_lazy_and_no_spatial_filter_is_present(self) -> None:
        source_path = Path(commutator.__file__)
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        eager_torch = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                eager_torch.extend(
                    alias.name for alias in node.names if alias.name == "torch"
                )
            elif isinstance(node, ast.ImportFrom) and node.module == "torch":
                eager_torch.append(node.module)
        self.assertEqual(eager_torch, [])
        self.assertNotIn("avg_pool2d", source)
        self.assertNotIn("conv2d", source)
        self.assertNotIn("spatial_hw", source)
        self.assertEqual(
            commutator.FIXED_TEMPORAL_SMOOTHING_KERNEL,
            (0.25, 0.5, 0.25),
        )

    def test_inference_apis_expose_no_oracle_condition(self) -> None:
        inference_apis = (
            commutator.build_motion_commutator,
            commutator.build_raw_motion_commutator,
            commutator.bound_motion_commutator_correction,
            commutator.execute_motion_commutator,
            commutator.apply_motion_commutator_to_official_tensor,
        )
        forbidden = (
            "mask",
            "flow",
            "pose",
            "target",
            "generator",
            "anchor",
            "first_frame",
        )
        for function in inference_apis:
            parameters = tuple(inspect.signature(function).parameters)
            with self.subTest(function=function.__name__, parameters=parameters):
                for name in parameters:
                    self.assertFalse(
                        any(token in name.lower() for token in forbidden),
                        msg=f"forbidden inference parameter: {name}",
                    )


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is unavailable")
class MotionCommutatorTensorTests(unittest.TestCase):
    @staticmethod
    def _phase() -> "torch.Tensor":
        return torch.arange(21, dtype=torch.float32).reshape(1, 21, 1, 1)

    @classmethod
    def _base_branches(cls) -> tuple["torch.Tensor", ...]:
        phase = cls._phase()
        spatial = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], dtype=torch.float32
        ).reshape(1, 1, 4, 1)
        channel = torch.tensor(
            [1.0, -1.0, 2.0], dtype=torch.float32
        ).reshape(1, 1, 1, 3)
        frozen_direction = 8.0 * phase * spatial * channel
        frozen_noop = torch.full_like(frozen_direction, 2.0)
        frozen_action = frozen_noop + frozen_direction
        common_drift = 4.0 * phase * torch.ones_like(spatial) * torch.ones_like(
            channel
        )
        adapted_noop = frozen_noop + common_drift
        adapted_action = frozen_action + common_drift
        return adapted_action, adapted_noop, frozen_action, frozen_noop

    def test_common_adapter_appearance_drift_cancels_exactly(self) -> None:
        fields = self._base_branches()
        result = commutator.build_motion_commutator(
            *fields,
            config=commutator.MotionCommutatorConfig(
                correction_increment_rms_floor=0.0
            ),
        )
        zero = torch.zeros_like(result.raw_commutator_correction)
        self.assertTrue(torch.equal(result.raw_commutator_correction, zero))
        self.assertTrue(torch.equal(result.bounded_commutator_correction, zero))
        self.assertTrue(
            torch.equal(result.final_direction, result.frozen_official_direction)
        )

    def test_training_and_deployment_stages_match_composed_api(self) -> None:
        adapted_action, adapted_noop, frozen_action, frozen_noop = (
            self._base_branches()
        )
        adapted_action = adapted_action + 2.0 * self._phase()
        config = commutator.MotionCommutatorConfig(
            max_correction_increment_ratio=0.1,
            correction_increment_rms_floor=0.0,
            temporal_smoothing=True,
        )
        raw = commutator.build_raw_motion_commutator(
            adapted_action,
            adapted_noop,
            frozen_action,
            frozen_noop,
        )
        bounded = commutator.bound_motion_commutator_correction(
            raw.frozen_official_direction,
            raw.raw_commutator_correction,
            config=config,
        )
        composed = commutator.build_motion_commutator(
            adapted_action,
            adapted_noop,
            frozen_action,
            frozen_noop,
            config=config,
        )
        self.assertTrue(
            torch.equal(
                raw.unbounded_final_direction,
                raw.frozen_official_direction + raw.raw_commutator_correction,
            )
        )
        for left, right in (
            (raw.frozen_official_direction, composed.frozen_official_direction),
            (raw.raw_commutator_correction, composed.raw_commutator_correction),
            (
                bounded.candidate_commutator_correction,
                composed.candidate_commutator_correction,
            ),
            (
                bounded.bounded_commutator_correction,
                composed.bounded_commutator_correction,
            ),
            (bounded.final_direction, composed.final_direction),
        ):
            self.assertTrue(torch.equal(left, right))

    def test_action_specific_adapter_residual_is_retained(self) -> None:
        adapted_action, adapted_noop, frozen_action, frozen_noop = (
            self._base_branches()
        )
        residual = 0.5 * self._phase() * torch.ones_like(frozen_action)
        adapted_action = adapted_action + residual
        result = commutator.build_motion_commutator(
            adapted_action,
            adapted_noop,
            frozen_action,
            frozen_noop,
            config=commutator.MotionCommutatorConfig(
                correction_increment_rms_floor=0.0
            ),
        )
        expected = commutator.causal_gauge(residual)
        self.assertTrue(
            torch.equal(result.raw_commutator_correction, expected)
        )
        self.assertTrue(
            torch.equal(result.bounded_commutator_correction, expected)
        )
        self.assertTrue(
            torch.equal(
                result.final_direction,
                result.frozen_official_direction + expected,
            )
        )

    def test_increment_domain_hard_bound_really_caps_every_phase(self) -> None:
        phase = self._phase()
        frozen_direction = phase * torch.ones(1, 1, 4, 3)
        frozen_noop = torch.zeros_like(frozen_direction)
        frozen_action = frozen_direction.clone()
        adapted_noop = frozen_noop.clone()
        adapted_action = frozen_action + 10.0 * phase * torch.ones_like(
            frozen_action
        )
        result = commutator.build_motion_commutator(
            adapted_action,
            adapted_noop,
            frozen_action,
            frozen_noop,
            config=commutator.MotionCommutatorConfig(
                max_correction_increment_ratio=0.2,
                correction_increment_rms_floor=0.0,
            ),
        )
        diagnostics = result.diagnostics
        self.assertTrue(
            bool(
                (
                    diagnostics.bounded_correction_increment_rms
                    <= diagnostics.correction_increment_rms_cap + 1.0e-6
                ).all()
            )
        )
        self.assertTrue(
            torch.allclose(
                diagnostics.correction_increment_rms_cap[:, 1:],
                torch.full_like(
                    diagnostics.correction_increment_rms_cap[:, 1:], 0.2
                ),
                atol=1.0e-6,
                rtol=0.0,
            )
        )
        self.assertTrue(
            torch.allclose(
                diagnostics.bounded_correction_increment_rms[:, 1:],
                diagnostics.correction_increment_rms_cap[:, 1:],
                atol=1.0e-6,
                rtol=0.0,
            )
        )
        self.assertTrue(bool((diagnostics.bound_scale[:, 1:] < 1.0).all()))

    def test_explicit_positive_floor_allows_new_locally_static_motion(self) -> None:
        phase = self._phase()
        zero = torch.zeros(1, 21, 4, 3, dtype=torch.float32)
        adapted_action = phase * torch.ones_like(zero)
        result = commutator.build_motion_commutator(
            adapted_action,
            zero,
            zero,
            zero,
            config=commutator.MotionCommutatorConfig(
                max_correction_increment_ratio=0.25,
                correction_increment_rms_floor=0.05,
            ),
        )
        diagnostics = result.diagnostics
        self.assertTrue(
            torch.allclose(
                diagnostics.correction_increment_rms_cap,
                torch.full_like(diagnostics.correction_increment_rms_cap, 0.05),
            )
        )
        self.assertTrue(
            bool(
                (
                    diagnostics.bounded_correction_increment_rms[:, 1:] > 0.0
                ).all()
            )
        )
        self.assertTrue(
            bool(
                (
                    diagnostics.bounded_correction_increment_rms[:, 1:]
                    <= 0.05 + 1.0e-6
                ).all()
            )
        )

    def test_optional_smoothing_moves_values_only_across_time(self) -> None:
        phase = self._phase()
        frozen_direction = 100.0 * phase * torch.ones(1, 1, 4, 2)
        frozen_noop = torch.zeros_like(frozen_direction)
        residual = torch.zeros_like(frozen_direction)
        residual[:, 10, 2, 1] = 4.0
        result = commutator.build_motion_commutator(
            frozen_direction + residual,
            frozen_noop,
            frozen_direction,
            frozen_noop,
            config=commutator.MotionCommutatorConfig(
                max_correction_increment_ratio=1.0,
                correction_increment_rms_floor=0.0,
                temporal_smoothing=True,
            ),
        )
        candidate = result.diagnostics.candidate_correction_increments
        self.assertTrue(result.diagnostics.temporal_smoothing_applied)
        # Temporal neighbors receive energy.
        active_phases = torch.nonzero(
            candidate[0, :, 2, 1], as_tuple=False
        ).flatten()
        self.assertGreaterEqual(int(active_phases.numel()), 3)
        # No other spatial token or channel can receive a value.
        elsewhere = candidate.clone()
        elsewhere[:, :, 2, 1] = 0.0
        self.assertTrue(torch.equal(elsewhere, torch.zeros_like(elsewhere)))

    def test_rho_zero_returns_exact_frozen_objects(self) -> None:
        result = commutator.build_motion_commutator(*self._base_branches())
        for step in (31, 32, 39):
            with self.subTest(step=step):
                execution = commutator.execute_motion_commutator(
                    result.frozen_official_direction,
                    result.bounded_commutator_correction,
                    step_index=step,
                )
                self.assertEqual(execution.rho, 0.0)
                self.assertIs(
                    execution.executed_direction,
                    result.frozen_official_direction,
                )

                official = self._base_branches()[2]
                official_execution = (
                    commutator.apply_motion_commutator_to_official_tensor(
                        official,
                        result.bounded_commutator_correction,
                        step_index=step,
                    )
                )
                self.assertIs(
                    official_execution.executed_official_tensor,
                    official,
                )
        early = commutator.execute_motion_commutator(
            result.frozen_official_direction,
            result.bounded_commutator_correction,
            step_index=0,
        )
        self.assertEqual(early.rho, 1.0)

    def test_training_losses_reach_both_adapted_branches(self) -> None:
        adapted_action, adapted_noop, frozen_action, frozen_noop = (
            self._base_branches()
        )
        residual = 0.25 * self._phase() * torch.ones_like(frozen_action)
        adapted_action = (adapted_action + residual).clone().requires_grad_()
        adapted_noop = adapted_noop.clone().requires_grad_()
        result = commutator.build_motion_commutator(
            adapted_action,
            adapted_noop,
            frozen_action,
            frozen_noop,
            config=commutator.MotionCommutatorConfig(
                max_correction_increment_ratio=10.0,
                correction_increment_rms_floor=0.0,
            ),
        )
        desired_correction = 0.5 * self._phase() * torch.ones_like(
            frozen_action
        )
        target_motion = result.frozen_official_direction + desired_correction
        target = commutator.build_target_correction(
            target_motion,
            result.frozen_official_direction.detach(),
        )
        correction_loss = commutator.target_correction_loss(
            result.bounded_commutator_correction,
            target.detach(),
        )
        noop_loss = commutator.adapted_noop_preservation_loss(
            adapted_noop,
            frozen_noop,
        )
        total = correction_loss + 0.1 * noop_loss
        total.backward()
        self.assertIsNotNone(adapted_action.grad)
        self.assertIsNotNone(adapted_noop.grad)
        self.assertTrue(bool(torch.isfinite(adapted_action.grad).all()))
        self.assertTrue(bool(torch.isfinite(adapted_noop.grad).all()))
        self.assertGreater(float(adapted_action.grad.abs().sum()), 0.0)
        self.assertGreater(float(adapted_noop.grad.abs().sum()), 0.0)
        self.assertIsNone(frozen_action.grad)
        self.assertIsNone(frozen_noop.grad)

    def test_training_targets_and_noop_loss_have_expected_zero_cases(self) -> None:
        fields = self._base_branches()
        result = commutator.build_motion_commutator(*fields)
        target = commutator.build_target_correction(
            result.final_direction,
            result.frozen_official_direction,
        )
        self.assertTrue(
            torch.equal(target, result.bounded_commutator_correction)
        )
        correction_loss = commutator.target_correction_loss(
            result.bounded_commutator_correction,
            target,
        )
        noop_loss = commutator.adapted_noop_preservation_loss(fields[3], fields[3])
        self.assertAlmostEqual(float(correction_loss), 0.0, places=7)
        self.assertAlmostEqual(float(noop_loss), 0.0, places=7)

    def test_shape_dtype_finite_and_phase_zero_contracts_are_strict(self) -> None:
        fields = list(self._base_branches())
        invalid_sets = []
        invalid_sets.append((torch.zeros(1, 20, 4, 3),) * 4)
        invalid_sets.append((torch.zeros(1, 21, 12),) * 4)
        invalid_sets.append(tuple(value.double() for value in fields))
        nonfinite = [value.clone() for value in fields]
        nonfinite[0][0, 5, 1, 1] = float("nan")
        invalid_sets.append(tuple(nonfinite))
        for invalid in invalid_sets:
            with self.subTest(shapes=[tuple(value.shape) for value in invalid]):
                with self.assertRaises(commutator.MotionCommutatorError):
                    commutator.build_motion_commutator(*invalid)

        result = commutator.build_motion_commutator(*fields)
        bad_direction = result.bounded_commutator_correction.clone()
        bad_direction[:, 0] = 1.0
        with self.assertRaisesRegex(
            commutator.MotionCommutatorError, "exact zero phase zero"
        ):
            commutator.execute_motion_commutator(
                result.frozen_official_direction,
                bad_direction,
                step_index=0,
            )
        official = fields[2]
        early = commutator.apply_motion_commutator_to_official_tensor(
            official,
            result.bounded_commutator_correction,
            step_index=0,
        )
        self.assertTrue(
            torch.equal(
                early.executed_official_tensor[:, 0],
                official[:, 0],
            )
        )


if __name__ == "__main__":
    unittest.main()
