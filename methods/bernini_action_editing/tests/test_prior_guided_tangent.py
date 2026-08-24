from __future__ import annotations

import ast
import inspect
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import prior_guided_tangent as pgt


class PriorGuidedTangentPureContractTests(unittest.TestCase):
    def test_method_and_main_arm_defaults_are_fixed(self) -> None:
        self.assertEqual(
            pgt.METHOD_NAME,
            "prior-guided-tangent-trust-region-lora-v5",
        )
        self.assertEqual(pgt.NUM_DENOISING_STEPS, 40)
        config = pgt.TangentTrustRegionConfig()
        config.validate()
        self.assertEqual(config.kappa_parallel, 0.5)
        self.assertEqual(config.kappa_perp, 0.15)
        self.assertEqual(config.phase_dim, 1)

    def test_exact_40_step_gamma_schedule(self) -> None:
        gamma = pgt.correction_gamma_schedule()
        self.assertEqual(len(gamma), 40)
        self.assertEqual(gamma[:24], (1.0,) * 24)
        self.assertEqual(gamma[24], 1.0)
        self.assertEqual(gamma[34], 0.0)
        self.assertEqual(gamma[35:], (0.0,) * 5)
        self.assertTrue(
            all(gamma[index] >= gamma[index + 1] for index in range(24, 34))
        )
        self.assertGreater(gamma[25], 0.0)
        self.assertLess(gamma[33], 1.0)
        for invalid in (-1, 40, 1.5, True):
            with self.subTest(invalid=invalid):
                with self.assertRaises(pgt.PriorGuidedTangentError):
                    pgt.correction_gamma(invalid)

    def test_student_api_cannot_receive_source_or_target(self) -> None:
        self.assertEqual(
            list(inspect.signature(pgt.student_executed_field).parameters),
            [
                "base_action_field",
                "base_noop_field",
                "adapted_action_field",
                "step_index",
                "config",
            ],
        )
        self.assertEqual(
            list(inspect.signature(pgt.teacher_executed_field).parameters)[:4],
            ["source", "target", "base_action_field", "base_noop_field"],
        )

    def test_configuration_rejects_unbounded_or_invalid_caps(self) -> None:
        invalid = (
            {"kappa_parallel": -0.1},
            {"kappa_perp": -0.1},
            {"kappa_parallel": float("inf")},
            {"epsilon": 0.0},
            {"phase_dim": 0},
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(pgt.PriorGuidedTangentError):
                    pgt.TangentTrustRegionConfig(**values).validate()

    def test_torch_is_not_a_module_import_dependency(self) -> None:
        tree = ast.parse(Path(pgt.__file__).read_text(encoding="utf-8"))
        eager_torch_imports = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                eager_torch_imports.extend(
                    alias.name for alias in node.names if alias.name == "torch"
                )
            elif isinstance(node, ast.ImportFrom) and node.module == "torch":
                eager_torch_imports.append(node.module)
        self.assertEqual(eager_torch_imports, [])
        # The schedule/configuration contract remains usable without invoking
        # any lazy tensor path.
        self.assertEqual(pgt.correction_gamma(0), 1.0)
        self.assertEqual(pgt.correction_gamma(39), 0.0)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is unavailable")
class PriorGuidedTangentTensorTests(unittest.TestCase):
    def test_q0_is_idempotent_and_phase_zero_exact(self) -> None:
        field = torch.randn(2, 4, 3, 2, dtype=torch.float32)
        projected = pgt.q0(field)
        self.assertTrue(torch.equal(projected[:, 0], torch.zeros_like(projected[:, 0])))
        self.assertTrue(torch.equal(pgt.q0(projected), projected))

    def test_zero_adapter_is_exact_prior_and_keeps_gradient(self) -> None:
        base_noop = torch.randn(1, 4, 2, 3, dtype=torch.float32)
        base_action = base_noop + torch.randn_like(base_noop)
        adapted_action = base_action.detach().clone().requires_grad_(True)
        prior = pgt.frozen_prior(base_action, base_noop)
        executed = pgt.student_executed_field(
            base_action,
            base_noop,
            adapted_action,
            step_index=0,
        )
        self.assertTrue(torch.equal(executed, prior))
        executed.sum().backward()
        self.assertIsNotNone(adapted_action.grad)
        self.assertTrue(bool(torch.isfinite(adapted_action.grad).all()))
        self.assertGreater(float(adapted_action.grad.abs().sum()), 0.0)

    def test_raw_prior_is_preserved_for_audit_but_executable_prior_is_causal(self) -> None:
        base_noop = torch.zeros(1, 3, 2, 1, dtype=torch.float32)
        base_action = torch.tensor(
            [[[[2.0], [-1.0]], [[3.0], [1.0]], [[4.0], [2.0]]]],
            dtype=torch.float32,
        )
        adapter_delta = torch.tensor(
            [[[[7.0], [-4.0]], [[9.0], [3.0]], [[1.0], [8.0]]]],
            dtype=torch.float32,
        )
        correction = pgt.adapter_correction(base_action + adapter_delta, base_action)
        raw_prior = pgt.raw_frozen_prior(base_action, base_noop)
        prior = pgt.frozen_prior(base_action, base_noop)
        result = pgt.execute_prior_guided_field(
            prior,
            correction,
            step_index=0,
        )
        self.assertTrue(
            torch.equal(correction[:, 0], torch.zeros_like(correction[:, 0]))
        )
        self.assertTrue(torch.equal(raw_prior, base_action - base_noop))
        self.assertGreater(float(raw_prior[:, 0].abs().sum()), 0.0)
        self.assertTrue(torch.equal(prior, pgt.q0(raw_prior)))
        self.assertTrue(torch.equal(prior[:, 0], torch.zeros_like(prior[:, 0])))
        self.assertTrue(
            torch.equal(result.executed_field[:, 0], prior[:, 0])
        )

    def test_execute_rejects_an_unprojected_prior(self) -> None:
        raw_prior = torch.randn(1, 3, 2, 1, dtype=torch.float32)
        raw_prior[:, 0] = 1.0
        with self.assertRaisesRegex(
            pgt.PriorGuidedTangentError, "exact zero first phase"
        ):
            pgt.execute_prior_guided_field(
                raw_prior,
                torch.zeros_like(raw_prior),
                step_index=0,
            )

    def test_phasewise_parallel_and_perpendicular_smooth_caps(self) -> None:
        prior = torch.tensor(
            [[[[2.0]], [[-1.0]]], [[[3.0]], [[-1.0]]], [[[2.0]], [[1.0]]]],
            dtype=torch.float32,
        ).unsqueeze(0)
        correction = torch.tensor(
            [[[[0.0]], [[0.0]]], [[[100.0]], [[100.0]]], [[[100.0]], [[100.0]]]],
            dtype=torch.float32,
        ).unsqueeze(0)
        config = pgt.TangentTrustRegionConfig(
            kappa_parallel=0.5,
            kappa_perp=0.15,
            epsilon=1e-6,
        )
        result = pgt.phasewise_trust_region(prior, correction, config)
        motion = result.motion_reference
        trusted = result.trusted_correction
        for phase in (1, 2):
            motion_phase = motion[:, phase].reshape(1, -1)
            trusted_phase = trusted[:, phase].reshape(1, -1)
            energy = motion_phase.square().sum(dim=1)
            coefficient = (trusted_phase * motion_phase).sum(dim=1) / energy
            perpendicular = trusted_phase - coefficient[:, None] * motion_phase
            perpendicular_rms = perpendicular.square().mean(dim=1).sqrt()
            motion_rms = motion_phase.square().mean(dim=1).sqrt()
            self.assertLessEqual(float(coefficient.abs().max()), 0.5 + 1e-5)
            self.assertLessEqual(
                float(perpendicular_rms.max()),
                float((0.15 * motion_rms + 1e-6).max()) + 1e-5,
            )
        parallel_only = pgt.phasewise_trust_region(
            prior,
            correction,
            pgt.TangentTrustRegionConfig(kappa_parallel=0.5, kappa_perp=0.0),
        )
        self.assertTrue(
            torch.equal(
                parallel_only.bounded_perpendicular_correction,
                torch.zeros_like(parallel_only.bounded_perpendicular_correction),
            )
        )

    def test_last_five_steps_are_exact_frozen_prior_alias(self) -> None:
        prior = pgt.q0(torch.randn(1, 4, 3, 2, dtype=torch.float32))
        correction = pgt.q0(torch.randn_like(prior))
        for step in range(35, 40):
            with self.subTest(step=step):
                result = pgt.execute_prior_guided_field(
                    prior,
                    correction,
                    step_index=step,
                )
                self.assertEqual(result.gamma, 0.0)
                self.assertIs(result.executed_field, prior)
                self.assertTrue(torch.equal(result.executed_field, prior))

    def test_teacher_and_student_use_the_same_executed_field_operator(self) -> None:
        source = torch.zeros(1, 3, 2, 1, dtype=torch.float32)
        base_noop = torch.zeros_like(source)
        base_action = torch.tensor(
            [[[[1.0], [2.0]], [[2.0], [4.0]], [[4.0], [8.0]]]],
            dtype=torch.float32,
        )
        correction = torch.tensor(
            [[[[0.0], [0.0]], [[1.0], [-1.0]], [[2.0], [1.0]]]],
            dtype=torch.float32,
        )
        adapted_action = base_action + correction
        prior = pgt.frozen_prior(base_action, base_noop)
        target = source + prior + correction
        config = pgt.TangentTrustRegionConfig()
        student = pgt.student_executed_field(
            base_action,
            base_noop,
            adapted_action,
            step_index=29,
            config=config,
        )
        teacher = pgt.teacher_executed_field(
            source,
            target,
            base_action,
            base_noop,
            step_index=29,
            config=config,
        )
        expected = pgt.execute_prior_guided_field(
            prior,
            correction,
            step_index=29,
            config=config,
        ).executed_field
        self.assertTrue(torch.equal(student, teacher))
        self.assertTrue(torch.equal(student, expected))

    def test_untrusted_teacher_algebra_does_not_reintroduce_raw_phase_zero(self) -> None:
        source = torch.randn(1, 3, 2, 1, dtype=torch.float32)
        base_noop = torch.randn_like(source)
        raw_prior = torch.tensor(
            [[[[5.0], [-3.0]], [[6.0], [1.0]], [[9.0], [4.0]]]],
            dtype=torch.float32,
        )
        base_action = base_noop + raw_prior
        desired_motion = torch.tensor(
            [[[[0.0], [0.0]], [[2.0], [-1.0]], [[3.0], [4.0]]]],
            dtype=torch.float32,
        )
        target = source + desired_motion
        causal_prior = pgt.frozen_prior(base_action, base_noop)
        correction = pgt.teacher_correction(source, target, causal_prior)
        untrusted_teacher = causal_prior + correction

        self.assertGreater(
            float(pgt.raw_frozen_prior(base_action, base_noop)[:, 0].abs().sum()),
            0.0,
        )
        self.assertTrue(
            torch.equal(causal_prior[:, 0], torch.zeros_like(causal_prior[:, 0]))
        )
        self.assertTrue(torch.equal(untrusted_teacher, desired_motion))
        self.assertTrue(
            torch.equal(
                (source + untrusted_teacher)[:, 0],
                source[:, 0],
            )
        )


if __name__ == "__main__":
    unittest.main()
