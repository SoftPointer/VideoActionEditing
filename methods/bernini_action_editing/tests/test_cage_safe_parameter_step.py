from __future__ import annotations

from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    import cage_safe_parameter_step as safe
else:  # pragma: no cover
    safe = None


@unittest.skipIf(torch is None, "torch is unavailable")
class CAGESafeParameterStepTests(unittest.TestCase):
    def test_unconstrained_action_step_is_exact_sgd(self) -> None:
        action = (torch.tensor([3.0, 4.0], dtype=torch.float32),)
        constraints = (
            (torch.tensor([0.0, 1.0], dtype=torch.float32),),
            (torch.tensor([1.0, 0.0], dtype=torch.float32),),
        )
        result = safe.project_safe_parameter_step(
            action,
            constraints,
            (-10.0, -10.0),
            ("identity", "camera"),
            step_size=0.1,
            trust_radius=2.0,
        )
        self.assertTrue(result.update_authorized)
        self.assertEqual(result.active_constraint_indices, ())
        self.assertTrue(torch.equal(result.displacement[0], -0.1 * action[0]))
        self.assertTrue(torch.equal(result.projected_gradient[0], action[0]))
        self.assertAlmostEqual(result.retention_ratio, 1.0)

    def test_active_constraint_removes_unsafe_component(self) -> None:
        action = (torch.tensor([-2.0, -1.0], dtype=torch.float32),)
        # d0=[2,1].  camera requires d_x<=0, leaving only y motion.
        constraints = ((torch.tensor([1.0, 0.0], dtype=torch.float32),),)
        result = safe.project_safe_parameter_step(
            action,
            constraints,
            (0.0,),
            ("camera",),
            step_size=1.0,
            trust_radius=10.0,
            minimum_retention=0.2,
        )
        self.assertTrue(result.update_authorized)
        self.assertEqual(result.active_constraint_indices, (0,))
        self.assertTrue(
            torch.allclose(
                result.displacement[0],
                torch.tensor([0.0, 1.0]),
                atol=1.0e-7,
                rtol=0.0,
            )
        )
        self.assertLessEqual(result.linearized_constraint_values[0], 1.0e-8)

    def test_retention_floor_returns_exact_zero_update(self) -> None:
        action = (torch.tensor([-10.0, -0.01], dtype=torch.float32),)
        constraints = ((torch.tensor([1.0, 0.0], dtype=torch.float32),),)
        result = safe.project_safe_parameter_step(
            action,
            constraints,
            (0.0,),
            ("identity",),
            step_size=1.0,
            trust_radius=20.0,
            minimum_retention=0.2,
        )
        self.assertFalse(result.update_authorized)
        self.assertEqual(
            result.block_reason,
            "projected_gradient_retention_below_registered_floor",
        )
        self.assertTrue(torch.equal(result.displacement[0], torch.zeros(2)))
        self.assertTrue(torch.equal(result.projected_gradient[0], torch.zeros(2)))

    def test_current_constraint_violation_is_not_hidden_by_recovery(self) -> None:
        result = safe.project_safe_parameter_step(
            (torch.tensor([1.0, 2.0], dtype=torch.float32),),
            ((torch.tensor([1.0, 0.0], dtype=torch.float32),),),
            (0.1,),
            ("quality",),
            step_size=0.1,
            trust_radius=1.0,
        )
        self.assertFalse(result.feasible)
        self.assertFalse(result.update_authorized)
        self.assertIn("recovery", result.block_reason)

    def test_trust_radius_scales_eta_before_projection(self) -> None:
        action = (torch.tensor([3.0, 4.0], dtype=torch.float32),)
        result = safe.project_safe_parameter_step(
            action,
            ((torch.tensor([0.0, 1.0], dtype=torch.float32),),),
            (-100.0,),
            ("identity",),
            step_size=10.0,
            trust_radius=0.5,
        )
        self.assertTrue(result.update_authorized)
        self.assertAlmostEqual(result.effective_step_size, 0.1)
        self.assertAlmostEqual(result.projected_step_norm, 0.5, places=6)

    def test_assigns_only_an_authorized_fp32_parameter_closure(self) -> None:
        parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
        result = safe.project_safe_parameter_step(
            (torch.tensor([1.0, -2.0], dtype=torch.float32),),
            ((torch.tensor([0.0, 1.0], dtype=torch.float32),),),
            (-10.0,),
            ("identity",),
            step_size=0.1,
            trust_radius=1.0,
        )
        safe.assign_projected_gradients((parameter,), result)
        self.assertTrue(torch.equal(parameter.grad, result.projected_gradient[0]))


if __name__ == "__main__":
    unittest.main()
