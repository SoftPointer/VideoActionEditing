import unittest
from pathlib import Path
import sys

try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover - environment dependent
    HAS_TORCH = False

METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import gauge_anchored_commutator as gauge  # noqa: E402
import motion_commutator as commutator  # noqa: E402


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for gauge tests")
class NoopGaugeAnchorTests(unittest.TestCase):
    def test_noop_phase_zero_and_action_increments_are_both_preserved(self):
        torch.manual_seed(7)
        action = torch.randn(2, 21, 3, 4, dtype=torch.float32)
        noop = torch.randn_like(action)
        result = gauge.build_noop_gauge_anchor(action, noop)
        self.assertTrue(torch.equal(result.anchored_action_field[:, 0], noop[:, 0]))
        self.assertTrue(
            torch.allclose(
                commutator.phase_increments(
                    commutator.causal_gauge(result.anchored_action_field)
                ),
                commutator.phase_increments(
                    commutator.causal_gauge(action)
                ),
                rtol=0.0,
                atol=4.0e-6,
            )
        )
        expected = (action[:, :1] - noop[:, :1]).expand_as(action)
        self.assertTrue(
            torch.allclose(result.removed_common_mode, expected, atol=4.0e-6)
        )
        self.assertGreater(result.phase_increment_tolerance, 0.0)
        self.assertTrue(
            torch.all(
                result.phase_increment_rms_error
                <= result.phase_increment_tolerance
            )
        )

    def test_early_correction_keeps_noop_phase_zero(self):
        torch.manual_seed(11)
        action = torch.randn(1, 21, 2, 3, dtype=torch.float32)
        noop = torch.randn_like(action)
        correction = commutator.causal_gauge(torch.randn_like(action))
        result = gauge.execute_gauge_anchored_commutator(
            action, noop, correction, step_index=0
        )
        self.assertEqual(result.rho, 1.0)
        self.assertTrue(torch.equal(result.executed_clean_field[:, 0], noop[:, 0]))
        self.assertTrue(
            torch.allclose(
                result.executed_clean_field,
                result.anchor.anchored_action_field + correction,
            )
        )

    def test_rho_zero_aliases_anchor_not_frozen_action(self):
        action = torch.ones(1, 21, 1, 2, dtype=torch.float32)
        noop = torch.zeros_like(action)
        correction = torch.zeros_like(action)
        result = gauge.execute_gauge_anchored_commutator(
            action, noop, correction, step_index=31
        )
        self.assertEqual(result.rho, 0.0)
        self.assertIs(result.executed_clean_field, result.anchor.anchored_action_field)
        self.assertIsNot(result.executed_clean_field, action)
        self.assertTrue(torch.equal(result.executed_clean_field[:, 0], noop[:, 0]))

    def test_nonzero_phase_correction_is_rejected(self):
        action = torch.zeros(1, 21, 1, 1, dtype=torch.float32)
        noop = torch.zeros_like(action)
        correction = torch.zeros_like(action)
        correction[:, 0] = 1.0
        with self.assertRaises(gauge.GaugeAnchoredCommutatorError):
            gauge.execute_gauge_anchored_commutator(
                action, noop, correction, step_index=0
            )

    def test_full_quotient_is_bounded_around_zero_and_lifted_from_noop(self):
        phase = torch.arange(21, dtype=torch.float32).reshape(1, 21, 1, 1)
        frozen_noop = 0.1 * phase
        frozen_action = frozen_noop + 0.2 * phase
        adapted_noop = frozen_noop.clone()
        adapted_action = frozen_noop + 4.0 * phase
        projection = gauge.project_complete_action_quotient(
            frozen_action_field=frozen_action,
            frozen_noop_field=frozen_noop,
            adapted_action_field=adapted_action,
            adapted_noop_field=adapted_noop,
        )
        diagnostics = projection.diagnostics
        self.assertTrue(
            torch.all(
                diagnostics.bounded_increment_rms
                <= diagnostics.radius + 1.0e-6
            )
        )
        self.assertTrue(torch.any(diagnostics.scale[:, 1:] < 1.0))
        execution = gauge.execute_feasible_quotient_transport(
            frozen_noop, projection.bounded_quotient, step_index=0
        )
        self.assertTrue(
            torch.equal(execution.executed_clean_field[:, 0], frozen_noop[:, 0])
        )

    def test_full_quotient_rho_zero_aliases_frozen_noop(self):
        noop = torch.randn(1, 21, 2, 2, dtype=torch.float32)
        quotient = commutator.causal_gauge(torch.randn_like(noop))
        execution = gauge.execute_feasible_quotient_transport(
            noop, quotient, step_index=31
        )
        self.assertIs(execution.executed_clean_field, noop)
        self.assertEqual(execution.rho, 0.0)


if __name__ == "__main__":
    unittest.main()
