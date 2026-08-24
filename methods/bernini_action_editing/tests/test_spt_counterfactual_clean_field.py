from __future__ import annotations

import inspect
from pathlib import Path
import sys
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from spt_v2 import counterfactual_clean_field as cf  # noqa: E402
from spt_v2 import phase_transport as spt  # noqa: E402


class PureCounterfactualContractTests(unittest.TestCase):
    def test_contract_is_same_state_source_text_only_and_not_peft(self) -> None:
        contract = cf.runtime_contract()
        self.assertEqual(contract["inference_conditions"], ["source_video", "edit_instruction"])
        self.assertEqual(
            contract["prediction_boundary"], "post_official_cfg_apg_clean_prediction"
        )
        self.assertIn("identical_noisy_y_and_sigma", contract["same_state_obligation"])
        self.assertEqual(contract["generate_gate_application_count"], 1)
        self.assertEqual(
            contract["parity_control_formula"],
            "x_noop_clean+delta_clean=x_action_clean",
        )
        self.assertTrue(
            {
                "delta_mean_square",
                "delta_rms",
                "source_anchor_displacement_rms",
                "preserve_gate_mass",
                "transport_gate_mass",
                "generate_gate_mass",
            }
            <= set(contract["diagnostics"])
        )
        self.assertEqual(contract["required_plan_provenance"], "student")
        self.assertEqual(
            contract["integrator"], "owned_externally_by_tri_branch_unipc_hook"
        )
        self.assertFalse(contract["same_state_enforced_here"])
        self.assertEqual(contract["same_state_authority"], "tri_branch_unipc_hook")
        self.assertEqual(
            contract["parity_control_scope"],
            "algebraic_identity_not_same_state_evidence",
        )
        self.assertFalse(contract["custom_integrator"])
        self.assertFalse(contract["peft_dependency"])
        self.assertEqual(contract["zero_sigma_policy"], "fail_before_velocity_projection")
        self.assertTrue(
            {"target_video", "mask", "track", "pose", "optical_flow"}
            <= set(contract["forbidden_conditions"])
        )

    def test_runtime_apis_cannot_receive_train_only_or_spatial_hints(self) -> None:
        forbidden = {
            "target",
            "target_video",
            "paired_target",
            "mask",
            "track",
            "pose",
            "flow",
            "trajectory",
            "anchor",
            "peft_model",
        }
        for function in (
            cf.same_state_clean_delta,
            cf.counterfactual_clean_field,
            cf.execute_counterfactual_clean_plan,
            cf.counterfactual_plan_velocity,
            cf.counterfactual_plan_velocity_with_diagnostics,
            cf.counterfactual_packed_velocity,
            cf.counterfactual_packed_velocity_with_diagnostics,
        ):
            with self.subTest(function=function.__name__):
                self.assertTrue(
                    forbidden.isdisjoint(inspect.signature(function).parameters)
                )


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is unavailable")
class TensorCounterfactualCleanFieldTests(unittest.TestCase):
    def _video(self, value: float = 0.0) -> "torch.Tensor":
        return torch.full((1, 21, 2, 3, 4), value, dtype=torch.float32)

    def _plan(self, gates=(1.0, 0.0, 0.0), *, provenance="student") -> spt.PhasePlan:
        offsets = torch.zeros(1, 3, 21, 2, 3, dtype=torch.float32)
        gate_probs = torch.zeros_like(offsets)
        for index, value in enumerate(gates):
            gate_probs[:, index].fill_(value)
        return spt.PhasePlan(offsets, gate_probs, provenance)

    def test_action_equal_noop_has_exact_zero_delta(self) -> None:
        action = torch.randn_like(self._video())
        delta = cf.same_state_clean_delta(action, action.clone())
        self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))

    def test_noop_plus_delta_exactly_recovers_action_prediction(self) -> None:
        noop = self._video(7.0)
        action = self._video(11.0)
        delta = cf.same_state_clean_delta(action, noop)
        self.assertTrue(torch.equal(noop.float() + delta, action.float()))

    def test_preserve_one_returns_source_independent_of_delta(self) -> None:
        source = torch.randn_like(self._video())
        action = torch.randn_like(source) * 10.0
        noop = torch.randn_like(source) * 10.0
        result = cf.execute_counterfactual_clean_plan(
            source=source,
            action_clean=action,
            noop_clean=noop,
            plan=self._plan((1.0, 0.0, 0.0)),
            alpha=3.0,
        )
        self.assertTrue(torch.equal(result, source))

    def test_generate_gate_is_applied_once_not_squared(self) -> None:
        source = self._video(2.0)
        noop = self._video(7.0)
        action = self._video(11.0)  # delta=4; alpha=.5; x_cf=4.
        plan = self._plan((0.75, 0.0, 0.25))
        field = cf.counterfactual_clean_field(source, action, noop, alpha=0.5)
        self.assertTrue(torch.equal(field, self._video(4.0)))
        result = cf.execute_counterfactual_clean_plan(
            source=source,
            action_clean=action,
            noop_clean=noop,
            plan=plan,
            alpha=0.5,
        )
        expected_once = self._video(2.5)  # .75*2 + .25*4
        expected_if_squared = self._video(1.75)  # .75*2 + .25*(.25*4)
        self.assertTrue(torch.allclose(result, expected_once))
        self.assertFalse(torch.allclose(result, expected_if_squared))

    def test_noop_anchor_plus_delta_parity_is_separate_from_source_anchor(self) -> None:
        source = self._video(2.0)
        noop = self._video(7.0)
        action = self._video(11.0)
        delta = cf.same_state_clean_delta(action, noop)
        self.assertTrue(torch.equal(noop + delta, action))
        source_anchored = cf.counterfactual_clean_field(source, action, noop)
        self.assertTrue(torch.equal(source_anchored, self._video(6.0)))

    def test_diagnostics_record_delta_energy_gate_mass_and_both_anchors(self) -> None:
        source = self._video(2.0)
        noop = self._video(7.0)
        action = self._video(11.0)
        noisy = self._video(9.0)
        _, record = cf.counterfactual_plan_velocity_with_diagnostics(
            noisy=noisy,
            sigma=0.5,
            source=source,
            action_clean=action,
            noop_clean=noop,
            plan=self._plan((0.2, 0.3, 0.5)),
            alpha=0.5,
        )
        self.assertEqual(record.sigma, 0.5)
        self.assertEqual(record.alpha, 0.5)
        self.assertAlmostEqual(record.delta_mean_square, 16.0)
        self.assertAlmostEqual(record.delta_rms, 4.0)
        self.assertAlmostEqual(record.source_anchor_displacement_rms, 2.0)
        self.assertEqual(record.noop_action_parity_rms_error, 0.0)
        self.assertEqual(record.noop_action_parity_max_abs_error, 0.0)
        self.assertAlmostEqual(record.preserve_gate_mass, 0.2, places=6)
        self.assertAlmostEqual(record.transport_gate_mass, 0.3, places=6)
        self.assertAlmostEqual(record.generate_gate_mass, 0.5, places=6)

    def test_algebra_core_does_not_own_a_scheduler_callback(self) -> None:
        self.assertFalse(hasattr(cf, "advance_counterfactual_scheduler_step"))

    def test_shape_mismatch_and_oracle_plan_fail_closed(self) -> None:
        video = self._video()
        wrong = torch.zeros(1, 21, 2, 4, 4)
        with self.assertRaisesRegex(cf.CounterfactualCleanFieldError, "shape"):
            cf.counterfactual_clean_field(video, wrong, video)
        with self.assertRaisesRegex(
            cf.CounterfactualCleanFieldError, "student plan"
        ):
            cf.execute_counterfactual_clean_plan(
                source=video,
                action_clean=video,
                noop_clean=video,
                plan=self._plan(provenance="oracle_pair_proxy"),
            )


if __name__ == "__main__":
    unittest.main()
