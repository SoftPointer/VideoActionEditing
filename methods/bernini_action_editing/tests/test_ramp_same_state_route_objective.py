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
    import ramp_same_state_route_objective as ramp
else:  # pragma: no cover
    ramp = None


def _identity():
    return ramp.SameStateInterventionIdentity(
        source_sha256="1" * 64,
        text_sha256="2" * 64,
        epsilon_sha256="3" * 64,
        noisy_target_sha256="3" * 64,
        timestep_token="sigma=1",
        program_a_sha256="4" * 64,
        program_b_sha256="5" * 64,
    )


def _fixture(*, perfect=True):
    target_a = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    target_b = torch.tensor([[3.0, 1.0]], dtype=torch.float32)
    epsilon = torch.tensor([[5.0, 7.0]], dtype=torch.float32)
    velocity_a = epsilon - target_a
    velocity_b = epsilon - target_b
    if perfect:
        pred_a_value, pred_b_value = velocity_a, velocity_b
    else:
        pred_a_value, pred_b_value = velocity_b, velocity_a
    pred_a = pred_a_value.clone().requires_grad_(True)
    pred_b = pred_b_value.clone().requires_grad_(True)
    donor_a = velocity_a.clone().requires_grad_(True)
    donor_b = velocity_a.clone().requires_grad_(True)
    order_a = velocity_b.clone().requires_grad_(True)
    order_b = velocity_b.clone().requires_grad_(True)
    logits = (torch.eye(3, dtype=torch.float32) * 8.0).unsqueeze(0).requires_grad_(True)
    transport = torch.eye(3, dtype=torch.float32)
    return {
        "prediction_a": pred_a,
        "prediction_b": pred_b,
        "clean_target_a": target_a,
        "clean_target_b": target_b,
        "shared_epsilon": epsilon,
        "identity": _identity(),
        "donor_identity_prediction_a": donor_a,
        "donor_identity_prediction_b": donor_b,
        "order_prediction_a": order_a,
        "order_prediction_b": order_b,
        "transport_logits": logits,
        "transport_target": transport,
    }


@unittest.skipIf(torch is None, "torch is unavailable")
class SameStateRouteObjectiveTests(unittest.TestCase):
    def test_native_delta_identity_and_perfect_route(self) -> None:
        values = _fixture(perfect=True)
        result = ramp.sigma_one_same_state_route_objective(**values)
        self.assertTrue(
            torch.equal(
                result.target_velocity_a - result.target_velocity_b,
                values["clean_target_b"] - values["clean_target_a"],
            )
        )
        self.assertTrue(torch.equal(result.prediction_delta.detach(), result.target_delta))
        self.assertEqual(float(result.flow_matching_loss.detach()), 0.0)
        self.assertEqual(float(result.route_loss.detach()), 0.0)
        self.assertEqual(float(result.donor_identity_invariance_loss.detach()), 0.0)
        self.assertEqual(float(result.order_invariance_loss.detach()), 0.0)
        self.assertTrue(torch.equal(result.route_explained_fraction, torch.ones(1)))
        self.assertTrue(bool(result.own_target_ranking.all()))
        self.assertEqual(result.receipt["sigma"], 1.0)
        self.assertFalse(result.receipt["natural_action_training_authorized"])

    def test_swapped_predictions_fail_route_and_own_target_ranking(self) -> None:
        result = ramp.sigma_one_same_state_route_objective(**_fixture(perfect=False))
        self.assertGreater(float(result.route_loss.detach()), 0.0)
        self.assertLess(float(result.route_explained_fraction.item()), 0.0)
        self.assertFalse(bool(result.own_target_ranking.any()))

    def test_combined_loss_backpropagates_to_every_trainable_path(self) -> None:
        values = _fixture(perfect=True)
        # Make invariance paths non-zero so they receive observable gradients.
        values["donor_identity_prediction_b"] = (
            values["donor_identity_prediction_b"].detach() + 0.25
        ).requires_grad_(True)
        values["order_prediction_b"] = (
            values["order_prediction_b"].detach() - 0.25
        ).requires_grad_(True)
        result = ramp.sigma_one_same_state_route_objective(**values)
        result.total_loss.backward()
        for name in (
            "prediction_a",
            "prediction_b",
            "donor_identity_prediction_a",
            "donor_identity_prediction_b",
            "order_prediction_a",
            "order_prediction_b",
            "transport_logits",
        ):
            self.assertIsNotNone(values[name].grad, name)
            self.assertTrue(bool(torch.isfinite(values[name].grad).all()), name)

    def test_weights_are_preregistered(self) -> None:
        values = _fixture()
        values["route_weight"] = 0.6
        with self.assertRaisesRegex(ramp.RAMPSameStateObjectiveError, "weights must equal"):
            ramp.sigma_one_same_state_route_objective(**values)

    def test_targets_epsilon_and_programs_fail_closed(self) -> None:
        values = _fixture()
        values["clean_target_b"] = values["clean_target_a"].clone()
        with self.assertRaisesRegex(ramp.RAMPSameStateObjectiveError, "byte-equal"):
            ramp.sigma_one_same_state_route_objective(**values)

        values = _fixture()
        values["shared_epsilon"] = values["shared_epsilon"].double()
        with self.assertRaisesRegex(ramp.RAMPSameStateObjectiveError, "must be FP32"):
            ramp.sigma_one_same_state_route_objective(**values)

        with self.assertRaisesRegex(ramp.RAMPSameStateObjectiveError, "distinct"):
            ramp.SameStateInterventionIdentity(
                source_sha256="1" * 64,
                text_sha256="2" * 64,
                epsilon_sha256="3" * 64,
                noisy_target_sha256="3" * 64,
                timestep_token="sigma=1",
                program_a_sha256="4" * 64,
                program_b_sha256="4" * 64,
            )

    def test_transport_target_must_be_row_stochastic(self) -> None:
        values = _fixture()
        values["transport_target"] = torch.ones(3, 3, dtype=torch.float32)
        with self.assertRaisesRegex(ramp.RAMPSameStateObjectiveError, "rows must be probabilities"):
            ramp.sigma_one_same_state_route_objective(**values)


if __name__ == "__main__":
    unittest.main()
