from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch  # noqa: E402
except ImportError:
    torch = None

if torch is not None:
    import dclr_preference_objective as objective  # noqa: E402
    import dclr_runtime_contract as runtime_contract  # noqa: E402
else:  # pragma: no cover - dependency-light environments
    objective = None
    runtime_contract = None


class DependencyLightSourceGuards(unittest.TestCase):
    def test_objective_has_no_model_optimizer_or_distributed_runtime(self) -> None:
        source = (METHOD_ROOT / "dclr_preference_objective.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("torch.distributed", source)
        self.assertNotIn("all_reduce(", source)
        self.assertNotIn("optimizer.step(", source)
        self.assertNotIn("shared_step(", source)


@unittest.skipIf(torch is None, "torch is unavailable")
class SharedPairFlowStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.winner = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
        self.loser = self.winner + 2.0
        self.epsilon = torch.flip(self.winner, dims=(2,)).contiguous()
        self.sigma = torch.tensor([0.375], dtype=torch.float32)

    def test_uses_one_exact_fp32_sigma_epsilon_and_velocity_target(self) -> None:
        state = objective.build_shared_pair_flow_state(
            self.winner, self.loser, self.epsilon, self.sigma
        )
        broadcast_sigma = self.sigma.reshape(1, 1, 1)
        expected_winner_x = (
            (torch.ones_like(broadcast_sigma) - broadcast_sigma) * self.winner
            + broadcast_sigma * self.epsilon
        )
        expected_loser_x = (
            (torch.ones_like(broadcast_sigma) - broadcast_sigma) * self.loser
            + broadcast_sigma * self.epsilon
        )

        self.assertIs(state.sigma, self.sigma)
        self.assertIs(state.epsilon, self.epsilon)
        self.assertTrue(torch.equal(state.winner_x_sigma, expected_winner_x))
        self.assertTrue(torch.equal(state.loser_x_sigma, expected_loser_x))
        self.assertTrue(
            torch.equal(state.winner_true_velocity, self.epsilon - self.winner)
        )
        self.assertTrue(
            torch.equal(state.loser_true_velocity, self.epsilon - self.loser)
        )
        self.assertTrue(
            torch.equal(
                state.timestep,
                runtime_contract.fp32_sigma_to_timestep(self.sigma),
            )
        )
        for tensor in (
            state.sigma,
            state.timestep,
            state.epsilon,
            state.winner_x_sigma,
            state.loser_x_sigma,
            state.winner_true_velocity,
            state.loser_true_velocity,
        ):
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertFalse(tensor.requires_grad)
            self.assertIsNone(tensor.grad_fn)

    def test_rejects_cast_shape_graph_and_sigma_contract_changes(self) -> None:
        with self.assertRaisesRegex(
            objective.DCLRPreferenceObjectiveError, "exact FP32"
        ):
            objective.build_shared_pair_flow_state(
                self.winner.double(),
                self.loser.double(),
                self.epsilon.double(),
                self.sigma.double(),
            )
        with self.assertRaisesRegex(
            objective.DCLRPreferenceObjectiveError, "shape/dtype/device/layout"
        ):
            objective.build_shared_pair_flow_state(
                self.winner, self.loser[:, :, :3], self.epsilon, self.sigma
            )
        with self.assertRaisesRegex(
            objective.DCLRPreferenceObjectiveError, "detached"
        ):
            objective.build_shared_pair_flow_state(
                self.winner.clone().requires_grad_(),
                self.loser,
                self.epsilon,
                self.sigma,
            )
        with self.assertRaisesRegex(
            objective.DCLRPreferenceObjectiveError, r"shape \[1\]"
        ):
            objective.build_shared_pair_flow_state(
                self.winner,
                self.loser,
                self.epsilon,
                torch.tensor([0.25, 0.75], dtype=torch.float32),
            )
        with self.assertRaisesRegex(
            objective.DCLRPreferenceObjectiveError, r"\[0, 1\]"
        ):
            objective.build_shared_pair_flow_state(
                self.winner,
                self.loser,
                self.epsilon,
                torch.tensor([1.1], dtype=torch.float32),
            )


@unittest.skipIf(torch is None, "torch is unavailable")
class TargetTailEnergyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.total = 4
        self.target_count = 2
        self.selector = torch.tensor([False, False, True, True])
        self.target = torch.ones(
            (1, self.target_count, runtime_contract.PINNED_PATCH_DIM),
            dtype=torch.float32,
        )

    def test_scores_only_target_tail_in_fp32_and_preserves_current_graph(self) -> None:
        current = torch.empty(
            (1, self.total, runtime_contract.PINNED_PATCH_DIM),
            dtype=torch.float32,
        )
        with torch.no_grad():
            current[:, : self.target_count, :] = 1000.0
            current[:, self.target_count :, :] = 2.0
        current.requires_grad_()
        reference = torch.empty_like(current)
        reference[:, : self.target_count, :] = -1000.0
        reference[:, self.target_count :, :] = 3.0

        energies = objective.candidate_current_reference_target_tail_mse(
            current, reference, self.target, self.selector
        )
        self.assertEqual(energies.current.dtype, torch.float32)
        self.assertEqual(float(energies.current.item()), 1.0)
        self.assertEqual(float(energies.reference.item()), 4.0)
        self.assertTrue(energies.current.requires_grad)
        self.assertFalse(energies.reference.requires_grad)

        energies.current.backward()
        self.assertIsNotNone(current.grad)
        self.assertTrue(
            torch.equal(
                current.grad[:, : self.target_count, :],
                torch.zeros_like(current.grad[:, : self.target_count, :]),
            )
        )
        self.assertGreater(
            float(current.grad[:, self.target_count :, :].abs().sum().item()),
            0.0,
        )

    def test_bfloat_prediction_accumulates_fp32(self) -> None:
        current = torch.full(
            (1, self.total, runtime_contract.PINNED_PATCH_DIM),
            2.0,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        reference = torch.full_like(current.detach(), 3.0)
        energies = objective.candidate_current_reference_target_tail_mse(
            current, reference, self.target, self.selector
        )
        self.assertEqual(energies.current.dtype, torch.float32)
        self.assertEqual(energies.reference.dtype, torch.float32)
        self.assertEqual(float(energies.current.item()), 1.0)
        self.assertEqual(float(energies.reference.item()), 4.0)

    def test_rejects_missing_or_leaked_graph_and_non_tail_mask(self) -> None:
        detached = torch.zeros(
            (1, self.total, runtime_contract.PINNED_PATCH_DIM),
            dtype=torch.float32,
        )
        with self.assertRaisesRegex(
            objective.DCLRPreferenceObjectiveError, "active adapter graph"
        ):
            objective.candidate_current_reference_target_tail_mse(
                detached, detached.clone(), self.target, self.selector
            )
        with self.assertRaisesRegex(
            objective.DCLRPreferenceObjectiveError, "detached collection-policy"
        ):
            objective.candidate_current_reference_target_tail_mse(
                detached.clone().requires_grad_(),
                detached.clone().requires_grad_(),
                self.target,
                self.selector,
            )
        with self.assertRaisesRegex(
            objective.DCLRPreferenceObjectiveError, "contiguous target tail"
        ):
            objective.candidate_current_reference_target_tail_mse(
                detached.clone().requires_grad_(),
                detached.clone(),
                self.target,
                torch.tensor([False, True, False, True]),
            )
        with self.assertRaisesRegex(
            objective.DCLRPreferenceObjectiveError, "exact FP32"
        ):
            objective.candidate_current_reference_target_tail_mse(
                detached.clone().requires_grad_(),
                detached.clone(),
                self.target.bfloat16(),
                self.selector,
            )


@unittest.skipIf(torch is None, "torch is unavailable")
class ReferenceCorrectedDPOTests(unittest.TestCase):
    @staticmethod
    def _energies(current: float, reference: float):
        return objective.CandidatePolicyEnergies(
            current=torch.tensor(current, dtype=torch.float32, requires_grad=True),
            reference=torch.tensor(reference, dtype=torch.float32),
        )

    def test_exact_reference_corrected_softplus_and_gradient_signs(self) -> None:
        winner = self._energies(1.0, 2.0)
        loser = self._energies(4.0, 3.0)
        result = objective.reference_corrected_dpo(winner, loser, beta=0.5)

        # Delta = (4 - 1) - (3 - 2) = 2.
        self.assertEqual(float(result.current_margin.item()), 3.0)
        self.assertEqual(float(result.reference_margin.item()), 1.0)
        self.assertEqual(float(result.delta.item()), 2.0)
        self.assertAlmostEqual(
            float(result.loss.item()),
            math.log1p(math.exp(-1.0)),
            places=6,
        )
        self.assertFalse(result.reference_margin.requires_grad)
        result.loss.backward()
        self.assertGreater(float(winner.current.grad.item()), 0.0)
        self.assertLess(float(loser.current.grad.item()), 0.0)
        self.assertIsNone(winner.reference.grad)
        self.assertIsNone(loser.reference.grad)

    def test_reference_revision_changes_delta_not_current_margin(self) -> None:
        winner = self._energies(1.0, 0.5)
        loser = self._energies(2.0, 3.5)
        result = objective.reference_corrected_dpo(winner, loser, beta=1.0)
        self.assertEqual(float(result.current_margin.item()), 1.0)
        self.assertEqual(float(result.reference_margin.item()), 3.0)
        self.assertEqual(float(result.delta.item()), -2.0)

    def test_rejects_nonpositive_beta_and_reference_graph(self) -> None:
        winner = self._energies(1.0, 2.0)
        loser = self._energies(4.0, 3.0)
        for beta in (0.0, -1.0, float("nan"), True):
            with self.subTest(beta=beta):
                with self.assertRaisesRegex(
                    objective.DCLRPreferenceObjectiveError, "beta"
                ):
                    objective.reference_corrected_dpo(
                        winner, loser, beta=beta
                    )
        leaked = objective.CandidatePolicyEnergies(
            current=torch.tensor(1.0, requires_grad=True),
            reference=torch.tensor(2.0, requires_grad=True),
        )
        with self.assertRaisesRegex(
            objective.DCLRPreferenceObjectiveError, "reference must be detached"
        ):
            objective.reference_corrected_dpo(leaked, loser, beta=1.0)


@unittest.skipIf(torch is None, "torch is unavailable")
class OneSidedRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.winner_action = {"actor": True, "order": True, "contact": True}
        self.winner_preservation = {"identity": True, "camera": True}

    def _route(self, pair_type: str, **overrides: object):
        values: dict[str, object] = {
            "winner_action_axis_pass": self.winner_action,
            "winner_preservation_axis_pass": self.winner_preservation,
            "loser_action_axis_pass": {
                "actor": True,
                "order": False,
                "contact": True,
            },
            "loser_preservation_axis_pass": {
                "identity": True,
                "camera": True,
            },
        }
        values.update(overrides)
        return objective.route_one_sided_nearmiss(pair_type, **values)

    def test_action_nearmiss_routes_action_adapter_only(self) -> None:
        route = self._route(objective.ACTION_NEARMISS)
        self.assertEqual(route.active_adapter, objective.ACTION_ADAPTER)
        self.assertTrue(route.action_adapter_trainable)
        self.assertFalse(route.identity_adapter_trainable)
        self.assertEqual(route.loser_failed_axis, "order")

        action_parameters = [torch.nn.Parameter(torch.ones(2))]
        identity_parameters = [
            torch.nn.Parameter(torch.ones(2), requires_grad=False)
        ]
        self.assertIs(
            objective.validate_adapter_trainability(
                route,
                action_adapter_parameters=action_parameters,
                identity_adapter_parameters=identity_parameters,
            ),
            route,
        )

    def test_preservation_nearmiss_routes_identity_adapter_only(self) -> None:
        route = self._route(
            objective.PRESERVATION_NEARMISS,
            loser_action_axis_pass={
                "actor": True,
                "order": True,
                "contact": True,
            },
            loser_preservation_axis_pass={
                "identity": False,
                "camera": True,
            },
        )
        self.assertEqual(route.active_adapter, objective.IDENTITY_ADAPTER)
        self.assertFalse(route.action_adapter_trainable)
        self.assertTrue(route.identity_adapter_trainable)
        self.assertEqual(route.loser_failed_axis, "identity")

        action_parameters = [
            torch.nn.Parameter(torch.ones(2), requires_grad=False)
        ]
        identity_parameters = [torch.nn.Parameter(torch.ones(2))]
        objective.validate_adapter_trainability(
            route,
            action_adapter_parameters=action_parameters,
            identity_adapter_parameters=identity_parameters,
        )

    def test_fail_closed_on_mislabeled_ambiguous_or_invalid_pairs(self) -> None:
        with self.assertRaisesRegex(
            objective.DCLRPreferenceObjectiveError, "winner must pass"
        ):
            self._route(
                objective.ACTION_NEARMISS,
                winner_action_axis_pass={
                    "actor": False,
                    "order": True,
                    "contact": True,
                },
            )
        with self.assertRaisesRegex(
            objective.DCLRPreferenceObjectiveError, "exactly one action axis"
        ):
            self._route(
                objective.ACTION_NEARMISS,
                loser_action_axis_pass={
                    "actor": False,
                    "order": False,
                    "contact": True,
                },
            )
        with self.assertRaisesRegex(
            objective.DCLRPreferenceObjectiveError, "exactly one action axis"
        ):
            self._route(
                objective.ACTION_NEARMISS,
                loser_preservation_axis_pass={
                    "identity": False,
                    "camera": True,
                },
            )
        with self.assertRaisesRegex(
            objective.DCLRPreferenceObjectiveError,
            "exactly one preservation axis",
        ):
            self._route(objective.PRESERVATION_NEARMISS)
        with self.assertRaisesRegex(
            objective.DCLRPreferenceObjectiveError, "unsupported"
        ):
            self._route("joint_nearmiss")

    def test_trainability_check_rejects_cross_adapter_gradient_leak(self) -> None:
        route = self._route(objective.ACTION_NEARMISS)
        action_parameter = torch.nn.Parameter(torch.ones(2))
        identity_parameter = torch.nn.Parameter(torch.ones(2))
        with self.assertRaisesRegex(
            objective.DCLRPreferenceObjectiveError,
            "identity adapter requires_grad",
        ):
            objective.validate_adapter_trainability(
                route,
                action_adapter_parameters=[action_parameter],
                identity_adapter_parameters=[identity_parameter],
            )
        identity_parameter.requires_grad_(False)
        action_parameter.requires_grad_(False)
        with self.assertRaisesRegex(
            objective.DCLRPreferenceObjectiveError,
            "action adapter requires_grad",
        ):
            objective.validate_adapter_trainability(
                route,
                action_adapter_parameters=[action_parameter],
                identity_adapter_parameters=[identity_parameter],
            )
        action_parameter.requires_grad_(True)
        with self.assertRaisesRegex(
            objective.DCLRPreferenceObjectiveError, "overlap"
        ):
            objective.validate_adapter_trainability(
                route,
                action_adapter_parameters=[action_parameter],
                identity_adapter_parameters=[action_parameter],
            )

    def test_combined_api_routes_before_returning_loss(self) -> None:
        winner = objective.CandidatePolicyEnergies(
            current=torch.tensor(1.0, requires_grad=True),
            reference=torch.tensor(1.5),
        )
        loser = objective.CandidatePolicyEnergies(
            current=torch.tensor(2.0, requires_grad=True),
            reference=torch.tensor(1.75),
        )
        routed = objective.compute_routed_reference_corrected_dpo(
            winner,
            loser,
            beta=0.25,
            pair_type=objective.ACTION_NEARMISS,
            winner_action_axis_pass=self.winner_action,
            winner_preservation_axis_pass=self.winner_preservation,
            loser_action_axis_pass={
                "actor": True,
                "order": False,
                "contact": True,
            },
            loser_preservation_axis_pass={
                "identity": True,
                "camera": True,
            },
        )
        self.assertEqual(routed.route.active_adapter, objective.ACTION_ADAPTER)
        self.assertTrue(routed.objective.loss.requires_grad)


if __name__ == "__main__":
    unittest.main()
