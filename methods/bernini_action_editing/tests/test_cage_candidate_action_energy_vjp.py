from __future__ import annotations

import inspect
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
    import cage_candidate_action_energy_vjp as cage  # noqa: E402
    import mace_candidate_action_energy as mace  # noqa: E402
else:  # pragma: no cover - dependency-light environments
    cage = None
    mace = None


def _prompts():
    return {
        branch: f"registered CAGE prompt for {branch}"
        for branch in mace.BRANCH_ORDER
    }


def _fixture():
    candidate = torch.tensor(
        [[0.20, -0.40, 0.70, -0.15]], dtype=torch.float32
    )
    coordinates = (
        cage.EnergyCoordinate(
            "sigma-low",
            0.25,
            torch.tensor([[0.90, -0.10, -0.60, 0.35]], dtype=torch.float32),
        ),
        cage.EnergyCoordinate(
            "sigma-high",
            0.70,
            torch.tensor([[-0.20, 0.80, 0.30, -0.75]], dtype=torch.float32),
        ),
    )
    return candidate, coordinates, _prompts()


if torch is not None:

    class ToyFrozenBridge:
        """Small nonlinear denoiser with a differentiable input path."""

        def __init__(self):
            self.calls = []

        @staticmethod
        def formula(x_sigma, sigma, branch):
            index = mace.BRANCH_ORDER.index(branch)
            scale = 0.22 + 0.035 * index
            quadratic = 0.045 + 0.002 * (index % 3)
            bias = -0.18 + 0.047 * index
            sigma_term = (0.13 - 0.009 * index) * float(sigma)
            return (
                scale * x_sigma
                + quadratic * x_sigma.square()
                + bias
                + sigma_term
            )

        def __call__(self, request):
            self.calls.append(
                {
                    "mode": request.mode,
                    "coordinate_index": request.coordinate_index,
                    "branch": request.branch,
                    "grad_enabled": torch.is_grad_enabled(),
                    "input_requires_grad": request.x_sigma.requires_grad,
                    "input_id": id(request.x_sigma),
                }
            )
            return self.formula(
                request.x_sigma, request.sigma, request.branch
            )


    class DirectOnlyBridge:
        """Graph-connected constant field: J_v is exactly zero."""

        def __init__(self):
            self.calls = []

        def __call__(self, request):
            self.calls.append((request.mode, request.branch))
            index = mace.BRANCH_ORDER.index(request.branch)
            value = -0.55 + 0.09 * index + 0.04 * float(request.sigma)
            return request.x_sigma * 0.0 + value


    class ExactTargetBridge:
        """Every branch equals epsilon-y and therefore has zero residual."""

        def __init__(self, candidate, coordinates):
            self.candidate = candidate
            self.coordinates = tuple(coordinates)

        def __call__(self, request):
            target = (
                self.coordinates[request.coordinate_index].epsilon
                - self.candidate
            )
            return request.x_sigma * 0.0 + target


@unittest.skipIf(torch is None, "torch is unavailable")
class CAGEBranchScanAndReplayTests(unittest.TestCase):
    def test_scans_all_sigma_branches_no_grad_then_replays_only_two(self) -> None:
        candidate, coordinates, prompts = _fixture()
        bridge = ToyFrozenBridge()
        result = cage.compute_candidate_action_energy_vjp(
            candidate, coordinates, prompts, bridge
        )

        scan_calls = [
            row for row in bridge.calls if row["mode"] == cage.SCAN_MODE
        ]
        replay_calls = [
            row for row in bridge.calls if row["mode"] == cage.REPLAY_MODE
        ]
        self.assertEqual(
            len(scan_calls), len(coordinates) * len(mace.BRANCH_ORDER)
        )
        self.assertTrue(all(not row["grad_enabled"] for row in scan_calls))
        self.assertTrue(
            all(not row["input_requires_grad"] for row in scan_calls)
        )
        self.assertEqual(len(replay_calls), 2)
        self.assertEqual(
            [row["branch"] for row in replay_calls],
            [mace.ACTION_BRANCH, result.scan.selected_negative_branch],
        )
        self.assertTrue(all(row["grad_enabled"] for row in replay_calls))
        self.assertTrue(
            all(row["input_requires_grad"] for row in replay_calls)
        )
        self.assertEqual(
            {row["coordinate_index"] for row in replay_calls},
            {result.scan.selected_coordinate_index},
        )
        expected_flat = int(
            torch.argmin(
                result.scan.negative_log_energy_ratios.reshape(-1)
            ).item()
        )
        negative_count = len(mace.HARD_NEGATIVE_BRANCHES)
        self.assertEqual(
            result.scan.selected_coordinate_index,
            expected_flat // negative_count,
        )
        self.assertEqual(
            result.scan.selected_negative_index,
            expected_flat % negative_count,
        )
        self.assertTrue(result.scan.selection_detached)
        self.assertEqual(
            tuple(result.scan.branch_energies.shape),
            (len(coordinates), len(mace.BRANCH_ORDER)),
        )
        self.assertTrue(result.finite)
        self.assertTrue(result.nonzero)
        self.assertGreater(result.gradient_norm, 0.0)

    def test_two_replay_vjp_matches_full_autograd(self) -> None:
        candidate, coordinates, prompts = _fixture()
        config = cage.EnergyVJPConfig(
            target_margin=0.17,
            temperature=0.73,
        )
        bridge = ToyFrozenBridge()
        result = cage.compute_candidate_action_energy_vjp(
            candidate,
            coordinates,
            prompts,
            bridge,
            config=config,
        )

        coordinate = coordinates[result.scan.selected_coordinate_index]
        y = candidate.clone().requires_grad_(True)
        sigma = float(coordinate.sigma)
        q = (1.0 - sigma) * y + sigma * coordinate.epsilon
        target = coordinate.epsilon - y
        action = ToyFrozenBridge.formula(q, sigma, mace.ACTION_BRANCH)
        negative = ToyFrozenBridge.formula(
            q, sigma, result.scan.selected_negative_branch
        )
        action_energy = (action - target).square().mean()
        negative_energy = (negative - target).square().mean()
        margin = torch.log(
            negative_energy + config.energy_epsilon
        ) - torch.log(action_energy + config.energy_epsilon)
        loss = torch.nn.functional.softplus(
            (config.target_margin - margin) / config.temperature
        )
        expected = torch.autograd.grad(loss, y)[0]

        self.assertTrue(
            torch.allclose(result.gradient, expected, rtol=2.0e-5, atol=2.0e-6)
        )
        recomposed = (
            result.direct_flow_target_gradient
            + (1.0 - sigma)
            * (result.action_input_vjp + result.negative_input_vjp)
        )
        self.assertTrue(torch.equal(result.gradient, recomposed))

    def test_flow_target_direct_gradient_is_present_when_teacher_jacobian_is_zero(self) -> None:
        candidate, coordinates, prompts = _fixture()
        bridge = DirectOnlyBridge()
        result = cage.compute_candidate_action_energy_vjp(
            candidate, coordinates, prompts, bridge
        )

        self.assertTrue(torch.equal(result.action_input_vjp, torch.zeros_like(candidate)))
        self.assertTrue(
            torch.equal(result.negative_input_vjp, torch.zeros_like(candidate))
        )
        self.assertTrue(
            torch.equal(result.gradient, result.direct_flow_target_gradient)
        )
        self.assertGreater(
            float(torch.linalg.vector_norm(result.direct_flow_target_gradient)),
            0.0,
        )


@unittest.skipIf(torch is None, "torch is unavailable")
class CAGEGradientAuditTests(unittest.TestCase):
    def test_central_difference_verifies_descent_sign_and_magnitude(self) -> None:
        candidate, coordinates, prompts = _fixture()
        config = cage.EnergyVJPConfig(
            target_margin=0.11,
            temperature=0.91,
            finite_difference_rtol=1.0e-2,
            finite_difference_atol=5.0e-5,
        )
        bridge = ToyFrozenBridge()
        result = cage.compute_candidate_action_energy_vjp(
            candidate,
            coordinates,
            prompts,
            bridge,
            config=config,
        )
        audit = cage.audit_candidate_action_energy_vjp_finite_difference(
            result,
            candidate,
            coordinates,
            prompts,
            bridge,
            step=1.0e-3,
            config=config,
        )

        self.assertTrue(audit.passed)
        self.assertTrue(audit.selection_stable)
        self.assertTrue(audit.descent_sign_passed)
        self.assertTrue(audit.magnitude_agreement_passed)
        self.assertLess(audit.analytic_directional_derivative, 0.0)
        self.assertLess(audit.numerical_directional_derivative, 0.0)
        self.assertLess(audit.plus_loss, audit.minus_loss)
        fd_calls = [
            row
            for row in bridge.calls
            if row["mode"]
            in {
                cage.FINITE_DIFFERENCE_PLUS_MODE,
                cage.FINITE_DIFFERENCE_MINUS_MODE,
            }
        ]
        self.assertEqual(
            len(fd_calls), 2 * len(coordinates) * len(mace.BRANCH_ORDER)
        )
        self.assertTrue(all(not row["grad_enabled"] for row in fd_calls))

    def test_zero_residual_blocks_optimizer_gradient(self) -> None:
        candidate, coordinates, prompts = _fixture()
        bridge = ExactTargetBridge(candidate, coordinates)
        with self.assertRaisesRegex(
            cage.CAGECandidateActionEnergyVJPError,
            "zero or below",
        ):
            cage.compute_candidate_action_energy_vjp(
                candidate, coordinates, prompts, bridge
            )

    def test_multi_sigma_and_public_api_fail_closed(self) -> None:
        candidate, coordinates, prompts = _fixture()
        bridge = ToyFrozenBridge()
        with self.assertRaisesRegex(
            cage.CAGECandidateActionEnergyVJPError,
            "at least two sigma",
        ):
            cage.compute_candidate_action_energy_vjp(
                candidate, coordinates[:1], prompts, bridge
            )
        bad = candidate.clone().requires_grad_(True)
        with self.assertRaisesRegex(
            cage.CAGECandidateActionEnergyVJPError,
            "detached finite FP32",
        ):
            cage.compute_candidate_action_energy_vjp(
                bad, coordinates, prompts, bridge
            )

        receipt = cage.contract_receipt()
        self.assertFalse(receipt["velocity_norm_score"])
        self.assertFalse(receipt["pure_t2v_media_consumed"])
        self.assertEqual(
            receipt["replay"],
            "selected_action_and_selected_hardest_negative_only",
        )
        forbidden = {
            "source_video",
            "proposal_video",
            "target_video",
            "mask",
            "flow",
            "pose",
            "track",
            "trajectory",
        }
        for function in (
            cage.scan_candidate_action_energy,
            cage.compute_candidate_action_energy_vjp,
            cage.audit_candidate_action_energy_vjp_finite_difference,
        ):
            self.assertTrue(
                set(inspect.signature(function).parameters).isdisjoint(forbidden)
            )


if __name__ == "__main__":
    unittest.main()
