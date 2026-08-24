from __future__ import annotations

import ast
from dataclasses import fields as dataclass_fields
import math
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import motion_commutator as commutator
import relational_commutator_objective as objective


class RelationalCommutatorObjectivePureContractTests(unittest.TestCase):
    def test_exact_seven_branch_and_two_graph_forward_contract(self) -> None:
        self.assertEqual(len(objective.FORWARD_BRANCH_ORDER), 7)
        self.assertEqual(
            objective.FORWARD_BRANCH_ORDER,
            (
                "frozen_editor_negative_full_source",
                "frozen_editor_noop_full_source",
                "frozen_editor_action_full_source",
                "adapted_editor_noop_full_source",
                "adapted_editor_action_full_source",
                "frozen_generator_negative_target_only",
                "frozen_generator_action_target_only",
            ),
        )
        self.assertEqual(
            objective.GRAPH_BRANCHES,
            (
                "adapted_editor_noop_full_source",
                "adapted_editor_action_full_source",
            ),
        )
        self.assertEqual(len(objective.GRAPH_FREE_MODEL_BRANCHES), 5)
        self.assertEqual(
            tuple(item.name for item in dataclass_fields(objective.SevenBranchCleanFields)),
            (
                "frozen_editor_negative",
                "frozen_editor_noop",
                "frozen_editor_action",
                "adapted_editor_noop",
                "adapted_editor_action",
                "frozen_generator_negative",
                "frozen_generator_action",
                "source_clean",
                "target_clean",
            ),
        )

    def test_target_only_is_default_and_contract_states_metric_limit(self) -> None:
        config = objective.RelationalCommutatorLossConfig()
        config.validate()
        contract = objective.immutable_objective_contract(config)
        self.assertTrue(contract["target_only_default"])
        self.assertTrue(contract["relational_auxiliary_fail_closed_when_enabled"])
        self.assertFalse(contract["cross_video_pointwise_coordinate_cosine"])
        self.assertIn("not proof", contract["relational_metric_limitation"])
        self.assertEqual(contract["model_forwards_per_candidate"], 7)
        self.assertEqual(contract["graph_forwards_per_candidate"], 2)
        self.assertIn("paired_target_video", contract["forbidden_inference_conditions"])
        self.assertIn("generator_branch", contract["forbidden_inference_conditions"])

    def test_configuration_fails_closed(self) -> None:
        invalid = (
            {"raw_target_weight": 0.0},
            {"raw_target_weight": -1.0},
            {"noop_preservation_weight": 0.0},
            {"residual_temporal_jitter_weight": -0.1},
            {"relational_auxiliary_weight": -0.1},
            {"charbonnier_epsilon": 0.0},
            {"normalization_floor": float("nan")},
            {"normalization_floor": True},
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(
                    objective.RelationalCommutatorObjectiveError
                ):
                    objective.RelationalCommutatorLossConfig(**values).validate()

    def test_torch_is_lazy_and_no_raw_cross_video_cosine_implementation(self) -> None:
        source = Path(objective.__file__).read_text(encoding="utf-8")
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
        self.assertNotIn("cosine_similarity(", source)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is unavailable")
class RelationalCommutatorObjectiveTensorTests(unittest.TestCase):
    @staticmethod
    def _phase() -> "torch.Tensor":
        return torch.arange(21, dtype=torch.float32).reshape(1, 21, 1, 1)

    @classmethod
    def _fields(
        cls,
        *,
        collapsed_teacher: bool,
        adapter_action_scale: float = 0.0,
        adapter_noop_scale: float = 0.0,
        target_scale: float = 2.0,
    ) -> objective.SevenBranchCleanFields:
        phase = cls._phase()
        coordinate = torch.tensor(
            [1.0, -0.7, 0.4, 1.3], dtype=torch.float32
        ).reshape(1, 1, 2, 2)
        source = 0.15 * phase * coordinate
        target_motion = target_scale * phase * coordinate
        target = source + target_motion

        frozen_negative = torch.full_like(source, -0.3)
        frozen_noop = source + 0.25
        frozen_action = frozen_noop + 0.4 * phase * coordinate
        adapted_noop = (
            frozen_noop + adapter_noop_scale * phase * torch.ones_like(coordinate)
        ).detach().requires_grad_(True)
        adapted_action = (
            frozen_action + adapter_action_scale * phase * coordinate
        ).detach().requires_grad_(True)

        generator_negative = torch.full_like(source, 0.1)
        generator_action = (
            generator_negative.clone()
            if collapsed_teacher
            else generator_negative + target_motion
        )
        return objective.SevenBranchCleanFields(
            frozen_editor_negative=frozen_negative,
            frozen_editor_noop=frozen_noop,
            frozen_editor_action=frozen_action,
            adapted_editor_noop=adapted_noop,
            adapted_editor_action=adapted_action,
            frozen_generator_negative=generator_negative,
            frozen_generator_action=generator_action,
            source_clean=source,
            target_clean=target,
        )

    def test_ineligible_teacher_does_not_block_target_only_gradients(self) -> None:
        fields = self._fields(collapsed_teacher=True)
        result = objective.compute_relational_commutator_objective(
            fields,
            step_index=0,
        )
        self.assertFalse(
            bool(result.diagnostics.teacher_eligibility.eligible.all().item())
        )
        self.assertFalse(result.diagnostics.relational_auxiliary_enabled)
        self.assertFalse(result.diagnostics.relational_auxiliary_active)
        self.assertTrue(result.total.requires_grad)
        result.total.backward()
        self.assertIsNotNone(fields.adapted_editor_action.grad)
        self.assertIsNotNone(fields.adapted_editor_noop.grad)
        self.assertGreater(float(fields.adapted_editor_action.grad.abs().sum()), 0.0)
        self.assertGreater(float(fields.adapted_editor_noop.grad.abs().sum()), 0.0)
        for frozen in (
            fields.frozen_editor_negative,
            fields.frozen_editor_noop,
            fields.frozen_editor_action,
            fields.frozen_generator_negative,
            fields.frozen_generator_action,
            fields.source_clean,
            fields.target_clean,
        ):
            self.assertFalse(frozen.requires_grad)
            self.assertIsNone(frozen.grad)

    def test_enabled_relational_auxiliary_fails_closed_on_same_teacher(self) -> None:
        fields = self._fields(collapsed_teacher=True)
        config = objective.RelationalCommutatorLossConfig(
            relational_auxiliary_weight=0.1
        )
        with self.assertRaises(objective.RelationalAuxiliaryIneligible):
            objective.compute_relational_commutator_objective(
                fields,
                step_index=0,
                config=config,
            )
        self.assertIsNone(fields.adapted_editor_action.grad)
        self.assertIsNone(fields.adapted_editor_noop.grad)

    def test_eligible_teacher_can_supply_only_invariant_relational_term(self) -> None:
        fields = self._fields(
            collapsed_teacher=False,
            adapter_action_scale=0.2,
        )
        config = objective.RelationalCommutatorLossConfig(
            relational_auxiliary_weight=0.1
        )
        result = objective.compute_relational_commutator_objective(
            fields,
            step_index=0,
            config=config,
        )
        self.assertTrue(
            bool(result.diagnostics.teacher_eligibility.eligible.all().item())
        )
        self.assertTrue(result.diagnostics.relational_auxiliary_enabled)
        self.assertTrue(result.diagnostics.relational_auxiliary_active)
        self.assertTrue(bool(torch.isfinite(result.relational_auxiliary)))
        result.total.backward()
        self.assertGreater(float(fields.adapted_editor_action.grad.abs().sum()), 0.0)
        self.assertGreater(float(fields.adapted_editor_noop.grad.abs().sum()), 0.0)

    def test_raw_target_is_supervised_while_bound_is_detached_diagnostic(self) -> None:
        fields = self._fields(
            collapsed_teacher=True,
            adapter_action_scale=100.0,
            target_scale=80.0,
        )
        config = objective.RelationalCommutatorLossConfig(
            commutator_config=commutator.MotionCommutatorConfig(
                max_correction_increment_ratio=0.10,
                correction_increment_rms_floor=0.0,
            )
        )
        result = objective.compute_relational_commutator_objective(
            fields,
            step_index=0,
            config=config,
        )
        for built in (
            result.diagnostics.deployment_projection,
            result.diagnostics.target_projection,
        ):
            bounded = built.diagnostics.bounded_correction_increment_rms
            cap = built.diagnostics.correction_increment_rms_cap
            self.assertTrue(bool((bounded <= cap + 2.0e-6).all()))
            self.assertTrue(bool((built.diagnostics.bound_scale[:, 1:] < 1.0).all()))
        self.assertEqual(result.rho, 1.0)
        self.assertFalse(
            result.diagnostics.deployment_projection.bounded_commutator_correction.requires_grad
        )
        self.assertFalse(
            result.diagnostics.target_projection.bounded_commutator_correction.requires_grad
        )
        receipt = objective.detached_receipt_diagnostics(result)
        self.assertEqual(receipt["model_forwards_per_candidate"], 7)
        self.assertEqual(receipt["graph_forwards_per_candidate"], 2)
        self.assertTrue(receipt["target_projection_matches_inference_operator"])
        self.assertFalse(receipt["hard_bound_contributes_target_gradient"])
        self.assertLessEqual(
            receipt["commutator_bound"]["max_postprojection_violation"],
            2.0e-6,
        )
        self.assertFalse(
            receipt["relational_auxiliary"][
                "cross_video_pointwise_coordinate_cosine"
            ]
        )
        # Saturating the detached deployment projection must not clip the raw
        # target gradient used by training.
        result.total.backward()
        self.assertGreater(float(fields.adapted_editor_action.grad.abs().sum()), 0.0)
        self.assertGreater(float(fields.adapted_editor_noop.grad.abs().sum()), 0.0)

    def test_required_kappa_statistics_are_exact_away_from_zero(self) -> None:
        phase = self._phase()
        ones = torch.ones(1, 1, 2, 2, dtype=torch.float32)
        source = torch.zeros(1, 21, 2, 2, dtype=torch.float32)
        frozen_noop = torch.ones_like(source)
        frozen_action = frozen_noop + phase * ones
        zero = torch.zeros_like(source)
        fields = objective.SevenBranchCleanFields(
            frozen_editor_negative=zero,
            frozen_editor_noop=frozen_noop,
            frozen_editor_action=frozen_action,
            adapted_editor_noop=frozen_noop.clone().requires_grad_(True),
            adapted_editor_action=frozen_action.clone().requires_grad_(True),
            frozen_generator_negative=zero,
            frozen_generator_action=zero.clone(),
            source_clean=source,
            target_clean=3.0 * phase * ones,
        )
        config = objective.RelationalCommutatorLossConfig(
            commutator_config=commutator.MotionCommutatorConfig(
                max_correction_increment_ratio=0.25,
                correction_increment_rms_floor=0.01,
                temporal_smoothing=False,
            )
        )
        result = objective.compute_relational_commutator_objective(
            fields, step_index=0, config=config
        )
        bound = objective.detached_receipt_diagnostics(result)["commutator_bound"]
        self.assertAlmostEqual(bound["target_bound_mean_scale_active"], 0.125)
        self.assertAlmostEqual(bound["target_mean_scale_active"], 0.125)
        self.assertEqual(bound["floor_dominated_fraction_active"], 0.0)
        self.assertEqual(bound["target_floor_sufficient_fraction_active"], 0.0)
        self.assertAlmostEqual(bound["target_required_kappa_median"], 2.0)
        self.assertAlmostEqual(bound["target_required_kappa_p90"], 2.0)
        self.assertAlmostEqual(bound["target_required_kappa_max"], 2.0)
        self.assertEqual(
            bound["target_required_kappa_near_zero_proxy_fraction_active"],
            0.0,
        )
        self.assertEqual(
            bound["frozen_increment_near_zero_fraction_active"], 0.0
        )
        self.assertEqual(
            bound[
                "target_required_kappa_exact_zero_unreachable_fraction_active"
            ],
            0.0,
        )

    def test_near_zero_frozen_increment_uses_finite_explicit_proxy(self) -> None:
        phase = self._phase()
        ones = torch.ones(1, 1, 2, 2, dtype=torch.float32)
        source = torch.zeros(1, 21, 2, 2, dtype=torch.float32)
        frozen_noop = torch.ones_like(source)
        zero = torch.zeros_like(source)
        fields = objective.SevenBranchCleanFields(
            frozen_editor_negative=zero,
            frozen_editor_noop=frozen_noop,
            frozen_editor_action=frozen_noop.clone(),
            adapted_editor_noop=frozen_noop.clone().requires_grad_(True),
            adapted_editor_action=frozen_noop.clone().requires_grad_(True),
            frozen_generator_negative=zero,
            frozen_generator_action=zero.clone(),
            source_clean=source,
            target_clean=phase * ones,
        )
        config = objective.RelationalCommutatorLossConfig(
            commutator_config=commutator.MotionCommutatorConfig(
                max_correction_increment_ratio=0.25,
                correction_increment_rms_floor=0.05,
                temporal_smoothing=False,
            )
        )
        result = objective.compute_relational_commutator_objective(
            fields, step_index=0, config=config
        )
        bound = objective.detached_receipt_diagnostics(result)["commutator_bound"]
        self.assertEqual(bound["floor_dominated_fraction_active"], 1.0)
        self.assertEqual(
            bound["target_required_kappa_near_zero_proxy_fraction_active"],
            1.0,
        )
        self.assertEqual(
            bound["frozen_increment_near_zero_fraction_active"], 1.0
        )
        self.assertEqual(
            bound[
                "target_required_kappa_exact_zero_unreachable_fraction_active"
            ],
            1.0,
        )
        self.assertIn("lower-bound proxy", bound["target_required_kappa_near_zero_handling"])
        self.assertIn("unreachable", bound["target_required_kappa_near_zero_handling"])
        for name in (
            "target_required_kappa_median",
            "target_required_kappa_p90",
            "target_required_kappa_max",
        ):
            self.assertTrue(math.isfinite(bound[name]))
            self.assertGreater(bound[name], 0.0)
        self.assertLessEqual(
            bound["target_required_kappa_median"],
            bound["target_required_kappa_p90"],
        )
        self.assertLessEqual(
            bound["target_required_kappa_p90"],
            bound["target_required_kappa_max"],
        )

    def test_positive_near_zero_frozen_increment_uses_exact_float64_ratio(self) -> None:
        phase = self._phase()
        ones = torch.ones(1, 1, 2, 2, dtype=torch.float32)
        zero = torch.zeros(1, 21, 2, 2, dtype=torch.float32)
        tiny_frozen_action = 1.0e-8 * phase * ones
        fields = objective.SevenBranchCleanFields(
            frozen_editor_negative=zero,
            frozen_editor_noop=zero.clone(),
            frozen_editor_action=tiny_frozen_action,
            adapted_editor_noop=zero.clone().requires_grad_(True),
            adapted_editor_action=tiny_frozen_action.clone().requires_grad_(True),
            frozen_generator_negative=zero,
            frozen_generator_action=zero.clone(),
            source_clean=zero,
            target_clean=phase * ones,
        )
        config = objective.RelationalCommutatorLossConfig(
            commutator_config=commutator.MotionCommutatorConfig(
                max_correction_increment_ratio=0.25,
                correction_increment_rms_floor=0.01,
                temporal_smoothing=False,
            )
        )
        result = objective.compute_relational_commutator_objective(
            fields, step_index=0, config=config
        )
        bound = objective.detached_receipt_diagnostics(result)["commutator_bound"]
        self.assertEqual(bound["frozen_increment_near_zero_fraction_active"], 1.0)
        self.assertEqual(
            bound["target_required_kappa_near_zero_proxy_fraction_active"],
            0.0,
        )
        self.assertEqual(
            bound[
                "target_required_kappa_exact_zero_unreachable_fraction_active"
            ],
            0.0,
        )
        self.assertGreater(bound["target_required_kappa_median"], 5.0e7)
        self.assertLess(bound["target_required_kappa_median"], 2.0e8)
        self.assertIn(
            "exact float64",
            bound["target_required_kappa_near_zero_handling"],
        )

    def test_common_adapter_drift_cancels_but_noop_anchor_penalizes_it(self) -> None:
        fields = self._fields(
            collapsed_teacher=True,
            adapter_action_scale=0.0,
            adapter_noop_scale=0.0,
            target_scale=0.4,
        )
        common = 0.8 * self._phase() * torch.ones(1, 1, 2, 2)
        fields = objective.SevenBranchCleanFields(
            frozen_editor_negative=fields.frozen_editor_negative,
            frozen_editor_noop=fields.frozen_editor_noop,
            frozen_editor_action=fields.frozen_editor_action,
            adapted_editor_noop=(
                fields.frozen_editor_noop + common
            ).detach().requires_grad_(True),
            adapted_editor_action=(
                fields.frozen_editor_action + common
            ).detach().requires_grad_(True),
            frozen_generator_negative=fields.frozen_generator_negative,
            frozen_generator_action=fields.frozen_generator_action,
            source_clean=fields.source_clean,
            target_clean=fields.target_clean,
        )
        result = objective.compute_relational_commutator_objective(
            fields,
            step_index=0,
        )
        self.assertTrue(
            torch.allclose(
                result.diagnostics.raw_commutator.raw_commutator_correction,
                torch.zeros_like(
                    result.diagnostics.raw_commutator.raw_commutator_correction
                ),
                atol=2.0e-6,
                rtol=0.0,
            )
        )
        self.assertGreater(float(result.noop_preservation), 0.0)

    def test_rho_zero_is_exact_frozen_replay_with_no_action_gradient(self) -> None:
        self.assertEqual(commutator.release_rho(32), 0.0)
        fields = self._fields(
            collapsed_teacher=True,
            adapter_action_scale=3.0,
            adapter_noop_scale=0.4,
        )
        config = objective.RelationalCommutatorLossConfig(
            relational_auxiliary_weight=0.5
        )
        # Ineligible relational evidence does not reject a step where rho=0,
        # because that teacher term is mathematically inactive.
        result = objective.compute_relational_commutator_objective(
            fields,
            step_index=32,
            config=config,
        )
        self.assertEqual(result.rho, 0.0)
        self.assertIs(
            result.diagnostics.predicted_execution.executed_direction,
            result.diagnostics.predicted_execution.frozen_official_direction,
        )
        self.assertFalse(result.diagnostics.relational_auxiliary_active)
        self.assertEqual(float(result.raw_target), 0.0)
        self.assertEqual(float(result.residual_temporal_jitter), 0.0)
        result.total.backward()
        self.assertIsNone(fields.adapted_editor_action.grad)
        self.assertIsNotNone(fields.adapted_editor_noop.grad)
        self.assertGreater(float(fields.adapted_editor_noop.grad.abs().sum()), 0.0)

    def test_graph_contract_rejects_teacher_or_target_leakage(self) -> None:
        base = self._fields(collapsed_teacher=True)
        leaking = objective.SevenBranchCleanFields(
            frozen_editor_negative=base.frozen_editor_negative,
            frozen_editor_noop=base.frozen_editor_noop,
            frozen_editor_action=base.frozen_editor_action,
            adapted_editor_noop=base.adapted_editor_noop,
            adapted_editor_action=base.adapted_editor_action,
            frozen_generator_negative=base.frozen_generator_negative,
            frozen_generator_action=(
                base.frozen_generator_action.clone().requires_grad_(True)
            ),
            source_clean=base.source_clean,
            target_clean=base.target_clean,
        )
        with self.assertRaises(objective.RelationalCommutatorObjectiveError):
            objective.compute_relational_commutator_objective(
                leaking, step_index=0
            )


if __name__ == "__main__":
    unittest.main()
