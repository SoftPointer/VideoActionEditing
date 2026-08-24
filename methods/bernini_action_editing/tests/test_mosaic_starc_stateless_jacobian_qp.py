from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch

    import mosaic_starc_stateless_jacobian_qp as qp

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    qp = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@unittest.skipUnless(_TORCH_AVAILABLE, "torch is required")
class MosaicStarcStatelessJacobianQPTests(unittest.TestCase):
    checkpoint_digest = _sha("frozen-bernini-checkpoint")

    @property
    def tensor_numel(self) -> int:
        return qp.HIDDEN_SIZE * qp.LORA_RANK

    @property
    def human_coordinate(self) -> int:
        return self.tensor_numel

    def parameters(self) -> list[tuple[str, "torch.Tensor"]]:
        result = []
        for name in qp.CANONICAL_PARAMETER_NAMES:
            result.append(
                (
                    name,
                    torch.zeros(
                        qp.CANONICAL_B_SHAPE,
                        dtype=torch.float32,
                        requires_grad=True,
                    ),
                )
            )
        return result

    def vector(self, layout, values: dict[int, float]) -> "torch.Tensor":
        result = torch.zeros(layout.total_numel, dtype=torch.float32)
        for index, value in values.items():
            result[index] = value
        return result

    def action_row(
        self,
        layout,
        *,
        arm: str,
        values: dict[int, float],
        lower: float = 0.1,
        suffix: str = "0",
    ):
        return qp.ActionConstraintRow(
            row_id=f"{arm}-action-{suffix}",
            actor_family=arm,
            values=self.vector(layout, values),
            minimum_dot=lower,
            layout_digest=layout.layout_digest,
            checkpoint_content_receipt_digest=self.checkpoint_digest,
            parameter_state_sha256=layout.parameter_state_sha256,
            gradient_computation_receipt_digest=_sha(
                f"{arm}-action-computation-{suffix}"
            ),
        )

    def preservation_row(
        self,
        layout,
        *,
        arm: str,
        family: str,
        values: dict[int, float],
        bound: float = 0.02,
    ):
        return qp.PreservationConstraintRow(
            row_id=f"{arm}-{family}",
            family=family,
            values=self.vector(layout, values),
            maximum_absolute_dot=bound,
            layout_digest=layout.layout_digest,
            checkpoint_content_receipt_digest=self.checkpoint_digest,
            parameter_state_sha256=layout.parameter_state_sha256,
            gradient_computation_receipt_digest=_sha(
                f"{arm}-{family}-computation"
            ),
        )

    def evidence(
        self,
        layout,
        *,
        action_vectors: dict[str, dict[int, float]] | None = None,
        action_lowers: dict[str, float] | None = None,
        preservation_overrides: dict[
            tuple[str, str], tuple[dict[int, float], float]
        ]
        | None = None,
    ):
        action_vectors = action_vectors or {
            "dog": {0: 1.0},
            "human": {self.human_coordinate: 1.0},
        }
        action_lowers = action_lowers or {"dog": 0.1, "human": 0.1}
        preservation_overrides = preservation_overrides or {}
        arms = []
        for arm_ordinal, arm in enumerate(qp.DP_ARM_ORDER):
            action_rows = (
                self.action_row(
                    layout,
                    arm=arm,
                    values=action_vectors[arm],
                    lower=action_lowers[arm],
                ),
            )
            preservation_rows = []
            base_parameter = 2 if arm == "dog" else 8
            for family_ordinal, family in enumerate(qp.PRESERVATION_FAMILIES):
                values, bound = preservation_overrides.get(
                    (arm, family),
                    (
                        {
                            (base_parameter + family_ordinal)
                            * self.tensor_numel: 1.0
                        },
                        0.02,
                    ),
                )
                preservation_rows.append(
                    self.preservation_row(
                        layout,
                        arm=arm,
                        family=family,
                        values=values,
                        bound=bound,
                    )
                )
            ranks = []
            for global_rank in qp.SP_GLOBAL_RANKS[arm]:
                ranks.append(
                    qp.SPRankEvidence(
                        global_rank=global_rank,
                        action_rows=action_rows,
                        preservation_rows=tuple(preservation_rows),
                        rank_evidence_receipt_digest=_sha(
                            f"rank-evidence-{global_rank}"
                        ),
                    )
                )
            arms.append(qp.DPArmEvidence(arm_id=arm, sp_ranks=tuple(ranks)))
        return qp.DP2SP4Evidence(
            dp_arms=tuple(arms),
            topology_receipt_digest=_sha("world8-dp2-sp4-topology"),
        )

    def trust(
        self,
        layout,
        *,
        overrides: dict[int, float] | None = None,
        a_scales: dict[int, float] | None = None,
    ):
        overrides = overrides or {}
        a_scales = a_scales or {}
        result = []
        for ordinal, name in enumerate(layout.names):
            fixed_a = torch.zeros(qp.CANONICAL_A_SHAPE, dtype=torch.float32)
            scale = a_scales.get(ordinal, 1.0)
            for rank_index in range(qp.LORA_RANK):
                fixed_a[rank_index, rank_index] = scale
            result.append(
                qp.LayerTrustRadius(
                    parameter_name=name,
                    fixed_lora_a_parameter_name=name.replace(
                        "action_lora_b.weight", "action_lora_a.weight"
                    ),
                    fixed_lora_a=fixed_a,
                    maximum_relative_delta=overrides.get(ordinal, 1.0),
                    reference_effective_weight_norm=1.0,
                    fixed_gauge_receipt_digest=_sha("fixed-a-gauge"),
                    reference_weight_receipt_digest=_sha(
                        f"effective-weight-{ordinal}"
                    ),
                )
            )
        return tuple(result)

    def live_fixed_a(self, trust):
        return [
            (
                item.fixed_lora_a_parameter_name,
                item.fixed_lora_a.detach().clone(),
            )
            for item in trust
        ]

    def solve(
        self,
        parameters=None,
        *,
        evidence=None,
        trust=None,
        global_radius: float = 1.0,
        config=None,
    ):
        parameters = parameters or self.parameters()
        layout = qp.FixedParameterLayout.from_ordered_parameters(parameters)
        evidence = evidence or self.evidence(layout)
        trust = trust or self.trust(layout)
        result = qp.solve_stateless_jacobian_qp(
            layout=layout,
            evidence=evidence,
            global_trust_radius=global_radius,
            layer_trust_radii=trust,
            config=config or qp.JacobianQPConfig(),
        )
        return layout, result

    def flat_delta(self, result):
        return torch.cat(
            [
                result.delta_by_parameter[name].reshape(-1)
                for name in result.layout.names
            ]
        )

    def test_parameter_layout_requires_canonical_order_and_fp32_finite(self) -> None:
        parameters = self.parameters()
        reversed_first = list(parameters)
        reversed_first[0], reversed_first[1] = reversed_first[1], reversed_first[0]
        with self.assertRaisesRegex(qp.MosaicStarcJacobianQPError, "ordering"):
            qp.FixedParameterLayout.from_ordered_parameters(reversed_first)

        wrong_dtype = self.parameters()
        wrong_dtype[0] = (
            wrong_dtype[0][0],
            torch.zeros(qp.CANONICAL_B_SHAPE, dtype=torch.float64),
        )
        with self.assertRaisesRegex(qp.MosaicStarcJacobianQPError, "exact FP32"):
            qp.FixedParameterLayout.from_ordered_parameters(wrong_dtype)

        nonfinite = self.parameters()
        nonfinite[0] = (
            nonfinite[0][0],
            torch.zeros(qp.CANONICAL_B_SHAPE, dtype=torch.float32),
        )
        nonfinite[0][1].reshape(-1)[0] = float("nan")
        with self.assertRaisesRegex(qp.MosaicStarcJacobianQPError, "non-finite"):
            qp.FixedParameterLayout.from_ordered_parameters(nonfinite)

        wrong_shape = self.parameters()
        wrong_shape[0] = (
            wrong_shape[0][0],
            torch.zeros((qp.HIDDEN_SIZE, 4), dtype=torch.float32),
        )
        with self.assertRaisesRegex(qp.MosaicStarcJacobianQPError, "canonical shape"):
            qp.FixedParameterLayout.from_ordered_parameters(wrong_shape)

        nonzero_b = self.parameters()
        with torch.no_grad():
            nonzero_b[0][1].reshape(-1)[0] = 1.0
        with self.assertRaisesRegex(qp.MosaicStarcJacobianQPError, "zero-init"):
            qp.FixedParameterLayout.from_ordered_parameters(nonzero_b)

        layout = qp.FixedParameterLayout.from_ordered_parameters(self.parameters())
        self.assertEqual(layout.total_numel, 393_216)
        self.assertEqual(layout.shapes, (qp.CANONICAL_B_SHAPE,) * 32)

    def test_rows_fail_closed_on_shape_dtype_finite_and_norm(self) -> None:
        parameters = self.parameters()
        layout = qp.FixedParameterLayout.from_ordered_parameters(parameters)
        cases = {
            "shape": torch.ones(layout.total_numel - 1, dtype=torch.float32),
            "exact FP32": torch.ones(layout.total_numel, dtype=torch.float64),
            "non-finite": torch.full(
                (layout.total_numel,), float("inf"), dtype=torch.float32
            ),
            "zero or below": torch.zeros(layout.total_numel, dtype=torch.float32),
        }
        for message, vector in cases.items():
            with self.subTest(message=message):
                evidence = self.evidence(layout)
                dog_arm = evidence.dp_arms[0]
                rank0 = dog_arm.sp_ranks[0]
                bad_row = replace(rank0.action_rows[0], values=vector)
                bad_ranks = list(dog_arm.sp_ranks)
                # Replicate malformed content across SP ranks so the row
                # validator, not consensus, is the failing gate.
                bad_ranks = [
                    replace(rank, action_rows=(bad_row,)) for rank in bad_ranks
                ]
                malformed = replace(
                    evidence,
                    dp_arms=(
                        replace(dog_arm, sp_ranks=tuple(bad_ranks)),
                        evidence.dp_arms[1],
                    ),
                )
                with self.assertRaisesRegex(qp.MosaicStarcJacobianQPError, message):
                    qp.solve_stateless_jacobian_qp(
                        layout=layout,
                        evidence=malformed,
                        global_trust_radius=1.0,
                        layer_trust_radii=self.trust(layout),
                    )

    def test_dp2_sp4_union_requires_both_arms_all_ranks_and_fixed_order(self) -> None:
        parameters = self.parameters()
        layout = qp.FixedParameterLayout.from_ordered_parameters(parameters)
        evidence = self.evidence(layout)
        with self.assertRaisesRegex(qp.MosaicStarcJacobianQPError, "dog then human"):
            qp.solve_stateless_jacobian_qp(
                layout=layout,
                evidence=replace(evidence, dp_arms=tuple(reversed(evidence.dp_arms))),
                global_trust_radius=1.0,
                layer_trust_radii=self.trust(layout),
            )
        dog = evidence.dp_arms[0]
        missing_rank = replace(
            evidence,
            dp_arms=(replace(dog, sp_ranks=dog.sp_ranks[:3]), evidence.dp_arms[1]),
        )
        with self.assertRaisesRegex(qp.MosaicStarcJacobianQPError, "exactly four SP ranks"):
            qp.solve_stateless_jacobian_qp(
                layout=layout,
                evidence=missing_rank,
                global_trust_radius=1.0,
                layer_trust_radii=self.trust(layout),
            )

    def test_sp4_tensor_consensus_is_required_before_union(self) -> None:
        parameters = self.parameters()
        layout = qp.FixedParameterLayout.from_ordered_parameters(parameters)
        evidence = self.evidence(layout)
        dog = evidence.dp_arms[0]
        ranks = list(dog.sp_ranks)
        changed = replace(
            ranks[3].action_rows[0],
            values=self.vector(layout, {0: 1.0, 7: 0.01}),
        )
        ranks[3] = replace(ranks[3], action_rows=(changed,))
        mismatched = replace(
            evidence,
            dp_arms=(replace(dog, sp_ranks=tuple(ranks)), evidence.dp_arms[1]),
        )
        with self.assertRaisesRegex(qp.MosaicStarcJacobianQPError, "SP4 row consensus"):
            qp.solve_stateless_jacobian_qp(
                layout=layout,
                evidence=mismatched,
                global_trust_radius=1.0,
                layer_trust_radii=self.trust(layout),
            )

    def test_world8_rank_receipt_digests_must_be_unique(self) -> None:
        parameters = self.parameters()
        layout = qp.FixedParameterLayout.from_ordered_parameters(parameters)
        evidence = self.evidence(layout)
        dog = evidence.dp_arms[0]
        ranks = list(dog.sp_ranks)
        ranks[1] = replace(
            ranks[1],
            rank_evidence_receipt_digest=ranks[0].rank_evidence_receipt_digest,
        )
        repeated = replace(
            evidence,
            dp_arms=(replace(dog, sp_ranks=tuple(ranks)), evidence.dp_arms[1]),
        )
        with self.assertRaisesRegex(qp.MosaicStarcJacobianQPError, "must be unique"):
            qp.solve_stateless_jacobian_qp(
                layout=layout,
                evidence=repeated,
                global_trust_radius=1.0,
                layer_trust_radii=self.trust(layout),
            )

    def test_preservation_rows_have_fixed_family_order_and_complete_families(self) -> None:
        parameters = self.parameters()
        layout = qp.FixedParameterLayout.from_ordered_parameters(parameters)
        evidence = self.evidence(layout)
        dog = evidence.dp_arms[0]
        ranks = [
            replace(rank, preservation_rows=tuple(reversed(rank.preservation_rows)))
            for rank in dog.sp_ranks
        ]
        reordered = replace(
            evidence,
            dp_arms=(replace(dog, sp_ranks=tuple(ranks)), evidence.dp_arms[1]),
        )
        with self.assertRaisesRegex(qp.MosaicStarcJacobianQPError, "ordering"):
            qp.solve_stateless_jacobian_qp(
                layout=layout,
                evidence=reordered,
                global_trust_radius=1.0,
                layer_trust_radii=self.trust(layout),
            )

    def test_feasible_solution_preserves_every_action_and_slab(self) -> None:
        layout, result = self.solve()
        self.assertTrue(result.authorized, result.receipt["failure_codes"])
        delta = self.flat_delta(result)
        self.assertGreaterEqual(float(delta[0]), 0.1)
        self.assertGreaterEqual(float(delta[self.human_coordinate]), 0.1)
        self.assertTrue(all(row["passed"] for row in result.receipt["action_rows"]))
        self.assertTrue(
            all(row["passed"] for row in result.receipt["preservation_rows"])
        )
        self.assertTrue(
            all(row["passed"] for row in result.receipt["per_layer_trust_radii"])
        )
        self.assertTrue(result.receipt["global_trust_radius"]["passed"])
        self.assertEqual(
            result.receipt["dp2_sp4_evidence_union_receipt"]["union_mode"],
            "raw_rows_after_per_arm_sp4_consensus_no_local_projection",
        )
        self.assertFalse(result.receipt["optimizer_step_allowed"])
        self.assertFalse(result.receipt["runtime_apply_authorized"])
        self.assertFalse(result.receipt["world8_apply_authorized"])
        self.assertTrue(result.receipt["external_scientific_gate_bundle_required"])
        self.assertFalse(result.receipt["checkpoint_retention_authorized"])
        self.assertFalse(result.receipt["training_executed"])
        self.assertGreaterEqual(result.receipt["linear_effective_rank"], 2)
        self.assertIn("linear_compact_gram_eigenvalues", result.receipt)
        self.assertIn("active_constraints", result.receipt)
        certificate = result.receipt["dykstra_optimality_certificate"]
        thresholds = result.receipt["thresholds"]
        self.assertLessEqual(
            certificate["guarded_primal_max_relative_violation"],
            thresholds["dykstra_primal_relative_tolerance"],
        )
        self.assertLessEqual(
            certificate["dykstra_dual_balance_relative_residual"],
            thresholds["dykstra_dual_relative_tolerance"],
        )
        self.assertLessEqual(
            certificate["guarded_complementarity_max_relative_residual"],
            thresholds["dykstra_complementarity_relative_tolerance"],
        )
        self.assertLessEqual(
            certificate["dykstra_correction_state_relative_residual"],
            thresholds["dykstra_dual_relative_tolerance"],
        )
        self.assertLessEqual(
            certificate["guarded_dual_cone_max_relative_residual"],
            thresholds["dykstra_dual_relative_tolerance"],
        )

    def test_solver_tolerance_cannot_be_loosened_to_accept_one_cycle(self) -> None:
        parameters = self.parameters()
        layout = qp.FixedParameterLayout.from_ordered_parameters(parameters)
        with self.assertRaisesRegex(
            qp.MosaicStarcJacobianQPError,
            "cycle tolerance may not be loosened",
        ):
            qp.solve_stateless_jacobian_qp(
                layout=layout,
                evidence=self.evidence(layout),
                global_trust_radius=1.0,
                layer_trust_radii=self.trust(layout),
                config=qp.JacobianQPConfig(dykstra_cycle_tolerance=1.0e9),
            )

    def test_two_sided_slab_projects_without_erasing_action(self) -> None:
        parameters = self.parameters()
        layout = qp.FixedParameterLayout.from_ordered_parameters(parameters)
        action_vectors = {
            "dog": {0: 1.0, self.human_coordinate: 1.0},
            "human": {0: 1.0, self.human_coordinate: 1.0},
        }
        evidence = self.evidence(
            layout,
            action_vectors=action_vectors,
            preservation_overrides={
                ("dog", "identity"): ({0: 1.0}, 0.05),
                ("human", "identity"): ({0: 1.0}, 0.05),
            },
        )
        result = qp.solve_stateless_jacobian_qp(
            layout=layout,
            evidence=evidence,
            global_trust_radius=1.0,
            layer_trust_radii=self.trust(layout),
        )
        self.assertTrue(result.authorized, result.receipt["failure_codes"])
        delta = self.flat_delta(result)
        self.assertLessEqual(abs(float(delta[0])), 0.05)
        for row in result.receipt["action_rows"]:
            self.assertGreaterEqual(row["actual_dot"], row["minimum_dot"])

    def test_per_layer_radius_binds_actual_fp32_candidate(self) -> None:
        parameters = self.parameters()
        layout = qp.FixedParameterLayout.from_ordered_parameters(parameters)
        result = qp.solve_stateless_jacobian_qp(
            layout=layout,
            evidence=self.evidence(layout),
            global_trust_radius=1.0,
            layer_trust_radii=self.trust(layout, overrides={0: 0.15}),
        )
        self.assertTrue(result.authorized, result.receipt["failure_codes"])
        delta = self.flat_delta(result)
        self.assertGreaterEqual(float(delta[0]), 0.1)
        self.assertLessEqual(abs(float(delta[0])), 0.15)
        self.assertLessEqual(
            result.receipt["per_layer_trust_radii"][0][
                "actual_effective_weight_delta_norm"
            ],
            0.15,
        )

    def test_deterministic_repeat_is_receipt_and_delta_identical(self) -> None:
        parameters = self.parameters()
        layout = qp.FixedParameterLayout.from_ordered_parameters(parameters)
        evidence = self.evidence(layout)
        trust = self.trust(layout)
        first = qp.solve_stateless_jacobian_qp(
            layout=layout,
            evidence=evidence,
            global_trust_radius=1.0,
            layer_trust_radii=trust,
        )
        second = qp.solve_stateless_jacobian_qp(
            layout=layout,
            evidence=evidence,
            global_trust_radius=1.0,
            layer_trust_radii=trust,
        )
        self.assertEqual(
            first.receipt["receipt_digest"], second.receipt["receipt_digest"]
        )
        self.assertTrue(torch.equal(self.flat_delta(first), self.flat_delta(second)))

    def test_distinct_infeasible_problems_return_byte_identical_null(self) -> None:
        parameters = self.parameters()
        layout = qp.FixedParameterLayout.from_ordered_parameters(parameters)
        trust = self.trust(layout)
        opposing = self.evidence(
            layout,
            action_vectors={"dog": {0: 1.0}, "human": {0: -1.0}},
            action_lowers={"dog": 0.2, "human": 0.2},
        )
        too_large = self.evidence(
            layout,
            action_lowers={"dog": 2.0, "human": 0.1},
        )
        first = qp.solve_stateless_jacobian_qp(
            layout=layout,
            evidence=opposing,
            global_trust_radius=1.0,
            layer_trust_radii=trust,
        )
        second = qp.solve_stateless_jacobian_qp(
            layout=layout,
            evidence=too_large,
            global_trust_radius=1.0,
            layer_trust_radii=trust,
        )
        self.assertFalse(first.authorized)
        self.assertFalse(second.authorized)
        self.assertTrue(torch.equal(self.flat_delta(first), self.flat_delta(second)))
        self.assertEqual(
            first.receipt["actual_fp32_candidate_delta_sha256"],
            second.receipt["actual_fp32_candidate_delta_sha256"],
        )
        self.assertEqual(
            first.receipt["actual_fp32_candidate_delta_sha256"],
            first.receipt["null_delta_sha256"],
        )

    def test_null_solution_cannot_mutate_parameters(self) -> None:
        parameters = self.parameters()
        layout = qp.FixedParameterLayout.from_ordered_parameters(parameters)
        evidence = self.evidence(
            layout,
            action_lowers={"dog": 2.0, "human": 0.1},
        )
        result = qp.solve_stateless_jacobian_qp(
            layout=layout,
            evidence=evidence,
            global_trust_radius=1.0,
            layer_trust_radii=self.trust(layout),
        )
        before = [tensor.detach().clone() for _, tensor in parameters]
        with self.assertRaisesRegex(qp.MosaicStarcJacobianQPError, "cannot be applied"):
            qp.apply_direct_delta_and_audit(
                solution=result,
                ordered_parameters=parameters,
                ordered_fixed_lora_a=self.live_fixed_a(self.trust(layout)),
            )
        for expected, (_, actual) in zip(before, parameters):
            self.assertTrue(torch.equal(expected, actual))

    def test_direct_add_recomputes_realized_displacement_and_hash(self) -> None:
        parameters = self.parameters()
        layout = qp.FixedParameterLayout.from_ordered_parameters(parameters)
        trust = self.trust(layout)
        result = qp.solve_stateless_jacobian_qp(
            layout=layout,
            evidence=self.evidence(layout),
            global_trust_radius=1.0,
            layer_trust_radii=trust,
        )
        audit = qp.apply_direct_delta_and_audit(
            solution=result,
            ordered_parameters=parameters,
            ordered_fixed_lora_a=self.live_fixed_a(trust),
        )
        self.assertTrue(audit.realized_geometry_safe, audit.receipt["failure_codes"])
        self.assertTrue(audit.rolled_back)
        self.assertFalse(audit.receipt["parameter_update_applied"])
        self.assertTrue(audit.receipt["local_probe_only"])
        self.assertFalse(audit.receipt["world8_apply_authorized"])
        self.assertEqual(
            audit.receipt["application_mechanism"],
            "torch_no_grad_direct_parameter_add_once",
        )
        self.assertFalse(
            audit.receipt["optimizer_instantiated_or_called_by_this_module"]
        )
        self.assertFalse(audit.receipt["checkpoint_retention_authorized"])
        for name, parameter in parameters:
            self.assertTrue(torch.count_nonzero(parameter.detach()).item() == 0)
            self.assertTrue(
                torch.equal(
                    audit.realized_delta_by_parameter[name],
                    result.delta_by_parameter[name],
                )
            )

    def test_effective_weight_ellipsoid_rejects_naked_b_norm_false_pass(self) -> None:
        parameters = self.parameters()
        layout = qp.FixedParameterLayout.from_ordered_parameters(parameters)
        # Old naked-B logic would allow |dB[0]| in [0.1, 0.15].  Frozen A is
        # scaled by 10, so the true effective-weight norm is at least 1.0 and
        # the contract-correct 0.15 ellipsoid is infeasible.
        trust = self.trust(
            layout,
            overrides={0: 0.15},
            a_scales={0: 10.0},
        )
        result = qp.solve_stateless_jacobian_qp(
            layout=layout,
            evidence=self.evidence(layout),
            global_trust_radius=1.0,
            layer_trust_radii=trust,
        )
        self.assertFalse(result.authorized)
        self.assertIn(
            "ACTION_LOWER_BOUND_FAILED:dog-action-0",
            result.receipt["failure_codes"],
        )
        self.assertEqual(
            result.receipt["actual_fp32_candidate_delta_sha256"],
            result.receipt["null_delta_sha256"],
        )

    def test_fixed_a_gauge_shape_rank_and_live_hash_are_fail_closed(self) -> None:
        parameters = self.parameters()
        layout = qp.FixedParameterLayout.from_ordered_parameters(parameters)
        trust = list(self.trust(layout))
        trust[0] = replace(
            trust[0],
            fixed_lora_a=torch.zeros((4, qp.HIDDEN_SIZE), dtype=torch.float32),
        )
        with self.assertRaisesRegex(qp.MosaicStarcJacobianQPError, "must have shape"):
            qp.solve_stateless_jacobian_qp(
                layout=layout,
                evidence=self.evidence(layout),
                global_trust_radius=1.0,
                layer_trust_radii=tuple(trust),
            )

        trust = self.trust(layout)
        result = qp.solve_stateless_jacobian_qp(
            layout=layout,
            evidence=self.evidence(layout),
            global_trust_radius=1.0,
            layer_trust_radii=trust,
        )
        live_a = self.live_fixed_a(trust)
        live_a[0][1][0, 0] += 0.25
        with self.assertRaisesRegex(qp.MosaicStarcJacobianQPError, "differs from the QP gauge"):
            qp.apply_direct_delta_and_audit(
                solution=result,
                ordered_parameters=parameters,
                ordered_fixed_lora_a=live_a,
            )
        self.assertTrue(
            all(
                int(torch.count_nonzero(parameter.detach()).item()) == 0
                for _, parameter in parameters
            )
        )

    def test_partial_direct_add_failure_hashes_attempt_then_rolls_back(self) -> None:
        class FailOnAddTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, data):
                return torch.Tensor._make_subclass(cls, data, True)

            def add_(self, *args, **kwargs):
                raise RuntimeError("injected-second-parameter-add-failure")

        parameters = self.parameters()
        parameters[1] = (
            parameters[1][0],
            FailOnAddTensor(
                torch.zeros(qp.CANONICAL_B_SHAPE, dtype=torch.float32)
            ),
        )
        layout = qp.FixedParameterLayout.from_ordered_parameters(parameters)
        trust = self.trust(layout)
        result = qp.solve_stateless_jacobian_qp(
            layout=layout,
            evidence=self.evidence(layout),
            global_trust_radius=1.0,
            layer_trust_radii=trust,
        )
        audit = qp.apply_direct_delta_and_audit(
            solution=result,
            ordered_parameters=parameters,
            ordered_fixed_lora_a=self.live_fixed_a(trust),
        )
        self.assertFalse(audit.realized_geometry_safe)
        self.assertTrue(audit.rolled_back)
        self.assertIn("DIRECT_PARAMETER_ADD_FAILED", audit.receipt["failure_codes"])
        self.assertIsNotNone(
            audit.receipt["attempted_after_parameter_state_sha256"]
        )
        realized = torch.cat(
            [audit.realized_delta_by_parameter[name].reshape(-1) for name in layout.names]
        )
        self.assertGreater(float(realized[0]), 0.1)
        self.assertEqual(float(realized[self.human_coordinate]), 0.0)
        self.assertNotEqual(
            audit.receipt["realized_fp32_displacement_sha256"],
            result.receipt["null_delta_sha256"],
        )
        self.assertTrue(audit.receipt["rollback_byte_identical_to_pre_step"])
        self.assertTrue(
            all(
                int(torch.count_nonzero(parameter.detach()).item()) == 0
                for _, parameter in parameters
            )
        )

    def test_candidate_tensor_tamper_fails_before_parameter_mutation(self) -> None:
        parameters = self.parameters()
        layout = qp.FixedParameterLayout.from_ordered_parameters(parameters)
        result = qp.solve_stateless_jacobian_qp(
            layout=layout,
            evidence=self.evidence(layout),
            global_trust_radius=1.0,
            layer_trust_radii=self.trust(layout),
        )
        result.delta_by_parameter[layout.names[0]].add_(1.0)
        before = [tensor.detach().clone() for _, tensor in parameters]
        with self.assertRaisesRegex(qp.MosaicStarcJacobianQPError, "no longer match"):
            qp.apply_direct_delta_and_audit(
                solution=result,
                ordered_parameters=parameters,
                ordered_fixed_lora_a=self.live_fixed_a(self.trust(layout)),
            )
        for expected, (_, actual) in zip(before, parameters):
            self.assertTrue(torch.equal(expected, actual))

    def test_receipt_schema_artifact_binds_both_schema_versions(self) -> None:
        schema_path = METHOD_ROOT / qp.RECEIPT_JSON_SCHEMA_RELATIVE_PATH
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(
            schema["$defs"]["candidate"]["properties"]["schema_version"]["const"],
            qp.CANDIDATE_RECEIPT_SCHEMA,
        )
        self.assertEqual(
            schema["$defs"]["realized"]["properties"]["schema_version"]["const"],
            qp.REALIZED_RECEIPT_SCHEMA,
        )
        self.assertFalse(
            schema["$defs"]["candidate"]["properties"]["optimizer_step_allowed"]["const"]
        )
        self.assertFalse(schema["$defs"]["candidate"]["additionalProperties"])
        self.assertFalse(schema["$defs"]["realized"]["additionalProperties"])
        self.assertEqual(
            set(schema["$defs"]["candidate"]["required"]),
            set(qp._CANDIDATE_RECEIPT_KEYS),
        )
        self.assertEqual(
            set(schema["$defs"]["realized"]["required"]),
            set(qp._REALIZED_RECEIPT_KEYS),
        )
        for definition in (
            "parameterLayoutItem",
            "optimizationProblem",
            "loraGauge",
            "actionAudit",
            "preservationAudit",
            "layerAudit",
            "globalAudit",
            "unionActionRow",
            "unionPreservationRow",
            "unionRank",
            "unionArm",
            "evidenceUnion",
            "dykstraCertificate",
            "thresholds",
        ):
            self.assertFalse(
                schema["$defs"][definition]["additionalProperties"],
                definition,
            )
        certificate_required = set(
            schema["$defs"]["dykstraCertificate"]["required"]
        )
        self.assertIn(
            "dykstra_correction_state_relative_residual",
            certificate_required,
        )
        self.assertIn(
            "guarded_dual_cone_max_relative_residual",
            certificate_required,
        )
        self.assertEqual(
            schema["$defs"]["candidate"]["properties"]
            ["linear_effective_rank"]["minimum"],
            0,
        )

    def test_runtime_receipt_validator_rejects_resealed_policy_tamper(self) -> None:
        parameters = self.parameters()
        layout = qp.FixedParameterLayout.from_ordered_parameters(parameters)
        trust = self.trust(layout)
        result = qp.solve_stateless_jacobian_qp(
            layout=layout,
            evidence=self.evidence(layout),
            global_trust_radius=1.0,
            layer_trust_radii=trust,
        )
        qp.validate_candidate_receipt_schema(result.receipt)

        tampered = dict(result.receipt)
        tampered["runtime_apply_authorized"] = True
        tampered.pop("receipt_digest")
        tampered["receipt_digest"] = qp.object_sha256(tampered)
        with self.assertRaisesRegex(qp.MosaicStarcJacobianQPError, "boolean policy"):
            qp.validate_candidate_receipt_schema(tampered)

        audit = qp.apply_direct_delta_and_audit(
            solution=result,
            ordered_parameters=parameters,
            ordered_fixed_lora_a=self.live_fixed_a(trust),
        )
        qp.validate_realized_receipt_schema(audit.receipt)
        tampered_realized = dict(audit.receipt)
        tampered_realized["parameter_update_applied"] = True
        tampered_realized.pop("receipt_digest")
        tampered_realized["receipt_digest"] = qp.object_sha256(tampered_realized)
        with self.assertRaisesRegex(qp.MosaicStarcJacobianQPError, "boolean policy"):
            qp.validate_realized_receipt_schema(tampered_realized)

    def test_negative_kkt_and_cycle_residuals_are_rejected(self) -> None:
        parameters = self.parameters()
        layout = qp.FixedParameterLayout.from_ordered_parameters(parameters)
        result = qp.solve_stateless_jacobian_qp(
            layout=layout,
            evidence=self.evidence(layout),
            global_trust_radius=1.0,
            layer_trust_radii=self.trust(layout),
        )
        for path in (
            ("dykstra_optimality_certificate", "guarded_primal_max_relative_violation"),
            ("dykstra_optimality_certificate", "dykstra_dual_balance_relative_residual"),
            ("dykstra_optimality_certificate", "dykstra_correction_state_relative_residual"),
            ("dykstra_optimality_certificate", "guarded_dual_cone_max_relative_residual"),
            (None, "dykstra_cycle_residual"),
        ):
            with self.subTest(path=path):
                tampered = copy.deepcopy(result.receipt)
                if path[0] is None:
                    tampered[path[1]] = -1.0
                else:
                    tampered[path[0]][path[1]] = -1.0
                tampered.pop("receipt_digest")
                tampered["receipt_digest"] = qp.object_sha256(tampered)
                with self.assertRaisesRegex(
                    qp.MosaicStarcJacobianQPError, "at least 0.0"
                ):
                    qp.validate_candidate_receipt_schema(tampered)

    def test_resealed_nested_layout_and_audit_tamper_are_rejected(self) -> None:
        parameters = self.parameters()
        layout = qp.FixedParameterLayout.from_ordered_parameters(parameters)
        trust = self.trust(layout)
        result = qp.solve_stateless_jacobian_qp(
            layout=layout,
            evidence=self.evidence(layout),
            global_trust_radius=1.0,
            layer_trust_radii=trust,
        )
        candidate = copy.deepcopy(result.receipt)
        candidate["parameter_layout"][0] = {}
        candidate.pop("receipt_digest")
        candidate["receipt_digest"] = qp.object_sha256(candidate)
        with self.assertRaisesRegex(qp.MosaicStarcJacobianQPError, "key closure"):
            qp.validate_candidate_receipt_schema(candidate)

        candidate = copy.deepcopy(result.receipt)
        candidate["action_rows"][0]["actual_dot"] = "fabricated"
        candidate.pop("receipt_digest")
        candidate["receipt_digest"] = qp.object_sha256(candidate)
        with self.assertRaisesRegex(qp.MosaicStarcJacobianQPError, "finite scalar"):
            qp.validate_candidate_receipt_schema(candidate)

        candidate = copy.deepcopy(result.receipt)
        candidate["optimization_problem"] = {}
        candidate.pop("receipt_digest")
        candidate["receipt_digest"] = qp.object_sha256(candidate)
        with self.assertRaisesRegex(qp.MosaicStarcJacobianQPError, "key closure"):
            qp.validate_candidate_receipt_schema(candidate)

        audit = qp.apply_direct_delta_and_audit(
            solution=result,
            ordered_parameters=parameters,
            ordered_fixed_lora_a=self.live_fixed_a(trust),
        )
        realized = copy.deepcopy(audit.receipt)
        realized["candidate_delta_norm"] = "fabricated"
        realized.pop("receipt_digest")
        realized["receipt_digest"] = qp.object_sha256(realized)
        with self.assertRaisesRegex(qp.MosaicStarcJacobianQPError, "finite scalar"):
            qp.validate_realized_receipt_schema(realized)

        realized = copy.deepcopy(audit.receipt)
        realized["action_rows"][0]["junk"] = True
        realized.pop("receipt_digest")
        realized["receipt_digest"] = qp.object_sha256(realized)
        with self.assertRaisesRegex(qp.MosaicStarcJacobianQPError, "key closure"):
            qp.validate_realized_receipt_schema(realized)

    def test_post_add_fixed_a_exception_still_rolls_back(self) -> None:
        parameters = self.parameters()
        layout = qp.FixedParameterLayout.from_ordered_parameters(parameters)
        trust = self.trust(layout)
        result = qp.solve_stateless_jacobian_qp(
            layout=layout,
            evidence=self.evidence(layout),
            global_trust_radius=1.0,
            layer_trust_radii=trust,
        )
        original = qp._validate_live_fixed_lora_a
        call_count = 0

        def fail_post_add(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise qp.MosaicStarcJacobianQPError("injected-post-A-failure")
            return original(*args, **kwargs)

        with mock.patch.object(
            qp, "_validate_live_fixed_lora_a", side_effect=fail_post_add
        ):
            with self.assertRaisesRegex(
                qp.MosaicStarcJacobianQPError,
                "failed after byte-identical rollback",
            ):
                qp.apply_direct_delta_and_audit(
                    solution=result,
                    ordered_parameters=parameters,
                    ordered_fixed_lora_a=self.live_fixed_a(trust),
                )
        self.assertEqual(call_count, 2)
        self.assertTrue(
            all(
                int(torch.count_nonzero(parameter.detach()).item()) == 0
                for _, parameter in parameters
            )
        )

    def test_post_rollback_offline_audit_exception_cannot_leak_delta(self) -> None:
        parameters = self.parameters()
        layout = qp.FixedParameterLayout.from_ordered_parameters(parameters)
        trust = self.trust(layout)
        result = qp.solve_stateless_jacobian_qp(
            layout=layout,
            evidence=self.evidence(layout),
            global_trust_radius=1.0,
            layer_trust_radii=trust,
        )
        original = qp._constraint_audit
        call_count = 0

        def fail_apply_audit(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise qp.MosaicStarcJacobianQPError("injected-offline-audit-failure")
            return original(*args, **kwargs)

        with mock.patch.object(qp, "_constraint_audit", side_effect=fail_apply_audit):
            with self.assertRaisesRegex(
                qp.MosaicStarcJacobianQPError, "injected-offline-audit-failure"
            ):
                qp.apply_direct_delta_and_audit(
                    solution=result,
                    ordered_parameters=parameters,
                    ordered_fixed_lora_a=self.live_fixed_a(trust),
                )
        self.assertEqual(call_count, 2)
        self.assertTrue(
            all(
                int(torch.count_nonzero(parameter.detach()).item()) == 0
                for _, parameter in parameters
            )
        )


if __name__ == "__main__":
    unittest.main()
