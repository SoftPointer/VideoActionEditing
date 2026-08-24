from __future__ import annotations

from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch

    import pair_v7_dual_coordinate_nullspace_transport as v7

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    v7 = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def _sha(character: str) -> str:
    return character * 64


@unittest.skipUnless(_TORCH_AVAILABLE, "torch is required")
class PairV7NullspaceTransportTests(unittest.TestCase):
    parameter_name = "blocks.0.attn2.to_q.action_lora_A"
    checkpoint_digest = _sha("2")
    parameter_state_digest = _sha("3")

    def named(self, values) -> dict[str, "torch.Tensor"]:
        return {
            self.parameter_name: torch.tensor(values, dtype=torch.float32)
        }

    def probes(self, vectors) -> list["v7.IdentityGradientProbe"]:
        rows = []
        families = tuple(v7.REQUIRED_IDENTITY_FAMILIES)
        for ordinal, vector in enumerate(vectors):
            family = families[ordinal % len(families)]
            feature_hex = ("a", "b", "c", "d")[ordinal]
            source_hex = ("e", "f", "0", "1")[ordinal]
            rows.append(
                v7.IdentityGradientProbe(
                    probe_id=f"{family}-{ordinal}",
                    family=family,
                    gradient_by_parameter=self.named(vector),
                    feature_sketch_sha256=_sha(feature_hex),
                    source_coordinate_receipt_digest=_sha(source_hex),
                    gradient_computation_receipt_digest=_sha(
                        ("7", "8", "9", "a")[ordinal]
                    ),
                    checkpoint_content_receipt_digest=self.checkpoint_digest,
                    parameter_state_sha256=self.parameter_state_digest,
                )
            )
        return rows

    def provenance(self, action) -> "v7.ActionGradientProvenance":
        tensor = torch.tensor(action, dtype=torch.float32)
        return v7.ActionGradientProvenance(
            candidate_ids=("fit-0",),
            action_families=("family-0",),
            event_digests=(_sha("4"),),
            component_gradient_sha256=(v7._tensor_sha256(tensor),),
            gradient_computation_receipt_digests=(_sha("5"),),
            fit_only_geometry_authority_digest=_sha("6"),
            aggregation="single_fit_only_geometry_event",
        )

    def project(self, action, vectors, **config_updates):
        config = v7.TransportConfig(**config_updates)
        return v7.project_action_gradient_to_identity_nullspace(
            action_gradient_by_parameter=self.named(action),
            action_gradient_provenance=self.provenance(action),
            identity_probes=self.probes(vectors),
            checkpoint_content_receipt_digest=self.checkpoint_digest,
            parameter_state_sha256=self.parameter_state_digest,
            config=config,
        )

    def flat_safe(self, result) -> "torch.Tensor":
        return result.safe_gradient_by_parameter[self.parameter_name]

    def test_gradient_layout_roundtrip_and_strict_closure(self) -> None:
        gradients = {
            "z.weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            "a.weight": torch.tensor([3.0], dtype=torch.float32),
        }
        layout = v7.GradientLayout.from_named_gradients(gradients)
        flat = layout.flatten(gradients, label="fixture")
        rebuilt = layout.unflatten(flat, label="fixture")
        self.assertEqual(layout.names, ("a.weight", "z.weight"))
        for name in gradients:
            self.assertTrue(torch.equal(gradients[name], rebuilt[name]))
        with self.assertRaisesRegex(v7.PairV7TransportError, "closure"):
            layout.flatten({"a.weight": gradients["a.weight"]}, label="missing")
        with self.assertRaisesRegex(v7.PairV7TransportError, "differs"):
            layout.flatten(
                {
                    "a.weight": gradients["a.weight"],
                    "z.weight": torch.ones(3, dtype=torch.float32),
                },
                label="shape",
            )

    def test_zero_action_is_no_go(self) -> None:
        result = self.project(
            [0.0, 0.0, 0.0],
            [[1.0, 0.0, 0.0]] * 4,
        )
        self.assertFalse(result.geometry_authorized)
        self.assertIn(
            "ACTION_GRADIENT_ZERO_OR_TOO_SMALL", result.receipt["failure_codes"]
        )
        self.assertFalse(result.receipt["optimizer_authorized"])

    def test_any_zero_required_probe_is_no_go(self) -> None:
        result = self.project(
            [0.0, 1.0, 1.0],
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
        )
        self.assertFalse(result.geometry_authorized)
        self.assertTrue(
            any(
                code.startswith("IDENTITY_PROBE_ZERO_OR_TOO_SMALL")
                for code in result.receipt["failure_codes"]
            )
        )

    def test_exact_collinear_scaled_and_antiparallel_rows_reduce_rank(self) -> None:
        result = self.project(
            [1.0, 2.0, 0.0],
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [-3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ],
        )
        self.assertTrue(result.geometry_authorized, result.receipt["failure_codes"])
        self.assertEqual(result.receipt["identity_effective_rank"], 1)
        self.assertEqual(result.receipt["identity_redundant_direction_count"], 3)
        self.assertTrue(
            torch.allclose(
                self.flat_safe(result),
                torch.tensor([0.0, 2.0, 0.0]),
                atol=1.0e-6,
                rtol=0.0,
            )
        )

    def test_near_collinear_retained_basis_is_ill_conditioned_no_go(self) -> None:
        epsilon = 1.0e-5
        result = self.project(
            [0.0, 0.0, 0.0, 1.0],
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, epsilon, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 2.0, 0.0],
            ],
        )
        self.assertFalse(result.geometry_authorized)
        self.assertIn(
            "IDENTITY_BASIS_ILL_CONDITIONED", result.receipt["failure_codes"]
        )
        self.assertGreater(
            result.receipt["effective_condition_number"],
            result.receipt["thresholds"]["maximum_effective_condition_number"],
        )

    def test_full_action_identity_conflict_fails_survival(self) -> None:
        result = self.project(
            [1.0, 1.0],
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [2.0, 0.0],
                [0.0, -2.0],
            ],
        )
        self.assertFalse(result.geometry_authorized)
        self.assertIn(
            "ACTION_GRADIENT_SURVIVAL_TOO_LOW", result.receipt["failure_codes"]
        )
        self.assertIn(
            "ACTION_DESCENT_COSINE_TOO_LOW", result.receipt["failure_codes"]
        )

    def test_partial_conflict_preserves_action_survival_and_descent(self) -> None:
        result = self.project(
            [1.0, 1.0, 2.0],
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-2.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
            ],
        )
        self.assertTrue(result.geometry_authorized, result.receipt["failure_codes"])
        expected_survival = 2.0 / (6.0**0.5)
        self.assertAlmostEqual(
            result.receipt["action_gradient_norm_survival"],
            expected_survival,
            places=6,
        )
        self.assertAlmostEqual(
            result.receipt["action_descent_cosine"], expected_survival, places=6
        )
        self.assertGreater(result.receipt["action_descent_gain"], 0.0)
        self.assertTrue(
            torch.allclose(
                self.flat_safe(result),
                torch.tensor([0.0, 0.0, 2.0]),
                atol=1.0e-6,
                rtol=0.0,
            )
        )

    def test_applied_fp32_direction_is_identity_null_for_every_probe(self) -> None:
        result = self.project(
            [3.0, -4.0, 5.0, 6.0],
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
                [0.0, -3.0, 0.0, 0.0],
            ],
        )
        self.assertTrue(result.geometry_authorized, result.receipt["failure_codes"])
        for row in result.receipt["identity_probes"]:
            self.assertLessEqual(
                row["absolute_unit_dot_with_applied_gradient"], row["dot_limit"]
            )
            self.assertLessEqual(
                row["absolute_cosine_with_applied_gradient"],
                result.receipt["thresholds"]["maximum_identity_cosine"],
            )

    def test_missing_identity_family_is_no_go(self) -> None:
        probes = self.probes([[1.0, 0.0]] * 4)[:1]
        result = v7.project_action_gradient_to_identity_nullspace(
            action_gradient_by_parameter=self.named([0.0, 1.0]),
            action_gradient_provenance=self.provenance([0.0, 1.0]),
            identity_probes=probes,
            checkpoint_content_receipt_digest=self.checkpoint_digest,
            parameter_state_sha256=self.parameter_state_digest,
        )
        self.assertFalse(result.geometry_authorized)
        self.assertIn(
            "MISSING_IDENTITY_FAMILY:deploy_camera_delta",
            result.receipt["failure_codes"],
        )

    def test_probe_must_share_action_checkpoint_and_parameter_state(self) -> None:
        probes = self.probes([[1.0, 0.0]] * 4)
        probe = probes[0]
        probes[0] = v7.IdentityGradientProbe(
            probe_id=probe.probe_id,
            family=probe.family,
            gradient_by_parameter=probe.gradient_by_parameter,
            feature_sketch_sha256=probe.feature_sketch_sha256,
            source_coordinate_receipt_digest=probe.source_coordinate_receipt_digest,
            gradient_computation_receipt_digest=(
                probe.gradient_computation_receipt_digest
            ),
            checkpoint_content_receipt_digest=probe.checkpoint_content_receipt_digest,
            parameter_state_sha256=_sha("0"),
        )
        with self.assertRaisesRegex(v7.PairV7TransportError, "state differs"):
            v7.project_action_gradient_to_identity_nullspace(
                action_gradient_by_parameter=self.named([0.0, 1.0]),
                action_gradient_provenance=self.provenance([0.0, 1.0]),
                identity_probes=probes,
                checkpoint_content_receipt_digest=self.checkpoint_digest,
                parameter_state_sha256=self.parameter_state_digest,
            )

    def test_nonfinite_input_raises_fail_closed(self) -> None:
        with self.assertRaisesRegex(v7.PairV7TransportError, "non-finite"):
            self.project(
                [float("nan"), 1.0],
                [[1.0, 0.0]] * 4,
            )

    def test_policy_cannot_be_silently_loosened(self) -> None:
        with self.assertRaisesRegex(v7.PairV7TransportError, "may not be loosened"):
            self.project(
                [0.0, 1.0],
                [[1.0, 0.0]] * 4,
                maximum_identity_cosine=0.5,
            )

    def test_stateless_delta_is_bounded_null_and_action_descent(self) -> None:
        result = self.project(
            [1.0, 1.0, 4.0],
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
        )
        delta = v7.build_stateless_trust_region_delta(
            transport=result,
            learning_rate=1.0,
            maximum_delta_norm=0.25,
            pre_step_parameter_state_sha256=self.parameter_state_digest,
        )
        vector = delta.delta_by_parameter[self.parameter_name]
        self.assertLessEqual(float(torch.linalg.vector_norm(vector)), 0.2500001)
        self.assertTrue(delta.receipt["action_descent"])
        self.assertLess(delta.receipt["action_directional_derivative"], 0.0)
        self.assertEqual(delta.receipt["application_contract"], "direct_parameter_add_only")
        self.assertFalse(delta.receipt["optimizer_step_allowed"])
        for row in delta.receipt["actual_fp32_delta_identity_audit"]:
            self.assertLessEqual(row["absolute_unit_dot"], row["dot_limit"])

    def state_bound_transport_and_delta(self, b_values):
        frozen_name = "blocks.0.attn2.to_q.action_lora_a.weight"
        before = {
            frozen_name: torch.tensor([2.0], dtype=torch.float32),
            self.parameter_name: torch.tensor(b_values, dtype=torch.float32),
        }
        state_digest = v7.named_parameter_state_sha256(before)
        action = [1.0, 1.0, 4.0]
        vectors = (
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        )
        probes = []
        families = tuple(v7.REQUIRED_IDENTITY_FAMILIES)
        for ordinal, vector in enumerate(vectors):
            family = families[ordinal % len(families)]
            probes.append(
                v7.IdentityGradientProbe(
                    probe_id=f"state-bound-{family}-{ordinal}",
                    family=family,
                    gradient_by_parameter=self.named(vector),
                    feature_sketch_sha256=_sha(("a", "b", "c", "d")[ordinal]),
                    source_coordinate_receipt_digest=_sha(
                        ("e", "f", "0", "1")[ordinal]
                    ),
                    gradient_computation_receipt_digest=_sha(
                        ("6", "7", "8", "9")[ordinal]
                    ),
                    checkpoint_content_receipt_digest=self.checkpoint_digest,
                    parameter_state_sha256=state_digest,
                )
            )
        transport = v7.project_action_gradient_to_identity_nullspace(
            action_gradient_by_parameter=self.named(action),
            action_gradient_provenance=self.provenance(action),
            identity_probes=probes,
            checkpoint_content_receipt_digest=self.checkpoint_digest,
            parameter_state_sha256=state_digest,
        )
        delta = v7.build_stateless_trust_region_delta(
            transport=transport,
            learning_rate=1.0,
            maximum_delta_norm=0.25,
            pre_step_parameter_state_sha256=state_digest,
        )
        return before, transport, delta

    def test_realized_theta_after_minus_before_is_reaudited(self) -> None:
        before, transport, candidate = self.state_bound_transport_and_delta(
            [0.0, 0.0, 0.0]
        )
        after = {name: tensor.detach().clone() for name, tensor in before.items()}
        for name, delta in candidate.delta_by_parameter.items():
            after[name] = after[name] + delta
        audit = v7.audit_realized_parameter_displacement(
            transport=transport,
            candidate=candidate,
            full_parameter_state_before=before,
            full_parameter_state_after=after,
        )
        self.assertTrue(audit.realized_displacement_safe, audit.receipt["failure_codes"])
        self.assertTrue(audit.receipt["audit_observed_theta_after_minus_theta_before"])
        self.assertFalse(audit.receipt["audit_executed_parameter_add"])

    def test_realized_fp32_rounding_to_zero_is_no_go(self) -> None:
        before, transport, candidate = self.state_bound_transport_and_delta(
            [1.0e20, 1.0e20, 1.0e20]
        )
        after = {name: tensor.detach().clone() for name, tensor in before.items()}
        for name, delta in candidate.delta_by_parameter.items():
            after[name] = after[name] + delta
        self.assertTrue(torch.equal(after[self.parameter_name], before[self.parameter_name]))
        audit = v7.audit_realized_parameter_displacement(
            transport=transport,
            candidate=candidate,
            full_parameter_state_before=before,
            full_parameter_state_after=after,
        )
        self.assertFalse(audit.realized_displacement_safe)
        self.assertIn(
            "REALIZED_PARAMETER_DISPLACEMENT_ROUNDED_TO_ZERO",
            audit.receipt["failure_codes"],
        )

    def test_mutated_candidate_tensor_cannot_reuse_sealed_delta_receipt(self) -> None:
        before, transport, candidate = self.state_bound_transport_and_delta(
            [0.0, 0.0, 0.0]
        )
        after = {name: tensor.detach().clone() for name, tensor in before.items()}
        for name, delta in candidate.delta_by_parameter.items():
            after[name] = after[name] + delta
        candidate.delta_by_parameter[self.parameter_name][2] += 0.125
        with self.assertRaisesRegex(v7.PairV7TransportError, "sealed receipt"):
            v7.audit_realized_parameter_displacement(
                transport=transport,
                candidate=candidate,
                full_parameter_state_before=before,
                full_parameter_state_after=after,
            )

    def test_no_go_transport_cannot_be_turned_into_delta(self) -> None:
        result = self.project([0.0, 0.0], [[1.0, 0.0]] * 4)
        with self.assertRaisesRegex(v7.PairV7TransportError, "NO-GO"):
            v7.build_stateless_trust_region_delta(
                transport=result,
                learning_rate=1.0,
                maximum_delta_norm=1.0,
                pre_step_parameter_state_sha256=self.parameter_state_digest,
            )

    def test_mutated_transport_tensor_cannot_reuse_a_sealed_receipt(self) -> None:
        result = self.project(
            [1.0, 2.0, 3.0],
            [[1.0, 0.0, 0.0]] * 4,
        )
        result.safe_gradient_by_parameter[self.parameter_name][1] += 1.0
        with self.assertRaisesRegex(v7.PairV7TransportError, "no longer match"):
            v7.build_stateless_trust_region_delta(
                transport=result,
                learning_rate=0.1,
                maximum_delta_norm=1.0,
                pre_step_parameter_state_sha256=self.parameter_state_digest,
            )

    def test_native_post_step_gate_accepts_and_requires_rollback(self) -> None:
        baseline = {family: 1.0 for family in v7.REQUIRED_IDENTITY_FAMILIES}
        stable = {family: 1.00001 for family in v7.REQUIRED_IDENTITY_FAMILIES}
        accepted = v7.build_post_step_native_rollback_receipt(
            pre_action_loss=1.0,
            post_action_loss=0.8,
            action_directional_derivative=-0.5,
            pre_identity_metric_by_family=baseline,
            post_identity_metric_by_family=stable,
            pre_step_parameter_state_sha256=_sha("1"),
            post_step_parameter_state_sha256=_sha("2"),
        )
        self.assertTrue(accepted["accepted"])
        self.assertFalse(accepted["rollback_required"])
        self.assertFalse(accepted["measurement_inputs_authoritatively_bound"])
        self.assertFalse(accepted["rollback_executed"])

        degraded = dict(stable)
        degraded["deploy_noop_identity"] = 1.1
        rejected = v7.build_post_step_native_rollback_receipt(
            pre_action_loss=1.0,
            post_action_loss=1.01,
            action_directional_derivative=-0.5,
            pre_identity_metric_by_family=baseline,
            post_identity_metric_by_family=degraded,
            pre_step_parameter_state_sha256=_sha("1"),
            post_step_parameter_state_sha256=_sha("3"),
        )
        self.assertFalse(rejected["accepted"])
        self.assertTrue(rejected["rollback_required"])
        self.assertIn("ACTION_ARMIJO_GATE_FAILED", rejected["failure_codes"])
        self.assertIn(
            "IDENTITY_NATIVE_FIELD_REGRESSION:deploy_noop_identity",
            rejected["failure_codes"],
        )

    def test_contract_closes_visual_transport_and_exact40_tail(self) -> None:
        receipt = v7.build_method_contract_receipt()
        self.assertEqual(receipt["frame_count"], 81)
        self.assertEqual(receipt["schedule_step_count"], 40)
        self.assertEqual(receipt["exact40_zero_update_indices"], [38, 39])
        self.assertEqual(
            receipt["coordinate_arms"]["coupling"],
            "shared_action_lora_parameter_names_and_parameter_space_only",
        )
        self.assertFalse(
            receipt["pure_t2v_visual_used_as_rv2v_target_noise_source_or_donor"]
        )
        self.assertFalse(
            receipt["optimization"][
                "adam_momentum_weight_decay_or_preconditioner_after_projection"
            ]
        )
        self.assertTrue(
            receipt["fit_only_geometry_evidence_must_be_create_only_and_sealed"]
        )
        self.assertFalse(receipt["population_confirmation_or_optimizer_go_consumed"])
        for field in (
            "global_population_go",
            "optimizer_authorized",
            "parameter_update_authorized",
            "action_success_claimed",
        ):
            self.assertFalse(receipt[field])
        unsigned = dict(receipt)
        digest = unsigned.pop("receipt_digest")
        self.assertEqual(v7.object_sha256(unsigned), digest)


if __name__ == "__main__":
    unittest.main()
