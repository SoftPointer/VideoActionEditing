#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path
import unittest

try:
    import torch
except ModuleNotFoundError:  # local contract-only environments need not ship torch
    torch = None


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if torch is not None:
    import pair_v7_dual_coordinate_nullspace_transport as core
    import pair_v7_phase_b_stateless_step as phase_b
else:
    core = None
    phase_b = None


SHA = "1" * 64
B_NAME = "blocks.0.attn2.to_q.action_lora_b.weight"
A_NAME = "blocks.0.attn2.to_q.action_lora_a.weight"


def seal(value):
    return {**value, "receipt_digest": core.object_sha256(value)}


@unittest.skipIf(torch is None, "torch is required for stateless mutation tests")
class PairV7TemporaryPhaseBTests(unittest.TestCase):
    def setUp(self) -> None:
        self.b = torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.float32))
        self.a = torch.nn.Parameter(
            torch.eye(2, dtype=torch.float32), requires_grad=False
        )
        self.base = torch.full((2, 2), 10.0, dtype=torch.bfloat16)
        self.state = {A_NAME: self.a, B_NAME: self.b}
        state_digest = core.named_parameter_state_sha256(self.state)
        action = {B_NAME: torch.tensor([[1.0, 0.0], [0.0, 0.0]])}
        action_layout = core.GradientLayout.from_named_gradients(action)
        action_digest = core._tensor_sha256(
            action_layout.flatten(action, label="test action").float()
        )
        provenance = core.ActionGradientProvenance(
            candidate_ids=("candidate",),
            action_families=("action",),
            event_digests=(SHA,),
            component_gradient_sha256=(action_digest,),
            gradient_computation_receipt_digests=(SHA,),
            fit_only_geometry_authority_digest=SHA,
            aggregation="single_fit_only_geometry_event",
        )
        probes = (
            core.IdentityGradientProbe(
                probe_id="noop",
                family="deploy_noop_identity",
                gradient_by_parameter={
                    B_NAME: torch.tensor([[0.0, 1.0], [0.0, 0.0]])
                },
                feature_sketch_sha256=SHA,
                source_coordinate_receipt_digest=SHA,
                gradient_computation_receipt_digest=SHA,
                checkpoint_content_receipt_digest=SHA,
                parameter_state_sha256=state_digest,
            ),
            core.IdentityGradientProbe(
                probe_id="camera",
                family="deploy_camera_delta",
                gradient_by_parameter={
                    B_NAME: torch.tensor([[0.0, 0.0], [1.0, 0.0]])
                },
                feature_sketch_sha256=SHA,
                source_coordinate_receipt_digest=SHA,
                gradient_computation_receipt_digest=SHA,
                checkpoint_content_receipt_digest=SHA,
                parameter_state_sha256=state_digest,
            ),
        )
        self.transport = core.project_action_gradient_to_identity_nullspace(
            action_gradient_by_parameter=action,
            action_gradient_provenance=provenance,
            identity_probes=probes,
            checkpoint_content_receipt_digest=SHA,
            parameter_state_sha256=state_digest,
        )
        self.assertTrue(self.transport.geometry_authorized)
        self.candidate = core.build_stateless_trust_region_delta(
            transport=self.transport,
            learning_rate=0.05,
            maximum_delta_norm=0.05,
            pre_step_parameter_state_sha256=state_digest,
        )
        union = seal(
            {
                "schema_version": phase_b.UNION_SCHEMA,
                "geometry_audit_passed": True,
                "transport_geometry_authorized": True,
                "failure_codes": [],
                "optimizer_authorized": False,
                "parameter_update_authorized": False,
                "parameter_mutation_performed": False,
                "scientific_action_editing_success_claim": False,
                "global_population_go": False,
            }
        )
        transport_receipt = dict(self.transport.receipt)
        top = {
            "schema_version": phase_b.PHASE_A_SCHEMA,
            "audit_complete": True,
            "geometry_audit_passed": True,
            "parameter_state_sha256": state_digest,
            "union_projection_receipt": union,
            "nullspace_transport_receipt": transport_receipt,
            "optimizer_authorized": False,
            "parameter_update_authorized": False,
            "parameter_mutation_performed": False,
            "scientific_action_editing_success_claim": False,
            "global_population_go": False,
        }
        self.phase_a = seal(top)
        self.authority = phase_b.build_temporary_canary_authority(
            phase_a_receipt=self.phase_a,
            expected_phase_a_receipt_digest=self.phase_a["receipt_digest"],
            maximum_delta_norm=0.05,
            maximum_effective_delta_w_relative_norm=0.01,
        )
        self.consensus = []

    def world_consensus(self, label: str, digest: str) -> None:
        self.consensus.append((label, digest))

    @staticmethod
    def evaluation(passed: bool = True):
        return seal(
            {
                "schema_version": "mock-native-exact81-evaluation-v1",
                "temporary_canary_evaluation_complete": True,
                "canary_gate_passed": passed,
                "candidate_latent_retained_only_after_rollback": True,
            }
        )

    def execute(self, **kwargs):
        return phase_b.run_temporary_stateless_canary(
            authority=self.authority,
            transport=self.transport,
            candidate=self.candidate,
            full_action_lora_state=self.state,
            live_b_parameters={B_NAME: self.b},
            base_projection_by_b_name={B_NAME: self.base},
            world_digest_consensus=self.world_consensus,
            evaluate_candidate=lambda: self.evaluation(),
            **kwargs,
        )

    def test_temporary_add_is_visible_only_to_evaluation_and_rolls_back(self) -> None:
        observed = {}

        def evaluate():
            observed["b"] = self.b.detach().clone()
            return self.evaluation()

        before = {name: value.detach().clone() for name, value in self.state.items()}
        result = phase_b.run_temporary_stateless_canary(
            authority=self.authority,
            transport=self.transport,
            candidate=self.candidate,
            full_action_lora_state=self.state,
            live_b_parameters={B_NAME: self.b},
            base_projection_by_b_name={B_NAME: self.base},
            world_digest_consensus=self.world_consensus,
            evaluate_candidate=evaluate,
        )
        self.assertGreater(torch.linalg.vector_norm(observed["b"]).item(), 0.0)
        self.assertTrue(result.execution_receipt["byte_exact_rollback_verified"])
        self.assertFalse(result.execution_receipt["retained_parameter_update"])
        self.assertFalse(result.execution_receipt["adapter_checkpoint_written"])
        for name in before:
            self.assertTrue(torch.equal(self.state[name], before[name]))
        self.assertEqual(len(self.consensus), 3)

    def test_mid_add_failure_still_restores_exact_state(self) -> None:
        before = {name: value.detach().clone() for name, value in self.state.items()}
        with self.assertRaisesRegex(phase_b.PairV7PhaseBError, "injected"):
            self.execute(_fault_after_add_index=0)
        for name in before:
            self.assertTrue(torch.equal(self.state[name], before[name]))
        self.assertEqual([row[0] for row in self.consensus], [
            "pair-v7 Phase-B pre-add state",
            "pair-v7 Phase-B post-rollback state",
        ])

    def test_evaluation_exception_still_restores_exact_state(self) -> None:
        before = self.b.detach().clone()

        def fail():
            raise RuntimeError("evaluation failed")

        with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
            phase_b.run_temporary_stateless_canary(
                authority=self.authority,
                transport=self.transport,
                candidate=self.candidate,
                full_action_lora_state=self.state,
                live_b_parameters={B_NAME: self.b},
                base_projection_by_b_name={B_NAME: self.base},
                world_digest_consensus=self.world_consensus,
                evaluate_candidate=fail,
            )
        self.assertTrue(torch.equal(self.b, before))

    def test_phase_a_no_go_cannot_create_authority(self) -> None:
        unsigned = dict(self.phase_a)
        unsigned.pop("receipt_digest")
        unsigned["geometry_audit_passed"] = False
        forged = seal(unsigned)
        with self.assertRaisesRegex(phase_b.PairV7PhaseBError, "geometry GO"):
            phase_b.build_temporary_canary_authority(
                phase_a_receipt=forged,
                expected_phase_a_receipt_digest=forged["receipt_digest"],
                maximum_delta_norm=0.05,
            )

    def test_authority_rejects_effective_delta_w_bound_above_one_percent(self) -> None:
        with self.assertRaisesRegex(
            phase_b.PairV7PhaseBError, "trust bound is too large"
        ):
            phase_b.build_temporary_canary_authority(
                phase_a_receipt=self.phase_a,
                expected_phase_a_receipt_digest=self.phase_a["receipt_digest"],
                maximum_delta_norm=0.05,
                maximum_effective_delta_w_relative_norm=0.0100001,
            )

    def test_effective_delta_w_trust_bound_fails_before_add(self) -> None:
        tiny_base = torch.full((2, 2), 0.01, dtype=torch.bfloat16)
        with self.assertRaisesRegex(phase_b.PairV7PhaseBError, "delta-W trust"):
            phase_b.run_temporary_stateless_canary(
                authority=self.authority,
                transport=self.transport,
                candidate=self.candidate,
                full_action_lora_state=self.state,
                live_b_parameters={B_NAME: self.b},
                base_projection_by_b_name={B_NAME: tiny_base},
                world_digest_consensus=self.world_consensus,
                evaluate_candidate=lambda: self.evaluation(),
            )
        self.assertEqual(torch.count_nonzero(self.b).item(), 0)

    def test_unsealed_evaluation_is_rejected_and_rolled_back(self) -> None:
        before = self.b.detach().clone()
        with self.assertRaisesRegex(phase_b.PairV7PhaseBError, "evaluation receipt"):
            phase_b.run_temporary_stateless_canary(
                authority=self.authority,
                transport=self.transport,
                candidate=self.candidate,
                full_action_lora_state=self.state,
                live_b_parameters={B_NAME: self.b},
                base_projection_by_b_name={B_NAME: self.base},
                world_digest_consensus=self.world_consensus,
                evaluate_candidate=lambda: {
                    "temporary_canary_evaluation_complete": True,
                    "canary_gate_passed": True,
                },
            )
        self.assertTrue(torch.equal(self.b, before))


if __name__ == "__main__":
    unittest.main()
