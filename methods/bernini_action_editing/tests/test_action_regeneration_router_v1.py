from __future__ import annotations

from dataclasses import replace
import pathlib
import sys
import types
import unittest


HERE = pathlib.Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]
MODULE_ROOT = HERE.parents[1]
for path in (str(REPO_ROOT), str(MODULE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    import torch
except ImportError:  # pragma: no cover - environment dependent
    torch = None

if torch is not None:
    import action_regeneration_router_v1 as regen


SHA_A = "1" * 64
SHA_B = "2" * 64
SHA_C = "3" * 64


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class ActionRegenerationRouterV1Test(unittest.TestCase):
    def _teacher_inputs(self, *, batch: int = 1, with_invalid: bool = False):
        source = torch.zeros(batch, 1, regen.PHASE_COUNT, 4, 5)
        target = torch.zeros_like(source)
        contact = torch.zeros_like(source)
        source[:, :, 1:, 1, 1] = 1.0
        target[:, :, 1:, 2, 3] = 1.0
        contact[:, :, 1:, 0, 4] = 1.0
        valid = torch.ones_like(source, dtype=torch.bool)
        if with_invalid:
            valid[:, :, 1:, 1, 2] = False
            valid[:, :, 1:, 0, 3] = False
        return source, target, contact, valid

    def _teacher_gate(self, *, batch: int = 1, dilation: int = 0, with_invalid: bool = False):
        source, target, contact, valid = self._teacher_inputs(
            batch=batch, with_invalid=with_invalid
        )
        samples = tuple("sample-%d" % index for index in range(batch))
        receipt = regen.build_teacher_gate_receipt_v1(
            source_occupancy=source,
            target_occupancy_in_source_coordinates=target,
            contact_corridor=contact,
            valid=valid,
            dilation=dilation,
            sample_ids=samples,
            producer_artifact_sha256=SHA_A,
            split_manifest_sha256=SHA_B,
        )
        gate = regen.build_teacher_state_change_gate_v1(
            source_occupancy=source,
            target_occupancy_in_source_coordinates=target,
            contact_corridor=contact,
            valid=valid,
            dilation=dilation,
            teacher_receipt=receipt,
        )
        return gate

    def _oracle_soft_gate(self, support: "torch.Tensor", *, value: float = 0.5):
        valid = torch.ones_like(support, dtype=torch.bool)
        source = torch.zeros_like(support, dtype=torch.float32)
        target = torch.zeros_like(source)
        target[support] = value
        contact = torch.zeros_like(source)
        receipt = regen.build_teacher_gate_receipt_v1(
            source_occupancy=source,
            target_occupancy_in_source_coordinates=target,
            contact_corridor=contact,
            valid=valid,
            dilation=0,
            sample_ids=tuple("sample-%d" % index for index in range(support.shape[0])),
            producer_artifact_sha256=SHA_A,
            split_manifest_sha256=SHA_B,
        )
        return regen.build_teacher_state_change_gate_v1(
            source_occupancy=source,
            target_occupancy_in_source_coordinates=target,
            contact_corridor=contact,
            valid=valid,
            dilation=0,
            teacher_receipt=receipt,
        )

    def _artifact(self, tensor, role, samples=("sample-0",)):
        derivation = (
            SHA_C
            if role in ("independent_regeneration_noise", "high_r2v_clean")
            else SHA_B
        )
        return regen.build_regeneration_tensor_artifact_receipt_v1(
            tensor=tensor,
            role=role,
            sample_ids=samples,
            producer_artifact_sha256=SHA_A,
            producer_checkpoint_sha256=SHA_B,
            producer_config_sha256=SHA_C,
            producer_frozen=True,
            input_payload_sha256=SHA_A,
            source_identity_sha256=SHA_B,
            external_manifest_sha256=SHA_B,
            derivation_key_sha256=derivation,
            solver_sigma=0.75,
            solver_step=0,
        )

    def test_teacher_gate_remasks_dilation_and_keeps_contact_separate(self) -> None:
        gate = self._teacher_gate(dilation=1, with_invalid=True)
        self.assertEqual(float(gate.delete[0, 0, 1, 1, 1]), 1.0)
        self.assertEqual(float(gate.create[0, 0, 1, 2, 3]), 1.0)
        self.assertEqual(float(gate.contact_permission[0, 0, 1, 0, 4]), 1.0)
        self.assertEqual(float(gate.create[0, 0, 1, 0, 4]), 0.0)
        self.assertEqual(float(gate.regenerate[~gate.valid].sum()), 0.0)
        self.assertFalse(bool(gate.hard_authorization_support[~gate.valid].any()))
        self.assertEqual(float(gate.regenerate[:, :, 0].sum()), 0.0)
        expected = torch.maximum(
            torch.maximum(gate.delete, gate.create), gate.contact_permission
        )
        self.assertTrue(torch.equal(gate.regenerate, expected))
        self.assertEqual(
            gate.provenance.hard_support_voxels,
            int(gate.hard_authorization_support.sum()),
        )

    def test_regenerate_union_is_exact_not_tolerance_forgeable(self) -> None:
        gate = self._teacher_gate()
        forged_regenerate = gate.regenerate.clone()
        forged_regenerate[0, 0, 2, 0, 0] = 5.0e-7
        forged = replace(
            gate,
            regenerate=forged_regenerate,
            preserve=1.0 - forged_regenerate,
        )
        with self.assertRaisesRegex(regen.ActionRegenerationRouterError, "exactly equal"):
            regen.validate_state_change_gate_v1(forged)

    def test_hard_support_sparsity_is_checked_per_sample_and_phase(self) -> None:
        source = torch.zeros(2, 1, regen.PHASE_COUNT, 4, 5)
        target = torch.zeros_like(source)
        target[0, 0, 1] = 1.0
        valid = torch.ones_like(source, dtype=torch.bool)
        receipt = regen.build_teacher_gate_receipt_v1(
            source_occupancy=source,
            target_occupancy_in_source_coordinates=target,
            valid=valid,
            dilation=0,
            sample_ids=("dense", "empty"),
            producer_artifact_sha256=SHA_A,
            split_manifest_sha256=SHA_B,
        )
        with self.assertRaisesRegex(regen.ActionRegenerationRouterError, "per sample/phase"):
            regen.build_teacher_state_change_gate_v1(
                source_occupancy=source,
                target_occupancy_in_source_coordinates=target,
                valid=valid,
                dilation=0,
                teacher_receipt=receipt,
            )

    def test_predicted_soft_gate_is_zero_outside_receipt_bound_hard_support(self) -> None:
        support = torch.zeros(1, 1, regen.PHASE_COUNT, 4, 5, dtype=torch.bool)
        support[:, :, 3, 2, 2] = True
        logits = torch.zeros_like(support, dtype=torch.float32)
        with self.assertRaisesRegex(regen.ActionRegenerationRouterError, "cannot self-authorize"):
            regen.build_predicted_gate_authorization_receipt_v1(
                delete_logits=logits,
                create_logits=logits,
                contact_logits=logits,
                hard_authorization_support=support,
                valid=torch.ones_like(support),
                sample_ids=("sample-0",),
                source_instruction_condition_sha256=SHA_A,
                predictor_artifact_sha256=SHA_B,
                split_manifest_sha256=SHA_C,
            )
        gate = self._oracle_soft_gate(support)
        self.assertEqual(float(gate.regenerate[0, 0, 3, 2, 2]), 0.5)
        self.assertTrue(torch.equal(gate.regenerate[~support], torch.zeros_like(gate.regenerate[~support])))
        self.assertTrue(torch.equal(gate.hard_authorization_support, support))
        anchor = replace(gate.provenance, origin="self_generated_anchor")
        with self.assertRaisesRegex(regen.ActionRegenerationRouterError, "anchor-derived"):
            regen.validate_state_change_gate_v1(replace(gate, provenance=anchor))

    def test_predictor_handles_nondivisible_legacy_group_count_and_rejects_int_q(self) -> None:
        config = regen.RegenerationGatePredictorConfigV1(
            source_channels=4, action_width=8, hidden_channels=30
        )
        predictor = regen.ActionRegenerationGatePredictorV1(config)
        source = torch.randn(2, 4, regen.PHASE_COUNT, 4, 5, requires_grad=True)
        phase = torch.randn(2, regen.PHASE_COUNT, 8, requires_grad=True)
        global_token = torch.randn(2, 8, requires_grad=True)
        delete_logits, create_logits, contact_logits = predictor(
            source, types.SimpleNamespace(phase_tokens=phase, global_token=global_token)
        )
        (delete_logits.mean() + create_logits.mean() + contact_logits.mean()).backward()
        self.assertGreater(float(phase.grad.abs().sum()), 0.0)
        with self.assertRaisesRegex(regen.ActionRegenerationRouterError, "floating-point dtype"):
            predictor(
                source.detach(),
                types.SimpleNamespace(
                    phase_tokens=torch.ones(2, regen.PHASE_COUNT, 8, dtype=torch.long),
                    global_token=torch.ones(2, 8, dtype=torch.long),
                ),
            )

    def test_gate_loss_uses_only_valid_and_rejects_non_qy_receipt(self) -> None:
        gate = self._teacher_gate(with_invalid=True)
        logits_a = torch.zeros_like(gate.delete, requires_grad=True)
        logits_b = logits_a.detach().clone()
        logits_b[~gate.valid] = 100.0
        loss_a, components = regen.state_change_gate_loss_v1(
            delete_logits=logits_a,
            create_logits=logits_a,
            contact_logits=logits_a,
            target=gate,
        )
        loss_b, _ = regen.state_change_gate_loss_v1(
            delete_logits=logits_b,
            create_logits=logits_b,
            contact_logits=logits_b,
            target=gate,
        )
        self.assertEqual(float(loss_a.detach()), float(loss_b.detach()))
        loss_a.backward()
        self.assertEqual(float(logits_a.grad[~gate.valid].abs().sum()), 0.0)
        self.assertFalse(components["invalid_voxels_used_as_negatives"])
        predicted_target = replace(
            gate,
            provenance=replace(
                gate.provenance,
                role="q_pred_hard_authorization",
                origin="source_instruction_support_predictor",
            ),
        )
        with self.assertRaisesRegex(regen.ActionRegenerationRouterError, "anchor-derived"):
            regen.state_change_gate_loss_v1(
                delete_logits=torch.zeros_like(gate.delete),
                create_logits=torch.zeros_like(gate.create),
                contact_logits=torch.zeros_like(gate.contact_permission),
                target=predicted_target,
            )

    def test_noise_route_rejects_shared_storage_and_dtype_mismatch(self) -> None:
        gate = self._teacher_gate()
        backing = torch.zeros(1, 6, regen.PHASE_COUNT, 4, 5)
        source = backing[:, :3]
        independent = backing[:, 3:]
        self.assertFalse(regen.tensor_byte_ranges_overlap_v1(source, independent))
        overlapping = backing[:, 1:4]
        self.assertTrue(regen.tensor_byte_ranges_overlap_v1(source, overlapping))
        if hasattr(torch, "frombuffer"):
            count = 1 * 3 * regen.PHASE_COUNT * 4 * 5
            raw = bytearray((count + 1) * 4)
            first_wrapper = torch.frombuffer(
                raw, dtype=torch.float32, count=count, offset=0
            ).reshape(1, 3, regen.PHASE_COUNT, 4, 5)
            second_wrapper = torch.frombuffer(
                raw, dtype=torch.float32, count=count, offset=4
            ).reshape(1, 3, regen.PHASE_COUNT, 4, 5)
            self.assertNotEqual(first_wrapper.data_ptr(), second_wrapper.data_ptr())
            self.assertTrue(
                regen.tensor_byte_ranges_overlap_v1(first_wrapper, second_wrapper)
            )
        with self.assertRaisesRegex(regen.ActionRegenerationRouterError, "cannot self-authorize"):
            self._artifact(source, "source_correlated_noise")
        owned_source = torch.zeros(1, 3, regen.PHASE_COUNT, 4, 5)
        separate = torch.ones_like(owned_source, dtype=torch.float64)
        with self.assertRaisesRegex(regen.ActionRegenerationRouterError, "external artifact authority"):
            regen.mix_regeneration_noise_v1(
                source_correlated_noise=owned_source,
                independent_regeneration_noise=separate,
                gate=gate,
                source_receipt=None,
                independent_receipt=None,
            )

    def test_noise_and_clean_use_exact_hard_selection(self) -> None:
        gate = self._teacher_gate()
        source_noise = torch.full((1, 3, regen.PHASE_COUNT, 4, 5), 0.0)
        independent = torch.zeros_like(source_noise)
        independent.view(torch.int32).fill_(-2147483648)
        with self.assertRaisesRegex(regen.ActionRegenerationRouterError, "external artifact authority"):
            regen.mix_regeneration_noise_v1(
                source_correlated_noise=source_noise,
                independent_regeneration_noise=independent,
                gate=gate,
                source_receipt=None,
                independent_receipt=None,
            )
        base = torch.full((1, 2, regen.PHASE_COUNT, 4, 5), 0.0)
        high = torch.zeros_like(base)
        high.view(torch.int32).fill_(-2147483648)
        with self.assertRaisesRegex(regen.ActionRegenerationRouterError, "external artifact authority"):
            regen.route_high_r2v_regeneration_v1(
                source_aware_clean=base,
                high_r2v_clean=high,
                gate=gate,
                source_receipt=None,
                high_r2v_receipt=None,
            )
        active_plan = regen.state_change_phase_plan_v1(gate)
        source_video = torch.zeros(1, regen.PHASE_COUNT, 4, 5, 2)
        with self.assertRaisesRegex(regen.ActionRegenerationRouterError, "active SPT generation"):
            regen.execute_state_change_phase_plan_clean_v1(
                source_clean=source_video,
                generated_clean=torch.ones_like(source_video, requires_grad=True),
                plan=active_plan,
            )

    def test_soft_route_is_local_numeric_blend_and_outside_uses_hard_support(self) -> None:
        support = torch.zeros(1, 1, regen.PHASE_COUNT, 4, 5, dtype=torch.bool)
        support[:, :, 4, 1, 2] = True
        gate = self._oracle_soft_gate(support)
        base = torch.zeros(1, 2, regen.PHASE_COUNT, 4, 5)
        base.view(torch.int32).fill_(-2147483648)
        high = torch.ones_like(base)
        with self.assertRaisesRegex(regen.ActionRegenerationRouterError, "external artifact authority"):
            regen.route_high_r2v_regeneration_v1(
                source_aware_clean=base,
                high_r2v_clean=high,
                gate=gate,
                source_receipt=None,
                high_r2v_receipt=None,
            )

    def test_fp16_cast_cannot_promote_soft_probability_to_hard_authority(self) -> None:
        support = torch.zeros(1, 1, regen.PHASE_COUNT, 4, 5, dtype=torch.bool)
        support[:, :, 5, 2, 2] = True
        gate = self._oracle_soft_gate(support, value=0.9999)
        base = torch.zeros(1, 2, regen.PHASE_COUNT, 4, 5, dtype=torch.float16)
        high = torch.ones_like(base)
        self.assertNotEqual(float(gate.regenerate[0, 0, 5, 2, 2]), 1.0)
        with self.assertRaisesRegex(regen.ActionRegenerationRouterError, "external artifact authority"):
            regen.route_high_r2v_regeneration_v1(
                source_aware_clean=base,
                high_r2v_clean=high,
                gate=gate,
                source_receipt=None,
                high_r2v_receipt=None,
            )

    def test_zero_gate_returns_same_object_bytes_and_spt_semantic_noop(self) -> None:
        support = torch.zeros(1, 1, regen.PHASE_COUNT, 4, 5, dtype=torch.bool)
        gate = self._oracle_soft_gate(support, value=0.0)
        base = torch.zeros(1, 2, regen.PHASE_COUNT, 4, 5)
        base.view(torch.int32).fill_(-2147483648)
        high = torch.ones_like(base)
        before = base.view(torch.uint8).clone()
        result = regen.route_high_r2v_regeneration_v1(
            source_aware_clean=base,
            high_r2v_clean=high,
            gate=gate,
            source_receipt=None,
            high_r2v_receipt=None,
        )
        self.assertIs(result.clean, base)
        self.assertTrue(torch.equal(result.clean.view(torch.uint8), before))
        source_noise = base.clone()
        source_noise_before = source_noise.view(torch.uint8).clone()
        mixed = regen.mix_regeneration_noise_v1(
            source_correlated_noise=source_noise,
            independent_regeneration_noise=torch.ones_like(source_noise),
            gate=gate,
            source_receipt=None,
            independent_receipt=None,
        )
        self.assertIs(mixed, source_noise)
        self.assertTrue(torch.equal(mixed.view(torch.uint8), source_noise_before))
        plan = regen.state_change_phase_plan_v1(gate)
        source_video = torch.zeros(1, regen.PHASE_COUNT, 4, 5, 2)
        source_video.view(torch.int32).fill_(-2147483648)
        source_video_before = source_video.view(torch.uint8).clone()
        generated_video = torch.ones_like(source_video)
        executed = regen.execute_state_change_phase_plan_clean_v1(
            source_clean=source_video,
            generated_clean=generated_video,
            plan=plan,
        )
        self.assertIs(executed, source_video)
        self.assertTrue(torch.equal(executed.view(torch.uint8), source_video_before))
        self.assertTrue(plan.diagnostics["semantic_noop"])
        forged_plan = types.SimpleNamespace(
            diagnostics={"schema_version": regen.SCHEMA_VERSION, "semantic_noop": True}
        )
        with self.assertRaisesRegex(regen.ActionRegenerationRouterError, "receipt is missing"):
            regen.execute_state_change_phase_plan_clean_v1(
                source_clean=source_video,
                generated_clean=generated_video,
                plan=forged_plan,
            )
        plan.diagnostics["semantic_noop"] = False
        executed_again = regen.execute_state_change_phase_plan_clean_v1(
            source_clean=source_video,
            generated_clean=generated_video,
            plan=plan,
        )
        self.assertIs(executed_again, source_video)
        plan.gate_probs[:, 0, 1, 0, 0] = 0.0
        plan.gate_probs[:, 2, 1, 0, 0] = 1.0
        with self.assertRaisesRegex(regen.ActionRegenerationRouterError, "gates changed"):
            regen.execute_state_change_phase_plan_clean_v1(
                source_clean=source_video,
                generated_clean=generated_video,
                plan=plan,
            )

    def test_receipts_reject_tensor_mutation_and_contract_is_fail_closed(self) -> None:
        source = torch.zeros(1, 2, regen.PHASE_COUNT, 4, 5)
        trainable_high = torch.ones_like(source, requires_grad=True)
        with self.assertRaisesRegex(regen.ActionRegenerationRouterError, "cannot self-authorize"):
            self._artifact(trainable_high, "high_r2v_clean")
        contract = regen.contract_v1()
        self.assertTrue(contract["soft_probability_separate_from_hard_authorization"])
        self.assertFalse(contract["predicted_gate_execution_authorized"])
        self.assertTrue(contract["spt_semantic_noop_wrapper_required"])
        self.assertFalse(contract["external_artifact_authority_implemented"])
        self.assertFalse(contract["active_high_r2v_routing_authorized"])
        self.assertFalse(contract["active_spt_execution_authorized"])
        self.assertFalse(contract["training_authorized"])
        self.assertFalse(contract["gpu_launch_authorized"])


if __name__ == "__main__":
    unittest.main()
