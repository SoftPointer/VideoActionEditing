from __future__ import annotations

from dataclasses import replace
import pathlib
import sys
import types
import unittest


HERE = pathlib.Path(__file__).resolve()
MODULE_ROOT = HERE.parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

try:
    import torch
except ImportError:  # pragma: no cover - environment dependent
    torch = None

if torch is not None:
    import action_state_reconstruction_v1 as state


SHA_A = "1" * 64
SHA_B = "2" * 64
SHA_C = "3" * 64
SHA_D = "4" * 64


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class ActionStateReconstructionV1Test(unittest.TestCase):
    def _target(self, batch: int = 3) -> "state.StructuredActionStateV1":
        generator = torch.Generator().manual_seed(17)
        phase = torch.randn(
            batch,
            state.PHASE_COUNT,
            len(state.PHASE_CONTINUOUS_AXES),
            generator=generator,
        ).requires_grad_(True)
        global_value = torch.randn(
            batch, len(state.GLOBAL_CONTINUOUS_AXES), generator=generator
        ).requires_grad_(True)
        phase_state = (
            torch.arange(batch * state.PHASE_COUNT, dtype=torch.long)
            .reshape(batch, state.PHASE_COUNT)
            % len(state.PHASE_STATE_CLASSES)
        )
        return state.StructuredActionStateV1(
            phase_continuous=phase,
            phase_continuous_valid=torch.ones_like(phase, dtype=torch.bool),
            phase_state=phase_state,
            phase_state_valid=torch.ones_like(phase_state, dtype=torch.bool),
            global_continuous=global_value,
            global_continuous_valid=torch.ones_like(global_value, dtype=torch.bool),
        )

    def _loss_fixture(self, batch: int = 3):
        target = self._target(batch)
        samples = tuple("sample-%d" % index for index in range(batch))
        target_receipt = state.build_action_state_point_teacher_receipt_v1(
            target=target,
            sample_ids=samples,
            split_manifest_sha256=SHA_A,
            producer_artifact_sha256=SHA_B,
        )
        decoder = state.ActionStateDecoderV1(
            state.ActionStateDecoderConfigV1(action_width=16, hidden_width=32)
        )
        decoder_receipt = state.bind_frozen_action_state_decoder_v1(
            decoder=decoder,
            checkpoint_artifact_sha256=SHA_C,
            fit_split_manifest_sha256=SHA_D,
        )
        phase = torch.randn(batch, state.PHASE_COUNT, 16, requires_grad=True)
        global_token = torch.randn(batch, 16, requires_grad=True)
        plan = types.SimpleNamespace(phase_tokens=phase, global_token=global_token)
        plan_receipt = state.build_predicted_action_plan_receipt_v1(
            plan=plan,
            sample_ids=samples,
            predictor_artifact_sha256=SHA_B,
        )
        return target, target_receipt, decoder, decoder_receipt, plan, plan_receipt

    def test_frozen_decoder_preserves_q_pred_gradients(self) -> None:
        target, target_receipt, decoder, decoder_receipt, plan, plan_receipt = self._loss_fixture()
        self.assertFalse(decoder.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in decoder.parameters()))
        loss, components = state.action_state_reconstruction_loss_v1(
            decoder=decoder,
            decoder_receipt=decoder_receipt,
            plan=plan,
            plan_receipt=plan_receipt,
            target=target,
            target_receipt=target_receipt,
        )
        loss.backward()
        self.assertGreater(float(plan.phase_tokens.grad.abs().sum()), 0.0)
        self.assertGreater(float(plan.global_token.grad.abs().sum()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in decoder.parameters()))
        self.assertIsNone(target.phase_continuous.grad)
        self.assertIsNone(target.global_continuous.grad)
        self.assertTrue(components["decoder_all_parameters_frozen"])

    def test_free_form_or_anchor_teacher_cannot_enter_point_loss(self) -> None:
        target, target_receipt, decoder, decoder_receipt, plan, plan_receipt = self._loss_fixture(2)
        anchor_receipt = replace(target_receipt, role="q_anchor")
        with self.assertRaisesRegex(state.ActionStateReconstructionError, "anchor-derived"):
            state.action_state_reconstruction_loss_v1(
                decoder=decoder,
                decoder_receipt=decoder_receipt,
                plan=plan,
                plan_receipt=plan_receipt,
                target=target,
                target_receipt=anchor_receipt,
            )
        with self.assertRaises(TypeError):
            state.action_state_reconstruction_loss_v1(
                decoder=decoder,
                decoder_receipt=decoder_receipt,
                plan=plan,
                plan_receipt=plan_receipt,
                target=target,
                target_receipt=target_receipt,
                teacher_role="q_y",
            )

    def test_receipts_bind_target_plan_and_sample_order(self) -> None:
        target, target_receipt, decoder, decoder_receipt, plan, plan_receipt = self._loss_fixture(2)
        with torch.no_grad():
            plan.phase_tokens[0, 0, 0].add_(1.0)
        with self.assertRaisesRegex(state.ActionStateReconstructionError, "plan bytes"):
            state.action_state_reconstruction_loss_v1(
                decoder=decoder,
                decoder_receipt=decoder_receipt,
                plan=plan,
                plan_receipt=plan_receipt,
                target=target,
                target_receipt=target_receipt,
            )
        changed_target = replace(
            target,
            phase_continuous=target.phase_continuous.detach().clone() + 0.5,
        )
        with self.assertRaisesRegex(state.ActionStateReconstructionError, "target bytes"):
            state.validate_action_state_point_teacher_receipt_v1(
                target_receipt, target=changed_target
            )

    def test_decoder_receipt_rejects_trainable_or_mutated_head(self) -> None:
        _, _, decoder, receipt, _, _ = self._loss_fixture(2)
        parameter = next(decoder.parameters())
        parameter.requires_grad_(True)
        with self.assertRaisesRegex(state.ActionStateReconstructionError, "require gradients"):
            state.validate_frozen_action_state_decoder_v1(decoder, receipt)
        parameter.requires_grad_(False)
        with torch.no_grad():
            parameter.reshape(-1)[0].add_(1.0)
        with self.assertRaisesRegex(state.ActionStateReconstructionError, "state differs"):
            state.validate_frozen_action_state_decoder_v1(decoder, receipt)
        with self.assertRaisesRegex(state.ActionStateReconstructionError, "frozen"):
            decoder.train(True)

    def _audit_fixture(self, batch: int = 24):
        generator = torch.Generator().manual_seed(23)
        phase_target = torch.randn(
            batch, state.PHASE_COUNT, len(state.PHASE_CONTINUOUS_AXES), generator=generator
        )
        global_target = torch.randn(
            batch, len(state.GLOBAL_CONTINUOUS_AXES), generator=generator
        )
        phase_state = (
            torch.arange(batch * state.PHASE_COUNT, dtype=torch.long)
            .reshape(batch, state.PHASE_COUNT)
            % len(state.PHASE_STATE_CLASSES)
        )
        logits = torch.full(
            (batch, state.PHASE_COUNT, len(state.PHASE_STATE_CLASSES)), -8.0
        )
        logits.scatter_(2, phase_state.unsqueeze(-1), 8.0)
        values = dict(
            phase_prediction=phase_target.clone(),
            phase_target=phase_target,
            phase_valid=torch.ones_like(phase_target, dtype=torch.bool),
            global_prediction=global_target.clone(),
            global_target=global_target,
            global_valid=torch.ones_like(global_target, dtype=torch.bool),
            phase_state_logits=logits,
            phase_state_target=phase_state,
            phase_state_valid=torch.ones_like(phase_state, dtype=torch.bool),
            action_codes=torch.randn(batch, 16, generator=generator),
            instruction_centroid_r2=0.1,
            source_only_r2=0.1,
            within_family_shuffle_r2=0.1,
            appearance_correlation=0.0,
        )
        receipt = state.build_local_action_state_audit_receipt_v1(
            sample_ids=tuple("heldout-%03d" % index for index in range(batch)),
            train_group_ids=("train-a", "train-b"),
            heldout_group_ids=("heldout-a", "heldout-b"),
            split_manifest_sha256=SHA_A,
            decoder_receipt_sha256=SHA_B,
            evaluator_artifact_sha256=SHA_C,
            **values,
        )
        return values, receipt

    def test_perfect_target_clone_is_local_only_never_qualified(self) -> None:
        values, receipt = self._audit_fixture()
        report = state.build_action_representation_audit_v1(audit_receipt=receipt, **values)
        self.assertTrue(report["local_checks_passed"])
        self.assertFalse(report["qualified"])
        self.assertFalse(report["formally_qualified"])
        self.assertFalse(report["qualification_authority_available"])
        self.assertFalse(report["training_authorized"])
        self.assertFalse(report["selection_authorized"])

    def test_audit_receipt_binds_scalars_valid_and_predictions(self) -> None:
        values, receipt = self._audit_fixture()
        modifications = []
        modifications.append({"appearance_correlation": 0.1})
        modifications.append({"phase_prediction": values["phase_prediction"] + 0.01})
        changed_valid = values["phase_valid"].clone()
        changed_valid[0, 0, 0] = False
        modifications.append({"phase_valid": changed_valid})
        for changed in modifications:
            modified = dict(values)
            modified.update(changed)
            with self.assertRaisesRegex(state.ActionStateReconstructionError, "differ from receipt"):
                state.build_action_representation_audit_v1(audit_receipt=receipt, **modified)

    def test_insufficient_axis_or_class_abstains_fail_closed(self) -> None:
        values, _ = self._audit_fixture(24)
        values["phase_valid"][:, :, 0] = False
        values["phase_state_valid"] = values["phase_state_target"] == 0
        receipt = state.build_local_action_state_audit_receipt_v1(
            sample_ids=tuple("heldout-%03d" % index for index in range(24)),
            train_group_ids=("train",),
            heldout_group_ids=("heldout",),
            split_manifest_sha256=SHA_A,
            decoder_receipt_sha256=SHA_B,
            evaluator_artifact_sha256=SHA_C,
            **values,
        )
        report = state.build_action_representation_audit_v1(audit_receipt=receipt, **values)
        self.assertFalse(report["local_checks_passed"])
        self.assertFalse(report["checks"]["all_phase_axes_estimable"])
        self.assertFalse(report["checks"]["all_phase_classes_estimable"])
        self.assertFalse(report["formally_qualified"])

    def test_constant_axes_and_overlapping_split_fail_closed(self) -> None:
        target = torch.ones(10, state.PHASE_COUNT, len(state.PHASE_CONTINUOUS_AXES))
        result = state.masked_axis_r2_v1(
            target.clone(),
            target,
            torch.ones_like(target, dtype=torch.bool),
            axis_names=state.PHASE_CONTINUOUS_AXES,
        )
        self.assertEqual(result["summary"]["estimable_axes"], 0)
        values, _ = self._audit_fixture()
        with self.assertRaisesRegex(state.ActionStateReconstructionError, "not content-disjoint"):
            state.build_local_action_state_audit_receipt_v1(
                sample_ids=tuple("s-%d" % index for index in range(24)),
                train_group_ids=("same",),
                heldout_group_ids=("same",),
                split_manifest_sha256=SHA_A,
                decoder_receipt_sha256=SHA_B,
                evaluator_artifact_sha256=SHA_C,
                **values,
            )

    def test_contract_exposes_typed_qy_hard_blocker(self) -> None:
        contract = state.contract_v1()
        self.assertFalse(contract["local_audit_can_formally_qualify"])
        self.assertTrue(contract["typed_q_y_encoder_hard_blocker"])
        self.assertFalse(contract["old_r7_random_lift_accepted_as_typed_q_y"])
        self.assertFalse(contract["selection_authorized"])


if __name__ == "__main__":
    unittest.main()
