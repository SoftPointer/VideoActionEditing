from __future__ import annotations

from dataclasses import replace
import inspect
import pathlib
import sys
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
    import source_relative_action_target_v2 as target_v2


SHA_A = "1" * 64
SHA_B = "2" * 64
SHA_C = "3" * 64
SHA_D = "4" * 64


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class SourceRelativeActionTargetV2Test(unittest.TestCase):
    def _evidence(
        self,
        batch: int = 2,
        *,
        raw_shift=(0.0, 0.0),
        target_motion_scale: float = 1.0,
    ):
        phases = torch.arange(target_v2.PHASE_COUNT, dtype=torch.float32)
        # The action settles at the terminal phase (10) and remains fixed
        # throughout the explicit hold window.
        progress = ((phases - 2.0) / 8.0).clamp(min=0.0, max=1.0)
        source_actor = torch.zeros(
            batch,
            target_v2.PHASE_COUNT,
            target_v2.ACTOR_SLOT_COUNT,
            2,
            dtype=torch.float32,
        )
        source_actor[:, :, 0, 0] = 10.0
        source_actor[:, :, 0, 1] = 5.0
        source_actor[:, :, 1, 0] = 14.0
        source_actor[:, :, 1, 1] = 5.0
        target_actor = source_actor.clone()
        target_actor[:, :, 0, 0] += target_motion_scale * 4.0 * progress
        target_actor[:, :, 0, 1] += target_motion_scale * 1.0 * progress
        target_actor[:, :, 1, 1] += target_motion_scale * 2.0 * progress

        source_object = torch.zeros(
            batch,
            target_v2.PHASE_COUNT,
            target_v2.OBJECT_SLOT_COUNT,
            2,
            dtype=torch.float32,
        )
        for slot, (x_value, y_value) in enumerate(
            ((12.0, 5.0), (16.0, 6.0), (18.0, 4.0), (22.0, 5.0))
        ):
            source_object[:, :, slot, 0] = x_value
            source_object[:, :, slot, 1] = y_value
        target_object = source_object.clone()
        target_object[:, :, 0, 0] += target_motion_scale * 6.0 * progress
        target_object[:, :, 1, 1] -= target_motion_scale * 2.0 * progress
        target_object[:, :, 2, 0] += target_motion_scale * 1.5 * progress

        shift = torch.tensor(raw_shift, dtype=torch.float32)
        source_actor += shift
        target_actor += shift
        source_object += shift
        target_object += shift
        source_h = torch.eye(3, dtype=torch.float32).reshape(1, 1, 3, 3).repeat(
            batch, target_v2.PHASE_COUNT, 1, 1
        )
        target_h = source_h.clone()
        source_h[..., 0, 2] = -float(raw_shift[0])
        source_h[..., 1, 2] = -float(raw_shift[1])
        target_h.copy_(source_h)

        actor_track = torch.ones(
            batch,
            target_v2.PHASE_COUNT,
            target_v2.ACTOR_SLOT_COUNT,
            dtype=torch.bool,
        )
        actor_track[:, 8, 1] = False
        object_track = torch.ones(
            batch,
            target_v2.PHASE_COUNT,
            target_v2.OBJECT_SLOT_COUNT,
            dtype=torch.bool,
        )
        scale = torch.full(
            (batch, target_v2.PHASE_COUNT, target_v2.ACTOR_SLOT_COUNT),
            2.0,
            dtype=torch.float32,
        )
        orientation = torch.zeros(
            batch,
            target_v2.PHASE_COUNT,
            target_v2.ACTOR_SLOT_COUNT,
            2,
            dtype=torch.float32,
        )
        orientation[..., 0] = 1.0
        source_hashes = tuple("%064x" % (index + 1) for index in range(batch))
        target_hashes = tuple("%064x" % (index + 101) for index in range(batch))
        return target_v2.SourceRelativeCameraEvidenceV2(
            sample_ids=tuple("sample-%d" % index for index in range(batch)),
            source_media_sha256=source_hashes,
            target_media_sha256=target_hashes,
            source_actor_xy=source_actor,
            target_actor_xy=target_actor,
            source_object_xy=source_object,
            target_object_xy=target_object,
            source_actor_track_valid=actor_track.clone(),
            target_actor_track_valid=actor_track.clone(),
            source_object_track_valid=object_track.clone(),
            target_object_track_valid=object_track.clone(),
            source_camera_to_stabilized=source_h,
            target_camera_to_stabilized=target_h,
            source_actor_scale=scale,
            source_actor_orientation=orientation,
        )

    def _annotations(self, batch: int = 2):
        actor_slot_valid = torch.ones(
            batch, target_v2.ACTOR_SLOT_COUNT, dtype=torch.bool
        )
        object_slot_valid = torch.ones(
            batch, target_v2.OBJECT_SLOT_COUNT, dtype=torch.bool
        )
        source_actor_presence = torch.ones(
            batch,
            target_v2.PHASE_COUNT,
            target_v2.ACTOR_SLOT_COUNT,
            dtype=torch.bool,
        )
        target_actor_presence = source_actor_presence.clone()
        source_object_presence = torch.ones(
            batch,
            target_v2.PHASE_COUNT,
            target_v2.OBJECT_SLOT_COUNT,
            dtype=torch.bool,
        )
        target_object_presence = source_object_presence.clone()
        source_presence_valid = torch.ones(
            batch,
            target_v2.PHASE_COUNT,
            target_v2.ENTITY_SLOT_COUNT,
            dtype=torch.bool,
        )
        target_presence_valid = source_presence_valid.clone()
        contact = torch.zeros(
            batch,
            target_v2.PHASE_COUNT,
            target_v2.ACTOR_SLOT_COUNT,
            target_v2.OBJECT_SLOT_COUNT,
            dtype=torch.bool,
        )
        contact[:, 3:10, 0, 0] = True
        contact_valid = torch.ones_like(contact)
        ownership = torch.full(
            (batch, target_v2.PHASE_COUNT, target_v2.OBJECT_SLOT_COUNT),
            target_v2.OWNER_ENVIRONMENT,
            dtype=torch.int64,
        )
        ownership[:, 3:10, 0] = target_v2.OWNER_PRIMARY_ACTOR
        ownership[:, 10:, 0] = target_v2.OWNER_GOAL_CONTAINER
        ownership_valid = torch.ones_like(ownership, dtype=torch.bool)
        phase_channels = torch.zeros(
            batch,
            target_v2.PHASE_COUNT,
            len(target_v2.PHASE_CHANNEL_NAMES),
            dtype=torch.bool,
        )
        phase_channels[:, 2, target_v2.ONSET_CHANNEL] = True
        phase_channels[:, 3:10, target_v2.TRANSITION_CHANNEL] = True
        phase_channels[:, 10, target_v2.TERMINAL_CHANNEL] = True
        phase_channels[:, 11:, target_v2.HOLD_CHANNEL] = True
        return target_v2.SourceRelativeActionAnnotationsV2(
            actor_slot_valid=actor_slot_valid,
            object_slot_valid=object_slot_valid,
            observed_actor_count=torch.full((batch,), 2, dtype=torch.int64),
            observed_object_count=torch.full((batch,), 4, dtype=torch.int64),
            role_assignment_unique=torch.ones(batch, dtype=torch.bool),
            ownership_unambiguous=torch.ones(batch, dtype=torch.bool),
            source_actor_presence=source_actor_presence,
            target_actor_presence=target_actor_presence,
            source_object_presence=source_object_presence,
            target_object_presence=target_object_presence,
            source_presence_valid=source_presence_valid,
            target_presence_valid=target_presence_valid,
            contact=contact,
            contact_valid=contact_valid,
            ownership=ownership,
            ownership_valid=ownership_valid,
            phase_channels=phase_channels,
            phase_valid=torch.ones(
                batch, target_v2.PHASE_COUNT, dtype=torch.bool
            ),
        )

    def _draft(
        self,
        batch: int = 2,
        *,
        evidence=None,
        annotations=None,
    ):
        evidence = evidence if evidence is not None else self._evidence(batch)
        annotations = annotations if annotations is not None else self._annotations(batch)
        camera = target_v2.build_source_relative_camera_bundle_v2(
            evidence,
            canonicalizer_artifact_sha256=SHA_A,
            annotation_manifest_sha256=SHA_B,
        )
        draft = target_v2.build_local_source_relative_action_target_draft_v2(
            camera,
            annotations,
            annotation_artifact_sha256=SHA_C,
            split_manifest_sha256=SHA_D,
        )
        return camera, annotations, draft

    def _transport_modules(self):
        encoder = target_v2.FrozenSourceRelativeSchemaEncoderV2()
        decoder = target_v2.FrozenSourceRelativeSchemaDecoderV2()
        evaluator = target_v2.FrozenSourceRelativeSchemaEvaluatorV2()
        receipt = target_v2.bind_frozen_source_relative_schema_transport_v2(
            encoder, decoder, evaluator, implementation_artifact_sha256=SHA_A
        )
        return encoder, decoder, evaluator, receipt

    def test_explicit_roles_visibility_contact_ownership_and_phase(self) -> None:
        _, _, draft = self._draft()
        target = draft.target
        self.assertTrue(bool(target.sample_valid.all()))
        self.assertEqual(
            target.actor_roles[0].tolist(),
            list(range(target_v2.ACTOR_SLOT_COUNT)),
        )
        self.assertEqual(
            target.object_roles[0].tolist(),
            list(range(target_v2.OBJECT_SLOT_COUNT)),
        )
        self.assertEqual(
            target_v2.OBJECT_ROLE_NAMES,
            (
                "primary_patient",
                "secondary_patient",
                "instrument",
                "goal_container",
            ),
        )
        self.assertEqual(
            tuple(target.contact.shape),
            (2, target_v2.PHASE_COUNT, 2, 4),
        )
        self.assertEqual(
            tuple(target.ownership.shape),
            (2, target_v2.PHASE_COUNT, 4),
        )
        self.assertEqual(tuple(target.actor_amplitude.shape), (2, 2))
        self.assertEqual(tuple(target.object_amplitude.shape), (2, 4))
        self.assertEqual(tuple(target.relative_amplitude.shape), (2, 8))
        self.assertEqual(tuple(target.mean_speed.shape), (2, 6))
        self.assertEqual(tuple(target.peak_speed.shape), (2, 6))
        self.assertTrue(bool(target.actor_presence[:, 8, 1].all()))
        self.assertFalse(bool(target.actor_delta_valid[:, 8, 1].any()))
        self.assertTrue(bool(target.contact[:, 3:10, 0, 0].all()))
        self.assertTrue(
            bool(
                (
                    target.ownership[:, 3:10, 0]
                    == target_v2.OWNER_PRIMARY_ACTOR
                ).all()
            )
        )
        self.assertTrue(
            bool(
                (
                    target.ownership[:, 10:, 0]
                    == target_v2.OWNER_GOAL_CONTAINER
                ).all()
            )
        )
        self.assertEqual(target.action_start_phase.tolist(), [2, 2])
        self.assertEqual(target.action_end_phase.tolist(), [10, 10])
        self.assertEqual(target.terminal_hold_start_phase.tolist(), [11, 11])
        self.assertEqual(target.duration_phases.tolist(), [8.0, 8.0])

    def test_camera_translation_invariance_and_source_only_scale(self) -> None:
        _, _, base = self._draft()
        shifted_evidence = self._evidence(raw_shift=(37.0, -19.0))
        _, _, shifted = self._draft(evidence=shifted_evidence)
        self.assertTrue(
            torch.allclose(
                base.target.actor_delta,
                shifted.target.actor_delta,
                rtol=0.0,
                atol=2.0e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                base.target.object_delta,
                shifted.target.object_delta,
                rtol=0.0,
                atol=2.0e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                base.target.actor_amplitude,
                shifted.target.actor_amplitude,
                rtol=0.0,
                atol=2.0e-6,
            )
        )

        doubled_evidence = self._evidence(target_motion_scale=2.0)
        _, _, doubled = self._draft(evidence=doubled_evidence)
        self.assertTrue(
            torch.allclose(
                doubled.target.actor_amplitude[:, 0],
                2.0 * base.target.actor_amplitude[:, 0],
                rtol=0.0,
                atol=1.0e-6,
            )
        )

    def test_camera_receipt_binds_raw_and_canonical_bytes(self) -> None:
        camera, _, _ = self._draft()
        changed_xy = camera.evidence.target_actor_xy.clone()
        changed_xy[0, 4, 0, 0] += 1.0
        tampered_evidence = replace(camera.evidence, target_actor_xy=changed_xy)
        with self.assertRaisesRegex(
            target_v2.SourceRelativeActionTargetError, "camera evidence bytes"
        ):
            target_v2.validate_source_relative_camera_bundle_v2(
                replace(camera, evidence=tampered_evidence)
            )
        changed_canonical = camera.coordinates.target_actor_xy.clone()
        changed_canonical[0, 4, 0, 0] += 1.0
        with self.assertRaisesRegex(
            target_v2.SourceRelativeActionTargetError, "canonical coordinate bytes"
        ):
            target_v2.validate_source_relative_camera_bundle_v2(
                replace(
                    camera,
                    coordinates=replace(
                        camera.coordinates, target_actor_xy=changed_canonical
                    ),
                )
            )

    def test_target_camera_transform_cannot_absorb_action_amplitude(self) -> None:
        evidence = self._evidence()
        target_h = evidence.target_camera_to_stabilized.clone()
        phases = torch.arange(target_v2.PHASE_COUNT, dtype=torch.float32)
        progress = ((phases - 2.0) / 8.0).clamp(min=0.0, max=1.0)
        target_h[0, :, 0, 2] -= 4.0 * progress
        target_h[0, :, 1, 2] -= 1.0 * progress
        _, _, draft = self._draft(
            evidence=replace(evidence, target_camera_to_stabilized=target_h)
        )
        self.assertTrue(bool(draft.target.abstain[0]))
        self.assertTrue(bool(draft.target.abstain_reasons[0, 0]))
        self.assertTrue(bool(draft.target.sample_valid[1]))

    def test_slot_overflow_abstains_and_clears_every_valid_axis(self) -> None:
        annotations = self._annotations()
        counts = annotations.observed_object_count.clone()
        counts[0] = target_v2.OBJECT_SLOT_COUNT + 1
        annotations = replace(annotations, observed_object_count=counts)
        _, _, draft = self._draft(annotations=annotations)
        target = draft.target
        self.assertTrue(bool(target.abstain[0]))
        self.assertTrue(bool(target.abstain_reasons[0, 1]))
        self.assertTrue(bool(target.sample_valid[1]))
        for name in (
            "actor_role_valid",
            "object_role_valid",
            "actor_delta_valid",
            "actor_target_position_valid",
            "object_delta_valid",
            "object_target_position_valid",
            "relative_delta_valid",
            "actor_presence_valid",
            "object_presence_valid",
            "source_presence_valid",
            "initial_source_presence_valid",
            "lifecycle_valid",
            "contact_valid",
            "ownership_valid",
            "phase_valid",
            "phase_summary_valid",
            "actor_amplitude_valid",
            "object_amplitude_valid",
            "relative_amplitude_valid",
            "speed_valid",
        ):
            self.assertFalse(bool(getattr(target, name)[0].any()), name)

    def test_bad_terminal_hold_abstains(self) -> None:
        annotations = self._annotations()
        phase = annotations.phase_channels.clone()
        phase[0, -1, target_v2.HOLD_CHANNEL] = False
        _, _, draft = self._draft(annotations=replace(annotations, phase_channels=phase))
        self.assertTrue(bool(draft.target.abstain[0]))
        self.assertTrue(bool(draft.target.abstain_reasons[0, 5]))
        self.assertTrue(bool(draft.target.sample_valid[1]))

    def test_invalid_source_phase0_camera_reference_abstains(self) -> None:
        evidence = self._evidence()
        scale = evidence.source_actor_scale.clone()
        scale[0, 0, 0] = 0.0
        _, _, draft = self._draft(evidence=replace(evidence, source_actor_scale=scale))
        self.assertTrue(bool(draft.target.abstain[0]))
        self.assertTrue(bool(draft.target.abstain_reasons[0, 0]))
        self.assertTrue(bool(draft.target.sample_valid[1]))

    def test_late_terminal_hold_track_loss_abstains(self) -> None:
        evidence = self._evidence()
        target_track = evidence.target_actor_track_valid.clone()
        target_track[0, 12:, 0] = False
        _, _, draft = self._draft(
            evidence=replace(evidence, target_actor_track_valid=target_track)
        )
        self.assertTrue(bool(draft.target.abstain[0]))
        self.assertTrue(bool(draft.target.abstain_reasons[0, 9]))
        self.assertTrue(bool(draft.target.sample_valid[1]))

    def test_goal_container_ownership_requires_container_presence(self) -> None:
        annotations = self._annotations()
        target_object = annotations.target_object_presence.clone()
        target_object[0, :, -1] = False
        contact_valid = annotations.contact_valid.clone()
        contact_valid[0, :, :, -1] = False
        ownership_valid = annotations.ownership_valid.clone()
        ownership_valid[0, :, -1] = False
        _, _, draft = self._draft(
            annotations=replace(
                annotations,
                target_object_presence=target_object,
                contact_valid=contact_valid,
                ownership_valid=ownership_valid,
            )
        )
        self.assertTrue(bool(draft.target.abstain[0]))
        self.assertTrue(bool(draft.target.abstain_reasons[0, 7]))

    def test_goal_container_slot_cannot_own_itself(self) -> None:
        annotations = self._annotations()
        ownership = annotations.ownership.clone()
        ownership[0, :, -1] = target_v2.OWNER_GOAL_CONTAINER
        _, _, draft = self._draft(
            annotations=replace(annotations, ownership=ownership)
        )
        self.assertTrue(bool(draft.target.abstain[0]))
        self.assertTrue(bool(draft.target.abstain_reasons[0, 3]))
        self.assertTrue(bool(draft.target.sample_valid[1]))

    def test_absent_contact_is_not_a_no_contact_negative(self) -> None:
        annotations = self._annotations()
        presence = annotations.target_object_presence.clone()
        presence[0, 5, 1] = False
        _, _, draft = self._draft(
            annotations=replace(annotations, target_object_presence=presence)
        )
        self.assertTrue(bool(draft.target.abstain[0]))
        self.assertTrue(bool(draft.target.abstain_reasons[0, 7]))

    def test_create_delete_are_derived_for_all_six_slots(self) -> None:
        annotations = self._annotations()
        source_object = annotations.source_object_presence.clone()
        target_object = annotations.target_object_presence.clone()
        source_object[:, :, 2] = False
        target_object[:, :5, 2] = False
        target_object[:, 5:, 2] = True
        target_object[:, :8, 1] = True
        target_object[:, 8:, 1] = False
        contact_valid = annotations.contact_valid.clone()
        ownership_valid = annotations.ownership_valid.clone()
        contact_valid[..., 2] &= target_object[:, :, 2, None]
        contact_valid[..., 1] &= target_object[:, :, 1, None]
        ownership_valid[:, :, 2] &= target_object[:, :, 2]
        ownership_valid[:, :, 1] &= target_object[:, :, 1]
        updated = replace(
            annotations,
            source_object_presence=source_object,
            target_object_presence=target_object,
            contact_valid=contact_valid,
            ownership_valid=ownership_valid,
        )
        _, _, draft = self._draft(annotations=updated)
        self.assertTrue(bool(draft.target.sample_valid.all()))
        object2 = target_v2.ACTOR_SLOT_COUNT + 2
        object1 = target_v2.ACTOR_SLOT_COUNT + 1
        self.assertTrue(bool(draft.target.entity_create[:, 5, object2].all()))
        self.assertEqual(int(draft.target.entity_create[:, :, object2].sum()), 2)
        self.assertTrue(bool(draft.target.entity_delete[:, 8, object1].all()))
        self.assertEqual(int(draft.target.entity_delete[:, :, object1].sum()), 2)
        self.assertFalse(bool(draft.target.object_delta_valid[:, 5:, 2].any()))
        self.assertTrue(
            bool(draft.target.object_target_position_valid[:, 5:, 2].all())
        )
        self.assertFalse(bool(draft.target.object_amplitude_valid[:, 2].any()))
        self.assertFalse(bool(draft.target.object_delta_valid[:, 8:, 1].any()))

    def test_phase0_create_is_not_a_pre_onset_source_baseline(self) -> None:
        annotations = self._annotations()
        source_object = annotations.source_object_presence.clone()
        source_object[0, :, 2] = False
        _, _, draft = self._draft(
            annotations=replace(
                annotations, source_object_presence=source_object
            )
        )
        self.assertTrue(bool(draft.target.abstain[0]))
        self.assertTrue(bool(draft.target.abstain_reasons[0, 10]))
        self.assertTrue(bool(draft.target.sample_valid[1]))

    def test_pre_onset_relation_change_abstains(self) -> None:
        annotations = self._annotations()
        contact = annotations.contact.clone()
        ownership = annotations.ownership.clone()
        contact[0, 0, 0, 0] = True
        ownership[0, 0, 0] = target_v2.OWNER_PRIMARY_ACTOR
        _, _, draft = self._draft(
            annotations=replace(
                annotations, contact=contact, ownership=ownership
            )
        )
        self.assertTrue(bool(draft.target.abstain[0]))
        self.assertTrue(bool(draft.target.abstain_reasons[0, 10]))
        self.assertTrue(bool(draft.target.sample_valid[1]))

    def test_constant_pre_onset_offset_abstains(self) -> None:
        evidence = self._evidence()
        target_actor = evidence.target_actor_xy.clone()
        target_object = evidence.target_object_xy.clone()
        offset = torch.tensor((3.0, 0.0), dtype=torch.float32)
        target_actor[0] = evidence.source_actor_xy[0] + offset
        target_object[0] = evidence.source_object_xy[0] + offset
        annotations = self._annotations()
        contact = annotations.contact.clone()
        ownership = annotations.ownership.clone()
        contact[0] = False
        ownership[0] = target_v2.OWNER_ENVIRONMENT
        _, _, draft = self._draft(
            evidence=replace(
                evidence,
                target_actor_xy=target_actor,
                target_object_xy=target_object,
            ),
            annotations=replace(
                annotations, contact=contact, ownership=ownership
            ),
        )
        self.assertTrue(bool(draft.target.abstain[0]))
        self.assertTrue(bool(draft.target.abstain_reasons[0, 10]))
        self.assertTrue(bool(draft.target.sample_valid[1]))

    def test_onset_only_jump_without_transition_evolution_abstains(self) -> None:
        evidence = self._evidence()
        target_actor = evidence.target_actor_xy.clone()
        target_object = evidence.target_object_xy.clone()
        target_actor[0] = evidence.source_actor_xy[0]
        target_object[0] = evidence.source_object_xy[0]
        target_actor[0, 2:, 0, 0] += 4.0
        target_actor[0, 2:, 0, 1] += 1.0
        target_actor[0, 2:, 1, 1] += 2.0
        target_object[0, 2:, 0, 0] += 6.0
        target_object[0, 2:, 1, 1] -= 2.0
        target_object[0, 2:, 2, 0] += 1.5
        annotations = self._annotations()
        contact = annotations.contact.clone()
        ownership = annotations.ownership.clone()
        contact[0] = False
        ownership[0] = target_v2.OWNER_ENVIRONMENT
        _, _, draft = self._draft(
            evidence=replace(
                evidence,
                target_actor_xy=target_actor,
                target_object_xy=target_object,
            ),
            annotations=replace(
                annotations, contact=contact, ownership=ownership
            ),
        )
        self.assertTrue(bool(draft.target.abstain[0]))
        self.assertTrue(bool(draft.target.abstain_reasons[0, 10]))
        self.assertTrue(bool(draft.target.sample_valid[1]))

    def test_validator_rejects_unilateral_delta_and_absent_position(self) -> None:
        _, _, draft = self._draft()
        source_presence = draft.target.source_presence.clone()
        source_presence[0, 5, 0] = False
        with self.assertRaisesRegex(
            target_v2.SourceRelativeActionTargetError,
            "both endpoints",
        ):
            target_v2.validate_source_relative_action_target_v2(
                replace(draft.target, source_presence=source_presence)
            )

        contact_valid = draft.target.contact_valid.clone()
        contact_valid[0, :2, 0, 1] = False
        with self.assertRaisesRegex(
            target_v2.SourceRelativeActionTargetError,
            "pre-onset relation-state evidence",
        ):
            target_v2.validate_source_relative_action_target_v2(
                replace(draft.target, contact_valid=contact_valid)
            )

        annotations = self._annotations()
        target_object = annotations.target_object_presence.clone()
        target_object[:, 8:, 1] = False
        contact_valid = annotations.contact_valid.clone()
        ownership_valid = annotations.ownership_valid.clone()
        contact_valid[..., 1] &= target_object[:, :, 1, None]
        ownership_valid[:, :, 1] &= target_object[:, :, 1]
        _, _, lifecycle_draft = self._draft(
            annotations=replace(
                annotations,
                target_object_presence=target_object,
                contact_valid=contact_valid,
                ownership_valid=ownership_valid,
            )
        )
        self.assertTrue(bool(lifecycle_draft.target.sample_valid.all()))
        position_valid = lifecycle_draft.target.object_target_position_valid.clone()
        position_valid[0, 8, 1] = True
        with self.assertRaisesRegex(
            target_v2.SourceRelativeActionTargetError,
            "known target presence",
        ):
            target_v2.validate_source_relative_action_target_v2(
                replace(
                    lifecycle_draft.target,
                    object_target_position_valid=position_valid,
                )
            )

    def test_e03_actor_to_goal_container_transition_roundtrips(self) -> None:
        _, _, draft = self._draft()
        encoder, decoder, evaluator, receipt = self._transport_modules()
        transport = encoder(draft, receipt)
        decoded = decoder(transport, draft.receipt, receipt)
        report = evaluator(draft, decoded, receipt)
        target = decoded.target
        self.assertTrue(bool(target.contact[:, 3:10, 0, 0].all()))
        self.assertFalse(bool(target.contact[:, 10:, 0, 0].any()))
        self.assertTrue(
            bool(
                (
                    target.ownership[:, 3:10, 0]
                    == target_v2.OWNER_PRIMARY_ACTOR
                ).all()
            )
        )
        self.assertTrue(
            bool(
                (
                    target.ownership[:, 10:, 0]
                    == target_v2.OWNER_GOAL_CONTAINER
                ).all()
            )
        )
        self.assertTrue(bool(target.phase_channels[:, 10, target_v2.TERMINAL_CHANNEL].all()))
        self.assertTrue(bool(target.phase_channels[:, 11:, target_v2.HOLD_CHANNEL].all()))
        self.assertTrue(report.exact_roundtrip)

    def test_transport_is_frozen_exact_and_never_qualification(self) -> None:
        _, _, draft = self._draft()
        encoder, decoder, evaluator, receipt = self._transport_modules()
        for module in (encoder, decoder, evaluator):
            self.assertFalse(module.training)
            self.assertEqual(tuple(module.parameters()), ())
            with self.assertRaisesRegex(
                target_v2.SourceRelativeActionTargetError, "permanently frozen"
            ):
                module.train(True)
        transport = encoder(draft, receipt)
        decoded = decoder(transport, draft.receipt, receipt)
        report = evaluator(draft, decoded, receipt)
        self.assertTrue(report.local_checks_passed)
        self.assertTrue(report.exact_roundtrip)
        self.assertFalse(report.representation_qualification_evidence)
        self.assertFalse(report.r2_evidence)
        self.assertFalse(report.formally_qualified)
        self.assertFalse(report.training_authorized)
        self.assertFalse(report.optimizer_authorized)
        self.assertFalse(report.selection_authorized)
        self.assertFalse(report.gate_authorized)
        target_v2.validate_local_schema_transport_evaluation_v2(report)
        with self.assertRaisesRegex(
            target_v2.SourceRelativeActionTargetError, "cannot authorize"
        ):
            target_v2.validate_local_schema_transport_evaluation_v2(
                replace(report, formally_qualified=True)
            )
        for name in target_v2.TARGET_FIELD_NAMES:
            self.assertTrue(
                torch.equal(getattr(draft.target, name), getattr(decoded.target, name)),
                name,
            )

    def test_transport_receipt_detects_frozen_module_mutation(self) -> None:
        encoder, decoder, evaluator, receipt = self._transport_modules()
        encoder._abi[0] += 1
        with self.assertRaisesRegex(
            target_v2.SourceRelativeActionTargetError, "canonical ABI buffer differs"
        ):
            target_v2.validate_frozen_schema_transport_receipt_v2(
                receipt, encoder=encoder, decoder=decoder, evaluator=evaluator
            )

    def test_transport_rejects_custom_execution_paths_and_presign_abi(self) -> None:
        class EvilEncoder(target_v2.FrozenSourceRelativeSchemaEncoderV2):
            def forward(self, *args, **kwargs):
                return object()

        class EvilDecoder(target_v2.FrozenSourceRelativeSchemaDecoderV2):
            def forward(self, *args, **kwargs):
                return object()

        class EvilEvaluator(target_v2.FrozenSourceRelativeSchemaEvaluatorV2):
            def forward(self, *args, **kwargs):
                return object()

        canonical = (
            target_v2.FrozenSourceRelativeSchemaEncoderV2(),
            target_v2.FrozenSourceRelativeSchemaDecoderV2(),
            target_v2.FrozenSourceRelativeSchemaEvaluatorV2(),
        )
        for modules in (
            (EvilEncoder(), canonical[1], canonical[2]),
            (canonical[0], EvilDecoder(), canonical[2]),
            (canonical[0], canonical[1], EvilEvaluator()),
        ):
            with self.assertRaisesRegex(
                target_v2.SourceRelativeActionTargetError, "exact frozen type"
            ):
                target_v2.bind_frozen_source_relative_schema_transport_v2(
                    *modules, implementation_artifact_sha256=SHA_A
                )

        encoder = target_v2.FrozenSourceRelativeSchemaEncoderV2()
        encoder.forward = lambda *args, **kwargs: object()
        with self.assertRaisesRegex(
            target_v2.SourceRelativeActionTargetError, "forward is shadowed"
        ):
            target_v2.bind_frozen_source_relative_schema_transport_v2(
                encoder,
                target_v2.FrozenSourceRelativeSchemaDecoderV2(),
                target_v2.FrozenSourceRelativeSchemaEvaluatorV2(),
                implementation_artifact_sha256=SHA_A,
            )

        encoder = target_v2.FrozenSourceRelativeSchemaEncoderV2()
        encoder.register_forward_hook(lambda module, inputs, output: object())
        with self.assertRaisesRegex(
            target_v2.SourceRelativeActionTargetError, "execution or state hooks"
        ):
            target_v2.bind_frozen_source_relative_schema_transport_v2(
                encoder,
                target_v2.FrozenSourceRelativeSchemaDecoderV2(),
                target_v2.FrozenSourceRelativeSchemaEvaluatorV2(),
                implementation_artifact_sha256=SHA_A,
            )

        encoder = target_v2.FrozenSourceRelativeSchemaEncoderV2()
        encoder._abi[0] = 999
        with self.assertRaisesRegex(
            target_v2.SourceRelativeActionTargetError, "canonical ABI buffer differs"
        ):
            target_v2.bind_frozen_source_relative_schema_transport_v2(
                encoder,
                target_v2.FrozenSourceRelativeSchemaDecoderV2(),
                target_v2.FrozenSourceRelativeSchemaEvaluatorV2(),
                implementation_artifact_sha256=SHA_A,
            )

        _, _, draft = self._draft()
        encoder, decoder, evaluator, receipt = self._transport_modules()
        encoder.register_forward_pre_hook(lambda module, inputs: inputs)
        with self.assertRaisesRegex(
            target_v2.SourceRelativeActionTargetError, "execution or state hooks"
        ):
            encoder(draft, receipt)

        encoder, _, _, receipt = self._transport_modules()
        global_events = []
        global_pre_handle = (
            torch.nn.modules.module.register_module_forward_pre_hook(
                lambda module, inputs: global_events.append("pre")
            )
        )
        global_forward_handle = torch.nn.modules.module.register_module_forward_hook(
            lambda module, inputs, output: global_events.append("forward") or object()
        )
        try:
            transport = encoder(draft, receipt)
        finally:
            global_forward_handle.remove()
            global_pre_handle.remove()
        self.assertIsInstance(transport, target_v2.LocalSchemaTransportV2)
        self.assertEqual(global_events, [])

    def test_transport_and_local_draft_tampering_fail_closed(self) -> None:
        _, _, draft = self._draft()
        encoder, decoder, _, receipt = self._transport_modules()
        transport = encoder(draft, receipt)
        changed_phase = transport.phase_code.clone()
        changed_phase[0, 0, 0] += 1.0
        with self.assertRaisesRegex(
            target_v2.SourceRelativeActionTargetError, "transport bytes"
        ):
            decoder(replace(transport, phase_code=changed_phase), draft.receipt, receipt)
        changed_delta = draft.target.actor_delta.clone()
        changed_delta[0, 8, 1, 0] = -0.0
        with self.assertRaisesRegex(
            target_v2.SourceRelativeActionTargetError, "local draft bytes"
        ):
            target_v2.validate_local_source_relative_action_target_draft_v2(
                replace(
                    draft,
                    target=replace(draft.target, actor_delta=changed_delta),
                )
            )
        with self.assertRaisesRegex(
            target_v2.SourceRelativeActionTargetError, "external authority"
        ):
            target_v2.validate_local_source_relative_action_target_draft_v2(
                replace(
                    draft,
                    receipt=replace(
                        draft.receipt, external_authority_verified=True
                    ),
                )
            )

    def test_q_anchor_and_optimizer_are_structurally_absent(self) -> None:
        contract = target_v2.source_relative_action_target_v2_contract()
        self.assertTrue(contract["qy_is_only_point_teacher_type"])
        self.assertFalse(contract["q_anchor_type_exposed"])
        self.assertFalse(contract["q_anchor_reconstruction_api_exists"])
        self.assertFalse(contract["q_anchor_gate_api_exists"])
        self.assertFalse(contract["reconstruction_loss_api_exists"])
        self.assertFalse(contract["optimizer_api_exists"])
        self.assertFalse(contract["representation_qualification_evidence"])
        self.assertFalse(contract["qy_promotion_implemented"])
        self.assertEqual(target_v2.OWNER_FREE, 0)
        self.assertEqual(target_v2.OWNER_NONE, target_v2.OWNER_FREE)
        self.assertEqual(
            contract["ownership_code_zero"],
            "known_free_only_when_ownership_valid_is_true",
        )
        self.assertIn("padding_not_free", contract["ownership_unknown_encoding"])
        public_names = tuple(target_v2.__all__)
        self.assertFalse(any("anchor" in name.lower() for name in public_names))
        signature = inspect.signature(
            target_v2.build_local_source_relative_action_target_draft_v2
        )
        self.assertNotIn("teacher_role", signature.parameters)
        self.assertNotIn("q_anchor", signature.parameters)
        camera, annotations, draft = self._draft()
        with self.assertRaises(TypeError):
            target_v2.build_local_source_relative_action_target_draft_v2(
                camera,
                annotations,
                annotation_artifact_sha256=SHA_C,
                split_manifest_sha256=SHA_D,
                q_anchor=torch.zeros(1),
            )
        with self.assertRaisesRegex(
            target_v2.SourceRelativeActionTargetError, "external clean-pair authority"
        ):
            target_v2.promote_local_draft_to_qy_source_relative_action_target_v2(
                draft, external_clean_pair_authority=object()
            )
        with self.assertRaisesRegex(
            target_v2.SourceRelativeActionTargetError, "external clean-pair authority"
        ):
            target_v2.QYSourceRelativeActionTargetV2()
        forged_qy = object.__new__(target_v2.QYSourceRelativeActionTargetV2)
        with self.assertRaisesRegex(
            target_v2.SourceRelativeActionTargetError, "external clean-pair authority"
        ):
            target_v2.validate_qy_source_relative_action_target_v2(forged_qy)

    def test_dtype_and_grad_bearing_teacher_evidence_are_rejected(self) -> None:
        evidence = self._evidence()
        with self.assertRaisesRegex(
            target_v2.SourceRelativeActionTargetError, "dtype"
        ):
            target_v2.build_source_relative_camera_bundle_v2(
                replace(evidence, source_actor_xy=evidence.source_actor_xy.double()),
                canonicalizer_artifact_sha256=SHA_A,
                annotation_manifest_sha256=SHA_B,
            )
        grad_value = evidence.source_actor_xy.clone().requires_grad_(True)
        with self.assertRaisesRegex(
            target_v2.SourceRelativeActionTargetError, "detached leaf"
        ):
            target_v2.build_source_relative_camera_bundle_v2(
                replace(evidence, source_actor_xy=grad_value),
                canonicalizer_artifact_sha256=SHA_A,
                annotation_manifest_sha256=SHA_B,
            )


if __name__ == "__main__":
    unittest.main()
