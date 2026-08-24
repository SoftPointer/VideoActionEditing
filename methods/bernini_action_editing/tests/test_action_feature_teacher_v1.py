from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace
import unittest

import numpy as np

from methods.bernini_action_editing import action_feature_teacher_v1 as teacher


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _temporal_teacher(
    *,
    direction: float = 1.0,
    amplitude: float = 0.2,
    duration: float = 1.0,
    event_start: float = 0.10,
    event_end: float = 0.70,
    camera_offset: float = 0.0,
    local_articulation: float = 0.03,
) -> SimpleNamespace:
    event_time = np.linspace(0.0, duration, teacher.INPUT_PHASES, dtype=np.float64)
    phase = np.linspace(0.0, 1.0, teacher.INPUT_PHASES, dtype=np.float32)
    trajectory = np.stack(
        (direction * amplitude * phase, 0.03 * np.sin(np.pi * phase)),
        axis=1,
    ).astype(np.float32)
    velocity64 = np.gradient(
        trajectory.astype(np.float64),
        event_time,
        axis=0,
        edge_order=2,
    )
    acceleration64 = np.gradient(
        velocity64,
        event_time,
        axis=0,
        edge_order=2,
    )
    velocity = velocity64.astype(np.float32)
    acceleration = acceleration64.astype(np.float32)
    energy = np.linalg.norm(velocity, axis=1).astype(np.float32)
    camera = np.zeros((teacher.INPUT_PHASES, 4), dtype=np.float32)
    camera[:, 0] = np.float32(camera_offset) * phase
    track_trajectories = np.zeros((8, teacher.INPUT_PHASES, 2), dtype=np.float32)
    for index in range(8):
        signed = -1.0 if index % 2 else 1.0
        track_trajectories[index] = trajectory
        track_trajectories[index, :, 1] += (
            np.float32(signed * local_articulation)
            * np.sin(np.pi * phase).astype(np.float32)
        )
    return SimpleNamespace(
        schema_version=teacher.SOURCE_TEACHER_SCHEMA,
        event_window=SimpleNamespace(
            normalized_start=float(event_start),
            normalized_end=float(event_end),
        ),
        actor_trajectory=trajectory,
        actor_velocity=velocity,
        actor_acceleration=acceleration,
        camera_trajectory=camera,
        phase_visibility=np.ones(teacher.INPUT_PHASES, dtype=np.float32),
        phase_uncertainty=np.zeros(teacher.INPUT_PHASES, dtype=np.float32),
        phase_energy=energy,
        actor_track_trajectories=track_trajectories,
        actor_track_mask=np.ones(8, dtype=np.bool_),
        event_duration=float(duration),
        background_residual_reduction=0.80,
        camera_explained_ratio=0.30,
        camera_inlier_fraction=0.90,
        camera_crossfit_valid=True,
        camera_crossfit_raw_median=0.10,
        camera_crossfit_residual_median=0.02,
        camera_crossfit_residual_reduction=0.80,
    )


def _authority(
    *,
    role: str,
    temporal: SimpleNamespace | None,
    content_group_id: str = "fixture-content-group",
    counterfactual_pair: str = "fixture-counterfactual-pair",
    actor: str = "fixture-actor",
    object_name: str = "fixture-object",
    instruction: str = "fixture-instruction",
) -> dict[str, object]:
    arrays_sha = (
        None
        if temporal is None
        else teacher.temporal_action_core(temporal).temporal_teacher_arrays_sha256
    )
    value: dict[str, object] = {
        "schema_version": teacher.UPSTREAM_AUTHORITY_SCHEMA,
        "media_role": role,
        "content_group_id": content_group_id,
        "counterfactual_pair_sha256": _sha(counterfactual_pair),
        "media_sha256": _sha(f"media:{role}"),
        "media_size": 4096,
        "track_cache_sha256": _sha(f"tracks:{role}"),
        "track_cache_size": 8192,
        "tracker_authority_sha256": _sha("tracker-authority"),
        "temporal_teacher_source_sha256": _sha("r7-source"),
        "temporal_teacher_config_sha256": _sha("r7-config"),
        "temporal_teacher_arrays_sha256": arrays_sha,
        "stability_receipt_sha256": _sha(f"stability:{role}"),
        "actor_binding_sha256": _sha(actor),
        "object_binding_sha256": _sha(object_name),
        "instruction_semantics_sha256": _sha(instruction),
        "temporal_teacher_present": temporal is not None,
        "static_noop_verified": temporal is None,
        "camera_crossfit_valid": temporal is not None,
        "perturbation_stability_passed": temporal is not None,
        "full_video_quality_passed": True,
    }
    value["authority_sha256"] = teacher.object_sha256(value)
    return value


def _redigest(value: dict[str, object]) -> dict[str, object]:
    result = copy.deepcopy(value)
    result.pop("authority_sha256", None)
    result["authority_sha256"] = teacher.object_sha256(result)
    return result


class ActionFeatureTeacherV1Test(unittest.TestCase):
    def _tokens(self, **kwargs: float) -> teacher.ActionFeatureTokens:
        action = _temporal_teacher(**kwargs)
        return teacher.build_action_feature_tokens(
            action,
            role="anchor",
            action_authority=_authority(role="anchor_action", temporal=action),
            baseline_authority=_authority(role="anchor_noop", temporal=None),
        )

    def test_exact_geometry_determinism_and_candidate_only_contract(self) -> None:
        first = self._tokens()
        second = self._tokens()
        self.assertEqual(first.phase_tokens.shape, (21, 256))
        self.assertEqual(first.global_token.shape, (256,))
        self.assertTrue(np.array_equal(first.phase_tokens, second.phase_tokens))
        self.assertTrue(np.array_equal(first.global_token, second.global_token))
        self.assertEqual(first.receipt, second.receipt)
        self.assertEqual(
            first.receipt["phase_tokens_sha256"],
            "9ad270e3bc2a2f9deed785f183cb90e05804e9224f6452746421af6f2620f33e",
        )
        self.assertEqual(
            first.receipt["global_token_sha256"],
            "caeea2fdfed9c8658e33d824455114bed8577f17a681ea00b701d25672e29a82",
        )
        self.assertEqual(
            first.receipt["receipt_sha256"],
            "1507a0952b23f6fee07de499ccb1d44edeb582af58bff4a9ee8e68c54605afbb",
        )
        self.assertEqual(first.receipt["teacher_qualification_status"], "candidate_unqualified")
        self.assertIs(first.receipt["point_distillation_authorized"], False)
        self.assertIs(first.receipt["action_following_claimed"], False)
        self.assertIs(first.receipt["camera_invariance_claimed"], False)
        self.assertIs(first.receipt["appearance_invariance_claimed"], False)
        self.assertRegex(teacher.PHASE_PROJECTION_SHA256, r"^[0-9a-f]{64}$")
        self.assertRegex(teacher.GLOBAL_PROJECTION_SHA256, r"^[0-9a-f]{64}$")

    def test_transform_global_replacement_is_detected_before_use(self) -> None:
        original = teacher._PHASE_PROJECTION
        hostile = original.copy()
        hostile[0, 0] *= np.float32(-1.0)
        teacher._PHASE_PROJECTION = hostile
        try:
            with self.assertRaisesRegex(
                teacher.ActionFeatureTeacherError,
                "transform bytes differ",
            ):
                self._tokens()
        finally:
            teacher._PHASE_PROJECTION = original

    def test_camera_is_audited_and_excluded_without_invariance_claim(self) -> None:
        still_camera = self._tokens(camera_offset=0.0)
        moving_camera = self._tokens(camera_offset=0.7)
        self.assertTrue(np.array_equal(still_camera.phase_tokens, moving_camera.phase_tokens))
        self.assertTrue(np.array_equal(still_camera.global_token, moving_camera.global_token))
        self.assertNotEqual(
            still_camera.receipt["action_camera_sha256_audit_only"],
            moving_camera.receipt["action_camera_sha256_audit_only"],
        )
        self.assertIs(still_camera.receipt["camera_invariance_claimed"], False)

    def test_direction_amplitude_and_event_onset_are_not_collapsed(self) -> None:
        forward = self._tokens(direction=1.0, amplitude=0.2)
        reverse = self._tokens(direction=-1.0, amplitude=0.2)
        larger = self._tokens(direction=1.0, amplitude=0.4)
        early = self._tokens(event_start=0.0, event_end=0.45)
        late = self._tokens(event_start=0.55, event_end=1.0)
        self.assertLess(teacher.token_cosine(forward, reverse), 0.75)
        self.assertGreater(
            float(np.linalg.norm(larger.phase_tokens)),
            float(np.linalg.norm(forward.phase_tokens)),
        )
        self.assertFalse(np.array_equal(early.phase_tokens, late.phase_tokens))
        self.assertGreater(float(np.linalg.norm(early.phase_tokens[0])), 0.0)
        self.assertEqual(float(np.linalg.norm(late.phase_tokens[0])), 0.0)

    def test_local_articulation_is_not_reduced_to_actor_centroid(self) -> None:
        rigid = self._tokens(local_articulation=0.0)
        articulated = self._tokens(local_articulation=0.08)
        self.assertFalse(np.array_equal(rigid.phase_tokens, articulated.phase_tokens))
        self.assertLess(teacher.token_cosine(rigid, articulated), 0.9999)

    def test_explicit_noop_delta_requires_matched_external_authorities(self) -> None:
        action = _temporal_teacher(direction=1.0, amplitude=0.2)
        baseline = _temporal_teacher(direction=1.0, amplitude=0.05)
        result = teacher.build_action_feature_tokens(
            action,
            role="target",
            action_authority=_authority(role="target_action", temporal=action),
            baseline_authority=_authority(role="target_noop", temporal=baseline),
            baseline_teacher=baseline,
        )
        self.assertEqual(result.receipt["baseline_mode"], "explicit_temporal_teacher")

        wrong_pair = _authority(
            role="target_noop",
            temporal=baseline,
            counterfactual_pair="different-pair",
        )
        with self.assertRaisesRegex(
            teacher.ActionFeatureTeacherError,
            "counterfactual_pair_sha256 differs",
        ):
            teacher.build_action_feature_tokens(
                action,
                role="target",
                action_authority=_authority(role="target_action", temporal=action),
                baseline_authority=wrong_pair,
                baseline_teacher=baseline,
            )

        static_claim_for_real_teacher = _authority(role="target_noop", temporal=None)
        with self.assertRaisesRegex(
            teacher.ActionFeatureTeacherError,
            "static-noop authority binding differs",
        ):
            teacher.build_action_feature_tokens(
                action,
                role="target",
                action_authority=_authority(role="target_action", temporal=action),
                baseline_authority=static_claim_for_real_teacher,
                baseline_teacher=baseline,
            )

    def test_authority_role_schema_and_redigested_mismatch_fail_closed(self) -> None:
        action = _temporal_teacher()
        wrong_role = _authority(role="target_action", temporal=action)
        with self.assertRaisesRegex(teacher.ActionFeatureTeacherError, "role differs"):
            teacher.build_action_feature_tokens(
                action,
                role="anchor",
                action_authority=wrong_role,
                baseline_authority=_authority(role="anchor_noop", temporal=None),
            )

        action_authority = _authority(role="anchor_action", temporal=action)
        forged_baseline = _authority(role="anchor_noop", temporal=None)
        forged_baseline["actor_binding_sha256"] = _sha("different-actor")
        forged_baseline = _redigest(forged_baseline)
        with self.assertRaisesRegex(teacher.ActionFeatureTeacherError, "actor_binding_sha256 differs"):
            teacher.build_action_feature_tokens(
                action,
                role="anchor",
                action_authority=action_authority,
                baseline_authority=forged_baseline,
            )

        aliased_baseline = _authority(role="anchor_noop", temporal=None)
        aliased_baseline["media_sha256"] = action_authority["media_sha256"]
        aliased_baseline = _redigest(aliased_baseline)
        with self.assertRaisesRegex(teacher.ActionFeatureTeacherError, "media_sha256 aliases"):
            teacher.build_action_feature_tokens(
                action,
                role="anchor",
                action_authority=action_authority,
                baseline_authority=aliased_baseline,
            )

        bad_digest = copy.deepcopy(action_authority)
        bad_digest["media_size"] = 4097
        with self.assertRaisesRegex(teacher.ActionFeatureTeacherError, "authority digest differs"):
            teacher.build_action_feature_tokens(
                action,
                role="anchor",
                action_authority=bad_digest,
                baseline_authority=_authority(role="anchor_noop", temporal=None),
            )

    def test_receipt_and_array_tampering_fail_closed(self) -> None:
        result = self._tokens()
        bad_phase = result.phase_tokens.copy()
        bad_phase[0, 0] += np.float32(1.0)
        with self.assertRaisesRegex(teacher.ActionFeatureTeacherError, "phase token digest"):
            teacher.ActionFeatureTokens(
                phase_tokens=bad_phase,
                global_token=result.global_token,
                receipt=result.receipt,
            ).validate()

        bad_receipt = copy.deepcopy(dict(result.receipt))
        bad_receipt["point_distillation_authorized"] = True
        unsigned = dict(bad_receipt)
        unsigned.pop("receipt_sha256")
        bad_receipt["receipt_sha256"] = teacher.object_sha256(unsigned)
        with self.assertRaisesRegex(teacher.ActionFeatureTeacherError, "semantic flags"):
            teacher.ActionFeatureTokens(
                phase_tokens=result.phase_tokens,
                global_token=result.global_token,
                receipt=bad_receipt,
            ).validate()

        extra = copy.deepcopy(dict(result.receipt))
        extra["forged_teacher_qualified"] = True
        unsigned = dict(extra)
        unsigned.pop("receipt_sha256")
        extra["receipt_sha256"] = teacher.object_sha256(unsigned)
        with self.assertRaisesRegex(teacher.ActionFeatureTeacherError, "receipt schema"):
            teacher.ActionFeatureTokens(
                phase_tokens=result.phase_tokens,
                global_token=result.global_token,
                receipt=extra,
            ).validate()

        bool_alias = copy.deepcopy(dict(result.receipt))
        bool_alias["input_phases"] = True
        unsigned = dict(bool_alias)
        unsigned.pop("receipt_sha256")
        bool_alias["receipt_sha256"] = teacher.object_sha256(unsigned)
        with self.assertRaisesRegex(teacher.ActionFeatureTeacherError, "fixed geometry"):
            teacher.ActionFeatureTokens(
                phase_tokens=result.phase_tokens,
                global_token=result.global_token,
                receipt=bool_alias,
            ).validate()

    def test_exact_array_types_camera_shape_and_derivative_consistency(self) -> None:
        valid = _temporal_teacher()
        action_authority = _authority(role="anchor_action", temporal=valid)
        baseline_authority = _authority(role="anchor_noop", temporal=None)

        float64 = copy.deepcopy(valid)
        float64.actor_trajectory = float64.actor_trajectory.astype(np.float64)
        with self.assertRaisesRegex(teacher.ActionFeatureTeacherError, "exact C-contiguous float32"):
            teacher.build_action_feature_tokens(
                float64,
                role="anchor",
                action_authority=action_authority,
                baseline_authority=baseline_authority,
            )

        complex_value = copy.deepcopy(valid)
        complex_value.actor_trajectory = complex_value.actor_trajectory.astype(np.complex64)
        with self.assertRaisesRegex(teacher.ActionFeatureTeacherError, "exact C-contiguous float32"):
            teacher.build_action_feature_tokens(
                complex_value,
                role="anchor",
                action_authority=action_authority,
                baseline_authority=baseline_authority,
            )

        wrong_camera = copy.deepcopy(valid)
        wrong_camera.camera_trajectory = np.zeros((teacher.INPUT_PHASES, 6), dtype=np.float32)
        with self.assertRaisesRegex(teacher.ActionFeatureTeacherError, "camera_trajectory"):
            teacher.build_action_feature_tokens(
                wrong_camera,
                role="anchor",
                action_authority=action_authority,
                baseline_authority=baseline_authority,
            )

        inconsistent = copy.deepcopy(valid)
        inconsistent.actor_velocity = inconsistent.actor_velocity.copy()
        inconsistent.actor_velocity[3, 0] += np.float32(0.1)
        with self.assertRaisesRegex(teacher.ActionFeatureTeacherError, "actor_velocity is inconsistent"):
            teacher.build_action_feature_tokens(
                inconsistent,
                role="anchor",
                action_authority=action_authority,
                baseline_authority=baseline_authority,
            )

        noncontiguous_mask = copy.deepcopy(valid)
        noncontiguous_mask.actor_track_mask = noncontiguous_mask.actor_track_mask[::-1]
        with self.assertRaisesRegex(teacher.ActionFeatureTeacherError, "C-contiguous bool"):
            teacher.build_action_feature_tokens(
                noncontiguous_mask,
                role="anchor",
                action_authority=action_authority,
                baseline_authority=baseline_authority,
            )

    def test_schema_camera_gate_and_degenerate_delta_reject(self) -> None:
        action = _temporal_teacher()
        action_authority = _authority(role="target_action", temporal=action)
        action.schema_version = "wrong"
        with self.assertRaisesRegex(teacher.ActionFeatureTeacherError, "schema differs"):
            teacher.build_action_feature_tokens(
                action,
                role="target",
                action_authority=action_authority,
                baseline_authority=_authority(role="target_noop", temporal=None),
            )

        bad_camera = _temporal_teacher()
        action_authority = _authority(role="target_action", temporal=bad_camera)
        bad_camera.camera_crossfit_valid = False
        with self.assertRaisesRegex(teacher.ActionFeatureTeacherError, "camera cross-fit"):
            teacher.build_action_feature_tokens(
                bad_camera,
                role="target",
                action_authority=action_authority,
                baseline_authority=_authority(role="target_noop", temporal=None),
            )

        same = _temporal_teacher()
        with self.assertRaisesRegex(teacher.ActionFeatureTeacherError, "degenerate"):
            teacher.build_action_feature_tokens(
                same,
                role="target",
                action_authority=_authority(role="target_action", temporal=same),
                baseline_authority=_authority(role="target_noop", temporal=same),
                baseline_teacher=same,
            )


if __name__ == "__main__":
    unittest.main()
