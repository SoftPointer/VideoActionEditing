from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch  # noqa: E402
except ImportError:
    torch = None

if torch is not None:
    import saic_event_reward_v1 as reward  # noqa: E402
    import saic_rollout_preference_set_v1 as rollout  # noqa: E402
    import saic_seven_axis_score_packet_v1 as audit  # noqa: E402
else:  # pragma: no cover
    reward = rollout = audit = None


def _sha(material: str) -> str:
    return hashlib.sha256(material.encode("ascii")).hexdigest()


def _seal(row: dict[str, object], field: str) -> dict[str, object]:
    result = deepcopy(row)
    result[field] = audit.object_sha256(result)
    return result


def _packet_value(value: object) -> dict[str, object]:
    return json.loads(audit.validate_candidate_score_packet(value).decode("ascii"))


def _receipt_value(value: bytes) -> dict[str, object]:
    return json.loads(
        audit.validate_source_hard_preference_set(value).decode("ascii")
    )


@unittest.skipIf(torch is None, "torch unavailable")
class SAICSevenAxisScorePacketTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.checkpoint = self.root / "critic.safetensors"
        self.checkpoint.write_bytes(b"qualified-detached-saic-event-critic\x00")
        self.checkpoint_sha = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.qualification = self._qualification()
        self.qualification_path = self.root / "qualification.json"
        self.qualification_path.write_bytes(
            reward.canonical_json_bytes(self.qualification) + b"\n"
        )
        self.boundary = reward.load_event_reward_boundary(
            self.checkpoint, self.qualification_path
        )
        self.current_policy = _sha("current-policy")
        self.base_policy = _sha("frozen-base-policy")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _qualification(self) -> dict[str, object]:
        margins = {phase: 0.30 for phase in reward.PHASE_ORDER}
        cells = [
            {
                "holdout_dimension": dimension,
                "negative_kind": negative,
                "sample_count": 3,
                "stage_margins": dict(margins),
                "weakest_margin": 0.30,
                "passed": True,
            }
            for dimension in reward.HOLDOUT_ORDER
            for negative in reward.NEGATIVE_ORDER
        ]
        unsigned = {
            "schema_version": reward.QUALIFICATION_SCHEMA_VERSION,
            "critic_checkpoint": {
                "content_sha256": self.checkpoint_sha,
                "byte_size": self.checkpoint.stat().st_size,
                "state_dict_kind": "critic_only_no_optimizer",
                "eval_mode": True,
                "requires_grad_parameter_count": 0,
                "optimizer_state_present": False,
            },
            "phase_order": list(reward.PHASE_ORDER),
            "negative_order": list(reward.NEGATIVE_ORDER),
            "holdout_order": list(reward.HOLDOUT_ORDER),
            "holdout_summary": {
                dimension: {
                    "held_out_unit_count": 2,
                    "fit_overlap_count": 0,
                    "passed": True,
                }
                for dimension in reward.HOLDOUT_ORDER
            },
            "coverage_cells": cells,
            "thresholds": {
                "qualification_margin_floor": 0.20,
                "bootstrap_relative_margin_floor": 0.10,
                "absolute_action_score_floors": {
                    "onset": 0.70,
                    "transition": 0.75,
                    "completion": 0.80,
                    "hold": 0.80,
                },
                "absolute_margin_floors": {
                    "onset": 0.20,
                    "transition": 0.20,
                    "completion": 0.25,
                    "hold": 0.25,
                },
            },
            "authority_contract": {
                "score_only_runtime_boundary": True,
                "receipt_alone_authorizes_optimizer": False,
                "receipt_alone_authorizes_inverse": False,
                "receipt_alone_authorizes_publication": False,
                "bootstrap_scope": "same_round_relative_pairing_only",
                "absolute_four_stage_pass_required_for_inverse": True,
                "absolute_four_stage_pass_required_for_publication": True,
                "external_source_constraints_still_required": True,
            },
        }
        return {**unsigned, "receipt_digest": reward.object_sha256(unsigned)}

    def _candidate(
        self,
        candidate_id: str,
        *,
        arm: str,
        policy_sha: str,
    ) -> dict[str, object]:
        source_id = f"{arm}-source"
        source_sha = _sha(source_id)
        output_sha = _sha(f"output-{candidate_id}")
        endpoint_sha = _sha(f"endpoint-{candidate_id}")
        legacy_rollout = _seal(
            {
                "schema_version": rollout.ROLLOUT_RECEIPT_SCHEMA_VERSION,
                "candidate_id": candidate_id,
                "generation_mode": rollout.GENERATION_MODE,
                "on_policy": True,
                "weights_frozen_during_rollout": True,
                "source_conditioned": True,
                "policy_sha256": policy_sha,
                "source_id": source_id,
                "source_media_sha256": source_sha,
                "output_media_sha256": output_sha,
                "frame_count": 81,
                "fps_numerator": 25,
                "fps_denominator": 1,
                "exact40_step_count": 40,
                "preference_update_indices": list(rollout.EXACT40_UPDATE_INDICES),
                "pure_t2v_media_read": False,
                "pure_t2v_latent_read": False,
                "pure_t2v_noise_read": False,
                "target_media_read": False,
                "paired_target_read": False,
            },
            "receipt_digest",
        )
        codec = _seal(
            {
                "schema_version": rollout.CODEC_RECEIPT_SCHEMA_VERSION,
                "candidate_id": candidate_id,
                "input_output_media_sha256": output_sha,
                "decoded_rgb24_sha256": _sha(f"decoded-{candidate_id}"),
                "codec_name": "h264-crf18",
                "codec_bitstream_sha256": _sha(f"codec-{candidate_id}"),
                "codec_decoded_rgb24_sha256": _sha(f"codec-rgb-{candidate_id}"),
                "vae_id": "bernini-vae",
                "vae_weights_sha256": _sha("vae"),
                "reencoded_latent_sha256": endpoint_sha,
                "frame_count": 81,
                "fps_numerator": 25,
                "fps_denominator": 1,
                "endpoint_detached": True,
            },
            "receipt_digest",
        )
        # These legacy scalar fields are required by the old endpoint schema,
        # but the new audit derives all optimizer-authoritative scores itself.
        return _seal(
            {
                "schema_version": rollout.SCHEMA_VERSION,
                "candidate_id": candidate_id,
                "arm": arm,
                "source_id": source_id,
                "instruction_id": f"{arm}-action",
                "policy_sha256": policy_sha,
                "source_media_sha256": source_sha,
                "output_media_sha256": output_sha,
                "endpoint_latent_sha256": endpoint_sha,
                "declared_role": rollout.ENDPOINT_ROLE,
                "event_score": 0.01,
                "axis_scores": {axis: 0.01 for axis in rollout.PRESERVATION_AXES},
                "rollout_receipt": legacy_rollout,
                "codec_reencode_receipt": codec,
            },
            "candidate_digest",
        )

    def _event_scores(
        self,
        candidate: dict[str, object],
        *,
        action_score: float,
        negative_score: float = 0.40,
    ) -> dict[str, object]:
        scores = {
            "action": {phase: action_score for phase in reward.PHASE_ORDER}
        }
        for negative in reward.NEGATIVE_ORDER:
            scores[negative] = {
                phase: negative_score for phase in reward.PHASE_ORDER
            }
        unsigned = {
            "schema_version": reward.CANDIDATE_SCORE_SCHEMA_VERSION,
            "candidate_id": candidate["candidate_id"],
            "rollout_id": f"event-{candidate['candidate_id']}",
            "action_family": candidate["instruction_id"],
            "policy_checkpoint_sha256": candidate["policy_sha256"],
            "critic_checkpoint_sha256": self.checkpoint_sha,
            "qualification_receipt_digest": self.qualification["receipt_digest"],
            "rollout_contract": {
                "on_policy": True,
                "fresh_after_latest_update": True,
                "source_coordinate": "current_policy_rv2v",
                "decoded_exact81": True,
                "frame_count": 81,
                "scores_computed_by_frozen_critic": True,
                "event_bank_candidate": False,
                "payload_kind": "scalar_stage_scores_only",
                "media_or_path_attached": False,
                "latent_attached": False,
                "noise_attached": False,
                "target_attached": False,
                "proposal_attached": False,
            },
            "phase_order": list(reward.PHASE_ORDER),
            "negative_order": list(reward.NEGATIVE_ORDER),
            "scores": scores,
        }
        return {**unsigned, "score_packet_digest": reward.object_sha256(unsigned)}

    def _event_binding(
        self,
        candidate: dict[str, object],
        event_scores: dict[str, object],
    ) -> dict[str, object]:
        unsigned = {
            "schema_version": audit.EVENT_MEDIA_BINDING_SCHEMA_VERSION,
            "candidate_id": candidate["candidate_id"],
            "legacy_candidate_digest": candidate["candidate_digest"],
            "rollout_receipt_digest": candidate["rollout_receipt"]["receipt_digest"],
            "source_media_sha256": candidate["source_media_sha256"],
            "output_media_sha256": candidate["output_media_sha256"],
            "evaluated_media_sha256": candidate["output_media_sha256"],
            "event_rollout_id": event_scores["rollout_id"],
            "event_score_packet_digest": event_scores["score_packet_digest"],
            "critic_checkpoint_sha256": self.checkpoint_sha,
            "critic_qualification_receipt_digest": self.qualification[
                "receipt_digest"
            ],
            "frozen_critic": True,
            "decoded_exact81": True,
            "frame_count": 81,
            "score_only_output": True,
            "pure_t2v_role": audit.PURE_T2V_ROLE,
            "pure_t2v_media_read_during_candidate_scoring": False,
            "pure_t2v_latent_read_during_candidate_scoring": False,
            "pure_t2v_noise_read_during_candidate_scoring": False,
            "pure_t2v_condition_used": False,
            "pure_t2v_target_used": False,
            "pure_t2v_donor_used": False,
            "paired_target_or_donor_read": False,
            "mask_pose_flow_track_trajectory_read": False,
        }
        return {**unsigned, "binding_digest": audit.object_sha256(unsigned)}

    @staticmethod
    def _component_values(**overrides: float) -> dict[str, float]:
        result = {
            "identity_preservation": 0.90,
            "camera_preservation": 0.90,
            "background_preservation": 0.90,
            "non_target_motion_preservation": 0.90,
            "appearance_preservation": 0.90,
            "technical_quality": 0.90,
            "temporal_consistency": 0.90,
            "correct_source_reconstruction_error": 0.10,
            "wrong_source_reconstruction_error": 0.40,
            "dropped_source_reconstruction_error": 0.40,
            "inverse_reconstruction_error": 0.10,
        }
        result.update(overrides)
        return result

    def _measurement_inputs(
        self,
        candidate: dict[str, object],
        components: dict[str, float],
        *,
        evaluator_suffix: str = "",
    ) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
        tensors = {
            component: torch.tensor(value, dtype=torch.float32)
            for component, value in components.items()
        }
        unsigned = {
            "schema_version": audit.MEASUREMENT_BUNDLE_SCHEMA_VERSION,
            "candidate_id": candidate["candidate_id"],
            "legacy_candidate_digest": candidate["candidate_digest"],
            "rollout_receipt_digest": candidate["rollout_receipt"][
                "receipt_digest"
            ],
            "codec_receipt_digest": candidate["codec_reencode_receipt"][
                "receipt_digest"
            ],
            "source_media_sha256": candidate["source_media_sha256"],
            "output_media_sha256": candidate["output_media_sha256"],
            "evaluator_set_sha256": _sha(
                f"whole-frame-evaluator-set{evaluator_suffix}"
            ),
            "upstream_receipt_digest_by_component": {
                component: _sha(
                    f"upstream-{candidate['candidate_id']}-{component}-{index}"
                )
                for index, component in enumerate(audit.MEASUREMENT_COMPONENTS)
            },
            "score_value_by_component": {
                component: float(tensors[component].item())
                for component in audit.MEASUREMENT_COMPONENTS
            },
            "whole_frame_exact81": True,
            "frame_count": 81,
            "score_tensor_detached_fp32": True,
            "input_closure": {
                "candidate_output_media_read": True,
                "registered_source_media_read": True,
                "unregistered_media_read": False,
                "pure_t2v_visual_condition_target_noise_or_donor_used": False,
                "paired_target_proposal_or_donor_read": False,
                "mask_pose_flow_track_trajectory_read": False,
                "whole_frame_scoring": True,
                "selected_actor_localization_used": False,
            },
        }
        return tensors, {
            **unsigned,
            "bundle_digest": audit.object_sha256(unsigned),
        }

    def _packet(
        self,
        candidate_id: str,
        *,
        arm: str,
        policy_sha: str,
        mode: str,
        action_score: float,
        negative_score: float = 0.40,
        components: dict[str, float] | None = None,
        evaluator_suffix: str = "",
    ) -> bytes:
        candidate = self._candidate(
            candidate_id, arm=arm, policy_sha=policy_sha
        )
        event_scores = self._event_scores(
            candidate, action_score=action_score, negative_score=negative_score
        )
        tensors, measurement_bundle = self._measurement_inputs(
            candidate,
            components or self._component_values(),
            evaluator_suffix=evaluator_suffix,
        )
        return audit.build_candidate_score_packet(
            candidate,
            event_boundary=self.boundary,
            event_candidate_scores=event_scores,
            event_media_binding_receipt=self._event_binding(candidate, event_scores),
            event_mode=mode,
            detached_measurements=tensors,
            measurement_bundle=measurement_bundle,
        )

    def _population(
        self,
        *,
        mode: str = audit.BOOTSTRAP_MODE,
        chosen_action: float = 0.90,
        rejected_action: float = 0.60,
        chosen_by_arm: dict[str, dict[str, float]] | None = None,
        rejected_by_arm: dict[str, dict[str, float]] | None = None,
    ) -> tuple[
        list[bytes],
        list[bytes],
    ]:
        candidates: list[bytes] = []
        baselines: list[bytes] = []
        for arm in audit.ARMS:
            candidates.extend(
                [
                    self._packet(
                        f"{arm}-chosen",
                        arm=arm,
                        policy_sha=self.current_policy,
                        mode=mode,
                        action_score=chosen_action,
                        components=(chosen_by_arm or {}).get(arm),
                    ),
                    self._packet(
                        f"{arm}-rejected",
                        arm=arm,
                        policy_sha=self.current_policy,
                        mode=mode,
                        action_score=rejected_action,
                        components=(rejected_by_arm or {}).get(arm),
                    ),
                ]
            )
            baselines.append(
                self._packet(
                    f"{arm}-base",
                    arm=arm,
                    policy_sha=self.base_policy,
                    mode=mode,
                    action_score=0.75,
                    components=self._component_values(
                        identity_preservation=0.85,
                        camera_preservation=0.85,
                        background_preservation=0.85,
                        non_target_motion_preservation=0.85,
                        appearance_preservation=0.85,
                        technical_quality=0.85,
                        temporal_consistency=0.85,
                    ),
                )
            )
        return candidates, baselines

    @staticmethod
    def _gate_kwargs(mode: str) -> dict[str, object]:
        return {
            "current_policy_sha256": _sha("current-policy"),
            "frozen_base_policy_sha256": _sha("frozen-base-policy"),
            "mode": mode,
        }

    def _gate(self, candidates, baselines, *, mode=audit.BOOTSTRAP_MODE):
        return audit.build_source_hard_preference_set(
            candidates,
            frozen_base_packets=baselines,
            **self._gate_kwargs(mode),
        )

    @staticmethod
    def _resign_packet(packet: dict[str, object]) -> bytes:
        packet = deepcopy(packet)
        bundle = packet["measurement_bundle"]
        bundle["bundle_digest"] = audit.object_sha256(
            {key: value for key, value in bundle.items() if key != "bundle_digest"}
        )
        event_binding = packet["event_media_binding_receipt"]
        event_binding["binding_digest"] = audit.object_sha256(
            {
                key: value
                for key, value in event_binding.items()
                if key != "binding_digest"
            }
        )
        packet["packet_digest"] = audit.object_sha256(
            {key: value for key, value in packet.items() if key != "packet_digest"}
        )
        return audit.validate_candidate_score_packet(packet)

    @staticmethod
    def _resign_receipt(receipt: dict[str, object]) -> bytes:
        receipt = deepcopy(receipt)
        receipt["receipt_digest"] = audit.object_sha256(
            {key: value for key, value in receipt.items() if key != "receipt_digest"}
        )
        return audit.canonical_json_bytes(receipt)

    def test_packet_derives_seven_distinct_axes_and_ignores_legacy_scalars(self) -> None:
        packet_bytes = self._packet(
            "dog-probe",
            arm="dog",
            policy_sha=self.current_policy,
            mode=audit.BOOTSTRAP_MODE,
            action_score=0.90,
        )
        packet = _packet_value(packet_bytes)
        self.assertIs(type(packet_bytes), bytes)
        self.assertEqual(set(packet["axis_scores"]), set(audit.PRESERVATION_AXES))
        self.assertEqual(len(packet["axis_scores"]), 7)
        self.assertAlmostEqual(packet["axis_scores"]["quality"], 0.90, places=6)
        self.assertAlmostEqual(packet["axis_scores"]["source_bind"], 0.30, places=6)
        self.assertAlmostEqual(packet["axis_scores"]["inverse"], 1.0 / 1.1, places=6)
        self.assertAlmostEqual(packet["event_evidence"]["weakest_margin"], 0.50)
        self.assertTrue(packet["event_evidence"]["absolute_four_stage_pass"])
        self.assertNotEqual(packet["axis_scores"]["identity"], 0.01)
        self.assertFalse(
            packet["authority_contract"]["media_evaluator_executed_by_this_module"]
        )
        self.assertEqual(
            packet["authority_contract"]["serialized_packet_provenance"],
            "none_caller_re_signable_diagnostic_only",
        )
        self.assertFalse(
            packet["authority_contract"][
                "serialized_builder_event_boundary_call_authenticated"
            ]
        )
        self.assertFalse(packet["authority_contract"]["optimizer_authority"])
        self.assertFalse(
            packet["authority_contract"][
                "pure_t2v_endpoint_condition_target_noise_or_donor_used"
            ]
        )

    def test_measurements_must_be_actual_detached_fp32_scalars(self) -> None:
        candidate = self._candidate(
            "dog-detach", arm="dog", policy_sha=self.current_policy
        )
        event_scores = self._event_scores(candidate, action_score=0.90)
        tensors, measurement_bundle = self._measurement_inputs(
            candidate, self._component_values()
        )
        tensors["identity_preservation"] = torch.tensor(
            0.90, dtype=torch.float32, requires_grad=True
        )
        with self.assertRaisesRegex(
            audit.SAICSevenAxisScorePacketError, "detached.*FP32"
        ):
            audit.build_candidate_score_packet(
                candidate,
                event_boundary=self.boundary,
                event_candidate_scores=event_scores,
                event_media_binding_receipt=self._event_binding(
                    candidate, event_scores
                ),
                event_mode=audit.BOOTSTRAP_MODE,
                detached_measurements=tensors,
                measurement_bundle=measurement_bundle,
            )

    def test_event_and_measurement_t2v_visual_routes_fail_closed(self) -> None:
        candidate = self._candidate(
            "dog-t2v", arm="dog", policy_sha=self.current_policy
        )
        event_scores = self._event_scores(candidate, action_score=0.90)
        tensors, measurement_bundle = self._measurement_inputs(
            candidate, self._component_values()
        )
        event_binding = self._event_binding(candidate, event_scores)
        event_binding["pure_t2v_condition_used"] = True
        event_binding["binding_digest"] = audit.object_sha256(
            {key: value for key, value in event_binding.items() if key != "binding_digest"}
        )
        with self.assertRaisesRegex(
            audit.SAICSevenAxisScorePacketError, "must be False"
        ):
            audit.build_candidate_score_packet(
                candidate,
                event_boundary=self.boundary,
                event_candidate_scores=event_scores,
                event_media_binding_receipt=event_binding,
                event_mode=audit.BOOTSTRAP_MODE,
                detached_measurements=tensors,
                measurement_bundle=measurement_bundle,
            )
        measurement_bundle["input_closure"][
            "pure_t2v_visual_condition_target_noise_or_donor_used"
        ] = True
        measurement_bundle["bundle_digest"] = audit.object_sha256(
            {
                key: value
                for key, value in measurement_bundle.items()
                if key != "bundle_digest"
            }
        )
        with self.assertRaisesRegex(
            audit.SAICSevenAxisScorePacketError, "must be False"
        ):
            audit.build_candidate_score_packet(
                candidate,
                event_boundary=self.boundary,
                event_candidate_scores=event_scores,
                event_media_binding_receipt=self._event_binding(
                    candidate, event_scores
                ),
                event_mode=audit.BOOTSTRAP_MODE,
                detached_measurements=tensors,
                measurement_bundle=measurement_bundle,
            )

    def test_bootstrap_relative_pair_is_distinct_from_strict_absolute_gate(self) -> None:
        # Both have strong action-vs-negative margins, but the winner's raw
        # action score remains below every strict absolute terminal floor.
        bootstrap_candidates, bootstrap_bases = self._population(
            mode=audit.BOOTSTRAP_MODE,
            chosen_action=0.65,
            rejected_action=0.35,
        )
        bootstrap = _receipt_value(
            self._gate(bootstrap_candidates, bootstrap_bases)
        )
        self.assertTrue(bootstrap["diagnostic_pairing_eligible"])
        self.assertFalse(bootstrap["optimizer_step_allowed"])
        self.assertFalse(
            _packet_value(bootstrap_candidates[0])["event_evidence"][
                "absolute_four_stage_pass"
            ]
        )
        strict_candidates, strict_bases = self._population(
            mode=audit.STRICT_MODE,
            chosen_action=0.65,
            rejected_action=0.35,
        )
        strict = _receipt_value(
            self._gate(
                strict_candidates, strict_bases, mode=audit.STRICT_MODE
            )
        )
        self.assertFalse(strict["diagnostic_pairing_eligible"])
        self.assertFalse(strict["optimizer_step_allowed"])
        self.assertEqual(strict["diagnostic_pairs"], [])

    def test_strict_two_arm_source_hard_set_reports_exactly_two_diagnostic_pairs(self) -> None:
        candidates, baselines = self._population(mode=audit.STRICT_MODE)
        result = _receipt_value(
            self._gate(candidates, baselines, mode=audit.STRICT_MODE)
        )
        self.assertTrue(result["diagnostic_pairing_eligible"])
        self.assertFalse(result["optimizer_step_allowed"])
        self.assertEqual(
            [row["arm"] for row in result["diagnostic_pairs"]],
            ["dog", "human"],
        )
        self.assertTrue(
            all(
                row["chosen_absolute_four_stage_event_pass"]
                for row in result["diagnostic_pairs"]
            )
        )
        self.assertFalse(result["scalar_reward_or_weighted_compensation_used"])
        self.assertFalse(
            result["pure_t2v_endpoint_condition_target_noise_or_donor_used"]
        )
        self.assertEqual(result["optimizer_authorized_pair_digests"], [])
        self.assertFalse(result["whole_frame_measurement_runtime_qualified"])

    def test_missing_human_pair_forces_global_exact_zero(self) -> None:
        candidates, baselines = self._population()
        candidates = [
            row
            for row in candidates
            if _packet_value(row)["candidate_binding"]["arm"] == "dog"
        ]
        baselines = [
            row
            for row in baselines
            if _packet_value(row)["candidate_binding"]["arm"] == "dog"
        ]
        result = _receipt_value(self._gate(candidates, baselines))
        self.assertFalse(result["diagnostic_pairing_eligible"])
        self.assertFalse(result["optimizer_step_allowed"])
        self.assertEqual(result["diagnostic_pairs"], [])
        self.assertGreater(
            result["diagnostic_admissible_pair_count_by_arm"]["dog"], 0
        )
        self.assertEqual(
            result["diagnostic_admissible_pair_count_by_arm"]["human"], 0
        )

    def test_appearance_shortcut_cannot_hide_behind_better_quality_minimum(self) -> None:
        chosen = {
            arm: self._component_values(
                appearance_preservation=0.86,
                technical_quality=0.95,
                temporal_consistency=0.95,
            )
            for arm in audit.ARMS
        }
        rejected = {
            arm: self._component_values(
                appearance_preservation=0.90,
                technical_quality=0.80,
                temporal_consistency=0.95,
            )
            for arm in audit.ARMS
        }
        candidates, baselines = self._population(
            chosen_by_arm=chosen, rejected_by_arm=rejected
        )
        chosen_packet = _packet_value(candidates[0])
        rejected_packet = _packet_value(candidates[1])
        # The aggregate quality axis favors chosen (0.86 > 0.80), while the
        # explicit appearance axis regresses beyond its 0.02 tolerance.
        self.assertGreater(
            chosen_packet["axis_scores"]["quality"],
            rejected_packet["axis_scores"]["quality"],
        )
        result = _receipt_value(self._gate(candidates, baselines))
        self.assertFalse(result["diagnostic_pairing_eligible"])
        self.assertFalse(result["optimizer_step_allowed"])
        self.assertEqual(
            result["diagnostic_admissible_pair_count_by_arm"],
            {"dog": 0, "human": 0},
        )

    def test_source_binding_and_inverse_are_noncompensating_axes(self) -> None:
        weak_bind = {
            arm: self._component_values(
                identity_preservation=1.0,
                camera_preservation=1.0,
                background_preservation=1.0,
                non_target_motion_preservation=1.0,
                appearance_preservation=1.0,
                technical_quality=1.0,
                temporal_consistency=1.0,
                correct_source_reconstruction_error=0.30,
                wrong_source_reconstruction_error=0.35,
                dropped_source_reconstruction_error=0.35,
                inverse_reconstruction_error=0.0,
            )
            for arm in audit.ARMS
        }
        candidates, baselines = self._population(chosen_by_arm=weak_bind)
        result = _receipt_value(self._gate(candidates, baselines))
        packet = _packet_value(candidates[0])
        self.assertFalse(result["diagnostic_pairing_eligible"])
        self.assertFalse(result["optimizer_step_allowed"])
        self.assertAlmostEqual(
            packet["axis_scores"]["source_bind"], 0.05, places=6
        )
        self.assertEqual(packet["axis_scores"]["inverse"], 1.0)

    def test_loaded_or_resealed_event_packet_remains_explicitly_untrusted(self) -> None:
        candidates, baselines = self._population()
        forged = _packet_value(candidates[0])
        for phase in reward.PHASE_ORDER:
            forged["event_evidence"]["stage_margins"][phase] = 0.60
        forged["event_evidence"]["weakest_margin"] = 0.60
        forged["event_evidence"]["event_decision_digest"] = _sha(
            "forged-loaded-event-decision"
        )
        forged["packet_digest"] = audit.object_sha256(
            {key: value for key, value in forged.items() if key != "packet_digest"}
        )
        diagnostic_bytes = audit.validate_candidate_score_packet(forged)
        self.assertIs(type(diagnostic_bytes), bytes)
        candidates[0] = diagnostic_bytes
        result = _receipt_value(self._gate(candidates, baselines))
        self.assertTrue(result["diagnostic_pairing_eligible"])
        self.assertFalse(result["builder_authentication_claimed"])
        self.assertFalse(result["whole_frame_measurement_runtime_qualified"])
        self.assertFalse(result["optimizer_step_allowed"])
        self.assertEqual(
            result["serialized_input_provenance"],
            "caller_supplied_digest_sealed_re_signable_diagnostic_only",
        )

    def test_current_policy_must_be_explicit_uniform_and_not_frozen_base(self) -> None:
        candidates, baselines = self._population()
        same_policy = self._gate_kwargs(audit.BOOTSTRAP_MODE)
        same_policy["current_policy_sha256"] = self.base_policy
        with self.assertRaisesRegex(
            audit.SAICSevenAxisScorePacketError, "must differ"
        ):
            audit.build_source_hard_preference_set(
                candidates,
                frozen_base_packets=baselines,
                **same_policy,
            )

        candidates[2] = self._packet(
            "human-chosen",
            arm="human",
            policy_sha=_sha("mixed-current-policy"),
            mode=audit.BOOTSTRAP_MODE,
            action_score=0.90,
        )
        with self.assertRaisesRegex(
            audit.SAICSevenAxisScorePacketError, "one explicit current policy"
        ):
            self._gate(candidates, baselines)

    def test_candidate_packet_cannot_be_reused_as_its_frozen_base(self) -> None:
        candidates, baselines = self._population()
        baselines[0] = candidates[0]
        with self.assertRaises(audit.SAICSevenAxisScorePacketError):
            self._gate(candidates, baselines)

    def test_source_contains_no_optimizer_true_construction(self) -> None:
        source = Path(audit.__file__).read_text(encoding="utf-8")
        self.assertNotIn('"optimizer_step_allowed": True', source)

    def test_module_global_contract_rebinding_cannot_weaken_literals(self) -> None:
        original_closure = audit._INPUT_CLOSURE
        original_authority = audit._AUTHORITY_CONTRACT
        with self.assertRaises(TypeError):
            original_closure["whole_frame_scoring"] = False
        with self.assertRaises(TypeError):
            original_authority["optimizer_authority"] = True
        try:
            audit._INPUT_CLOSURE = {"whole_frame_scoring": False}
            audit._AUTHORITY_CONTRACT = {"optimizer_authority": True}
            packet = _packet_value(
                self._packet(
                    "dog-global-rebind",
                    arm="dog",
                    policy_sha=self.current_policy,
                    mode=audit.BOOTSTRAP_MODE,
                    action_score=0.90,
                )
            )
            self.assertTrue(
                packet["measurement_bundle"]["input_closure"][
                    "whole_frame_scoring"
                ]
            )
            self.assertFalse(packet["authority_contract"]["optimizer_authority"])
        finally:
            audit._INPUT_CLOSURE = original_closure
            audit._AUTHORITY_CONTRACT = original_authority

    def test_arm_global_rebinding_cannot_remove_required_human_arm(self) -> None:
        candidates, baselines = self._population()
        candidates = [
            packet
            for packet in candidates
            if _packet_value(packet)["candidate_binding"]["arm"] == "dog"
        ]
        baselines = [
            packet
            for packet in baselines
            if _packet_value(packet)["candidate_binding"]["arm"] == "dog"
        ]
        original_arms = audit.ARMS
        try:
            audit.ARMS = ("dog",)
            result = _receipt_value(self._gate(candidates, baselines))
            self.assertEqual(result["required_arms"], ["dog", "human"])
            self.assertEqual(
                result["diagnostic_admissible_pair_count_by_arm"],
                {"dog": 1, "human": 0},
            )
            self.assertFalse(result["diagnostic_pairing_eligible"])
            self.assertFalse(result["optimizer_step_allowed"])
        finally:
            audit.ARMS = original_arms

    def test_quality_global_rebinding_cannot_drop_technical_or_temporal(self) -> None:
        bad_quality = self._component_values(
            appearance_preservation=0.90,
            technical_quality=0.01,
            temporal_consistency=0.01,
        )
        original_components = audit.EXPLICIT_QUALITY_COMPONENTS
        try:
            audit.EXPLICIT_QUALITY_COMPONENTS = ("appearance_preservation",)
            candidates, baselines = self._population(
                chosen_by_arm={"dog": bad_quality, "human": bad_quality}
            )
            result = _receipt_value(self._gate(candidates, baselines))
            chosen = _packet_value(candidates[0])
            self.assertAlmostEqual(chosen["axis_scores"]["quality"], 0.01)
            self.assertEqual(
                result["diagnostic_admissible_pair_count_by_arm"],
                {"dog": 0, "human": 0},
            )
            self.assertFalse(result["diagnostic_pairing_eligible"])
            self.assertFalse(result["optimizer_step_allowed"])
        finally:
            audit.EXPLICIT_QUALITY_COMPONENTS = original_components

    def test_other_public_contract_rebinding_cannot_change_a_valid_gate(self) -> None:
        candidates, baselines = self._population()
        originals = {
            "BOOTSTRAP_MODE": audit.BOOTSTRAP_MODE,
            "STRICT_MODE": audit.STRICT_MODE,
            "MODES": audit.MODES,
            "FRAME_COUNT": audit.FRAME_COUNT,
            "PURE_T2V_ROLE": audit.PURE_T2V_ROLE,
            "PRESERVATION_AXES": audit.PRESERVATION_AXES,
            "UNIT_COMPONENTS": audit.UNIT_COMPONENTS,
            "ERROR_COMPONENTS": audit.ERROR_COMPONENTS,
            "MEASUREMENT_COMPONENTS": audit.MEASUREMENT_COMPONENTS,
        }
        original_phases = reward.PHASE_ORDER
        try:
            audit.BOOTSTRAP_MODE = "evil-bootstrap"
            audit.STRICT_MODE = "evil-strict"
            audit.MODES = ("evil-bootstrap", "evil-strict")
            audit.FRAME_COUNT = 1
            audit.PURE_T2V_ROLE = "forged-role"
            audit.PRESERVATION_AXES = ("identity",)
            audit.UNIT_COMPONENTS = ()
            audit.ERROR_COMPONENTS = audit.MEASUREMENT_COMPONENTS
            audit.MEASUREMENT_COMPONENTS = ("identity_preservation",)
            reward.PHASE_ORDER = ("onset",)

            self.assertEqual(
                audit.validate_candidate_score_packet(candidates[0]),
                candidates[0],
            )
            receipt = audit.build_source_hard_preference_set(
                candidates,
                frozen_base_packets=baselines,
                current_policy_sha256=self.current_policy,
                frozen_base_policy_sha256=self.base_policy,
                mode="bootstrap",
            )
            result = _receipt_value(receipt)
            self.assertEqual(result["required_arms"], ["dog", "human"])
            self.assertTrue(result["diagnostic_pairing_eligible"])
            self.assertFalse(result["optimizer_step_allowed"])
        finally:
            for name, value in originals.items():
                setattr(audit, name, value)
            reward.PHASE_ORDER = original_phases

    def test_resealed_event_action_family_must_match_candidate_instruction(self) -> None:
        packet = _packet_value(
            self._packet(
                "dog-action-family-mix",
                arm="dog",
                policy_sha=self.current_policy,
                mode=audit.BOOTSTRAP_MODE,
                action_score=0.90,
            )
        )
        packet["event_evidence"]["action_family"] = "human-unrelated-action"
        with self.assertRaisesRegex(
            audit.SAICSevenAxisScorePacketError,
            "action family differs from candidate instruction",
        ):
            self._resign_packet(packet)

    def test_threshold_policy_is_pinned_and_builder_has_no_override(self) -> None:
        candidates, baselines = self._population()
        result = _receipt_value(self._gate(candidates, baselines))
        self.assertEqual(result["thresholds"]["minimum_event_delta"], 0.20)
        self.assertEqual(
            result["thresholds"]["axis_absolute_floors"],
            {
                "identity": 0.75,
                "camera": 0.75,
                "background": 0.75,
                "non_target": 0.75,
                "quality": 0.75,
                "source_bind": 0.10,
                "inverse": 0.75,
            },
        )
        with self.assertRaises(TypeError):
            audit.build_source_hard_preference_set(
                candidates,
                frozen_base_packets=baselines,
                **self._gate_kwargs(audit.BOOTSTRAP_MODE),
                minimum_event_delta=0.0,
            )

    def test_cross_arm_evaluator_or_event_critic_mixing_is_rejected(self) -> None:
        mutations = (
            ("evaluator", "measurement_bundle", "evaluator_set_sha256"),
            ("critic", "event_evidence", "critic_checkpoint_sha256"),
            (
                "qualification",
                "event_evidence",
                "critic_qualification_receipt_digest",
            ),
        )
        for name, section, key in mutations:
            with self.subTest(name=name):
                candidates, baselines = self._population()
                for collection in (candidates, baselines):
                    for index, packet_bytes in enumerate(collection):
                        packet = _packet_value(packet_bytes)
                        if packet["candidate_binding"]["arm"] != "human":
                            continue
                        replacement = _sha(f"cross-arm-{name}")
                        packet[section][key] = replacement
                        if section == "event_evidence":
                            packet["event_media_binding_receipt"][key] = replacement
                        collection[index] = self._resign_packet(packet)
                with self.assertRaisesRegex(
                    audit.SAICSevenAxisScorePacketError, "one global"
                ):
                    self._gate(candidates, baselines)

    def test_receipt_validator_recomputes_counts_pairs_policies_and_floors(self) -> None:
        candidates, baselines = self._population(mode=audit.STRICT_MODE)
        root = _receipt_value(
            self._gate(candidates, baselines, mode=audit.STRICT_MODE)
        )
        variants: dict[str, dict[str, object]] = {}

        counts = deepcopy(root)
        counts["diagnostic_admissible_pair_count_by_arm"]["dog"] += 1
        variants["counts"] = counts

        bool_count = deepcopy(root)
        bool_count["diagnostic_admissible_pair_count_by_arm"]["dog"] = True
        variants["counts-bool-is-not-one"] = bool_count

        policy = deepcopy(root)
        policy["diagnostic_pairs"][0]["current_policy_sha256"] = _sha(
            "fabricated-pair-policy"
        )
        pair = policy["diagnostic_pairs"][0]
        pair["pair_digest"] = audit.object_sha256(
            {key: value for key, value in pair.items() if key != "pair_digest"}
        )
        variants["pair-policy"] = policy

        floor = deepcopy(root)
        floor["diagnostic_pairs"][0]["axis_effective_floors"]["identity"] = 0.99
        pair = floor["diagnostic_pairs"][0]
        pair["pair_digest"] = audit.object_sha256(
            {key: value for key, value in pair.items() if key != "pair_digest"}
        )
        variants["pair-floor"] = floor

        thresholds = deepcopy(root)
        thresholds["thresholds"]["axis_absolute_floors"]["identity"] = 0.99
        variants["root-threshold"] = thresholds

        for name, forged in variants.items():
            with self.subTest(name=name), self.assertRaises(
                audit.SAICSevenAxisScorePacketError
            ):
                audit.validate_source_hard_preference_set(
                    self._resign_receipt(forged)
                )

    def test_packet_and_gate_receipt_are_canonical_immutable_bytes(self) -> None:
        candidates, baselines = self._population()
        packet_bytes = candidates[0]
        original = _packet_value(packet_bytes)
        loaded = _packet_value(packet_bytes)
        loaded["axis_scores"]["identity"] = 0.0
        self.assertNotEqual(
            loaded["axis_scores"]["identity"],
            _packet_value(packet_bytes)["axis_scores"]["identity"],
        )
        self.assertEqual(original, _packet_value(packet_bytes))
        receipt = self._gate(candidates, baselines)
        self.assertIs(type(receipt), bytes)
        with self.assertRaises(TypeError):
            receipt[0] = 0

    def test_source_bind_uses_both_wrong_and_drop_bottlenecks(self) -> None:
        common = {
            "arm": "dog",
            "policy_sha": self.current_policy,
            "mode": audit.BOOTSTRAP_MODE,
            "action_score": 0.90,
        }
        balanced = _packet_value(
            self._packet(
                "dog-bind-balanced",
                components=self._component_values(
                    correct_source_reconstruction_error=0.10,
                    wrong_source_reconstruction_error=0.40,
                    dropped_source_reconstruction_error=0.40,
                ),
                **common,
            )
        )
        wrong_bottleneck = _packet_value(
            self._packet(
                "dog-bind-wrong",
                components=self._component_values(
                    correct_source_reconstruction_error=0.10,
                    wrong_source_reconstruction_error=0.20,
                    dropped_source_reconstruction_error=0.40,
                ),
                **common,
            )
        )
        drop_bottleneck = _packet_value(
            self._packet(
                "dog-bind-drop",
                components=self._component_values(
                    correct_source_reconstruction_error=0.10,
                    wrong_source_reconstruction_error=0.40,
                    dropped_source_reconstruction_error=0.20,
                ),
                **common,
            )
        )
        self.assertAlmostEqual(balanced["axis_scores"]["source_bind"], 0.30, places=6)
        self.assertAlmostEqual(
            wrong_bottleneck["axis_scores"]["source_bind"], 0.10, places=6
        )
        self.assertAlmostEqual(
            drop_bottleneck["axis_scores"]["source_bind"], 0.10, places=6
        )

    def test_inverse_score_is_detached_and_strictly_monotone_in_error(self) -> None:
        low_error = _packet_value(
            self._packet(
                "dog-inverse-low",
                arm="dog",
                policy_sha=self.current_policy,
                mode=audit.BOOTSTRAP_MODE,
                action_score=0.90,
                components=self._component_values(inverse_reconstruction_error=0.10),
            )
        )
        high_error = _packet_value(
            self._packet(
                "dog-inverse-high",
                arm="dog",
                policy_sha=self.current_policy,
                mode=audit.BOOTSTRAP_MODE,
                action_score=0.90,
                components=self._component_values(inverse_reconstruction_error=1.00),
            )
        )
        self.assertGreater(
            low_error["axis_scores"]["inverse"],
            high_error["axis_scores"]["inverse"],
        )
        self.assertAlmostEqual(high_error["axis_scores"]["inverse"], 0.5)

    def test_tamper_and_cross_evaluator_comparison_fail_closed(self) -> None:
        candidates, baselines = self._population()
        tampered = _packet_value(candidates[0])
        tampered["axis_scores"]["identity"] = 1.0
        with self.assertRaisesRegex(
            audit.SAICSevenAxisScorePacketError, "derivation differs|digest differs"
        ):
            audit.validate_candidate_score_packet(tampered)

        candidates[0] = self._packet(
            "dog-chosen",
            arm="dog",
            policy_sha=self.current_policy,
            mode=audit.BOOTSTRAP_MODE,
            action_score=0.90,
            evaluator_suffix="-other",
        )
        with self.assertRaisesRegex(
            audit.SAICSevenAxisScorePacketError, "one global"
        ):
            self._gate(candidates, baselines)


if __name__ == "__main__":
    unittest.main()
