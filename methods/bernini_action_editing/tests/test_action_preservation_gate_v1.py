from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
import stat
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import action_preservation_gate_v1 as gate
import action_preservation_decoded_eval_verified_release_v1 as verified_release


def digest(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def sign(row: dict, field: str) -> dict:
    row = copy.deepcopy(row)
    row.pop(field, None)
    row[field] = gate.object_sha256(row)
    return row


def calibration() -> dict:
    row = {
        "schema_version": gate.CALIBRATION_SCHEMA,
        "calibration_id": "heldout-human-v1",
        "heldout_manifest_sha256": digest("manifest"),
        "human_labels_sha256": digest("labels"),
        "thresholds": {
            "face_similarity_min": 0.75,
            "face_coverage_min": 0.60,
            "background_similarity_min": 0.80,
            "background_coverage_min": 0.45,
            "camera_translation_max": 0.04,
            "camera_log_scale_abs_max": 0.03,
            "camera_rotation_degrees_max": 1.5,
            "camera_reprojection_error_max": 2.0,
            "quality_min": 0.70,
            "action_order_min": 0.65,
            "onset_score_min": 0.70,
            "onset_timing_error_frames_max": 3,
            "terminal_hold_score_min": 0.75,
            "terminal_hold_frames_min": 8,
        },
        "validation": {
            "positive_count": 40,
            "negative_count": 80,
            "face_fixed_fpr_recall": 0.8,
            "background_fixed_fpr_recall": 0.8,
            "camera_fixed_fpr_recall": 0.8,
            "onset_fixed_fpr_recall": 0.8,
            "terminal_hold_fixed_fpr_recall": 0.8,
            "human_agreement_report_sha256": digest("agreement"),
            "worst_group_reported": True,
            "domain_gap_reported": True,
            "hacking_failures_reported": True,
        },
        "controlled_negatives": {key: True for key in gate._NEGATIVE_FIELDS},
        "thresholds_frozen_before_candidate_generation": True,
    }
    return sign(row, "calibration_digest")


def measurement(*, human: bool = True) -> dict:
    row = {
        "schema_version": gate.MEASUREMENT_SCHEMA,
        "candidate_id": "case-a-arm-b-u20",
        "candidate_video_sha256": digest("candidate"),
        "source_video_sha256": digest("source"),
        "scope": {
            "single_subject": True,
            "human_subject": human,
            "source_face_visible": human,
            "output_face_visible": human,
            "static_or_weak_camera": True,
            "no_shot_cut": True,
            "background_expected_unchanged": True,
        },
        "face": {
            "available": human,
            "similarity": 0.88 if human else None,
            "coverage": 0.90 if human else None,
            "source_face_pool_size": 4 if human else 0,
            "receipt_sha256": digest("face") if human else None,
        },
        "background": {
            "available": True,
            "similarity": 0.91,
            "valid_coverage": 0.72,
            "source_and_output_masks_independent": True,
            "union_exclusion_used": True,
            "receipt_sha256": digest("background"),
        },
        "camera": {
            "available": True,
            "translation": 0.01,
            "log_scale_abs": 0.01,
            "rotation_degrees_abs": 0.5,
            "reprojection_error": 0.7,
            "background_registration_used": True,
            "receipt_sha256": digest("camera"),
        },
        "quality": {
            "available": True,
            "score": 0.9,
            "receipt_sha256": digest("quality"),
        },
        "onset": {
            "available": True,
            "anchor_frame": 10,
            "candidate_frame": 12,
            "timing_error_frames": 2,
            "score": 0.85,
            "receipt_sha256": digest("onset"),
        },
        "action_order": {
            "available": True,
            "score": 0.8,
            "reverse_rejected": True,
            "truncation_rejected": True,
            "terminal_hold_score": 0.9,
            "terminal_hold_start_frame": 71,
            "terminal_hold_end_frame": 80,
            "terminal_hold_frames": 10,
            "receipt_sha256": digest("action"),
        },
        "input_closure": {
            "target_video_read": False,
            "anchor_appearance_used_for_preservation": False,
            "whole_frame_dino_used_as_identity_gate": False,
            "fixed_source_mask_used_as_background_gate": False,
            "training_loss_used_as_decoded_gate": False,
        },
    }
    return sign(row, "measurement_digest")


def action_contract(index: int) -> dict:
    description = f"Perform opaque action sequence {index}, then hold the terminal pose."
    row = {
        "schema_version": gate.ACTION_REVIEW_CONTRACT_SCHEMA,
        "action_order_description": description,
        "action_order_description_sha256": hashlib.sha256(
            description.encode("utf-8")
        ).hexdigest(),
        "expected_onset_frame_min": 4,
        "expected_onset_frame_max": 20,
        "terminal_hold_start_frame_min": 65,
        "terminal_hold_end_frame": 80,
        "full_video_frame_count": 81,
        "fps_num": 25,
        "fps_den": 1,
    }
    return sign(row, "contract_digest")


def packet_authority(decision: dict) -> tuple[dict, dict, dict]:
    public_rows = []
    private_rows = []
    for index in range(256):
        blind_id = f"blind-candidate-{index:03d}"
        candidate_id = decision["candidate_id"] if index == 0 else f"candidate-{index:03d}"
        source_sha = (
            decision["source_video_sha256"] if index == 0 else digest(f"source:{index}")
        )
        candidate_sha = (
            decision["candidate_video_sha256"]
            if index == 0
            else digest(f"candidate-video:{index}")
        )
        instruction = f"Perform the fitted action for blind row {index}."
        contract = action_contract(index)
        public_row = {
            "blind_candidate_id": blind_id,
            "source_media_sha256": source_sha,
            "source_receipt_sha256": digest(f"source-receipt:{index}"),
            "source_media_relpath": f"media/{source_sha}.mp4",
            "review_media_sha256": candidate_sha,
            "review_media_relpath": f"media/{candidate_sha}.mp4",
            "review_output_digest": digest(f"candidate-output:{index}"),
            "full_video_receipt_sha256": digest(f"candidate-receipt:{index}"),
            "matched_base_media_sha256": digest(f"base-video:{index}"),
            "matched_base_media_relpath": f"media/{digest(f'base-video:{index}')}.mp4",
            "matched_base_output_digest": digest(f"base-output:{index}"),
            "matched_base_full_video_receipt_sha256": digest(
                f"base-receipt:{index}"
            ),
            "instruction": instruction,
            "instruction_sha256": hashlib.sha256(
                instruction.encode("utf-8")
            ).hexdigest(),
            "action_review_contract": contract,
            "action_review_contract_digest": contract["contract_digest"],
            "required_axes": list(gate.AXES),
            "minimum_independent_reviewer_count": 2,
            "full_81_frame_video_required": True,
        }
        public_row = sign(public_row, "blind_row_digest")
        public_rows.append(public_row)
        private_row = {
            "blind_candidate_id": blind_id,
            "blind_row_digest": public_row["blind_row_digest"],
            "order_digest": digest(f"order:{index}"),
            "candidate_id": candidate_id,
            "arm": "v2_onset_all",
            "checkpoint_step": 20,
            "iid": f"iid-{index:03d}",
            "onset_policy": "none",
            "matched_control_id": f"control-{index:03d}",
            "candidate_output_path": f"/outputs/candidate-{index:03d}.mp4",
            "candidate_output_receipt_path": f"/outputs/candidate-{index:03d}.receipt.json",
            "candidate_output_receipt_sha256": public_row[
                "full_video_receipt_sha256"
            ],
            "candidate_output_digest": public_row["review_output_digest"],
            "matched_base_output_receipt_path": f"/outputs/base-{index:03d}.receipt.json",
            "matched_base_output_receipt_sha256": public_row[
                "matched_base_full_video_receipt_sha256"
            ],
            "matched_base_output_digest": public_row[
                "matched_base_output_digest"
            ],
            "instruction_sha256": public_row["instruction_sha256"],
            "action_review_contract_digest": public_row[
                "action_review_contract_digest"
            ],
        }
        private_rows.append(sign(private_row, "private_row_digest"))
    private = {
        "schema_version": gate.PRIVATE_PACKET_SCHEMA,
        "evaluation_id": "gate-review-evaluation",
        "evaluation_manifest_digest": digest("evaluation-manifest"),
        "blinding_key_sha256": digest("blinding-key"),
        "rows": private_rows,
        "row_count": 256,
        "method_arm_checkpoint_policy_private": True,
    }
    private = sign(private, "private_mapping_digest")
    public = {
        "schema_version": gate.PUBLIC_PACKET_SCHEMA,
        "packet_id": "packet-gate-review",
        "review_contract_digest": digest("review-contract"),
        "private_mapping_digest": private["private_mapping_digest"],
        "rows": public_rows,
        "row_count": 256,
        "method_hidden": True,
        "arm_hidden": True,
        "checkpoint_hidden": True,
        "onset_policy_hidden": True,
        "private_key_in_public_packet": False,
        "training_loss_present": False,
    }
    public = sign(public, "public_packet_digest")
    holder_summaries = []
    holder_authorities = []
    for index in range(4):
        authority = {
            "job_id": f"holder-job-{index}",
            "model_capture_digest": digest(f"model-capture-{index}"),
            "model_final_digest": digest(f"model-final-{index}"),
            "task_consumption_set_digest": digest(
                f"task-consumption-set-{index}"
            ),
            "ordered_chain_digests_digest": digest(
                f"ordered-chain-digests-{index}"
            ),
        }
        holder_authorities.append(authority)
        holder_summaries.append(
            {
                "job_id": authority["job_id"],
                "node": f"holder-node-{index}",
                "summary_path": f"/sealed/holder-{index}/summary.json",
                "summary_sha256": digest(f"holder-summary-file-{index}"),
                "summary_digest": digest(f"holder-summary-{index}"),
                "holder_execution_digest": digest(
                    f"holder-execution-{index}"
                ),
                "executor_verified_release_capture": {
                    "receipt_path": (
                        f"/sealed/holder-{index}/executor-capture.json"
                    ),
                    "receipt_sha256": digest(
                        f"executor-capture-file-{index}"
                    ),
                    "capture_digest": digest(f"executor-capture-{index}"),
                    "target": (
                        "action_preservation_decoded_eval_executor_v2.py"
                    ),
                    "target_arguments_sha256": digest(
                        f"executor-arguments-{index}"
                    ),
                },
                "model_capture_path": (
                    f"/sealed/holder-{index}/model_capture.json"
                ),
                "model_capture_sha256": digest(
                    f"model-capture-file-{index}"
                ),
                "model_capture_digest": authority[
                    "model_capture_digest"
                ],
                "model_final_path": (
                    f"/sealed/holder-{index}/model_final.json"
                ),
                "model_final_sha256": digest(
                    f"model-final-file-{index}"
                ),
                "model_final_digest": authority["model_final_digest"],
                "task_consumption_set_digest": authority[
                    "task_consumption_set_digest"
                ],
                "ordered_chain_digests_digest": authority[
                    "ordered_chain_digests_digest"
                ],
                "holder_authority_digest": gate.object_sha256(authority),
                "all_task_fd_inheritance_evidence_verified": True,
            }
        )
    aggregate = {
        "schema_version": gate.EVALUATION_AGGREGATE_SCHEMA,
        "evaluation_id": private["evaluation_id"],
        "evaluation_manifest_digest": private["evaluation_manifest_digest"],
        "physical_bindings_digest": digest("physical-bindings"),
        "holder_summaries": holder_summaries,
        "holder_count": 4,
        "holder_authority_set_digest": gate.object_sha256(
            holder_authorities
        ),
        "ordered_task_authority_chain_set_digest": digest(
            "ordered-task-authority-chain-set"
        ),
        "candidate_output_count": 256,
        "matched_base_output_count": 8,
        "total_output_count": 264,
        "exact_full81_at_25fps_pts_verified": True,
        "all_native_inference_receipts_verified": True,
        "all_model_and_adapter_consumption_authority_verified_offline": True,
        "all_fd_inheritance_evidence_verified": True,
        "all_consumption_publication_gates_verified": True,
        "all_outputs_create_only_and_sealed": True,
        "aggregate_verified_release_capture": {
            "receipt_path": "/sealed/aggregate-runtime-capture.json",
            "receipt_sha256": digest("aggregate-capture-file"),
            "capture_digest": digest("aggregate-capture"),
            "target": "action_preservation_decoded_eval_aggregate_v2.py",
            "target_arguments_sha256": digest("aggregate-arguments"),
        },
        "automatic_retry_count": 0,
        "training_loss_read_or_used": False,
        "checkpoint_loss_ranking": False,
        "private_mapping_digest": private["private_mapping_digest"],
        "public_packet_digest": public["public_packet_digest"],
        "blinding_key_sha256": private["blinding_key_sha256"],
        "machine_calibration_digest": (
            None
            if decision["calibration"] is None
            else decision["calibration"]["calibration_digest"]
        ),
        "machine_status": (
            "ABSTAIN_CALIBRATION_MISSING"
            if decision["calibration"] is None
            else "WAIT_FOR_MACHINE_MEASUREMENT"
        ),
        "blind_review_status": "WAIT_FOR_BLIND_REVIEW",
        "next_action": "WAIT_FOR_BLIND_REVIEW",
        "scientific_promotion_authorized": False,
    }
    aggregate = sign(aggregate, "aggregate_digest")
    return aggregate, public, private


def production_media_projection(
    public: dict, *, collide_sources_and_bases_with_outputs: bool = False
) -> dict:
    """Shape the synthetic packet like exact264 publication media reuse."""
    value = copy.deepcopy(public)
    candidate_shas = [row["review_media_sha256"] for row in value["rows"]]
    for index, row in enumerate(value["rows"]):
        source_sha = (
            candidate_shas[index % 4]
            if collide_sources_and_bases_with_outputs
            else digest(f"production-source:{index % 4}")
        )
        base_sha = (
            candidate_shas[index % 8]
            if collide_sources_and_bases_with_outputs
            else digest(f"production-base:{index % 8}")
        )
        row["source_media_sha256"] = source_sha
        row["source_media_relpath"] = f"media/{source_sha}.mp4"
        row["matched_base_media_sha256"] = base_sha
        row["matched_base_media_relpath"] = f"media/{base_sha}.mp4"
        value["rows"][index] = sign(row, "blind_row_digest")
    return sign(value, "public_packet_digest")


def blind_review(
    decision: dict,
    *, labels: dict | None = None,
    second_labels: dict | None = None,
) -> tuple[dict, dict]:
    aggregate, public, private = packet_authority(decision)
    first = labels if labels is not None else {axis: "pass" for axis in gate.AXES}
    second = second_labels if second_labels is not None else copy.deepcopy(first)
    blind_id = private["rows"][0]["blind_candidate_id"]
    ballots = [
        gate.build_blind_ballot(
            public_packet=public,
            blind_candidate_id=blind_id,
            reviewer_id=reviewer_id,
            labels=ballot_labels,
        )
        for reviewer_id, ballot_labels in (
            ("reviewer-a", first), ("reviewer-b", second)
        )
    ]
    review = gate.build_blind_review(
        decision=decision,
        evaluation_aggregate=aggregate,
        public_packet=public,
        private_mapping=private,
        ballots=ballots,
    )
    evidence = {
        "evaluation_aggregate": aggregate,
        "public_packet": public,
        "private_mapping": private,
    }
    return review, evidence


def promote(decision: dict, review: dict, evidence: dict) -> dict:
    return gate.promotion_decision(decision, review, **evidence)


class CalibrationTests(unittest.TestCase):
    def test_controlled_negative_coverage_is_mandatory(self):
        value = calibration()
        value["controlled_negatives"]["reverse_action"] = False
        value = sign(value, "calibration_digest")
        with self.assertRaisesRegex(gate.ActionPreservationGateError, "incomplete"):
            gate.validate_calibration(value)


class DecisionTests(unittest.TestCase):
    def test_missing_calibration_abstains_on_every_axis(self):
        result = gate.decide(measurement())
        self.assertEqual(result["status"], "abstain")
        self.assertFalse(result["motion_ranking_allowed"])
        self.assertIsNone(result["weighted_score"])
        self.assertEqual(
            {row["state"] for row in result["axes"].values()}, {"undetermined"}
        )

    def test_all_axes_pass_before_motion_ranking(self):
        result = gate.decide(measurement(), calibration())
        self.assertEqual(tuple(result["axes"]), gate.AXES)
        self.assertEqual(result["status"], "eligible_for_motion_ranking")
        self.assertTrue(result["motion_ranking_allowed"])
        self.assertFalse(result["training_promotion_authorized"])
        self.assertIsNone(result["weighted_score"])
        self.assertTrue(all(row["state"] == "pass" for row in result["axes"].values()))

    def test_nonhuman_identity_abstains_instead_of_using_dino_fallback(self):
        result = gate.decide(measurement(human=False), calibration())
        self.assertEqual(result["status"], "abstain")
        self.assertEqual(result["axes"]["source_identity"]["state"], "undetermined")
        self.assertIn(
            "source_identity_scope_unsupported",
            result["axes"]["source_identity"]["reasons"],
        )

    def test_low_face_coverage_is_undetermined_not_a_pass(self):
        value = measurement()
        value["face"]["coverage"] = 0.2
        value = sign(value, "measurement_digest")
        result = gate.decide(value, calibration())
        self.assertEqual(result["status"], "abstain")
        self.assertEqual(result["axes"]["source_identity"]["state"], "undetermined")

    def test_onset_and_terminal_hold_are_separate_auditable_hard_gates(self):
        late = measurement()
        late["onset"]["candidate_frame"] = 20
        late["onset"]["timing_error_frames"] = 10
        late = sign(late, "measurement_digest")
        result = gate.decide(late, calibration())
        self.assertEqual(result["axes"]["onset"]["state"], "fail")

        no_hold = measurement()
        no_hold["action_order"]["terminal_hold_start_frame"] = 79
        no_hold["action_order"]["terminal_hold_frames"] = 2
        no_hold = sign(no_hold, "measurement_digest")
        result = gate.decide(no_hold, calibration())
        self.assertEqual(result["axes"]["action_order"]["state"], "fail")
        self.assertIn(
            "terminal_hold_frames_below_calibrated_minimum",
            result["axes"]["action_order"]["reasons"],
        )

    def test_onset_timing_error_must_equal_observed_frame_delta(self):
        value = measurement()
        value["onset"]["timing_error_frames"] = 1
        value = sign(value, "measurement_digest")
        with self.assertRaisesRegex(gate.ActionPreservationGateError, "frame-auditable"):
            gate.validate_measurement(value)

        value = measurement()
        value["action_order"]["terminal_hold_frames"] = 9
        value = sign(value, "measurement_digest")
        with self.assertRaisesRegex(gate.ActionPreservationGateError, "frame-auditable"):
            gate.validate_measurement(value)

    def test_background_failure_cannot_be_compensated_by_action(self):
        value = measurement()
        value["background"]["similarity"] = 0.2
        value["action_order"]["score"] = 1.0
        value = sign(value, "measurement_digest")
        result = gate.decide(value, calibration())
        self.assertEqual(result["status"], "reject")
        self.assertEqual(result["axes"]["background"]["state"], "fail")
        self.assertFalse(result["motion_ranking_allowed"])

    def test_forbidden_whole_frame_identity_shortcut_is_rejected(self):
        value = measurement()
        value["input_closure"]["whole_frame_dino_used_as_identity_gate"] = True
        value = sign(value, "measurement_digest")
        with self.assertRaisesRegex(gate.ActionPreservationGateError, "shortcut"):
            gate.decide(value, calibration())


class PromotionTests(unittest.TestCase):
    def test_public_media_authority_counts_distinct_exact268_projection(self):
        _, public, _ = packet_authority(gate.decide(measurement()))
        projected = production_media_projection(public)
        self.assertEqual(len(gate._public_media_sha256_set(projected)), 268)

    def test_public_media_authority_deduplicates_by_sha256(self):
        _, public, _ = packet_authority(gate.decide(measurement()))
        projected = production_media_projection(
            public, collide_sources_and_bases_with_outputs=True
        )
        self.assertEqual(len(gate._public_media_sha256_set(projected)), 256)

    def test_machine_pass_still_requires_blind_full_video_authority(self):
        decision = gate.decide(measurement(), calibration())
        review, evidence = blind_review(decision)
        result = promote(decision, review, evidence)
        self.assertEqual(result["status"], "eligible_for_next_20_update_stage")
        self.assertFalse(result["automatic_model_update"])
        self.assertTrue(result["requires_fresh_create_only_training_stage"])

    def test_human_preservation_failure_forces_stop(self):
        decision = gate.decide(measurement(), calibration())
        labels = {axis: "pass" for axis in gate.AXES}
        labels["background"] = "fail"
        review, evidence = blind_review(decision, labels=labels)
        result = promote(decision, review, evidence)
        self.assertEqual(result["status"], "stop_and_rollback")
        self.assertFalse(result["requires_fresh_create_only_training_stage"])

    def test_dog_review_cannot_hide_machine_abstention(self):
        decision = gate.decide(measurement(human=False), calibration())
        review, evidence = blind_review(decision)
        result = promote(decision, review, evidence)
        self.assertEqual(result["status"], "hold_for_more_evidence")

    def test_reviewer_ids_and_independent_full_video_ballots_are_enforced(self):
        decision = gate.decide(measurement(), calibration())
        review, evidence = blind_review(decision)
        review["reviewers"][1]["reviewer_id"] = "reviewer-a"
        review["reviewers"][1] = sign(review["reviewers"][1], "ballot_digest")
        review = sign(review, "review_digest")
        with self.assertRaisesRegex(gate.ActionPreservationGateError, "not unique"):
            promote(decision, review, evidence)

        review, evidence = blind_review(decision)
        review["reviewers"][1]["full_video_reviewed"] = False
        review["reviewers"][1] = sign(review["reviewers"][1], "ballot_digest")
        review = sign(review, "review_digest")
        with self.assertRaisesRegex(gate.ActionPreservationGateError, "full video"):
            promote(decision, review, evidence)

        review, evidence = blind_review(decision)
        review["reviewers"][1]["independent_review"] = False
        review["reviewers"][1] = sign(review["reviewers"][1], "ballot_digest")
        review = sign(review, "review_digest")
        with self.assertRaisesRegex(gate.ActionPreservationGateError, "independent"):
            promote(decision, review, evidence)

    def test_ballot_disagreement_is_deterministically_undetermined(self):
        decision = gate.decide(measurement(), calibration())
        first = {axis: "pass" for axis in gate.AXES}
        second = copy.deepcopy(first)
        second["background"] = "fail"
        review, evidence = blind_review(
            decision, labels=first, second_labels=second
        )
        self.assertEqual(review["axis_resolution"]["background"], "undetermined")
        result = promote(decision, review, evidence)
        self.assertEqual(result["status"], "hold_for_more_evidence")

        forged = copy.deepcopy(review)
        forged["axis_resolution"]["background"] = "pass"
        forged = sign(forged, "review_digest")
        with self.assertRaisesRegex(gate.ActionPreservationGateError, "differs"):
            promote(decision, forged, evidence)

    def test_forged_or_incomplete_decision_is_rejected_before_promotion(self):
        decision = gate.decide(measurement(), calibration())
        forged = copy.deepcopy(decision)
        forged["axes"]["background"]["state"] = "fail"
        review, evidence = blind_review(decision)
        with self.assertRaisesRegex(gate.ActionPreservationGateError, "digest"):
            promote(forged, review, evidence)

        resigned = copy.deepcopy(decision)
        resigned["axes"]["background"]["state"] = "fail"
        resigned["axes"]["background"]["reasons"] = [
            "background_similarity_below_calibrated_threshold"
        ]
        resigned = sign(resigned, "decision_digest")
        with self.assertRaisesRegex(gate.ActionPreservationGateError, "status/axis"):
            gate.promotion_decision(
                resigned,
                blind_review(decision)[0],
                **blind_review(decision)[1],
            )

    def test_resigned_packet_tampering_cannot_reuse_review_or_machine_decision(self):
        decision = gate.decide(measurement(), calibration())
        review, evidence = blind_review(decision)

        def close_chain(changed: dict) -> dict:
            changed["private_mapping"] = sign(
                changed["private_mapping"], "private_mapping_digest"
            )
            changed["public_packet"]["private_mapping_digest"] = changed[
                "private_mapping"
            ]["private_mapping_digest"]
            changed["public_packet"] = sign(
                changed["public_packet"], "public_packet_digest"
            )
            changed["evaluation_aggregate"]["private_mapping_digest"] = changed[
                "private_mapping"
            ]["private_mapping_digest"]
            changed["evaluation_aggregate"]["public_packet_digest"] = changed[
                "public_packet"
            ]["public_packet_digest"]
            changed["evaluation_aggregate"] = sign(
                changed["evaluation_aggregate"], "aggregate_digest"
            )
            return changed

        for mutation in ("instruction", "timing", "matched_base_and_receipts"):
            changed = copy.deepcopy(evidence)
            public_row = changed["public_packet"]["rows"][0]
            private_row = changed["private_mapping"]["rows"][0]
            if mutation == "instruction":
                public_row["instruction"] = "A different opaque action instruction."
                public_row["instruction_sha256"] = hashlib.sha256(
                    public_row["instruction"].encode("utf-8")
                ).hexdigest()
                private_row["instruction_sha256"] = public_row["instruction_sha256"]
            elif mutation == "timing":
                contract = public_row["action_review_contract"]
                contract["expected_onset_frame_min"] = 5
                public_row["action_review_contract"] = sign(
                    contract, "contract_digest"
                )
                public_row["action_review_contract_digest"] = public_row[
                    "action_review_contract"
                ]["contract_digest"]
                private_row["action_review_contract_digest"] = public_row[
                    "action_review_contract_digest"
                ]
            else:
                public_row["matched_base_media_sha256"] = digest("hostile-base")
                public_row["matched_base_media_relpath"] = (
                    f"media/{public_row['matched_base_media_sha256']}.mp4"
                )
                public_row["matched_base_output_digest"] = digest(
                    "hostile-base-output"
                )
                public_row["matched_base_full_video_receipt_sha256"] = digest(
                    "hostile-base-receipt"
                )
                public_row["full_video_receipt_sha256"] = digest(
                    "hostile-candidate-receipt"
                )
                private_row["candidate_output_receipt_sha256"] = public_row[
                    "full_video_receipt_sha256"
                ]
                private_row["matched_base_output_receipt_sha256"] = public_row[
                    "matched_base_full_video_receipt_sha256"
                ]
                private_row["matched_base_output_digest"] = public_row[
                    "matched_base_output_digest"
                ]
            changed["public_packet"]["rows"][0] = sign(
                public_row, "blind_row_digest"
            )
            private_row["blind_row_digest"] = changed["public_packet"]["rows"][0][
                "blind_row_digest"
            ]
            changed["private_mapping"]["rows"][0] = sign(
                private_row, "private_row_digest"
            )
            changed = close_chain(changed)
            with self.subTest(mutation=mutation), self.assertRaises(
                gate.ActionPreservationGateError
            ):
                promote(decision, review, changed)

        changed = copy.deepcopy(evidence)
        public_zero = changed["public_packet"]["rows"][0]
        public_one = changed["public_packet"]["rows"][1]
        private_zero = changed["private_mapping"]["rows"][0]
        private_one = changed["private_mapping"]["rows"][1]
        binding_keys = {
            "blind_candidate_id": "blind_candidate_id",
            "blind_row_digest": "blind_row_digest",
            "candidate_output_receipt_sha256": "full_video_receipt_sha256",
            "candidate_output_digest": "review_output_digest",
            "matched_base_output_receipt_sha256": (
                "matched_base_full_video_receipt_sha256"
            ),
            "matched_base_output_digest": "matched_base_output_digest",
            "instruction_sha256": "instruction_sha256",
            "action_review_contract_digest": "action_review_contract_digest",
        }
        for private_row, public_row in (
            (private_zero, public_one), (private_one, public_zero)
        ):
            for private_key, public_key in binding_keys.items():
                private_row[private_key] = public_row[public_key]
        changed["private_mapping"]["rows"][0] = sign(
            private_zero, "private_row_digest"
        )
        changed["private_mapping"]["rows"][1] = sign(
            private_one, "private_row_digest"
        )
        changed = close_chain(changed)
        with self.assertRaisesRegex(
            gate.ActionPreservationGateError,
            "decision/public row binding",
        ):
            gate.build_blind_review(
                decision=decision,
                ballots=review["reviewers"],
                **changed,
            )


class CliPublicationTests(unittest.TestCase):
    class WorkRootAuthority:
        def __init__(self, root: pathlib.Path) -> None:
            self.root = root
            self.authority_parent = root / "authority"
            self.authority_parent.mkdir()
            self.work = self.authority_parent / "work"
            self.work.mkdir(mode=0o700)
            self.work.chmod(0o700)
            self.inputs = root / "inputs"
            self.inputs.mkdir()

            creation = verified_release._work_root_identity_value(
                self.work.stat()
            )
            parent_creation = verified_release._work_root_identity_value(
                self.authority_parent.stat()
            )
            immutable_fields = (
                "device", "inode", "uid", "gid", "mode", "rdev",
            )
            authority = {
                "schema_version": verified_release.WORK_ROOT_AUTHORITY_SCHEMA,
                "path": str(self.work),
                "parent_path": str(self.authority_parent),
                "creation_identity": creation,
                "immutable_identity": {
                    key: creation[key] for key in immutable_fields
                },
                "parent_immutable_identity": {
                    key: parent_creation[key] for key in immutable_fields
                },
                "initial_entries": [],
                "retained_parent_fd_through_request_publication": True,
                "retained_root_fd_through_request_publication": True,
            }
            authority["authority_digest"] = gate.object_sha256(authority)
            deployment_value = {
                "work_root_authority": authority,
                "receipt_digest": digest("gate deployment receipt"),
            }
            deployment_path, deployment_sha = self._write_at(
                self.work, "deployment.json", deployment_value, sealed=True
            )
            source_value = {
                "work_root_authority": authority,
                "deployment_receipt_digest": deployment_value["receipt_digest"],
                "receipt_digest": digest("gate source authority"),
            }
            source_path, source_sha = self._write_at(
                self.work, "source-authority.json", source_value, sealed=True
            )

            flags = (
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0)
            )
            self.parent_fd = os.open(self.authority_parent, flags)
            self.root_fd = os.open(
                self.work.name, flags, dir_fd=self.parent_fd
            )
            os.set_inheritable(self.parent_fd, False)
            os.set_inheritable(self.root_fd, False)
            binding = {
                "schema_version": verified_release.WORK_ROOT_BINDING_SCHEMA,
                "path": str(self.work),
                "parent_path": str(self.authority_parent),
                "parent_fd": self.parent_fd,
                "root_fd": self.root_fd,
                "parent_identity": verified_release._work_root_identity_value(
                    os.fstat(self.parent_fd)
                ),
                "root_identity": verified_release._work_root_identity_value(
                    os.fstat(self.root_fd)
                ),
                "parent_immutable_identity": authority[
                    "parent_immutable_identity"
                ],
                "root_immutable_identity": authority["immutable_identity"],
                "entries": sorted(os.listdir(self.root_fd)),
                "work_root_authority_digest": authority["authority_digest"],
                "work_root_authority": authority,
                "deployment_receipt": {
                    "path": str(deployment_path), "sha256": deployment_sha,
                },
                "source_spec_authority": {
                    "path": str(source_path), "sha256": source_sha,
                },
                "deployment_receipt_digest": deployment_value["receipt_digest"],
                "source_spec_authority_digest": source_value["receipt_digest"],
                "target": gate.GATE_RUNTIME_TARGET,
                "capture_receipt_path": str(
                    self.work / "unused-runtime-capture.json"
                ),
                "exact_two_directory_fds": True,
                "fds_inheritable_only_across_verified_exec": True,
            }
            binding["binding_digest"] = gate.object_sha256(binding)
            verified_release.validate_inherited_work_root_binding(
                binding,
                verify_open_fds=True,
                expected_inheritable=False,
                verify_entries=True,
            )
            self.binding = binding

        @staticmethod
        def _write_at(
            parent: pathlib.Path, name: str, value: dict, *, sealed: bool = False,
        ) -> tuple[pathlib.Path, str]:
            path = parent / name
            raw = gate.canonical_json_bytes(value) + b"\n"
            path.write_bytes(raw)
            if sealed:
                path.chmod(0o444)
            return path, hashlib.sha256(raw).hexdigest()

        def write_input(self, name: str, value: dict) -> tuple[pathlib.Path, str]:
            return self._write_at(self.inputs, name, value)

        def environment(
            self, *, binding: dict | None = None, mixed: bool = False,
        ):
            selected = self.binding if binding is None else binding
            environment = {
                gate.WORK_ROOT_BINDING_ENV: gate.canonical_json_bytes(
                    selected
                ).decode("utf-8")
            }
            if mixed:
                environment[gate.TASK_FD_BINDING_ENV] = "{}"
            return mock.patch.dict(os.environ, environment, clear=True)

        def close(self) -> None:
            os.close(self.root_fd)
            os.close(self.parent_fd)

    def authority(self, root: pathlib.Path) -> "CliPublicationTests.WorkRootAuthority":
        value = self.WorkRootAuthority(root)
        self.addCleanup(value.close)
        return value

    @staticmethod
    def aggregate_anchor_fixture(
        authority: "CliPublicationTests.WorkRootAuthority",
    ) -> dict:
        aggregate, public, private = packet_authority(gate.decide(measurement()))
        media_raw = b"one-content-addressed-test-video"
        media_sha = hashlib.sha256(media_raw).hexdigest()
        public = copy.deepcopy(public)
        private = copy.deepcopy(private)
        for index, public_row in enumerate(public["rows"]):
            for prefix in ("source", "review", "matched_base"):
                public_row[f"{prefix}_media_sha256"] = media_sha
                public_row[f"{prefix}_media_relpath"] = f"media/{media_sha}.mp4"
            public_row = sign(public_row, "blind_row_digest")
            public["rows"][index] = public_row
            private_row = private["rows"][index]
            private_row["blind_row_digest"] = public_row["blind_row_digest"]
            private["rows"][index] = sign(
                private_row, "private_row_digest"
            )
        private = sign(private, "private_mapping_digest")
        public["private_mapping_digest"] = private["private_mapping_digest"]
        public = sign(public, "public_packet_digest")
        aggregate["private_mapping_digest"] = private[
            "private_mapping_digest"
        ]
        aggregate["public_packet_digest"] = public["public_packet_digest"]
        aggregate = sign(aggregate, "aggregate_digest")

        aggregate_root = authority.work / "aggregate"
        media_root = aggregate_root / "media"
        aggregate_root.mkdir(mode=0o700)
        media_root.mkdir(mode=0o700)
        media_path = media_root / f"{media_sha}.mp4"
        media_path.write_bytes(media_raw)
        media_path.chmod(0o444)

        def write_json(name: str, value: dict, mode: int) -> tuple[pathlib.Path, str]:
            path = aggregate_root / name
            raw = gate.canonical_json_bytes(value) + b"\n"
            path.write_bytes(raw)
            path.chmod(mode)
            return path, hashlib.sha256(raw).hexdigest()

        aggregate_path, aggregate_sha = write_json(
            "evaluation_complete.json", aggregate, 0o444
        )
        private_path, private_sha = write_json(
            "private_blind_mapping.json", private, 0o400
        )
        public_path, public_sha = write_json(
            "blind_review_packet.json", public, 0o444
        )
        media_root.chmod(0o555)
        aggregate_root.chmod(0o555)

        def file_binding(
            path: pathlib.Path, sha256: str, mode: int, object_digest: str,
        ) -> dict:
            return {
                "relative_path": path.name,
                "sha256": sha256,
                "size": path.stat().st_size,
                "mode": mode,
                "identity": gate._file_identity_value(path.stat()),
                "object_digest": object_digest,
            }

        media_rows = [
            {
                "relative_path": f"media/{media_path.name}",
                "sha256": media_sha,
                "size": len(media_raw),
                "mode": 0o444,
                "identity": gate._file_identity_value(media_path.stat()),
            }
        ]
        anchor = {
            "schema_version": verified_release.AGGREGATE_COMPLETION_ANCHOR_SCHEMA,
            "evaluation_id": aggregate["evaluation_id"],
            "aggregate_root": str(aggregate_root),
            "aggregate_root_identity": gate._file_identity_value(
                aggregate_root.stat()
            ),
            "aggregate_file": file_binding(
                aggregate_path, aggregate_sha, 0o444,
                aggregate["aggregate_digest"],
            ),
            "private_file": file_binding(
                private_path, private_sha, 0o400,
                private["private_mapping_digest"],
            ),
            "public_file": file_binding(
                public_path, public_sha, 0o444,
                public["public_packet_digest"],
            ),
            "media_directory_identity": gate._file_identity_value(
                media_root.stat()
            ),
            "media_file_count": 1,
            "media_rows_digest": gate.object_sha256(media_rows),
        }
        anchor["media_tree_digest"] = gate.object_sha256(
            {
                "media_directory_identity": anchor[
                    "media_directory_identity"
                ],
                "media_file_count": anchor["media_file_count"],
                "media_rows_digest": anchor["media_rows_digest"],
            }
        )
        anchor["anchor_digest"] = gate.object_sha256(anchor)
        verified_release.validate_aggregate_completion_anchor(anchor)
        return {
            "aggregate": aggregate,
            "aggregate_path": aggregate_path,
            "aggregate_sha": aggregate_sha,
            "public": public,
            "public_path": public_path,
            "public_sha": public_sha,
            "private": private,
            "private_path": private_path,
            "private_sha": private_sha,
            "media_path": media_path,
            "media_sha": media_sha,
            "anchor": anchor,
        }

    def replay_aggregate_fixture(
        self,
        authority: "CliPublicationTests.WorkRootAuthority",
        fixture: dict,
        **overrides,
    ) -> None:
        arguments = {
            "completion_anchor": fixture["anchor"],
            "work_root": authority.binding,
            "aggregate_path": str(fixture["aggregate_path"]),
            "aggregate_sha256": fixture["aggregate_sha"],
            "aggregate": fixture["aggregate"],
            "public_path": str(fixture["public_path"]),
            "public_sha256": fixture["public_sha"],
            "public": fixture["public"],
            "private_path": str(fixture["private_path"]),
            "private_sha256": fixture["private_sha"],
            "private": fixture["private"],
            "expected_media_sha256": frozenset({fixture["media_sha"]}),
            "expected_work_root_target": gate.GATE_RUNTIME_TARGET,
        }
        arguments.update(overrides)
        gate._replay_aggregate_completion_publication(**arguments)

    def test_dynamic_aggregate_anchor_replays_exact_held_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            authority = self.authority(pathlib.Path(temporary).resolve())
            fixture = self.aggregate_anchor_fixture(authority)
            self.replay_aggregate_fixture(authority, fixture)

    def test_dynamic_aggregate_anchor_rejects_same_bytes_at_another_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            authority = self.authority(pathlib.Path(temporary).resolve())
            fixture = self.aggregate_anchor_fixture(authority)
            copy_path = authority.inputs / "blind_review_packet.json"
            copy_path.write_bytes(fixture["public_path"].read_bytes())
            copy_path.chmod(0o444)
            with self.assertRaisesRegex(
                gate.ActionPreservationGateError, "path/SHA differs"
            ):
                self.replay_aggregate_fixture(
                    authority, fixture, public_path=str(copy_path)
                )

    def test_dynamic_aggregate_anchor_rejects_same_bytes_new_inode(self):
        with tempfile.TemporaryDirectory() as temporary:
            authority = self.authority(pathlib.Path(temporary).resolve())
            fixture = self.aggregate_anchor_fixture(authority)
            aggregate_root = fixture["public_path"].parent
            raw = fixture["public_path"].read_bytes()
            aggregate_root.chmod(0o755)
            fixture["public_path"].unlink()
            fixture["public_path"].write_bytes(raw)
            fixture["public_path"].chmod(0o444)
            aggregate_root.chmod(0o555)
            with self.assertRaisesRegex(
                gate.ActionPreservationGateError,
                "aggregate root differs|dynamic completion anchor inode",
            ):
                self.replay_aggregate_fixture(authority, fixture)

    def test_dynamic_aggregate_anchor_rejects_media_inode_rewrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            authority = self.authority(pathlib.Path(temporary).resolve())
            fixture = self.aggregate_anchor_fixture(authority)
            fixture["media_path"].chmod(0o644)
            fixture["media_path"].write_bytes(b"same-uid-intervening-media")
            fixture["media_path"].chmod(0o444)
            with self.assertRaisesRegex(
                gate.ActionPreservationGateError, "aggregate media differs"
            ):
                self.replay_aggregate_fixture(authority, fixture)

    def test_dynamic_aggregate_anchor_rejects_resigned_media_rows_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            authority = self.authority(pathlib.Path(temporary).resolve())
            fixture = self.aggregate_anchor_fixture(authority)
            hostile = copy.deepcopy(fixture["anchor"])
            hostile["media_rows_digest"] = digest("forged media rows")
            hostile["media_tree_digest"] = gate.object_sha256(
                {
                    "media_directory_identity": hostile[
                        "media_directory_identity"
                    ],
                    "media_file_count": hostile["media_file_count"],
                    "media_rows_digest": hostile["media_rows_digest"],
                }
            )
            hostile = sign(hostile, "anchor_digest")
            verified_release.validate_aggregate_completion_anchor(hostile)
            with self.assertRaisesRegex(
                gate.ActionPreservationGateError,
                "aggregate tree final replay differs",
            ):
                self.replay_aggregate_fixture(
                    authority, fixture, completion_anchor=hostile
                )

    @staticmethod
    def gate_argv(
        authority: "CliPublicationTests.WorkRootAuthority",
        measurement_path: pathlib.Path,
        measurement_sha: str,
        *, calibration_path: pathlib.Path | None = None,
        calibration_sha: str | None = None,
        output_name: str = "decision.json",
    ) -> list[str]:
        argv = [
            "gate",
            "--measurement", str(measurement_path),
            "--measurement-sha256", measurement_sha,
        ]
        if calibration_path is not None:
            argv.extend(["--calibration", str(calibration_path)])
        if calibration_sha is not None:
            argv.extend(["--calibration-sha256", calibration_sha])
        argv.extend(["--output", str(authority.work / output_name)])
        return argv

    def test_gate_cli_output_is_create_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            authority = self.authority(root)
            measurement_path, measurement_sha = authority.write_input(
                "measurement.json", measurement()
            )
            calibration_path, calibration_sha = authority.write_input(
                "calibration.json", calibration()
            )
            output = authority.work / "decision.json"
            argv = self.gate_argv(
                authority, measurement_path, measurement_sha,
                calibration_path=calibration_path,
                calibration_sha=calibration_sha,
            )
            with authority.environment():
                self.assertEqual(gate.main(argv), 0)
            before = output.read_bytes()
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o444)
            with authority.environment():
                with self.assertRaisesRegex(
                    gate.ActionPreservationGateError, "overwrite"
                ):
                    gate.main(argv)
            self.assertEqual(output.read_bytes(), before)

    def test_all_cli_subcommands_reject_missing_A_before_input_io(self):
        commands = {
            "gate": [
                "gate", "--measurement", "/abs/missing.json",
                "--measurement-sha256", digest("missing"),
                "--output", "/abs/output.json",
            ],
            "ballot": [
                "ballot", "--public-packet", "/abs/missing.json",
                "--public-packet-sha256", digest("missing"),
                "--blind-candidate-id", "blind-candidate-000",
                "--reviewer-id", "reviewer-a",
                "--labels", "/abs/labels.json",
                "--labels-sha256", digest("labels"),
                "--output", "/abs/output.json",
            ],
            "aggregate-review": [
                "aggregate-review", "--decision", "/abs/decision.json",
                "--decision-sha256", digest("decision"),
                "--evaluation-complete", "/abs/evaluation.json",
                "--evaluation-complete-sha256", digest("evaluation"),
                "--public-packet", "/abs/public.json",
                "--public-packet-sha256", digest("public"),
                "--private-mapping", "/abs/private.json",
                "--private-mapping-sha256", digest("private"),
                "--physical-bindings", "/abs/physical.json",
                "--physical-bindings-sha256", digest("physical"),
                "--aggregate-completion-anchor", "{}",
                "--ballot", "/abs/ballot.json",
                "--ballot-sha256", digest("ballot"),
                "--output", "/abs/output.json",
            ],
            "promote": [
                "promote", "--decision", "/abs/decision.json",
                "--decision-sha256", digest("decision"),
                "--blind-review", "/abs/review.json",
                "--blind-review-sha256", digest("review"),
                "--evaluation-complete", "/abs/evaluation.json",
                "--evaluation-complete-sha256", digest("evaluation"),
                "--public-packet", "/abs/public.json",
                "--public-packet-sha256", digest("public"),
                "--private-mapping", "/abs/private.json",
                "--private-mapping-sha256", digest("private"),
                "--physical-bindings", "/abs/physical.json",
                "--physical-bindings-sha256", digest("physical"),
                "--aggregate-completion-anchor", "{}",
                "--output", "/abs/output.json",
            ],
        }
        for label, argv in commands.items():
            with self.subTest(command=label):
                with mock.patch.dict(os.environ, {}, clear=True):
                    with mock.patch.object(
                        gate, "_stable_expected_bytes",
                        side_effect=AssertionError("input was read"),
                    ):
                        with self.assertRaisesRegex(
                            gate.ActionPreservationGateError, "A authority is absent"
                        ):
                            gate.main(argv)

    def test_gate_rejects_mixed_or_wrong_target_before_input_io(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            authority = self.authority(root)
            measurement_path, measurement_sha = authority.write_input(
                "measurement.json", measurement()
            )
            argv = self.gate_argv(
                authority, measurement_path, measurement_sha
            )
            wrong = copy.deepcopy(authority.binding)
            wrong["target"] = "action_preservation_decoded_eval_aggregate_v2.py"
            wrong = sign(wrong, "binding_digest")
            for label, context, expected in (
                ("mixed", authority.environment(mixed=True), "mixed WORK_ROOT"),
                ("wrong", authority.environment(binding=wrong), "not the gate"),
            ):
                with self.subTest(case=label):
                    with context:
                        with mock.patch.object(
                            gate, "_stable_expected_bytes",
                            side_effect=AssertionError("input was read"),
                        ):
                            with self.assertRaisesRegex(
                                gate.ActionPreservationGateError, expected
                            ):
                                gate.main(argv)

    def test_gate_rejects_unpaired_calibration_authority(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            authority = self.authority(root)
            measurement_path, measurement_sha = authority.write_input(
                "measurement.json", measurement()
            )
            calibration_path, _ = authority.write_input(
                "calibration.json", calibration()
            )
            argv = self.gate_argv(
                authority, measurement_path, measurement_sha,
                calibration_path=calibration_path,
            )
            with authority.environment():
                with mock.patch.object(
                    gate, "_stable_expected_bytes",
                    side_effect=AssertionError("input was read"),
                ):
                    with self.assertRaisesRegex(
                        gate.ActionPreservationGateError, "supplied together"
                    ):
                        gate.main(argv)

    def test_gate_rejects_input_leaf_rename_hardlink_and_symlink(self):
        for attack in ("rename", "hardlink", "symlink"):
            with self.subTest(attack=attack), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary).resolve()
                authority = self.authority(root)
                measurement_path, measurement_sha = authority.write_input(
                    "measurement.json", measurement()
                )
                argv = self.gate_argv(
                    authority, measurement_path, measurement_sha,
                    output_name=f"{attack}-decision.json",
                )
                original_read = gate._read_fd
                triggered = False

                def hostile_read(descriptor: int) -> bytes:
                    nonlocal triggered
                    raw = original_read(descriptor)
                    if not triggered:
                        triggered = True
                        if attack == "hardlink":
                            os.link(
                                measurement_path,
                                authority.inputs / "measurement-hardlink.json",
                            )
                        else:
                            displaced = authority.inputs / "measurement-held.json"
                            os.rename(measurement_path, displaced)
                            if attack == "symlink":
                                os.symlink(displaced.name, measurement_path)
                            else:
                                authority._write_at(
                                    authority.inputs,
                                    measurement_path.name,
                                    measurement(),
                                )
                    return raw

                with authority.environment():
                    with mock.patch.object(
                        gate, "_read_fd", side_effect=hostile_read
                    ):
                        with self.assertRaisesRegex(
                            gate.ActionPreservationGateError,
                            "same-FD double read|hard link",
                        ):
                            gate.main(argv)
                self.assertFalse((authority.work / f"{attack}-decision.json").exists())

    def test_gate_output_uses_held_root_and_rejects_work_root_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            authority = self.authority(root)
            measurement_path, measurement_sha = authority.write_input(
                "measurement.json", measurement()
            )
            output_name = "root-replacement-decision.json"
            argv = self.gate_argv(
                authority, measurement_path, measurement_sha,
                output_name=output_name,
            )
            displaced = authority.authority_parent / "work-held"
            original_replay = gate._replay_gate_work_root_authority
            triggered = False

            def hostile_replay(value: dict) -> dict:
                nonlocal triggered
                row = original_replay(value)
                if not triggered:
                    triggered = True
                    os.rename(authority.work, displaced)
                    authority.work.mkdir(mode=0o700)
                    authority.work.chmod(0o700)
                return row

            with authority.environment():
                with mock.patch.object(
                    gate, "_replay_gate_work_root_authority",
                    side_effect=hostile_replay,
                ):
                    with self.assertRaisesRegex(
                        gate.ActionPreservationGateError,
                        "FD identity|replay",
                    ):
                        gate.main(argv)
            self.assertFalse((authority.work / output_name).exists())
            self.assertTrue((displaced / output_name).is_file())

    def test_gate_rejects_output_leaf_rename_hardlink_and_symlink(self):
        for attack in ("rename", "hardlink", "symlink"):
            with self.subTest(attack=attack), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary).resolve()
                authority = self.authority(root)
                measurement_path, measurement_sha = authority.write_input(
                    "measurement.json", measurement()
                )
                output_name = f"output-{attack}.json"
                output = authority.work / output_name
                argv = self.gate_argv(
                    authority, measurement_path, measurement_sha,
                    output_name=output_name,
                )
                original_read = gate._read_fd
                read_count = 0

                def hostile_read(descriptor: int) -> bytes:
                    nonlocal read_count
                    raw = original_read(descriptor)
                    read_count += 1
                    # Measurement consumes calls one and two.  The third call
                    # is the first same-FD replay of the newly written output.
                    if read_count == 3:
                        if attack == "hardlink":
                            os.link(output, authority.work / "extra-hardlink.json")
                        else:
                            displaced = authority.work / "output-held.json"
                            os.rename(output, displaced)
                            if attack == "symlink":
                                os.symlink(displaced.name, output)
                            else:
                                output.write_bytes(b"{}\n")
                    return raw

                with authority.environment():
                    with mock.patch.object(
                        gate, "_read_fd", side_effect=hostile_read
                    ):
                        with self.assertRaisesRegex(
                            gate.ActionPreservationGateError,
                            "same-FD write replay",
                        ):
                            gate.main(argv)

    def test_ballot_cli_requires_literal_labels_sha_and_held_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            authority = self.authority(root)
            decision = gate.decide(measurement(), calibration())
            _, public, _ = packet_authority(decision)
            packet_path, packet_sha = authority.write_input(
                "public.json", public
            )
            labels = {axis: "pass" for axis in gate.AXES}
            labels_path, labels_sha = authority.write_input(
                "labels.json", labels
            )
            output = authority.work / "ballot.json"
            argv = [
                "ballot",
                "--public-packet", str(packet_path),
                "--public-packet-sha256", packet_sha,
                "--blind-candidate-id", "blind-candidate-000",
                "--reviewer-id", "reviewer-a",
                "--labels", str(labels_path),
                "--labels-sha256", labels_sha,
                "--output", str(output),
            ]
            with authority.environment():
                self.assertEqual(gate.main(argv), 0)
            expected = gate.build_blind_ballot(
                public_packet=public,
                blind_candidate_id="blind-candidate-000",
                reviewer_id="reviewer-a",
                labels=labels,
            )
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")), expected
            )

            hostile = list(argv)
            hostile[hostile.index("--labels-sha256") + 1] = digest(
                "wrong labels"
            )
            hostile[-1] = str(authority.work / "hostile-ballot.json")
            with authority.environment():
                with self.assertRaisesRegex(
                    gate.ActionPreservationGateError, "review labels SHA differs"
                ):
                    gate.main(hostile)
            self.assertFalse((authority.work / "hostile-ballot.json").exists())

    def test_gate_rejects_output_outside_inherited_work_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            authority = self.authority(root)
            measurement_path, measurement_sha = authority.write_input(
                "measurement.json", measurement()
            )
            argv = self.gate_argv(
                authority, measurement_path, measurement_sha
            )
            outside = root / "outside.json"
            argv[-1] = str(outside)
            with authority.environment():
                with self.assertRaisesRegex(
                    gate.ActionPreservationGateError, "direct child"
                ):
                    gate.main(argv)
            self.assertFalse(outside.exists())

    def test_aggregate_review_cli_requires_expected_physical_packet_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            authority = self.authority(root)
            decision = gate.decide(measurement(), calibration())
            review, evidence = blind_review(decision)

            def write_artifact(name: str, value: dict) -> tuple[pathlib.Path, str]:
                return authority.write_input(name, value)

            decision_path, decision_sha = write_artifact("decision.json", decision)
            aggregate_path, aggregate_sha = write_artifact(
                "evaluation_complete.json", evidence["evaluation_aggregate"]
            )
            public_path, public_sha = write_artifact(
                "public.json", evidence["public_packet"]
            )
            private_path, private_sha = write_artifact(
                "private.json", evidence["private_mapping"]
            )
            ballot_paths = [
                write_artifact(f"ballot-{index}.json", ballot)
                for index, ballot in enumerate(review["reviewers"])
            ]
            output = authority.work / "blind-review.json"
            argv = [
                "aggregate-review",
                "--decision", str(decision_path),
                "--decision-sha256", decision_sha,
                "--evaluation-complete", str(aggregate_path),
                "--evaluation-complete-sha256", aggregate_sha,
                "--public-packet", str(public_path),
                "--public-packet-sha256", public_sha,
                "--private-mapping", str(private_path),
                "--private-mapping-sha256", private_sha,
                "--physical-bindings", str(root / "physical_bindings.json"),
                "--physical-bindings-sha256", digest("physical-bindings-file"),
                "--aggregate-completion-anchor", "{}",
            ]
            for path, sha in ballot_paths:
                argv.extend(["--ballot", str(path), "--ballot-sha256", sha])
            argv.extend(["--output", str(output)])
            with authority.environment():
                with mock.patch.object(gate, "_verify_cli_eval_release"):
                    self.assertEqual(gate.main(argv), 0)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")), review
            )

            resigned_public = copy.deepcopy(evidence["public_packet"])
            resigned_public["packet_id"] = "packet-self-resigned-hostile"
            resigned_public = sign(resigned_public, "public_packet_digest")
            public_path.write_bytes(
                gate.canonical_json_bytes(resigned_public) + b"\n"
            )
            hostile_output = authority.work / "hostile-review.json"
            hostile_argv = list(argv)
            hostile_argv[-1] = str(hostile_output)
            with authority.environment():
                with self.assertRaisesRegex(
                    gate.ActionPreservationGateError, "SHA differs"
                ):
                    with mock.patch.object(gate, "_verify_cli_eval_release"):
                        gate.main(hostile_argv)
            self.assertFalse(hostile_output.exists())


if __name__ == "__main__":
    unittest.main()
