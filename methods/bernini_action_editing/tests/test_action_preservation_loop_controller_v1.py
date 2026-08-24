from __future__ import annotations

import copy
import hashlib
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TEST_ROOT = pathlib.Path(__file__).resolve().parent
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

import action_preservation_gate_v1 as gate
import action_preservation_loop_controller_v1 as loop
import test_action_preservation_gate_v1 as gate_fixtures


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def sign(row: dict, field: str) -> dict:
    result = copy.deepcopy(row)
    result.pop(field, None)
    result[field] = gate.object_sha256(result)
    return result


def provenance(*, base: str = "base") -> dict[str, str]:
    result = {key: digest(f"{key}:{base}") for key in loop.PROVENANCE_FIELDS}
    result["calibration_digest"] = calibration()["calibration_digest"]
    return result


def calibration() -> dict:
    row = {
        "schema_version": gate.CALIBRATION_SCHEMA,
        "calibration_id": "heldout-human-v1",
        "heldout_manifest_sha256": digest("calibration-manifest"),
        "human_labels_sha256": digest("calibration-labels"),
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
            "human_agreement_report_sha256": digest("calibration-agreement"),
            "worst_group_reported": True,
            "domain_gap_reported": True,
            "hacking_failures_reported": True,
        },
        "controlled_negatives": {key: True for key in gate._NEGATIVE_FIELDS},
        "thresholds_frozen_before_candidate_generation": True,
    }
    return sign(row, "calibration_digest")


def measurement(*, background_similarity: float = 0.91) -> dict:
    row = {
        "schema_version": gate.MEASUREMENT_SCHEMA,
        "candidate_id": "stage0-u20-candidate",
        "candidate_video_sha256": digest("candidate-video"),
        "source_video_sha256": digest("source-video"),
        "scope": {
            "single_subject": True,
            "human_subject": True,
            "source_face_visible": True,
            "output_face_visible": True,
            "static_or_weak_camera": True,
            "no_shot_cut": True,
            "background_expected_unchanged": True,
        },
        "face": {
            "available": True,
            "similarity": 0.88,
            "coverage": 0.90,
            "source_face_pool_size": 4,
            "receipt_sha256": digest("face-measurement"),
        },
        "background": {
            "available": True,
            "similarity": background_similarity,
            "valid_coverage": 0.72,
            "source_and_output_masks_independent": True,
            "union_exclusion_used": True,
            "receipt_sha256": digest("background-measurement"),
        },
        "camera": {
            "available": True,
            "translation": 0.01,
            "log_scale_abs": 0.01,
            "rotation_degrees_abs": 0.5,
            "reprojection_error": 0.7,
            "background_registration_used": True,
            "receipt_sha256": digest("camera-measurement"),
        },
        "quality": {
            "available": True,
            "score": 0.9,
            "receipt_sha256": digest("quality-measurement"),
        },
        "onset": {
            "available": True,
            "anchor_frame": 10,
            "candidate_frame": 12,
            "timing_error_frames": 2,
            "score": 0.85,
            "receipt_sha256": digest("onset-measurement"),
        },
        "action_order": {
            "available": True,
            "score": 1.0,
            "reverse_rejected": True,
            "truncation_rejected": True,
            "terminal_hold_score": 0.9,
            "terminal_hold_start_frame": 71,
            "terminal_hold_end_frame": 80,
            "terminal_hold_frames": 10,
            "receipt_sha256": digest("action-measurement"),
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


def blind_review(decision: dict, *, failing_axis: str | None = None) -> tuple[dict, dict]:
    labels = {axis: "pass" for axis in gate.AXES}
    if failing_axis is not None:
        labels[failing_axis] = "fail"
    return gate_fixtures.blind_review(decision, labels=labels)


class LoopControllerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.parent = pathlib.Path(self.temporary.name)
        self.stage_root = self.parent / "stage-000"
        self.plan = loop.build_stage_plan(
            stage_id="preservation-v2-stage-000",
            stage_index=0,
            stage_root=self.stage_root,
            input_provenance=provenance(),
        )
        checkpoints = [
            {
                "relative_step": step,
                "checkpoint_sha256": digest(f"checkpoint-{step}"),
                "checkpoint_receipt_sha256": digest(f"checkpoint-receipt-{step}"),
            }
            for step in loop.CHECKPOINT_STEPS
        ]
        self.stage_receipt = loop.build_stage_receipt(
            self.plan,
            checkpoints=checkpoints,
            training_completion_receipt_sha256=digest("training-complete"),
            candidate_id="stage0-u20-candidate",
            candidate_video_sha256=digest("candidate-video"),
            source_video_sha256=digest("source-video"),
            decode_receipt_sha256=digest("decode-receipt"),
            loss_receipt_sha256=digest("loss-diagnostic"),
        )

    def transition(
        self,
        measured: dict | None,
        *,
        review: tuple[dict, dict] | None = None,
        decision: dict | None = None,
        stage_receipt: dict | None = None,
    ) -> dict:
        authority = calibration()
        if decision is None and measured is not None:
            decision = gate.decide(measured, authority)
        review_value = review[0] if review is not None else None
        review_evidence = review[1] if review is not None else {}
        return loop.decide_next_action(
            self.plan,
            stage_receipt=self.stage_receipt if stage_receipt is None else stage_receipt,
            measurement=measured,
            calibration=authority,
            decision=decision,
            blind_review=review_value,
            **review_evidence,
        )

    def test_fresh_root_and_artifacts_are_create_only(self):
        plan_path = loop.publish_stage_plan(self.plan)
        self.assertEqual(plan_path, self.stage_root / loop.PLAN_FILENAME)
        with self.assertRaisesRegex(loop.ActionPreservationLoopError, "not fresh"):
            loop.publish_stage_plan(self.plan)
        receipt_path = loop.publish_stage_receipt(self.plan, self.stage_receipt)
        self.assertTrue(receipt_path.is_file())
        with self.assertRaisesRegex(loop.ActionPreservationLoopError, "overwrite"):
            loop.publish_stage_receipt(self.plan, self.stage_receipt)

        measured = measurement()
        decision = gate.decide(measured, calibration())
        waiting = self.transition(measured, decision=decision)
        wait_path = loop.publish_transition_receipt(self.plan, waiting)
        self.assertEqual(wait_path.name, loop.WAIT_TRANSITION_FILENAME)
        with self.assertRaisesRegex(loop.ActionPreservationLoopError, "overwrite"):
            loop.publish_transition_receipt(self.plan, waiting)

        changed = measurement()
        changed["quality"]["score"] = 0.95
        changed = sign(changed, "measurement_digest")
        changed_decision = gate.decide(changed, calibration())
        changed_final = self.transition(
            changed,
            decision=changed_decision,
            review=blind_review(changed_decision),
        )
        with self.assertRaisesRegex(loop.ActionPreservationLoopError, "machine evidence"):
            loop.publish_transition_receipt(self.plan, changed_final)

        transition = self.transition(
            measured, decision=decision, review=blind_review(decision)
        )
        transition_path = loop.publish_transition_receipt(self.plan, transition)
        self.assertEqual(transition_path.name, loop.TRANSITION_FILENAME)
        self.assertTrue(transition_path.is_file())
        with self.assertRaisesRegex(loop.ActionPreservationLoopError, "overwrite"):
            loop.publish_transition_receipt(self.plan, transition)

    def test_missing_evidence_fail_closes_to_stop(self):
        result = loop.decide_next_action(
            self.plan,
            stage_receipt=None,
            measurement=None,
            calibration=None,
            decision=None,
        )
        self.assertEqual(result["next_action"], loop.STOP)
        self.assertIn("stage_receipt_missing", result["reasons"])
        loop.publish_stage_plan(self.plan)
        self.assertEqual(
            loop.publish_transition_receipt(self.plan, result).name,
            loop.TRANSITION_FILENAME,
        )

        result = loop.decide_next_action(
            self.plan,
            stage_receipt=self.stage_receipt,
            measurement=None,
            calibration=calibration(),
            decision=None,
        )
        self.assertEqual(result["next_action"], loop.WAIT_FOR_MACHINE_EVIDENCE)
        self.assertEqual(result["machine_status"], "abstain")
        self.assertEqual(result["human_status"], "review_may_proceed_in_parallel")

    def test_machine_failure_forces_stop(self):
        measured = measurement(background_similarity=0.20)
        result = self.transition(measured)
        self.assertEqual(result["next_action"], loop.STOP)
        self.assertEqual(result["machine_status"], "reject")
        self.assertFalse(result["next_stage_requires_fresh_create_only_root"])

    def test_machine_pass_waits_for_blinded_full_video_review(self):
        result = self.transition(measurement())
        self.assertEqual(result["next_action"], loop.WAIT_FOR_BLIND_REVIEW)
        self.assertEqual(result["human_status"], "pending")
        self.assertFalse(result["automatic_model_update"])

    def test_human_failure_forces_stop(self):
        measured = measurement()
        decision = gate.decide(measured, calibration())
        result = self.transition(
            measured,
            decision=decision,
            review=blind_review(decision, failing_axis="background"),
        )
        self.assertEqual(result["next_action"], loop.STOP)
        self.assertEqual(result["human_status"], "fail_or_undetermined")

    def test_successful_review_is_only_eligible_for_fresh_next_stage(self):
        measured = measurement()
        decision = gate.decide(measured, calibration())
        result = self.transition(
            measured, decision=decision, review=blind_review(decision)
        )
        self.assertEqual(result["next_action"], loop.ELIGIBLE_FOR_NEXT_20)
        self.assertFalse(result["automatic_model_update"])
        self.assertFalse(result["controller_performed_remote_launch"])
        self.assertTrue(result["next_stage_requires_fresh_create_only_root"])

        next_pins = provenance(base="next")
        next_pins["base_checkpoint_sha256"] = digest("checkpoint-20")
        next_plan = loop.build_next_stage_plan(
            self.plan,
            result,
            stage_id="preservation-v2-stage-001",
            stage_root=self.parent / "stage-001",
            input_provenance=next_pins,
        )
        self.assertEqual(next_plan["stage_index"], 1)
        self.assertEqual(next_plan["lineage"]["prior_stage_roots"], [str(self.stage_root)])
        self.assertEqual(
            next_plan["lineage"]["parent_transition_digest"],
            result["transition_digest"],
        )

    def test_no_weighted_compensation_or_loss_promotion(self):
        measured = measurement(background_similarity=0.20)
        result = self.transition(measured)
        self.assertEqual(result["next_action"], loop.STOP)
        self.assertIsNone(result["weighted_score"])
        self.assertFalse(result["loss_used_for_transition"])

        tampered = copy.deepcopy(self.stage_receipt)
        tampered["loss_diagnostics"]["used_for_transition"] = True
        tampered = sign(tampered, "stage_receipt_digest")
        result = self.transition(measurement(), stage_receipt=tampered)
        self.assertEqual(result["next_action"], loop.STOP)
        self.assertIn("stage_receipt_invalid", result["reasons"])

    def test_provenance_hashes_and_candidate_binding_are_enforced(self):
        invalid_pins = provenance()
        invalid_pins["teacher_cache_sha256"] = "not-a-sha"
        with self.assertRaisesRegex(loop.ActionPreservationLoopError, "SHA-256"):
            loop.build_stage_plan(
                stage_id="invalid-stage",
                stage_index=0,
                stage_root=self.parent / "invalid-stage",
                input_provenance=invalid_pins,
            )

        measured = measurement()
        measured["candidate_video_sha256"] = digest("different-video")
        measured = sign(measured, "measurement_digest")
        result = self.transition(measured)
        self.assertEqual(result["next_action"], loop.STOP)
        self.assertIn(
            "decoded_candidate_measurement_binding_differs", result["reasons"]
        )

        wrong_plan = copy.deepcopy(self.plan)
        wrong_plan["input_provenance"]["calibration_digest"] = digest(
            "different-calibration"
        )
        wrong_plan = sign(wrong_plan, "plan_digest")
        wrong_receipt = loop.build_stage_receipt(
            wrong_plan,
            checkpoints=self.stage_receipt["training"]["checkpoints"],
            training_completion_receipt_sha256=digest("training-complete"),
            candidate_id="stage0-u20-candidate",
            candidate_video_sha256=digest("candidate-video"),
            source_video_sha256=digest("source-video"),
            decode_receipt_sha256=digest("decode-receipt"),
        )
        measured = measurement()
        authority = calibration()
        result = loop.decide_next_action(
            wrong_plan,
            stage_receipt=wrong_receipt,
            measurement=measured,
            calibration=authority,
            decision=gate.decide(measured, authority),
        )
        self.assertEqual(result["next_action"], loop.STOP)
        self.assertIn("calibration_plan_binding_differs", result["reasons"])

    def test_supplied_gate_decision_must_equal_strict_recomputation(self):
        measured = measurement()
        supplied = gate.decide(measured, calibration())
        supplied["motion_ranking_allowed"] = False
        result = self.transition(measured, decision=supplied)
        self.assertEqual(result["next_action"], loop.STOP)
        self.assertIn(
            "strict_gate_decision_differs_from_recomputation", result["reasons"]
        )

    def test_previous_root_cannot_be_reused(self):
        with self.assertRaisesRegex(loop.ActionPreservationLoopError, "reuses"):
            loop.build_stage_plan(
                stage_id="preservation-v2-stage-001",
                stage_index=1,
                stage_root=self.stage_root,
                input_provenance=provenance(base="next"),
                prior_stage_roots=[self.stage_root],
                parent_stage_id=self.plan["stage_id"],
                parent_plan_digest=self.plan["plan_digest"],
                parent_transition_digest=digest("parent-transition"),
            )


class HeldLoopAuthorityTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = pathlib.Path(temporary.name).resolve() / "work"
        self.work.mkdir(mode=0o700)
        self.work_fd = os.open(
            self.work,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0),
        )
        os.set_inheritable(self.work_fd, False)
        self.addCleanup(os.close, self.work_fd)
        self.work_authority = {
            "path": str(self.work),
            "root_fd": self.work_fd,
            "work_root_authority_digest": digest("work-root-authority"),
        }
        replay = mock.patch.object(
            loop,
            "_replay_loop_work_root_authority",
            side_effect=lambda value: self.work_authority,
        )
        replay.start()
        self.addCleanup(replay.stop)
        self.plan = loop.build_stage_plan(
            stage_id="held-stage-000",
            stage_index=0,
            stage_root=self.work / "held-stage-000",
            input_provenance=provenance(),
        )
        (
            self.stage_authority,
            self.stage_authority_file,
            self.plan_file,
        ) = loop._create_and_publish_stage_plan(
            work_root=self.work_authority, plan=self.plan
        )
        checkpoints = [
            {
                "relative_step": step,
                "checkpoint_sha256": digest(f"held-checkpoint-{step}"),
                "checkpoint_receipt_sha256": digest(
                    f"held-checkpoint-receipt-{step}"
                ),
            }
            for step in loop.CHECKPOINT_STEPS
        ]
        self.stage_receipt = loop.build_stage_receipt(
            self.plan,
            checkpoints=checkpoints,
            training_completion_receipt_sha256=digest(
                "held-training-complete"
            ),
            candidate_id="stage0-u20-candidate",
            candidate_video_sha256=digest("candidate-video"),
            source_video_sha256=digest("source-video"),
            decode_receipt_sha256=digest("held-decode-receipt"),
        )

    def advance_argv(self, receipt_path: pathlib.Path, receipt_sha: str):
        return [
            "advance",
            "--plan", self.plan_file["path"],
            "--plan-sha256", self.plan_file["sha256"],
            "--stage-authority", self.stage_authority_file["path"],
            "--stage-authority-sha256", self.stage_authority_file["sha256"],
            "--stage-receipt", str(receipt_path),
            "--stage-receipt-sha256", receipt_sha,
        ]

    def test_external_valid_stage_receipt_cannot_skip_seal_stage(self) -> None:
        external = self.work / "external-stage-receipt.json"
        payload = loop.canonical_json_bytes(self.stage_receipt) + b"\n"
        external.write_bytes(payload)
        external.chmod(0o444)
        captured = []
        original = loop._open_stage_context

        def capture(**kwargs):
            result = original(**kwargs)
            captured.append(result[0])
            return result

        with mock.patch.object(
            loop, "_load_loop_work_root_authority",
            return_value=self.work_authority,
        ), mock.patch.object(
            loop, "_open_stage_context", side_effect=capture
        ), self.assertRaisesRegex(
            loop.ActionPreservationLoopError,
            "outside the held stage root",
        ):
            loop.main(
                self.advance_argv(
                    external, hashlib.sha256(payload).hexdigest()
                )
            )
        self.assertEqual(loop._HELD_STAGE_LIFETIMES, [])
        self.assertEqual(len(captured), 1)
        with self.assertRaises(OSError):
            os.fstat(captured[0].descriptor)

    def test_in_root_receipt_path_is_rejected_when_seal_stage_was_skipped(
        self,
    ) -> None:
        missing = pathlib.Path(self.plan["stage_root"]) / loop.STAGE_RECEIPT_FILENAME
        with mock.patch.object(
            loop, "_load_loop_work_root_authority",
            return_value=self.work_authority,
        ), self.assertRaisesRegex(
            loop.ActionPreservationLoopError,
            "cannot read held published stage receipt",
        ):
            loop.main(self.advance_argv(missing, digest("missing receipt")))
        self.assertEqual(loop._HELD_STAGE_LIFETIMES, [])

    def test_held_sealed_stage_receipt_can_publish_machine_wait(self) -> None:
        held, _, _ = loop._open_stage_context(
            work_root=self.work_authority,
            plan_path=self.plan_file["path"],
            plan_sha256=self.plan_file["sha256"],
            authority_path=self.stage_authority_file["path"],
            authority_sha256=self.stage_authority_file["sha256"],
        )
        try:
            receipt_file = held.write_json(
                loop.STAGE_RECEIPT_FILENAME, self.stage_receipt
            )
        finally:
            held.close()
        with mock.patch.object(
            loop, "_load_loop_work_root_authority",
            return_value=self.work_authority,
        ):
            self.assertEqual(
                loop.main(
                    self.advance_argv(
                        pathlib.Path(receipt_file["path"]),
                        receipt_file["sha256"],
                    )
                ),
                0,
            )
        self.assertTrue(
            (pathlib.Path(self.plan["stage_root"])
             / loop.MACHINE_WAIT_TRANSITION_FILENAME).is_file()
        )


if __name__ == "__main__":
    unittest.main()
