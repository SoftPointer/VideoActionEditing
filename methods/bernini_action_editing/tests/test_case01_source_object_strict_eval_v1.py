#!/usr/bin/env python3
"""Contract tests for the case01 fail-closed source-object evaluator."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest

from methods.bernini_action_editing import case01_source_object_strict_eval_v1 as evaluator


REPO_ROOT = Path(__file__).resolve().parents[3]


def passing_variant(name: str = "exact_original") -> dict:
    return {
        "variant": name,
        "review_coverage": {
            "all_81_decoded_frames_reviewed": True,
            "source_and_output_pair_reviewed": True,
            "frame_range": [0, 80],
            "frame_count": 81,
        },
        "dog_identity": {
            "subject_track_id": "dog#1",
            "identity_switch_observed": False,
            "first_mismatch_frame": None,
            "cues": [
                {
                    "name": name,
                    "source": f"source-{name}",
                    "output": f"source-{name}",
                    "preserved": True,
                }
                for name in evaluator.IDENTITY_CUES
            ],
        },
        "source_bone": {
            "patient_track_id": "bone#1",
            "input_patient_available": True,
            "same_instance_continuity": "PROVEN",
            "left_initial_support": True,
            "entered_effector_region": True,
            "terminal_hold": True,
            "source_instance_remains_in_background": False,
            "duplicate_or_substitute_prop": {
                "observed": False,
                "frame_interval": None,
                "description": "No second object is present.",
            },
            "observed_state": "bone#1 moves continuously from ground into dog#1.mouth",
        },
        "action_trace": {
            "patient_track_id": "bone#1",
            "effector_region_id": "dog#1.mouth",
            "minimum_hold_frames": 10,
            "stages": [
                {
                    "name": stage,
                    "observed": True,
                    "frame_interval": interval,
                    "evidence": f"Synthetic evidence for {stage} of bone#1.",
                }
                for stage, interval in zip(
                    evaluator.ACTION_STAGES,
                    ([0, 15], [20, 22], [25, 30], [35, 42], [50, 80]),
                )
            ],
        },
    }


class Case01SourceObjectStrictEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("ffprobe") is None:
            raise unittest.SkipTest("ffprobe is required for real-bundle replay")
        cls.report = evaluator.evaluate_bundle(REPO_ROOT)

    def test_real_exact5_is_explicit_five_of_five_failure(self) -> None:
        report = self.report
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["evaluation_status"], "COMPLETE_FAIL_CLOSED")
        self.assertTrue(report["all_five_failed"])
        self.assertEqual(
            report["counts"],
            {"variant_count": 5, "pass_count": 0, "fail_count": 5},
        )
        self.assertEqual(
            tuple(row["variant"] for row in report["variants"]),
            evaluator.VARIANT_ORDER,
        )
        for row in report["variants"]:
            self.assertEqual(row["status"], "FAIL")
            self.assertEqual(row["gate_statuses"]["review_coverage"], "PASS")
            self.assertEqual(row["gate_statuses"]["dog_identity_retention"], "FAIL")
            self.assertEqual(row["gate_statuses"]["same_source_bone_reuse"], "FAIL")
            self.assertEqual(row["gate_statuses"]["ordered_source_bone_action"], "FAIL")
            self.assertEqual(
                row["gates"]["ordered_source_bone_action"]["observed_stage_count"],
                0,
            )

    def test_real_failures_name_identity_switch_background_bone_and_second_prop(self) -> None:
        rows = {row["variant"]: row for row in self.report["variants"]}
        for row in rows.values():
            identity_reasons = row["gates"]["dog_identity_retention"]["reasons"]
            self.assertIn("identity_switch_observed_at_frame_0", identity_reasons)
            self.assertEqual(
                row["gates"]["dog_identity_retention"]["preserved_cue_count"],
                0,
            )

        for name in (
            "exact_original",
            "codec_only_present",
            "bone_translated_up150",
            "sham_control_up150",
        ):
            self.assertIn(
                "source_bone_remains_in_background",
                rows[name]["gates"]["same_source_bone_reuse"]["reasons"],
            )
        for name in ("exact_original", "codec_only_present", "sham_control_up150"):
            self.assertIn(
                "duplicate_or_substitute_prop_observed",
                rows[name]["gates"]["same_source_bone_reuse"]["reasons"],
            )
        self.assertIn(
            "source_bone_not_available_in_intervention_input",
            rows["bone_removed"]["gates"]["same_source_bone_reuse"]["reasons"],
        )

    def test_source_authority_replays_all_81_masklets_and_g0(self) -> None:
        replay = self.report["source_grounding_replay"]
        self.assertEqual(replay["source_dog_mask_count"], 81)
        self.assertEqual(replay["source_bone_mask_count"], 81)
        self.assertEqual(replay["source_joint_iou_zero_frames"], 81)
        self.assertEqual(replay["g0_sparse_frame_count"], 9)
        self.assertEqual(replay["g0_independent_admission"], "PASS")
        self.assertEqual(len(self.report["media_replay"]), 5)
        for row in self.report["media_replay"]:
            self.assertEqual(row["media_probe"], evaluator.EXPECTED_MEDIA_PROBE)

    def test_conjunction_is_reachable_but_no_gate_can_compensate(self) -> None:
        baseline = passing_variant()
        passed = evaluator.evaluate_variant(baseline)
        self.assertEqual(passed["status"], "PASS")
        self.assertTrue(all(value == "PASS" for value in passed["gate_statuses"].values()))

        identity_failure = copy.deepcopy(baseline)
        identity_failure["dog_identity"]["identity_switch_observed"] = True
        identity_failure["dog_identity"]["first_mismatch_frame"] = 0
        failed = evaluator.evaluate_variant(identity_failure)
        self.assertEqual(failed["status"], "FAIL")
        self.assertEqual(failed["gate_statuses"]["dog_identity_retention"], "FAIL")
        self.assertEqual(failed["gate_statuses"]["same_source_bone_reuse"], "PASS")
        self.assertEqual(failed["gate_statuses"]["ordered_source_bone_action"], "PASS")

    def test_action_requires_order_same_patient_and_persistent_hold(self) -> None:
        wrong_order = passing_variant()
        wrong_order["action_trace"]["stages"][1]["frame_interval"] = [40, 42]
        report = evaluator.evaluate_variant(wrong_order)
        self.assertIn(
            "source_bone_stage_onsets_not_strictly_ordered",
            report["gates"]["ordered_source_bone_action"]["reasons"],
        )

        short_hold = passing_variant()
        short_hold["action_trace"]["stages"][-1]["frame_interval"] = [75, 80]
        report = evaluator.evaluate_variant(short_hold)
        self.assertIn(
            "source_bone_hold_too_short:6<10",
            report["gates"]["ordered_source_bone_action"]["reasons"],
        )

        wrong_patient = passing_variant()
        wrong_patient["action_trace"]["patient_track_id"] = "generated-prop#2"
        with self.assertRaisesRegex(evaluator.StrictEvalError, "patient must be bone#1"):
            evaluator.evaluate_variant(wrong_patient)

    def test_missing_or_tampered_evidence_fails_closed(self) -> None:
        missing_interval = passing_variant()
        missing_interval["action_trace"]["stages"][2]["frame_interval"] = None
        with self.assertRaisesRegex(evaluator.StrictEvalError, "evidence/interval"):
            evaluator.evaluate_variant(missing_interval)

        expected = b"immutable evidence\n"
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            path = Path(temporary) / "evidence.bin"
            path.write_bytes(expected)
            expected_hash = hashlib.sha256(expected).hexdigest()
            evaluator._stable_file(
                path.resolve(),
                expected_sha256=expected_hash,
                expected_size=len(expected),
            )
            path.write_bytes(expected + b"tampered\n")
            with self.assertRaisesRegex(evaluator.StrictEvalError, "SHA-256 differs"):
                evaluator._stable_file(
                    path.resolve(),
                    expected_sha256=expected_hash,
                    expected_size=len(expected),
                )

    def test_report_digest_and_claim_limits_are_closed(self) -> None:
        report = copy.deepcopy(self.report)
        digest = report.pop("report_digest")
        self.assertEqual(digest, evaluator.object_sha256(report))
        self.assertTrue(report["claim_limits"]["diagnostic_failure_report_only"])
        self.assertFalse(report["claim_limits"]["formal_causal_claim_authorized"])
        self.assertFalse(report["claim_limits"]["scientific_claim_authorized"])


if __name__ == "__main__":
    unittest.main()
