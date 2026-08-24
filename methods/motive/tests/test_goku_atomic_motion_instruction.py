from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from motive import goku_atomic_motion_instruction as atomic
from motive import goku_natural_motion_instruction as natural_v5


IID_GOOD = "coherent_multi_subject"
IID_23C = "23c7c93a3219452b"
SUBJECT_IDS = ["subject_01", "subject_02", "subject_03", "subject_04"]
ROLES = [["agent"], ["tool"], ["patient"], ["effect"]]


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _passed(iid: str, *, conflicting_23c: bool = False) -> dict:
    source_subjects = [
        {
            "subject_id": "subject_01",
            "dynamic": True,
            "stable_reference": "the cook beside the grill",
            "i0_state": "a cook stands beside the grill",
            "source_motion": "the cook handles food near the grill",
        },
        {
            "subject_id": "subject_02",
            "dynamic": True,
            "stable_reference": "the metal tongs in the cook's hand",
            "i0_state": "metal tongs are visible in the cook's hand",
            "source_motion": "the tongs move with the cook's hand",
        },
        {
            "subject_id": "subject_03",
            "dynamic": True,
            "stable_reference": "the skewered sausage on the grill",
            "i0_state": "a skewered sausage rests on the grill",
            "source_motion": "the sausage shifts near the grill",
        },
        {
            "subject_id": "subject_04",
            "dynamic": True,
            "stable_reference": "the steam above the grill",
            "i0_state": "steam is visible above the grill",
            "source_motion": "steam rises above the grill",
        },
    ]
    if conflicting_23c:
        motions = [
            (
                "from frame 0 to frame 20, the cook uses the tongs to transfer the "
                "first sausage into the container; from frame 20 to frame 40, the cook "
                "uses a bare right hand to transfer the skewered sausage into it"
            ),
            (
                "from frame 0 to frame 20, the tongs transfer the first sausage into "
                "the container; from frame 20 to frame 80, the tongs remain still"
            ),
            (
                "from frame 0 to frame 20, the skewered sausage moves from the right "
                "side into the container; from frame 20 to frame 80, it remains there"
            ),
            (
                "from frame 0 to frame 40, steam rises above the grill; from frame 40 "
                "to frame 80, it dissipates"
            ),
        ]
    else:
        motions = [
            (
                "from frame 0 to frame 40, the cook uses the tongs to transfer the "
                "skewered sausage into the tray, causing steam to rise"
            ),
            (
                "from frame 0 to frame 40, the tongs carry the skewered sausage into "
                "the tray under the cook's control"
            ),
            (
                "from frame 0 to frame 40, the skewered sausage moves with the tongs "
                "from the grill into the tray"
            ),
            (
                "from frame 0 to frame 40, steam rises as the hot sausage is transferred "
                "from the grill into the tray"
            ),
        ]
    target_subjects = [
        {
            "subject_id": subject_id,
            "substantive_change": True,
            "target_motion": motion,
            "target_action_signature": "transfer_sausage",
        }
        for subject_id, motion in zip(SUBJECT_IDS, motions)
    ]
    old_prompt = "Starting from the exact first frame: " + " ".join(motions)
    return {
        "schema_version": natural_v5.EXPECTED_PASSED_SCHEMA,
        "iid": iid,
        "action_change_substantive": True,
        "all_dynamic_subjects_covered": True,
        "camera_covered": True,
        "edit_instruction": old_prompt,
        "edit_instruction_sha256": _sha_text(old_prompt),
        "compiled_instruction": {"instruction": old_prompt},
        "source_census": {
            "iid": iid,
            "dynamic_subjects": source_subjects,
            "camera": {"motion_class": "locked_off", "source_motion": "fixed"},
        },
        "target_plan": {
            "iid": iid,
            "dynamic_subject_targets": target_subjects,
            "coverage": {
                "dynamic_subject_ids": SUBJECT_IDS,
                "camera_covered": True,
            },
            "camera_target": {
                "motion_class": "locked_off",
                "relation": "preserve_static",
                "target_motion": "from frame 0 to frame 80, the camera remains locked off",
            },
        },
        "resolved_source_video": f"/data/{iid}/source.mp4",
        "source_video_sha256": "a" * 64,
    }


def _input_row(
    iid: str = IID_GOOD, *, conflicting_23c: bool = False, candidate_index: int = 0
) -> dict:
    row = {
        "schema_version": natural_v5.INPUT_SCHEMA,
        "iid": iid,
        "original_candidate_index": candidate_index,
        "candidates_path": "/data/candidates.jsonl",
        "candidates_sha256": "b" * 64,
        "source_passed_path": f"/data/passed/{iid}.jsonl",
        "source_passed_sha256": "c" * 64,
        "passed_row": _passed(iid, conflicting_23c=conflicting_23c),
        "row_digest": None,
    }
    row["row_digest"] = atomic._object_digest(row, omit="row_digest")
    return row


def _plan_audit(iid: str, *, pass_plan: bool = True) -> dict:
    checks = {
        "schema_version": atomic.PLAN_GLOBAL_CHECKS_SCHEMA,
        "single_causal_event": True,
        "all_dynamic_subjects_in_event": True,
        "no_independent_action_thread": True,
        "controller_tool_patient_roles_consistent": True,
        "cross_subject_timing_consistent": pass_plan,
        "cross_subject_contact_transfer_consistent": pass_plan,
        "physically_coherent": pass_plan,
        "camera_compatible": True,
    }
    return {
        "schema_version": atomic.PLAN_AUDIT_SCHEMA,
        "iid": iid,
        "atomic_event": {
            "schema_version": atomic.PLAN_EVENT_SCHEMA,
            "event_name": "sausage_transfer",
            "event_summary": "the cook transfers one sausage with the tongs into the tray",
            "participant_subject_ids": SUBJECT_IDS,
        },
        "subject_roles": [
            {
                "schema_version": atomic.PLAN_SUBJECT_ROLE_SCHEMA,
                "subject_id": subject_id,
                "roles": roles,
                "same_event_participant": True,
            }
            for subject_id, roles in zip(SUBJECT_IDS, ROLES)
        ],
        "global_checks": checks,
        "overall_verdict": "pass" if pass_plan else "fail",
        "reason_codes": (
            []
            if pass_plan
            else ["cross_subject_transfer_intervals_contradict"]
        ),
        "confidence": "high",
    }


def _rewrite(iid: str = IID_GOOD) -> dict:
    references = ["the cook", "the tongs", "the skewered sausage", "the steam"]
    summaries = [
        "agent controlling the transfer",
        "tool carrying the sausage",
        "patient transferred into the tray",
        "direct heat effect rising from the transfer",
    ]
    return {
        "schema_version": atomic.REWRITE_SCHEMA,
        "iid": iid,
        "atomic_event": {
            "schema_version": atomic.REWRITE_EVENT_SCHEMA,
            "event_name": "sausage_transfer",
            "event_summary": "the cook transfers the sausage into the tray with the tongs",
            "participant_subject_ids": SUBJECT_IDS,
        },
        "atomic_action_instruction": (
            "Have the cook transfer the skewered sausage into the tray with the tongs, "
            "causing the steam to rise."
        ),
        "subject_mappings": [
            {
                "schema_version": atomic.SUBJECT_MAPPING_SCHEMA,
                "subject_id": subject_id,
                "natural_reference": reference,
                "event_roles": roles,
                "participation_summary": summary,
            }
            for subject_id, reference, roles, summary in zip(
                SUBJECT_IDS, references, ROLES, summaries
            )
        ],
        "camera_instruction": "Keep the camera fixed.",
        "preservation_instruction": "Keep all visible appearances unchanged.",
    }


def _semantic_audit(iid: str = IID_GOOD) -> dict:
    return {
        "schema_version": atomic.AUDIT_SCHEMA,
        "iid": iid,
        "subject_audits": [
            {
                "schema_version": atomic.SUBJECT_AUDIT_SCHEMA,
                "subject_id": subject_id,
                "explicitly_grounded": True,
                "same_event_participation_entailed": True,
                "role_match": True,
                "motion_direction_endpoint_match": True,
                "no_independent_action": True,
            }
            for subject_id in SUBJECT_IDS
        ],
        "global_audit": {
            "schema_version": atomic.GLOBAL_AUDIT_SCHEMA,
            "single_atomic_event": True,
            "all_dynamic_subjects_covered": True,
            "one_causal_graph": True,
            "agent_tool_patient_effect_consistent": True,
            "cross_subject_temporal_consistency_preserved": True,
            "no_independent_action_thread": True,
            "physically_coherent": True,
            "no_sequence_or_concurrency_stitching": True,
        },
        "camera_audit": {
            "schema_version": atomic.CAMERA_AUDIT_SCHEMA,
            "explicit": True,
            "class_direction_match": True,
            "compatible_with_atomic_event": True,
            "no_contradiction": True,
        },
        "appearance_content_preserved": True,
        "natural_atomic_imperative": True,
        "overall_verdict": "pass",
        "reason_codes": [],
        "confidence": "high",
    }


class FakeBackend:
    model_path = "/models/Qwen3-VL-32B-Instruct"
    model_revision = "test-revision"
    transformers_version = "test"
    mode = "text"

    def __init__(self, responses: list[dict]):
        self.responses = [json.dumps(value) for value in responses]
        self.calls = 0
        self.requests: list[dict[str, str]] = []

    def generate_text(self, *, system: str, user: str) -> str:
        self.requests.append({"system": system, "user": user})
        response = self.responses[self.calls]
        self.calls += 1
        return response


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _worker_args(manifest: Path, output: Path, *, allow_errors: bool) -> argparse.Namespace:
    return argparse.Namespace(
        input=manifest,
        output_root=output,
        model="/models/Qwen3-VL-32B-Instruct",
        worker_index=0,
        num_workers=1,
        num_rows=1,
        max_new_tokens=2048,
        max_attempts=2,
        attn_implementation="sdpa",
        allow_download=False,
        allow_errors=allow_errors,
        skip_source_revalidation=True,
    )


class AtomicMotionInstructionTest(unittest.TestCase):
    def test_23c_cross_subject_timing_conflict_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            row = _input_row(IID_23C, conflicting_23c=True)
            payload = atomic._plan_payload(row)
            serialized = json.dumps(payload, sort_keys=True)
            self.assertIn("frame 20 to frame 40", serialized)
            self.assertIn("frame 0 to frame 20", serialized)
            self.assertTrue(
                payload["raw_target_subject_plans"][0][
                    "shared_frame_interval_evidence"
                ]
            )
            self.assertTrue(
                payload["raw_target_subject_plans"][2][
                    "shared_frame_interval_evidence"
                ]
            )

            manifest = tmp_path / "input.jsonl"
            _write_manifest(manifest, [row])
            output = tmp_path / "output"
            backend = FakeBackend([_plan_audit(IID_23C, pass_plan=False)])

            self.assertEqual(
                atomic.run_worker(
                    _worker_args(manifest, output, allow_errors=True),
                    backend_factory=lambda **_: backend,
                ),
                0,
            )
            self.assertEqual(backend.calls, 1)
            result = json.loads(
                (output / "rows" / IID_23C / "result.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["error"]["type"], "AtomicTargetPlanRejected")
            self.assertEqual(result["rewrite_attempts"], [])
            receipt = json.loads(
                (output / "terminal" / f"{IID_23C}.receipt.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(receipt["status"], "error")
            self.assertIsNone(receipt["atomic_action_instruction_path"])
            self.assertIsNone(receipt["full_edit_instruction_path"])
            self.assertFalse((output / "instructions" / IID_23C).exists())
            self.assertFalse((output / "full_instructions" / IID_23C).exists())

            manifest_output = tmp_path / "dataset.jsonl"
            summary_output = tmp_path / "summary.json"
            self.assertEqual(
                atomic.verify_outputs(
                    argparse.Namespace(
                        input=manifest,
                        output_root=output,
                        expected_rows=1,
                        min_ok=0,
                        manifest_output=manifest_output,
                        summary_output=summary_output,
                    )
                ),
                0,
            )
            self.assertEqual(manifest_output.read_bytes(), b"")
            summary = json.loads(summary_output.read_text(encoding="utf-8"))
            self.assertEqual(summary["ok_rows"], 0)
            self.assertEqual(summary["error_rows"], 1)

    def test_atomic_action_rejects_frame_time_and_stitched_actions(self) -> None:
        bad_actions = [
            "Have the cook lift the sausage from frame 0 to frame 20.",
            "Have the cook lift the sausage for two seconds.",
            "Have the cook move the sausage frame by frame.",
            "Have the cook lift the sausage, then place it in the tray.",
            "Have the cook lift the sausage while the other person waves.",
        ]
        for bad_action in bad_actions:
            with self.subTest(bad_action=bad_action):
                rewrite = _rewrite()
                rewrite["atomic_action_instruction"] = bad_action
                with self.assertRaises(atomic.AtomicMotionInstructionError):
                    atomic._validate_rewrite(
                        rewrite,
                        iid=IID_GOOD,
                        subject_ids=SUBJECT_IDS,
                        source_subjects=_passed(IID_GOOD)["source_census"][
                            "dynamic_subjects"
                        ],
                        camera_class="locked_off",
                        plan_audit=_plan_audit(IID_GOOD),
                    )

    def test_spatial_frame_word_is_allowed_but_plan_roles_are_bound(self) -> None:
        rewrite = _rewrite()
        rewrite["atomic_action_instruction"] = (
            "Have the cook carry the sausage out of the frame with the tongs, "
            "causing the steam to trail behind."
        )
        validated = atomic._validate_rewrite(
            rewrite,
            iid=IID_GOOD,
            subject_ids=SUBJECT_IDS,
            source_subjects=_passed(IID_GOOD)["source_census"]["dynamic_subjects"],
            camera_class="locked_off",
            plan_audit=_plan_audit(IID_GOOD),
        )
        self.assertIn("out of the frame", validated["atomic_action_instruction"])

        mismatched = _rewrite()
        mismatched["subject_mappings"][1]["event_roles"] = ["patient"]
        rebound = atomic._validate_rewrite(
            mismatched,
            iid=IID_GOOD,
            subject_ids=SUBJECT_IDS,
            source_subjects=_passed(IID_GOOD)["source_census"][
                "dynamic_subjects"
            ],
            camera_class="locked_off",
            plan_audit=_plan_audit(IID_GOOD),
        )
        self.assertEqual(rebound["subject_mappings"][1]["event_roles"], ["tool"])

    def test_nonpublished_rewrite_metadata_is_bound_to_plan_and_source(self) -> None:
        rewrite = _rewrite()
        rewrite["atomic_event"]["event_name"] = "different_event_spelling"
        rewrite["atomic_event"]["event_summary"] = (
            "the cook reaches first and then transfers the sausage"
        )
        rewrite["atomic_event"]["participant_subject_ids"] = ["wrong"]
        for mapping in rewrite["subject_mappings"]:
            mapping["subject_id"] = "wrong_subject"
            mapping["natural_reference"] = "the participant visible at I0"
            mapping["event_roles"] = ["patient"]
            mapping["participation_summary"] = (
                "the participant reaches and then performs the focal role"
            )

        plan = _plan_audit(IID_GOOD)
        source_subjects = _passed(IID_GOOD)["source_census"]["dynamic_subjects"]
        validated = atomic._validate_rewrite(
            rewrite,
            iid=IID_GOOD,
            subject_ids=SUBJECT_IDS,
            source_subjects=source_subjects,
            camera_class="locked_off",
            plan_audit=plan,
        )

        self.assertEqual(
            validated["atomic_event"]["event_name"],
            plan["atomic_event"]["event_name"],
        )
        self.assertEqual(
            validated["atomic_event"]["event_summary"],
            plan["atomic_event"]["event_summary"],
        )
        self.assertEqual(
            validated["atomic_event"]["participant_subject_ids"], SUBJECT_IDS
        )
        for index, mapping in enumerate(validated["subject_mappings"]):
            self.assertEqual(mapping["subject_id"], SUBJECT_IDS[index])
            self.assertEqual(
                mapping["natural_reference"],
                source_subjects[index]["stable_reference"],
            )
            self.assertEqual(mapping["event_roles"], ROLES[index])
            self.assertNotIn(" then ", f" {mapping['participation_summary'].lower()} ")

    def test_plan_camera_only_disagreement_is_compiled_not_a_causal_veto(self) -> None:
        plan = _plan_audit(IID_GOOD)
        plan["global_checks"]["camera_compatible"] = False
        plan["overall_verdict"] = "fail"
        plan["reason_codes"] = ["camera_compatible"]
        camera_target = _passed(IID_GOOD)["target_plan"]["camera_target"]
        validated = atomic._validate_plan_audit(
            plan,
            iid=IID_GOOD,
            subject_ids=SUBJECT_IDS,
            camera_target=camera_target,
        )
        self.assertTrue(validated["global_checks"]["camera_compatible"])
        self.assertEqual(validated["overall_verdict"], "pass")
        self.assertEqual(validated["reason_codes"], [])

        causal_failure = _plan_audit(IID_GOOD)
        causal_failure["global_checks"]["single_causal_event"] = False
        causal_failure["overall_verdict"] = "fail"
        causal_failure["reason_codes"] = ["multiple_causal_events"]
        with self.assertRaises(atomic.AtomicTargetPlanRejected):
            atomic._validate_plan_audit(
                causal_failure,
                iid=IID_GOOD,
                subject_ids=SUBJECT_IDS,
                camera_target=camera_target,
            )

    def test_deterministic_imperative_owns_only_the_redundant_audit_bit(self) -> None:
        rewrite = atomic._validate_rewrite(
            _rewrite(),
            iid=IID_GOOD,
            subject_ids=SUBJECT_IDS,
            source_subjects=_passed(IID_GOOD)["source_census"][
                "dynamic_subjects"
            ],
            camera_class="locked_off",
            plan_audit=_plan_audit(IID_GOOD),
        )
        audit = _semantic_audit(IID_GOOD)
        audit["natural_atomic_imperative"] = False
        audit["overall_verdict"] = "fail"
        audit["reason_codes"] = ["not_natural_atomic_imperative"]
        audit["confidence"] = "medium"
        effective = atomic._validate_semantic_audit(
            audit,
            iid=IID_GOOD,
            subject_ids=SUBJECT_IDS,
            atomic_action_instruction=rewrite["atomic_action_instruction"],
            camera_instruction=rewrite["camera_instruction"],
            preservation_instruction=rewrite["preservation_instruction"],
            full_edit_instruction=rewrite["full_edit_instruction"],
        )
        self.assertTrue(effective["natural_atomic_imperative"])

        rejected_camera = _semantic_audit(IID_GOOD)
        rejected_camera["camera_audit"]["no_contradiction"] = False
        rejected_camera["overall_verdict"] = "fail"
        rejected_camera["reason_codes"] = ["camera_contradiction"]
        with self.assertRaisesRegex(
            atomic.AtomicMotionInstructionError, "rejected camera"
        ):
            atomic._validate_semantic_audit(
                rejected_camera,
                iid=IID_GOOD,
                subject_ids=SUBJECT_IDS,
                atomic_action_instruction=rewrite["atomic_action_instruction"],
                camera_instruction=rewrite["camera_instruction"],
                preservation_instruction=rewrite["preservation_instruction"],
                full_edit_instruction=rewrite["full_edit_instruction"],
            )

    def test_stitched_published_action_is_retried_without_relaxing_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            row = _input_row()
            manifest = tmp_path / "input.jsonl"
            _write_manifest(manifest, [row])
            bad_rewrite = _rewrite(IID_GOOD)
            bad_rewrite["atomic_action_instruction"] = (
                "Have the cook lift the sausage, then place it in the tray."
            )
            backend = FakeBackend(
                [
                    _plan_audit(IID_GOOD),
                    bad_rewrite,
                    _rewrite(IID_GOOD),
                    _semantic_audit(IID_GOOD),
                ]
            )
            output = tmp_path / "output"
            self.assertEqual(
                atomic.run_worker(
                    _worker_args(manifest, output, allow_errors=False),
                    backend_factory=lambda **_: backend,
                ),
                0,
            )
            self.assertEqual(backend.calls, 4)
            self.assertIn("one root action verb", backend.requests[2]["user"])
            result = json.loads(
                (output / "rows" / IID_GOOD / "result.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(result["status"], "ok")
            self.assertIn(
                "temporal stitching", result["rewrite_attempts"][0]["error"]
            )
            self.assertEqual(result["rewrite_attempts"][1]["status"], "ok")

    def test_coherent_event_publishes_separate_labels_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            row = _input_row()
            manifest = tmp_path / "input.jsonl"
            _write_manifest(manifest, [row])
            output = tmp_path / "output"
            backend = FakeBackend(
                [
                    _plan_audit(IID_GOOD),
                    _rewrite(IID_GOOD),
                    _semantic_audit(IID_GOOD),
                ]
            )
            args = _worker_args(manifest, output, allow_errors=False)

            self.assertEqual(
                atomic.run_worker(args, backend_factory=lambda **_: backend), 0
            )
            self.assertEqual(backend.calls, 3)
            action_path = (
                output
                / "instructions"
                / IID_GOOD
                / "atomic_action_instruction.txt"
            )
            full_path = (
                output
                / "full_instructions"
                / IID_GOOD
                / "full_edit_instruction.txt"
            )
            action = action_path.read_text(encoding="utf-8").strip()
            full = full_path.read_text(encoding="utf-8").strip()
            self.assertEqual(atomic._sentence_count(action), 1)
            self.assertEqual(atomic._sentence_count(full), 3)
            self.assertEqual(
                full,
                f"{action} Keep the camera fixed. "
                f"{atomic.CANONICAL_PRESERVATION_INSTRUCTION}",
            )
            self.assertNotIn("frame 0", full.casefold())
            self.assertNotIn(" then ", f" {full.casefold()} ")
            self.assertNotIn(" while ", f" {full.casefold()} ")

            result = json.loads(
                (output / "rows" / IID_GOOD / "result.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["atomic_action_instruction"], action)
            self.assertEqual(result["camera_instruction"], "Keep the camera fixed.")
            self.assertEqual(
                result["preservation_instruction"],
                atomic.CANONICAL_PRESERVATION_INSTRUCTION,
            )
            self.assertEqual(result["full_edit_instruction"], full)

            receipt_path = output / "terminal" / f"{IID_GOOD}.receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(
                receipt["atomic_action_instruction_sha256"],
                atomic._sha256_file(action_path),
            )
            self.assertEqual(
                receipt["full_edit_instruction_sha256"],
                atomic._sha256_file(full_path),
            )
            atomic._validate_receipt(receipt, row=row)
            with self.assertRaisesRegex(
                atomic.AtomicMotionInstructionError, "receipt keys differ"
            ):
                atomic._validate_receipt(
                    {**receipt, "unbound_extra": True}, row=row
                )

            dataset_path = tmp_path / "atomic_dataset.jsonl"
            summary_path = tmp_path / "summary.json"
            self.assertEqual(
                atomic.verify_outputs(
                    argparse.Namespace(
                        input=manifest,
                        output_root=output,
                        expected_rows=1,
                        min_ok=1,
                        manifest_output=dataset_path,
                        summary_output=summary_path,
                    )
                ),
                0,
            )
            dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
            self.assertEqual(
                dataset["primary_training_label_field"],
                "atomic_action_instruction",
            )
            self.assertEqual(dataset["wan_prompt_field"], "full_edit_instruction")
            self.assertEqual(dataset["atomic_action_instruction"], action)
            self.assertEqual(
                atomic._sentence_count(dataset["atomic_action_instruction"]), 1
            )
            self.assertEqual(
                dataset["camera_instruction"], "Keep the camera fixed."
            )
            self.assertEqual(dataset["full_edit_instruction"], full)
            self.assertIn(
                "frame 0",
                dataset["source_generation_provenance"]["frame_gridded_prompt"],
            )

            def forbidden_backend(**_: object) -> object:
                raise AssertionError("resume must not load Qwen")

            self.assertEqual(
                atomic.run_worker(args, backend_factory=forbidden_backend), 0
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
