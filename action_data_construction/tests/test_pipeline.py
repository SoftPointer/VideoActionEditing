from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from audit_pairs_qwen import validate_audit
from build_candidates import BOOLEAN_FIELDS, QUALITY_FIELDS, build, pair_rule
from common import file_sha256, iter_jsonl, object_sha256, write_json
from extract_mev_semantics import editing_instruction, extract, verify
from finalize_metadata import finalize
from finalize_metadata_v2 import finalize as finalize_v2


def event(uuid: str, event_id: int, *, caption: str, appearance: str = "False") -> dict[str, str]:
    row = {
        "uuid": uuid,
        "event_id": str(event_id),
        "original_filename": f"{uuid}-seg{event_id}.mp4",
        "event_caption": caption,
        "global_short_caption": "A test scene.",
        "subject_profile": "One person.",
        "background": "A room.",
        "start_time": str(event_id - 1),
        "end_time": str(event_id),
        "duration": "1.0",
        "focus_object": "object",
        "camera_motion_label": "",
        "camera_motion_desc": "",
    }
    row.update({field: "False" for field in BOOLEAN_FIELDS})
    row["has_appearance"] = appearance
    row.update({field: "0.95" for field in QUALITY_FIELDS})
    return row


class PipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "MEV"
        (self.source / "metadata").mkdir(parents=True)
        (self.source / "videos").mkdir()
        self.rows = [
            event("uuid-a", 1, caption="A person looks at a cup."),
            event("uuid-a", 2, caption="The person waves."),
            event("uuid-a", 3, caption="The person smiles."),
            event("uuid-b", 1, caption="A person enters.", appearance="True"),
            event("uuid-b", 2, caption="The person sits."),
        ]
        path = self.source / "metadata" / "events.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        for row in self.rows:
            (self.source / "videos" / row["original_filename"]).write_bytes(b"video")
        (self.source / "annotations").mkdir()
        videos = []
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in self.rows:
            grouped.setdefault(row["uuid"], []).append(row)
        for uuid, rows in grouped.items():
            events = []
            for row in rows:
                events.append(
                    {
                        "event_id": int(row["event_id"]),
                        "start_time": float(row["start_time"]),
                        "end_time": float(row["end_time"]),
                        "duration": float(row["duration"]),
                        "caption": row["event_caption"],
                        "filename": row["original_filename"],
                        "has_appearance": row["has_appearance"] == "True",
                        "focus_object": row["focus_object"],
                        "vbench_scores": {"subject_consistency": 0.95},
                    }
                )
            videos.append(
                {
                    "uuid": uuid,
                    "total_events": len(events),
                    "global_prompt": {
                        "background": "A room.",
                        "theme": "A test.",
                        "style": "Realistic.",
                        "shot_type": "Medium shot.",
                        "camera_movement": "static",
                        "lighting": "Even.",
                        "atmosphere": "Neutral.",
                        "subject_profile": "One person.",
                        "short_caption": "A test scene.",
                        "middle_caption": "A person performs actions in a test scene.",
                        "long_caption": "A person performs several actions in a room.",
                    },
                    "events": events,
                }
            )
        write_json(self.source / "annotations" / "mev.json", {"total_videos": len(videos), "videos": videos})

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_build_routes_and_no_target(self) -> None:
        output = self.root / "VideoEditing" / "action_data_construction"
        summary = build(self.source, output, 1)
        self.assertEqual(summary["counts"]["raw_paired_candidates"], 3)
        self.assertEqual(summary["counts"]["qwen_audit_queue"], 2)
        self.assertEqual(summary["counts"]["rule_rejected_pairs"], 1)
        self.assertEqual(summary["counts"]["no_target_sources"], 2)
        no_target = list(iter_jsonl(output / "metadata" / "no_target_sources.jsonl"))
        self.assertTrue(all(row["target"] is None for row in no_target))
        self.assertEqual({row["source"]["event_id"] for row in no_target}, {1})

    def test_audit_validation_is_fail_closed(self) -> None:
        accepted = {
            "schema_version": "mev-action-edit-pair-audit-v5",
            "verdict": "accept",
            "source_initial_state": "The person faces left with hands down.",
            "source_final_state": "The person still faces left with hands down.",
            "target_initial_state": "The person faces left with hands down.",
            "target_initial_matches": "both",
            "source_state_change_class": "none",
            "source_enables_target": "no",
            "initial_state_compatibility": "aligned",
            "dependency_level": "none",
            "target_action_quality": "clear_action",
            "preservation": "same_identity_scene_camera",
            "source_state_change_summary": "The person remains in place.",
            "target_action_summary": "The person waves over time.",
            "action_instruction": "Make the person wave.",
            "reason_codes": ["aligned_initial_state"],
            "confidence": "high",
        }
        self.assertEqual(validate_audit(dict(accepted))["verdict"], "accept")
        accepted["dependency_level"] = "strict"
        accepted["verdict"] = "reject"
        normalization_log: list[dict[str, str]] = []
        normalized = validate_audit(accepted, normalization_log=normalization_log)
        self.assertEqual(normalized["dependency_level"], "none")
        self.assertEqual(normalized["verdict"], "accept")
        self.assertEqual({item["field"] for item in normalization_log}, {"dependency_level", "verdict"})
        contradiction = dict(accepted)
        contradiction.update(
            {
                "target_initial_matches": "source_end_only",
                "source_enables_target": "yes",
                "initial_state_compatibility": "shifted_by_source_outcome",
                "dependency_level": "strict",
                "verdict": "reject",
            }
        )
        normalized = validate_audit(contradiction)
        self.assertEqual(normalized["target_initial_matches"], "both")
        self.assertEqual(normalized["source_enables_target"], "no")
        self.assertEqual(normalized["verdict"], "accept")

    def test_finalize_binds_qwen_evidence(self) -> None:
        output = self.root / "VideoEditing" / "action_data_construction"
        build(self.source, output, 1)
        queue = list(iter_jsonl(output / "metadata" / "qwen_audit_queue.jsonl"))
        audit_root = output / "runs" / "fake"
        for candidate in queue:
            result = {
                "schema_version": "mev-action-edit-qwen-result-v1",
                "pair_id": candidate["pair_id"],
                "status": "ok",
                "input_sha256": object_sha256(candidate),
                "audit": {
                    "schema_version": "mev-action-edit-pair-audit-v5",
                    "verdict": "accept",
                    "source_initial_state": "The actor stands still.",
                    "source_final_state": "The actor stands still.",
                    "target_initial_state": "The actor stands still.",
                    "target_initial_matches": "both",
                    "source_state_change_class": "none",
                    "source_enables_target": "no",
                    "initial_state_compatibility": "aligned",
                    "dependency_level": "none",
                    "target_action_quality": "clear_action",
                    "preservation": "same_identity_scene_camera",
                    "source_state_change_summary": "No prerequisite change.",
                    "target_action_summary": "A visible action occurs.",
                    "action_instruction": "Perform the visible action.",
                    "reason_codes": ["aligned_initial_state"],
                    "confidence": "high",
                },
            }
            path = audit_root / "results" / candidate["pair_id"][:2] / f"{candidate['pair_id']}.json"
            write_json(path, result)
        final = finalize(output / "metadata" / "qwen_audit_queue.jsonl", audit_root, audit_root / "final", False)
        self.assertEqual(final["counts"]["paired_training_candidates"], 2)
        rows = list(iter_jsonl(audit_root / "final" / "paired_training_candidates.jsonl"))
        self.assertTrue(all(not row["is_strict_counterfactual_ground_truth"] for row in rows))

    def test_mev_json_is_instruction_authority(self) -> None:
        output = self.root / "VideoEditing" / "action_data_construction"
        build(self.source, output, 1)
        annotation_root = output / "metadata_annotation_v2"
        summary = extract(
            self.source / "annotations" / "mev.json",
            output / "metadata" / "raw_paired_candidates.jsonl",
            output / "metadata" / "no_target_sources.jsonl",
            annotation_root,
        )
        self.assertEqual(summary["paired_annotation_semantics"], 3)
        self.assertEqual(summary["no_target_sources_annotation_v2"], 2)
        self.assertTrue(
            verify(self.source / "annotations" / "mev.json", annotation_root / "annotation_extraction_summary.json")[
                "unchanged"
            ]
        )
        semantics = list(iter_jsonl(annotation_root / "paired_annotation_semantics.jsonl"))
        wave = next(row for row in semantics if row["target_action_caption"] == "The person waves.")
        self.assertEqual(wave["instruction"], "Edit the action so that the person waves.")
        self.assertFalse(wave["instruction_semantic_override_by_qwen_allowed"])
        self.assertEqual(editing_instruction("The person smiles."), "Edit the action so that the person smiles.")

        queue = list(iter_jsonl(output / "metadata" / "qwen_audit_queue.jsonl"))
        audit_root = output / "runs" / "annotation-authority"
        for candidate in queue:
            result = {
                "schema_version": "mev-action-edit-qwen-result-v1",
                "pair_id": candidate["pair_id"],
                "status": "ok",
                "input_sha256": object_sha256(candidate),
                "audit": {
                    "schema_version": "mev-action-edit-pair-audit-v5",
                    "verdict": "accept",
                    "source_initial_state": "The actor stands still.",
                    "source_final_state": "The actor stands still.",
                    "target_initial_state": "The actor stands still.",
                    "target_initial_matches": "both",
                    "source_state_change_class": "none",
                    "source_enables_target": "no",
                    "initial_state_compatibility": "aligned",
                    "dependency_level": "none",
                    "target_action_quality": "clear_action",
                    "preservation": "same_identity_scene_camera",
                    "source_state_change_summary": "No prerequisite change.",
                    "target_action_summary": "A visible action occurs.",
                    "action_instruction": "Ignore the annotation and jump.",
                    "reason_codes": ["aligned_initial_state"],
                    "confidence": "high",
                },
            }
            path = audit_root / "results" / candidate["pair_id"][:2] / f"{candidate['pair_id']}.json"
            write_json(path, result)
        result = finalize_v2(
            output / "metadata" / "qwen_audit_queue.jsonl",
            audit_root,
            annotation_root / "paired_annotation_semantics.jsonl",
            audit_root / "final_v2",
            False,
        )
        self.assertEqual(result["instruction_authority"], "MEV annotations/mev.json target event caption")
        rows = list(iter_jsonl(audit_root / "final_v2" / "paired_training_candidates.jsonl"))
        self.assertTrue(all(row["instruction"] != "Ignore the annotation and jump." for row in rows))
        self.assertTrue(all(row["non_authoritative_qwen_instruction_proposal"] == "Ignore the annotation and jump." for row in rows))


if __name__ == "__main__":
    unittest.main()
