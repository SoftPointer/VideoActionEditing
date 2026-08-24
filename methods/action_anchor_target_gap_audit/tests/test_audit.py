from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from methods.action_anchor_target_gap_audit import audit


def observation(anchor_slot: str = "A", anchor_score: int = 4, base_score: int = 2):
    def candidate(score: int):
        return {
            "action_semantics": score,
            "temporal_order": score,
            "action_completion": score,
            "reference_motion_match": score,
            "action_observable": "yes",
            "artifact_blocks_action": "no",
            "evidence": ["F0-F9 show the visible action."],
        }

    values = {anchor_slot: candidate(anchor_score)}
    values["B" if anchor_slot == "A" else "A"] = candidate(base_score)
    return {
        "schema_version": audit.QWEN_OBSERVATION_SCHEMA,
        "reference_action_valid": "yes",
        "source_target_initial_comparable": "yes",
        "candidate_A": values["A"],
        "candidate_B": values["B"],
        "closer_to_reference_action": anchor_slot,
        "confidence": "high",
        "comparison_evidence": ["The anchor completes the action."],
    }


class AuditTests(unittest.TestCase):
    def test_qwen_schema_rejects_out_of_range_score(self):
        value = observation()
        value["candidate_A"]["action_completion"] = 5
        with self.assertRaises(ValueError):
            audit.validate_qwen_observation(value)

    def test_slot_maps_are_opposites(self):
        first, second = audit._slot_maps("a" * 64)
        self.assertEqual(first["A"], second["B"])
        self.assertEqual(first["B"], second["A"])
        self.assertEqual(set(first.values()), {"anchor", "frozen_base"})

    def test_all_task_writes_reject_protected_mev_tree(self):
        with tempfile.TemporaryDirectory() as raw:
            protected = Path(raw) / "MEV"
            protected.mkdir()
            with mock.patch.object(audit, "MEV_PROTECTED_ROOT", protected):
                with self.assertRaisesRegex(ValueError, "protected MEV"):
                    audit.write_json(protected / "forbidden.json", {})
                with self.assertRaisesRegex(ValueError, "protected MEV"):
                    audit.write_jsonl(protected / "forbidden.jsonl", [])
            self.assertEqual(list(protected.iterdir()), [])

    def test_qwen_summary_requires_slot_swap_stability(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            records = root / "records"
            records.mkdir()
            row = {
                "schema_version": audit.QWEN_RECORD_SCHEMA,
                "pair_id": "1" * 64,
                "pair_prefix": "1" * 12,
                "instruction": "Do the action.",
                "passes": [
                    {
                        "pass_index": 0,
                        "slot_map": {"A": "anchor", "B": "frozen_base"},
                        "mosaic_sha256": "2" * 64,
                        "raw_output": "{}",
                        "parse_error": None,
                        "observation": observation("A"),
                    },
                    {
                        "pass_index": 1,
                        "slot_map": {"A": "frozen_base", "B": "anchor"},
                        "mosaic_sha256": "3" * 64,
                        "raw_output": "{}",
                        "parse_error": None,
                        "observation": observation("B"),
                    },
                ],
            }
            audit.write_jsonl(records / "qwen-shard-0.jsonl", [row])
            output = root / "summary.json"
            args = argparse.Namespace(records_dir=str(records), output=str(output))
            self.assertEqual(audit.qwen_summarize(args), 0)
            summary = audit.load_json(output)
            self.assertEqual(summary["winner_counts"], {"anchor": 1})
            self.assertEqual(summary["pairs"][0]["gate_pass_winners"], ["anchor", "anchor"])

    def test_build_manifest_uses_mev_annotations_and_never_copies_media(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            mev_root = root / "MEV"
            mev_root.mkdir()
            prefixes = [f"{index:012x}" for index in range(16)]
            selection = root / "selection.json"
            audit.write_json(selection, {"pair_id_prefixes": prefixes})
            metadata = root / "metadata.jsonl"
            rows = []
            for ordinal, prefix in enumerate(prefixes):
                source = mev_root / f"source-{ordinal}.mp4"
                target = mev_root / f"target-{ordinal}.mp4"
                source.write_bytes(f"source-{ordinal}".encode())
                target.write_bytes(f"target-{ordinal}".encode())
                rows.append({
                    "pair_id": prefix + "a" * 52,
                    "uuid": f"uuid-{ordinal}",
                    "split": "test",
                    "instruction": f"Edit the action so that action {ordinal} occurs.",
                    "instruction_source": "mev.json target event caption",
                    "source_action_caption": "The subject waits.",
                    "target_action_caption": f"The subject performs action {ordinal}.",
                    "source_video_path": str(source),
                    "target_video_path": str(target),
                    "global_prompt": {"short_caption": "A subject in a fixed scene."},
                    "source_event_annotation": {"event_id": 1},
                    "target_event_annotation": {
                        "event_id": 2, "has_appearance": False,
                        "has_camera_motion": False, "has_disappearance": False,
                        "has_environmental_change": False, "has_lighting_change": False,
                        "is_multi_person": False,
                    },
                    "automatic_visual_audit": {
                        "verdict": "accept", "dependency_level": "none",
                        "initial_state_compatibility": "aligned",
                        "target_action_quality": "clear_action",
                        "preservation": "same_identity_scene_camera",
                    },
                })
            audit.write_jsonl(metadata, rows)
            output = root / "manifest.json"
            args = argparse.Namespace(
                metadata=str(metadata), selection=str(selection),
                experiment_root=str(root / "experiment"), seed_base=100,
                output=str(output),
            )
            with mock.patch.object(audit, "MEV_PROTECTED_ROOT", mev_root):
                self.assertEqual(audit.build_manifest(args), 0)
            manifest = json.loads(output.read_text())
            self.assertEqual(len(manifest["samples"]), 16)
            self.assertFalse(manifest["protected_source_contract"]["videos_copied"])
            self.assertEqual(manifest["samples"][0]["generation"]["normalized_source"]["frame_count"], 81)
            self.assertIn("action 0", manifest["samples"][0]["generation_caption"])
            self.assertEqual(manifest["samples"][0]["instruction_source"], "mev.json target event caption")


if __name__ == "__main__":
    unittest.main()
