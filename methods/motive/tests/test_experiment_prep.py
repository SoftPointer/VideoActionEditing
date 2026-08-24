from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from motive.experiment_prep import (
    EXPERIMENT_PSEUDO_LABEL_SCHEMA,
    LEGACY_QWEN_POLICY,
    _canonical_digest,
    prepare_legacy_qwen_pilot,
)
from motive.qwen_filter import _object_digest
from motive.train_action_repr import _signature_base


def _row(index: int, split: str, *, verdict: str = "valid_action") -> dict:
    iid = f"case-{index:03d}"
    row = {
        "schema_version": "motive-action-cascade-v1",
        "iid": iid,
        "prompt": f"Make the actor perform action {index}.",
        "src_video": f"videos/{iid}/source.mp4",
        "tgt_video": f"videos/{iid}/edited.mp4",
        "source_caption": "an actor",
        "edited_caption": "an actor moving",
        "split": split,
        "group_id": f"group-{index}",
        "final_triage": {
            "decision": "review",
            "action_signature": "coarse-rule-family",
        },
    }
    digest_fields = {
        key: row[key]
        for key in (
            "iid",
            "prompt",
            "src_video",
            "tgt_video",
            "source_caption",
            "edited_caption",
        )
    }
    row["input_digest"] = _canonical_digest(digest_fields)
    observation = {
        "schema_version": "qwen-motion-observation-v2",
        "source_action": "standing",
        "target_action": f"jumping {index}",
        "source_actor_motion": "none",
        "target_actor_motion": "clear",
        "camera_dominance": "low",
        "background_dominance": "low",
        "artifact_level": "low",
        "preservation_quality": "acceptable",
        "temporal_evidence": ["target actor moves from T0 to T5"],
        "uncertainty_codes": [],
    }
    result = {
        "schema_version": "qwen-motion-judge-v4",
        "verdict": verdict,
        "edit_effect": "started" if verdict == "valid_action" else "none",
        "action_signature": (
            f"jumping {index}" if verdict == "valid_action" else "unknown"
        ),
        "reason_codes": ["target_actor_motion_clear"],
        "uncertainty_codes": [],
        "confidence": "high",
    }
    if verdict == "static":
        observation["target_actor_motion"] = "none"
        observation["target_action"] = "standing"
    row["qwen_evidence"] = {
        "visual": {
            "iid": iid,
            "input_digest": row["input_digest"],
            "status": "ok",
            "observation_validated_from": "original",
            "result_validated_from": "original",
            "observation_repairs": [],
            "alignment_repairs": [],
            "observation": observation,
            "observation_digest": _object_digest(observation),
            "result": result,
        }
    }
    return row


class ExperimentPrepTests(unittest.TestCase):
    def test_legacy_qwen_pilot_is_strict_and_provenance_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "fused.jsonl"
            rows = [
                _row(0, "train"),
                _row(1, "train"),
                _row(2, "train"),
                _row(3, "validation"),
                _row(4, "test"),
                _row(5, "train", verdict="static"),
            ]
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            output_dir = root / "pilot"
            args = argparse.Namespace(
                input=input_path,
                output_dir=output_dir,
                max_lucy_train=2,
                seed=260108828,
                overwrite=False,
            )
            self.assertEqual(prepare_legacy_qwen_pilot(args), 0)
            representation = [
                json.loads(line)
                for line in (
                    output_dir / "representation_manifest.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            lucy_train = (
                output_dir / "lucy_train_manifest.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            lucy_eval = (
                output_dir / "lucy_eval_manifest.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(representation), 5)
            self.assertEqual(len(lucy_train), 2)
            self.assertEqual(len(lucy_eval), 2)
            self.assertEqual(summary["selected_rows"], 5)
            self.assertEqual(
                summary["selection_reason_counts"]["verdict:static"],
                1,
            )
            self.assertTrue(summary["legacy_non_content_split"])
            self.assertEqual(summary["legacy_missing_result_digest_rows"], 5)
            pseudo = representation[0]["experiment_pseudo_label"]
            self.assertEqual(
                pseudo["schema_version"],
                EXPERIMENT_PSEUDO_LABEL_SCHEMA,
            )
            self.assertEqual(pseudo["policy"], LEGACY_QWEN_POLICY)
            self.assertFalse(pseudo["human_approved"])
            self.assertFalse(pseudo["production_eligible"])
            with self.assertRaises(FileExistsError):
                prepare_legacy_qwen_pilot(args)

    def test_qwen_observation_digest_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "fused.jsonl"
            row = _row(0, "train")
            row["qwen_evidence"]["visual"]["observation_digest"] = "0" * 64
            input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "observation digest"):
                prepare_legacy_qwen_pilot(
                    argparse.Namespace(
                        input=input_path,
                        output_dir=root / "pilot",
                        max_lucy_train=0,
                        seed=1,
                        overwrite=False,
                    )
                )

    def test_human_signature_remains_authoritative_over_pseudo(self) -> None:
        row = _row(0, "train")
        row["experiment_pseudo_label"] = {
            "schema_version": EXPERIMENT_PSEUDO_LABEL_SCHEMA,
            "policy": LEGACY_QWEN_POLICY,
            "action_signature": "pseudo jump",
            "source_manifest_sha256": "1" * 64,
            "observation_digest": "2" * 64,
            "result_object_digest": "3" * 64,
            "legacy_result_digest_missing": True,
            "human_approved": False,
            "production_eligible": False,
        }
        self.assertEqual(_signature_base(row, row["iid"]), "pseudo_jump")
        row["human_review"] = {"action_signature": "turn left"}
        self.assertEqual(_signature_base(row, row["iid"]), "turn_left")


if __name__ == "__main__":
    unittest.main()
