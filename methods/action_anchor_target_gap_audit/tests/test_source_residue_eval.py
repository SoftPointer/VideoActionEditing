from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import tempfile
import unittest

from methods.action_anchor_target_gap_audit import corrected_eval
from methods.action_anchor_target_gap_audit import source_residue_eval


ROOT = Path(__file__).resolve().parents[1]


def neutral_trace() -> dict:
    return {
        "schema_version": corrected_eval.TRACE_SCHEMA,
        "visual_quality": "yes",
        "dense_temporal_observations": [
            {
                "index": index,
                "phase": "early" if index < 4 else "middle" if index < 8 else "late",
                "actor_orientation": "front",
                "body_pose": "standing upright",
                "hands_and_objects": "hands visible",
                "continuity_from_previous": "start" if index == 0 else "continuous",
                "observation": "hands move while torso remains upright",
            }
            for index in range(12)
        ],
        "ambiguous_or_unseen": ["contact sound is unavailable"],
        "neutral_summary": "Hands move together while the torso stays upright.",
    }


class SourceResidueEvaluationTest(unittest.TestCase):
    def test_contracts_cover_frozen_mev_pair_set(self) -> None:
        target = corrected_eval.load_contracts(ROOT / "manual_action_contracts_v2.json")
        source = source_residue_eval.load_source_contracts(ROOT / "manual_source_residue_contracts_v3.json")
        self.assertEqual(set(source), set(target))
        self.assertEqual(len(source), 16)
        ids = [row["source_residue"]["id"] for row in source.values()]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("source_bow_replayed_or_retained", ids)
        self.assertIn("source_tablet_only_use_persists", ids)

    def test_prompt_uses_metadata_behavior_but_not_source_visual_context(self) -> None:
        prefix = "d908552fb691"
        target = corrected_eval.load_contracts(ROOT / "manual_action_contracts_v2.json")[prefix]
        source = source_residue_eval.load_source_contracts(ROOT / "manual_source_residue_contracts_v3.json")[prefix]
        prompt = source_residue_eval.make_source_aware_prompt(target, source, neutral_trace(), False)
        self.assertIn(source["source_action_caption"], prompt)
        self.assertIn(source["source_residue"]["description"], prompt)
        self.assertIn(
            "Never count the same person, object, scene, clothing, camera, spatial layout",
            " ".join(prompt.split()),
        )
        self.assertIn("initial checkpoint alone", prompt)
        self.assertIn("A clear matching checkpoint forces result=yes", " ".join(prompt.split()))
        self.assertIn("You do not see any video in this stage", source_residue_eval.SOURCE_AWARE_SYSTEM_PROMPT)
        self.assertIn("does NOT mean the behavior must remain at the end", source_residue_eval.SOURCE_AWARE_SYSTEM_PROMPT)
        self.assertIn(
            "Predicate ids are opaque",
            " ".join(source_residue_eval.SOURCE_AWARE_SYSTEM_PROMPT.split()),
        )
        self.assertNotIn("You see exactly one anonymized video", source_residue_eval.SOURCE_AWARE_SYSTEM_PROMPT)
        self.assertNotIn(target["target_action"], prompt)
        self.assertNotIn("Candidate A", prompt)
        self.assertNotIn("Candidate B", prompt)

    def test_source_only_schema_forces_explicit_temporal_rule(self) -> None:
        prefix = "d908552fb691"
        source = source_residue_eval.load_source_contracts(
            ROOT / "manual_source_residue_contracts_v3.json"
        )[prefix]
        observation = {
            "schema_version": source_residue_eval.SOURCE_ONLY_OBSERVATION_SCHEMA,
            "source_action_residue": {
                "id": source["source_residue"]["id"],
                "result": "yes",
                "evidence": [{"phase": "early", "observation": "the torso bends forward"}],
            },
            "temporal_rule_application": (
                "This is any-occurrence logic; the early bow permanently triggers the violation."
            ),
            "summary": "A source bow occurs before the clap.",
        }
        parsed = source_residue_eval.validate_source_only_observation(observation, source)
        self.assertEqual(parsed["source_action_residue"]["result"], "yes")

    def test_v3_jsonl_reader_does_not_apply_v2_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v3.jsonl"
            row = {"schema_version": source_residue_eval.RECORD_SCHEMA, "value": 1}
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            self.assertEqual(source_residue_eval._read_jsonl_any(path), [row])

    def test_source_residue_is_noncompensatory(self) -> None:
        prefix = "d908552fb691"
        target = corrected_eval.load_contracts(ROOT / "manual_action_contracts_v2.json")[prefix]
        source = source_residue_eval.load_source_contracts(ROOT / "manual_source_residue_contracts_v3.json")[prefix]
        observation = {
            "schema_version": source_residue_eval.OBSERVATION_SCHEMA,
            "action_observable": "yes",
            "requested_action_complete": "yes",
            "required_predicates": [
                {"id": row["id"], "result": "yes", "evidence": [{"phase": "middle", "observation": "visible target transition"}]}
                for row in target["required_predicates"]
            ],
            "forbidden_behaviors": [
                {"id": row["id"], "result": "no", "evidence": [{"phase": "late", "observation": "target substitute absent"}]}
                for row in target["forbidden_behaviors"]
            ],
            "source_action_residue": {
                "id": source["source_residue"]["id"],
                "result": "yes",
                "evidence": [{"phase": "early", "observation": "source bow is visible"}],
            },
            "summary": "Target occurs but source bow remains.",
        }
        parsed = source_residue_eval.validate_observation(observation, target, source)
        self.assertEqual(source_residue_eval.observation_gate(parsed), 0)
        observation["source_action_residue"]["result"] = "uncertain"
        self.assertEqual(source_residue_eval.observation_gate(observation), 2)
        observation["source_action_residue"]["result"] = "no"
        self.assertEqual(source_residue_eval.observation_gate(observation), 4)

    def test_every_contract_names_a_unique_source_behavior(self) -> None:
        contracts = source_residue_eval.load_source_contracts(ROOT / "manual_source_residue_contracts_v3.json")
        descriptions = [row["source_residue"]["description"].lower() for row in contracts.values()]
        captions = [row["source_action_caption"] for row in contracts.values()]
        self.assertEqual(len(captions), len(set(captions)))
        self.assertTrue(all(len(description.split()) >= 12 for description in descriptions))
        self.assertGreaterEqual(Counter("not residue" in value for value in descriptions)[True], 10)


if __name__ == "__main__":
    unittest.main()
