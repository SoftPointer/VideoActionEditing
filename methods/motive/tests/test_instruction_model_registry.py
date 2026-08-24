from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from motive.instruction_model_registry import (
    InstructionModelRegistryError,
    availability_report,
    load_registry,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
REGISTRY = (
    REPO_ROOT
    / "methods"
    / "motive"
    / "configs"
    / "instruction_video_editor_registry_v1.json"
)


class InstructionModelRegistryTests(unittest.TestCase):
    def test_registry_is_instruction_only_and_vace_is_excluded(self) -> None:
        registry = load_registry(REGISTRY)
        self.assertEqual(
            registry["scope"]["required_user_inputs"],
            ["source_video", "instruction"],
        )
        self.assertIn(
            "vace",
            {
                row["id"]
                for row in registry["explicitly_excluded"]
            },
        )
        self.assertNotIn(
            "vace",
            {
                row["id"]
                for row in registry["models"]
                if row["primary_eligible"]
            },
        )
        self.assertGreaterEqual(
            len(
                {
                    row["architecture_family"]
                    for row in registry["models"]
                    if row["primary_eligible"]
                }
            ),
            2,
        )

    def test_gate_blocks_every_model_even_if_artifacts_exist(self) -> None:
        registry = load_registry(REGISTRY)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "checkpoints" / "LucyEdit").mkdir(parents=True)
            report = availability_report(
                registry,
                workspace=root,
                representation_gate_passed=False,
            )
        self.assertTrue(
            all(
                row["scheduling_status"]
                == "blocked_by_representation_gate"
                for row in report["models"]
            )
        )
        lucy = next(
            row for row in report["models"]
            if row["id"] == "lucy_edit_1_1"
        )
        self.assertTrue(lucy["available"])
        self.assertFalse(report["mutations_performed"])
        self.assertFalse(report["downloads_performed"])

    def test_mask_interface_is_rejected(self) -> None:
        payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
        broken = copy.deepcopy(payload)
        broken["models"][0]["interface"] = (
            "source_video_plus_instruction_plus_mask"
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "broken.json"
            path.write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaises(InstructionModelRegistryError):
                load_registry(path)


if __name__ == "__main__":
    unittest.main()
