#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import unittest


ASSET = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "bernini_braid_exact81_arms_v1.json"
)


class BerniniBraidExact81ArmRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(ASSET.read_text(encoding="utf-8"))
        cls.arms = {row["arm_id"]: row for row in cls.payload["arms"]}

    def test_registry_is_closed_and_non_authoritative(self) -> None:
        self.assertEqual(
            list(self.arms), ["A", "B", "C", "D0", "D1", "E", "F", "H", "G"]
        )
        self.assertFalse(self.payload["scientific_authority"])
        self.assertFalse(self.payload["training_update_authority"])
        self.assertEqual(self.payload["runtime"]["frames"], 81)
        self.assertEqual(self.payload["runtime"]["scheduler_steps"], 40)
        self.assertEqual(self.payload["runtime"]["scheduler_advances_per_step"], 1)
        self.assertEqual(
            self.payload["runtime"]["apg_state"],
            "independent_branch_local_identical_initial_bytes",
        )

    def test_anchor_arm_uses_c0_and_action_arms_use_c0_and_ca(self) -> None:
        self.assertEqual(self.arms["A"]["prompt"], "ca")
        self.assertEqual(self.arms["B"]["prompt"], "c0")
        for arm_id in ("C", "D0", "D1", "E", "F", "H", "G"):
            self.assertEqual(self.arms[arm_id]["prompt"], "c0_and_ca")

    def test_declared_single_variable_contrasts_are_exact(self) -> None:
        fields = (
            "A_motion",
            "A_text",
            "I_source",
            "basis",
            "branch",
            "g",
            "plan",
            "prompt",
            "reset",
            "source_gate",
        )
        expected = {
            ("C", "D0"): "A_motion",
            ("D0", "D1"): "reset",
            ("D1", "E"): "basis",
            ("D1", "F"): "plan",
            ("D1", "H"): "I_source",
            ("D1", "G"): "source_gate",
        }
        for pair, changed in expected.items():
            with self.subTest(pair=pair):
                left, right = (self.arms[value] for value in pair)
                differences = [field for field in fields if left[field] != right[field]]
                self.assertEqual(differences, [changed])


if __name__ == "__main__":
    unittest.main()
