from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    METHOD_ROOT / "tools" / "materialize_saic_t2v_anchor_semantics_matrix_v1.py"
)
SPEC = importlib.util.spec_from_file_location("saic_anchor_matrix", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SaicAnchorSemanticsMatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        root = json.loads(
            (METHOD_ROOT / "assets" / "saic_reversible_source_set_v1.json")
            .read_text(encoding="ascii")
        )
        self.sources = {row["iid"]: row for row in root["rows"]}
        self.records = []
        for source in self.sources.values():
            for branch in MODULE.BRANCHES:
                for ordinal, seed in enumerate(source["rollout_seeds"]):
                    passed = ordinal == 0 and (
                        branch == "forward"
                        or (branch == "noop" and source["actor_family"] == "dog")
                    )
                    self.records.append({
                        "candidate_id": (
                            f"saic-{source['iid']}-{branch}-s{seed}"
                        ),
                        "iid": source["iid"],
                        "branch": branch,
                        "seed": seed,
                        "analysis_split": source["analysis_split"],
                        "actor_family": source["actor_family"],
                        "action_family_id": source["action_family_id"],
                        "validated_observation": {},
                        "deterministic_branch_gate_passed": passed,
                        "deterministic_failure_codes": [] if passed else ["failed"],
                    })

    def test_matrix_preserves_source_branch_seed_coverage(self) -> None:
        matrix, aggregate = MODULE.build_matrix(self.sources, self.records)
        self.assertEqual(len(matrix), 8)
        self.assertEqual(aggregate["source_coverage"]["forward"], 8)
        self.assertEqual(aggregate["source_coverage"]["noop"], 4)
        self.assertEqual(aggregate["source_coverage"]["reverse"], 0)
        self.assertEqual(
            aggregate["event_evidence_source_coverage_by_split"]["fit"][
                "forward"
            ],
            0,
        )
        fit = next(item for item in matrix if item["analysis_split"] == "fit")
        confirmation = next(
            item for item in matrix if item["analysis_split"] == "confirmation"
        )
        self.assertEqual(fit["branches"]["forward"]["record_count"], 2)
        self.assertEqual(
            confirmation["branches"]["forward"]["record_count"], 3
        )

    def test_event_evidence_ignores_only_appearance_change(self) -> None:
        row = {
            "branch": "forward",
            "deterministic_branch_gate_passed": False,
            "validated_observation": {
                "start_state_match": "yes",
                "requested_branch_change_present": "yes",
                "requested_change_fidelity": "exact",
                "target_action_progress": "full",
                "terminal_state_reached": "yes",
                "temporal_order_coherent": "yes",
                "identity_geometry_stable": "yes",
                "protected_scene_stable": "yes",
                "camera_motion_level": "none",
                "appearance_change_level": "global",
            },
        }
        self.assertTrue(MODULE.is_event_evidence_candidate(row))
        row["validated_observation"]["camera_motion_level"] = "conspicuous"
        self.assertFalse(MODULE.is_event_evidence_candidate(row))

    def test_noop_event_evidence_keeps_strict_gate(self) -> None:
        self.assertFalse(MODULE.is_event_evidence_candidate({
            "branch": "noop", "deterministic_branch_gate_passed": False
        }))
        self.assertTrue(MODULE.is_event_evidence_candidate({
            "branch": "noop", "deterministic_branch_gate_passed": True
        }))

    def test_split_leak_is_rejected(self) -> None:
        self.records[0] = {**self.records[0], "analysis_split": "confirmation"}
        with self.assertRaises(SystemExit):
            MODULE.build_matrix(self.sources, self.records)


if __name__ == "__main__":
    unittest.main()
