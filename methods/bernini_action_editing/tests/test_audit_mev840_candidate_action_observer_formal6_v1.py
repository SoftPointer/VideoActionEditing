from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import audit_mev840_candidate_action_observer_formal6_v1 as audit  # noqa: E402


MANIFEST = ROOT / "assets" / "mev840_candidate_action_observer_formal6_v1.json"
RUNNER = ROOT / "run_mev840_candidate_action_observer_batch_v1.py"
REVIEW = ROOT.parents[1] / "md" / "action_editing" / "20260822_mev840_native_target_action_matrix_review"


class FormalSixObserverAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.receipts = {
            2027: json.loads(
                (REVIEW / "formal_runs" / "seed2027" / "receipt.json").read_text(
                    encoding="utf-8"
                )
            ),
            2028: json.loads(
                (REVIEW / "formal_runs" / "seed2028" / "receipt.json").read_text(
                    encoding="utf-8"
                )
            ),
        }

    def run_audit(self, value: dict) -> dict:
        def fake_regular(path_value, _sha_value, label):
            if label == "batch runner":
                return RUNNER.resolve()
            return Path(str(path_value))

        def fake_read(path: Path):
            if path == MANIFEST.resolve():
                return value
            for row in value["formal6_interface"]["formal_seed_receipts"]:
                if str(path) == row["path"]:
                    return self.receipts[row["seed"]]
            raise AssertionError(path)

        with mock.patch.object(audit, "regular_exact", side_effect=fake_regular), mock.patch.object(
            audit, "read_json", side_effect=fake_read
        ):
            return audit.audit(MANIFEST, RUNNER, audit.file_sha256(RUNNER))

    def test_sealed_formal6_closes_receipts_candidates_and_external_gates(self):
        result = self.run_audit(copy.deepcopy(self.manifest))
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["candidate_ids"], audit.IDS)
        self.assertIsNone(result["appearance_quality_gate_passed"])
        self.assertIsNone(result["single_bottle_gate_passed"])
        self.assertFalse(result["selection_authorized"])

    def test_receipt_to_candidate_sha_mutation_is_rejected(self):
        value = copy.deepcopy(self.manifest)
        value["candidates"][2]["sha256"] = "0" * 64
        with self.assertRaises(audit.FormalSixAuditError):
            self.run_audit(value)

    def test_single_bottle_or_selection_premature_resolution_is_rejected(self):
        for key, mutation in (
            ("single_bottle_gate_passed", True),
            ("selection_authorized", True),
        ):
            value = copy.deepcopy(self.manifest)
            value["formal6_interface"]["external_gate_contract"][key] = mutation
            with self.subTest(key=key), self.assertRaises(audit.FormalSixAuditError):
                self.run_audit(value)


if __name__ == "__main__":
    unittest.main()
