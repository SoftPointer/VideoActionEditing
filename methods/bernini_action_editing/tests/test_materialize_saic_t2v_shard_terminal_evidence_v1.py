from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools/materialize_saic_t2v_shard_terminal_evidence_v1.py"
)
SPEC = importlib.util.spec_from_file_location("saic_shard_terminal", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def receipt(value: dict) -> dict:
    value = dict(value)
    value["receipt_digest"] = MODULE.canonical_digest(value)
    return value


class SaicShardTerminalEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.rows = [
            {
                "candidate_index": index,
                "candidate_id": f"candidate-{index}",
                "attempt_receipt_sha256": f"{index + 1:064x}",
                "completion_receipt_sha256": f"{index + 11:064x}",
                "mp4_sha256": f"{index + 21:064x}",
            }
            for index in range(3)
        ]
        self.audit = receipt({
            "root": str(self.root),
            "group_id": "sp4-a",
            "planned_candidate_count": 3,
            "completed_prefix_count": 3,
            "completed_candidate_indices": [0, 1, 2],
            "deep_generation_receipt_validation": True,
            "deep_rendezvous_completion_validation": True,
            "root_spec_raw_sha256": "a" * 64,
            "rows": self.rows,
            "authority": {"training": False, "optimizer": False},
        })

    def test_exact_deep_audit_and_partial_receipt_pass(self) -> None:
        rows = MODULE.validate_deep_audit(
            self.audit, root=self.root, group_id="sp4-a", expected_count=3
        )
        partial = receipt({
            "group_id": "sp4-a",
            "candidate_count": 3,
            "root_spec_raw_sha256": "a" * 64,
            "authority": {"training": False},
            "rows": [
                {key: row[key] for key in (
                    "candidate_index", "candidate_id", "attempt_receipt_sha256",
                    "completion_receipt_sha256",
                )}
                for row in self.rows
            ],
        })
        MODULE.validate_partial_receipt(
            partial, self.audit, rows, group_id="sp4-a", expected_count=3
        )

    def test_duplicate_mp4_is_rejected(self) -> None:
        self.audit["rows"][2]["mp4_sha256"] = self.audit["rows"][1]["mp4_sha256"]
        with self.assertRaises(SystemExit):
            MODULE.validate_deep_audit(
                self.audit, root=self.root, group_id="sp4-a", expected_count=3
            )

    def test_partial_row_substitution_is_rejected(self) -> None:
        rows = MODULE.validate_deep_audit(
            self.audit, root=self.root, group_id="sp4-a", expected_count=3
        )
        partial = receipt({
            "group_id": "sp4-a",
            "candidate_count": 3,
            "root_spec_raw_sha256": "a" * 64,
            "authority": {"training": False},
            "rows": [
                {key: row[key] for key in (
                    "candidate_index", "candidate_id", "attempt_receipt_sha256",
                    "completion_receipt_sha256",
                )}
                for row in self.rows
            ],
        })
        partial["rows"][1]["candidate_id"] = "substituted"
        with self.assertRaises(SystemExit):
            MODULE.validate_partial_receipt(
                partial, self.audit, rows, group_id="sp4-a", expected_count=3
            )

    def test_receipt_digest_is_canonical(self) -> None:
        value = receipt({"b": 2, "a": 1})
        expected = hashlib.sha256(b'{"a":1,"b":2}').hexdigest()
        self.assertEqual(value["receipt_digest"], expected)


if __name__ == "__main__":
    unittest.main()
