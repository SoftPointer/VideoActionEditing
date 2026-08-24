from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools/materialize_saic_t2v_full60_terminal_evidence_v1.py"
)
SPEC = importlib.util.spec_from_file_location("saic_full60_terminal", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def master_rows(root: Path) -> list[dict]:
    branches = ("appearance_only", "camera_only", "incomplete")
    return [
        {
            "candidate_id": f"candidate-{index}",
            "branch": branches[index % 3],
            "receipt_path": str(root / f"candidate-{index}" / "receipt.json"),
            "receipt_sha256": f"{index + 1:064x}",
            "receipt_digest": f"{index + 101:064x}",
            "mp4_path": str(root / f"candidate-{index}" / "t2v.mp4"),
            "mp4_sha256": f"{index + 201:064x}",
            "event_audit_status": "pending_detached_full81_review",
        }
        for index in range(60)
    ]


class SaicFull60TerminalEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp()).resolve()
        self.master_rows = master_rows(self.root)
        self.master = {
            "schema_version": MODULE.MASTER_SCHEMA,
            "topology": "two_concurrent_world4_sp4_groups_on_one_8gpu_node",
            "attempt_count": 60,
            "seed_cell_count": 20,
            "six_branch_spec_merge_cell_count": 20,
            "attempts": self.master_rows,
            "same_seed_official_gaussian_proofs": [{} for _ in range(20)],
            "detached_full81_event_review_complete": False,
            "event_verified": False,
            "identity_preservation_verified": False,
            "seed_selection_authorized": False,
            "training_target_authorized": False,
            "optimizer_or_parameter_update_authorized": False,
        }

    def deep(self, group: str, offset: int) -> dict:
        rows = [
            {
                "candidate_index": local,
                "candidate_id": source["candidate_id"],
                "branch": source["branch"],
                "attempt_receipt_sha256": source["receipt_sha256"],
                "attempt_receipt_digest": source["receipt_digest"],
                "mp4_sha256": source["mp4_sha256"],
            }
            for local, source in enumerate(self.master_rows[offset:offset + 30])
        ]
        return {
            "schema_version": MODULE.DEEP_SCHEMA,
            "root": str(self.root),
            "group_id": group,
            "slurm_job_id": "123",
            "planned_candidate_count": 30,
            "completed_prefix_count": 30,
            "completed_candidate_indices": list(range(30)),
            "deep_generation_receipt_validation": True,
            "deep_rendezvous_completion_validation": True,
            "same_cell_gaussian_prefix_validation": True,
            "authority": {"training": False, "optimizer": False},
            "rows": rows,
        }

    def test_master_and_two_deep_audits_bind_exact60(self) -> None:
        rows = MODULE.validate_master(self.master, root=self.root)
        a = MODULE.validate_deep_audit(
            self.deep("sp4-a", 0), root=self.root, group_id="sp4-a", job_id="123"
        )
        b = MODULE.validate_deep_audit(
            self.deep("sp4-b", 30), root=self.root, group_id="sp4-b", job_id="123"
        )
        MODULE.bind_master_to_deep(rows, a + b)

    def test_deep_substitution_is_rejected(self) -> None:
        a = self.deep("sp4-a", 0)
        b = self.deep("sp4-b", 30)
        b["rows"][1]["attempt_receipt_sha256"] = "f" * 64
        with self.assertRaises(SystemExit):
            MODULE.bind_master_to_deep(self.master_rows, a["rows"] + b["rows"])

    def test_cross_group_duplicate_mp4_is_rejected(self) -> None:
        a = self.deep("sp4-a", 0)
        b = self.deep("sp4-b", 30)
        b["rows"][0]["mp4_sha256"] = a["rows"][0]["mp4_sha256"]
        with self.assertRaises(SystemExit):
            MODULE.bind_master_to_deep(self.master_rows, a["rows"] + b["rows"])

    def test_incomplete_prefix_is_rejected(self) -> None:
        audit = self.deep("sp4-a", 0)
        audit["completed_prefix_count"] = 29
        with self.assertRaises(SystemExit):
            MODULE.validate_deep_audit(
                audit, root=self.root, group_id="sp4-a", job_id="123"
            )

    def test_receipt_digest_is_canonical(self) -> None:
        expected = hashlib.sha256(b'{"a":1,"b":2}').hexdigest()
        self.assertEqual(MODULE.object_sha256({"b": 2, "a": 1}), expected)


if __name__ == "__main__":
    unittest.main()
