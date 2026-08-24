from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import tempfile
import unittest

import saic_t2v_partial_group_pair_audit_v1 as audit


def _shard(group_id: str, offset: int) -> dict:
    rows = [
        {
            "group_id": group_id,
            "candidate_index": index,
            "candidate_id": f"candidate-{offset + index:02d}",
            "iid": f"iid-{(offset + index) // 3:02d}",
            "seed": 1000 + (offset + index) // 3,
            "branch": ("incomplete", "camera_only", "appearance_only")[index % 3],
            "attempt_receipt_path": f"/root/{group_id}/attempt-{index}.json",
            "attempt_receipt_sha256": f"{offset + index + 1:064x}",
            "attempt_receipt_digest": f"{offset + index + 101:064x}",
            "completion_receipt_path": f"/root/{group_id}/completion-{index}.json",
            "completion_receipt_sha256": f"{offset + index + 201:064x}",
            "mp4_path": f"/root/{group_id}/{index}.mp4",
            "mp4_sha256": f"{offset + index + 301:064x}",
        }
        for index in range(30)
    ]
    proofs = [
        {
            "group_id": group_id,
            "iid": f"iid-{offset // 3 + index:02d}",
            "seed": 1000 + offset // 3 + index,
            "branch_order": ["incomplete", "camera_only", "appearance_only"],
            "official_gaussian_identity_digest": f"{offset + index + 401:064x}",
        }
        for index in range(10)
    ]
    return {
        "group_id": group_id,
        "root": f"/root/{group_id}",
        "slurm_job_id": str(100 + offset),
        "source_revision": "a" * 40,
        "source_archive_sha256": "b" * 64,
        "root_spec_raw_sha256": "c" * 64,
        "partial_receipt_path": f"/root/{group_id}/partial.json",
        "partial_receipt_sha256": "d" * 64,
        "partial_receipt_digest": "e" * 64,
        "candidate_count": 30,
        "permanent_port_claim_count": 30,
        "candidate_rows": rows,
        "gaussian_proofs": proofs,
    }


class SAICPartialGroupPairAuditTests(unittest.TestCase):
    def test_exact60_pair_is_review_only_and_canonical(self) -> None:
        receipt = audit.assemble_pair_receipt(
            [_shard("sp4-a", 0), _shard("sp4-b", 30)]
        )
        self.assertEqual(receipt["candidate_count"], 60)
        self.assertEqual(receipt["seed_cell_count"], 20)
        self.assertEqual(receipt["group_order"], ["sp4-a", "sp4-b"])
        self.assertTrue(receipt["authority"]["detached_decoded_event_review_input"])
        self.assertFalse(receipt["authority"]["training"])
        self.assertFalse(receipt["training_target_authorized"])
        unsigned = dict(receipt)
        digest = unsigned.pop("receipt_digest")
        self.assertEqual(digest, audit.object_sha256(unsigned))

    def test_overlap_and_contract_mismatch_fail_closed(self) -> None:
        shard_a = _shard("sp4-a", 0)
        shard_b = _shard("sp4-b", 30)
        shard_b["candidate_rows"][0]["candidate_id"] = "candidate-00"
        with self.assertRaisesRegex(audit.SAICPartialPairAuditError, "exact60"):
            audit.assemble_pair_receipt([shard_a, shard_b])
        shard_b = _shard("sp4-b", 30)
        shard_b["source_revision"] = "f" * 40
        with self.assertRaisesRegex(audit.SAICPartialPairAuditError, "source_revision"):
            audit.assemble_pair_receipt([shard_a, shard_b])

    def test_swapped_or_same_root_shards_fail_closed(self) -> None:
        shard_a = _shard("sp4-a", 0)
        shard_b = _shard("sp4-b", 30)
        with self.assertRaisesRegex(audit.SAICPartialPairAuditError, "order"):
            audit.assemble_pair_receipt([shard_b, shard_a])
        shard_b["root"] = shard_a["root"]
        with self.assertRaisesRegex(audit.SAICPartialPairAuditError, "disjoint"):
            audit.assemble_pair_receipt([shard_a, shard_b])

    def test_create_only_receipt_is_mode_0444_and_refuses_reuse(self) -> None:
        receipt = audit.assemble_pair_receipt(
            [_shard("sp4-a", 0), _shard("sp4-b", 30)]
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(strict=True)
            output = root / "pair.json"
            original_umask = os.umask(0o077)
            try:
                audit.write_create_only(output, receipt)
            finally:
                os.umask(original_umask)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o444)
            self.assertEqual(
                output.read_bytes(), audit.canonical_bytes(receipt) + b"\n"
            )
            with self.assertRaisesRegex(audit.SAICPartialPairAuditError, "fresh"):
                audit.write_create_only(output, receipt)

    def test_deep_audit_calls_are_present_on_production_path(self) -> None:
        source = Path(audit.__file__).read_text(encoding="utf-8")
        self.assertIn("generation._load_attempt_receipt", source)
        self.assertIn("rendezvous._validate_completion", source)
        self.assertIn("permanent claim coverage differs", source)
        self.assertIn("same-cell Gaussian differs", source)
        self.assertIn("single_root_master_receipt_emulated", source)


if __name__ == "__main__":
    unittest.main()
