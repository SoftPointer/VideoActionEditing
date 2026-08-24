from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = METHOD_ROOT / "tools"
for entry in (str(METHOD_ROOT), str(TOOLS_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import build_stage_a_source_edge_confirmation_review_html_v1 as builder  # noqa: E402
import stage_a_source_edge_confirmation_contract_v1 as contract  # noqa: E402


class ConfirmationReviewHtmlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.manifest_path = self.root / "confirmation.json"
        self.manifest_path.write_text("{}\n", encoding="utf-8")
        self.manifest_sha = contract.file_sha256(self.manifest_path)
        self.run_root = self.root / "run"
        (self.run_root / "outputs").mkdir(parents=True)
        self.cell = dict(contract.admitted_cell(35, "late_middle"))
        self.plan = [dict(row) for row in contract.build_confirmation_plan(35, "late_middle")]
        persistent = self.root / "persistent-review.json"
        authorization = self.root / "authorization.json"
        persistent.write_text("{}\n", encoding="utf-8")
        authorization.write_text("{}\n", encoding="utf-8")
        formal_records = []
        for family in ("dog", "human"):
            path = self.root / f"a1-{family}.json"
            path.write_text("{}\n", encoding="utf-8")
            formal_records.append(
                {
                    "family": family,
                    "receipt_path": str(path),
                    "receipt_file_sha256": contract.file_sha256(path),
                }
            )
        self.manifest = {
            "manifest_digest": "a" * 64,
            "admitted_cell": self.cell,
            "review_manifest": {
                "path": str(persistent),
                "file_sha256": contract.file_sha256(persistent),
            },
            "confirmation_authorization": {
                "path": str(authorization),
                "file_sha256": contract.file_sha256(authorization),
            },
            "a1_formal_receipts": formal_records,
        }
        self.receipts = []
        for index, sentinel_id in enumerate(contract.SENTINEL_ORDER):
            self.receipts.append(self._shard(index, sentinel_id))
        self.manifest["sentinels"] = [
            receipt["sentinel"] for receipt in self.receipts
        ]

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _shard(self, index: int, sentinel_id: str) -> dict:
        shard = self.run_root / "outputs" / sentinel_id
        shard.mkdir()
        source_records = {}
        for role in ("correct", "wrong_owner"):
            path = shard / f"source-{role}.mp4"
            path.write_bytes(f"{sentinel_id}-{role}".encode("utf-8"))
            source_records[role] = {
                "relative_mp4": path.name,
                "mp4_sha256": contract.file_sha256(path),
                "frame_count": 81,
                "fps": 25,
            }
        source_records["wrong_owner"].update(
            {
                "equal_latent_geometry": True,
                "pure_identity_control": False,
                "action_scene_entity_confound_acknowledged": True,
            }
        )
        records = []
        for row in self.plan:
            path = shard / f"{row['key']}.mp4"
            path.write_bytes(f"{sentinel_id}-{row['key']}".encode("utf-8"))
            records.append(
                {
                    **row,
                    "sentinel_id": sentinel_id,
                    "seed": 7000 + index,
                    "instruction": f"FULL {sentinel_id} {row['text_branch']} instruction",
                    "relative_mp4": path.name,
                    "mp4_sha256": contract.file_sha256(path),
                }
            )
        receipt = {
            "sentinel": {
                "sentinel_id": sentinel_id,
                "diversity_role": f"role-{index}",
                "source_entity_type": f"entity-{index}",
                "iid": f"iid-{index}",
                "action_family": f"action-{index}",
                "source_caption": f"full source caption {index}",
                "source_video_sha256": source_records["correct"]["mp4_sha256"],
                "wrong_owner_sentinel_id": contract.SENTINEL_ORDER[(index + 2) % 4],
                "wrong_owner_iid": f"wrong-{index}",
                "wrong_owner_source_video_sha256": source_records["wrong_owner"]["mp4_sha256"],
                "latent_shape": [1, 16, 21, 60, 62],
                "seed": 7000 + index,
            },
            "source_snapshots": source_records,
            "records": records,
        }
        receipt_path = shard / "receipt.json"
        receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
        return receipt

    def _build(self, output: Path) -> Path:
        calls = []
        for sentinel_id, receipt in zip(contract.SENTINEL_ORDER, self.receipts):
            path = self.run_root / "outputs" / sentinel_id / "receipt.json"
            calls.append((receipt, path, contract.file_sha256(path)))
        with mock.patch.object(
            builder.contract, "load_manifest", return_value=self.manifest
        ), mock.patch.object(
            builder.contract, "load_receipt", side_effect=calls
        ):
            return builder.build(
                manifest_path_value=self.manifest_path,
                expected_manifest_sha256=self.manifest_sha,
                run_root_value=self.run_root,
                output_dir_value=output,
            )

    def test_self_contained_exact56_review(self) -> None:
        output = self.root / "review"
        index = self._build(output)
        self.assertEqual(index, output / "index.html")
        page = index.read_text(encoding="utf-8")
        self.assertEqual(page.count("<video controls"), 64)
        first_screen = page.split("</section>", 1)[0]
        for text in (
            "4-source-disjoint confirmation",
            "s35 × late_middle",
            "sigma 0.41860657930374146",
            "blocks 16–22",
            "Native",
            "source-on",
            "source-off",
            "wrong-owner",
            "不是 Stage-B admission",
        ):
            self.assertIn(text, first_screen)
        for sentinel_id in contract.SENTINEL_ORDER:
            self.assertIn(sentinel_id, page)
            self.assertIn(f"FULL {sentinel_id} forward instruction", page)
        lowered = page.lower()
        for forbidden in ("scalar", "reward", "ranking"):
            self.assertNotIn(forbidden, lowered)
        self.assertNotIn('src="/', page)
        output_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(output_manifest["outputs_per_sentinel"], 14)
        self.assertEqual(sum(len(row["outputs"]) for row in output_manifest["cells"]), 56)
        self.assertFalse(output_manifest["stage_b_admission"])
        self.assertEqual(len(output_manifest["evidence_files"]), 5)

    def test_copy_time_byte_drift_fails_without_partial_publish(self) -> None:
        sentinel_id = contract.SENTINEL_ORDER[1]
        target = self.run_root / "outputs" / sentinel_id / f"{self.plan[3]['key']}.mp4"
        target.write_bytes(b"drift")
        output = self.root / "failed"
        with self.assertRaisesRegex(builder.ConfirmationReviewError, "source bytes differ"):
            self._build(output)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
