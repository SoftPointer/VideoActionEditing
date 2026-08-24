from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_self_generated_action_fullfield_v4 as trainer


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


class Full644ActionAnchorTests(unittest.TestCase):
    def test_legacy_optimizer_entrypoint_is_permanently_fail_closed(self):
        with self.assertRaisesRegex(
            trainer.FullFieldTrainingError,
            trainer.LEGACY_TRAINING_BLOCKED_STATUS,
        ):
            trainer.main([])

    def manifest(self, parquet: Path):
        parquet = parquet.resolve(strict=True)
        rows = []
        for index in range(644):
            iid = f"{index:016x}"
            rows.append(
                {
                    "iid": iid,
                    "instruction": f"perform action {index}",
                    "noop_instruction": "keep source unchanged",
                    "posterior_pair": {
                        "parquet_path": str(parquet),
                        "parquet_sha256": "1" * 64,
                        "source_role_index": 0,
                        "source_blob_sha256": "2" * 64,
                        "action_anchor_role_index": 1,
                        "action_anchor_blob_sha256": "3" * 64,
                    },
                }
            )
        value = {
            "schema_version": trainer.FULL644_MANIFEST_SCHEMA,
            "authorization_label": trainer.FULL644_AUTHORIZATION,
            "row_count": 644,
            "rows": rows,
            "optimizer_schedule": "exact644_unique_rows_once",
            "source_anchor_role": "identity_appearance_background_camera_and_non_target_preservation",
            "self_generated_action_anchor_role": "dense_action_trajectory_supervision",
            "paired_ground_truth_claimed": False,
            "qwen_or_other_verifier_controls_optimizer_admission": False,
            "production_claim_forbidden": True,
            "scientific_claim_authorized": False,
        }
        value["manifest_digest"] = trainer.data.object_sha(value)
        return value

    def write_manifest(self, root: Path, value):
        path = root / "manifest.json"
        raw = canonical(value) + b"\n"
        path.write_bytes(raw)
        return path, hashlib.sha256(raw).hexdigest()

    def test_exact644_manifest_and_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parquet = root / "row.parquet"
            parquet.write_bytes(b"fixture")
            path, digest = self.write_manifest(root, self.manifest(parquet))
            value, rows = trainer.load_source_manifest(path, digest)
            self.assertEqual(value["row_count"], 644)
            self.assertEqual(len(rows), 644)
            self.assertEqual(rows[0]["posterior_pair"]["source_role_index"], 0)
            self.assertEqual(rows[0]["posterior_pair"]["action_anchor_role_index"], 1)

    def test_exact643_and_role_swap_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parquet = root / "row.parquet"
            parquet.write_bytes(b"fixture")
            for mutation in ("drop", "swap"):
                value = self.manifest(parquet)
                if mutation == "drop":
                    value["rows"].pop()
                else:
                    value["rows"][0]["posterior_pair"]["source_role_index"] = 1
                    value["rows"][0]["posterior_pair"]["action_anchor_role_index"] = 0
                value.pop("manifest_digest")
                value["manifest_digest"] = trainer.data.object_sha(value)
                path = root / f"{mutation}.json"
                raw = canonical(value) + b"\n"
                path.write_bytes(raw)
                with self.assertRaises(trainer.FullFieldTrainingError):
                    trainer.load_source_manifest(path, hashlib.sha256(raw).hexdigest())

    def test_full_schedule_contract(self):
        base = dict(
            max_steps=644,
            micro_records=1,
            overfit_row=None,
            max_grad_norm=100.0,
            source_manifest_sha256="a" * 64,
            method_source_revision="b" * 40,
            method_source_archive_sha256="c" * 64,
        )
        trainer.validate_args(argparse.Namespace(**base))
        base["overfit_row"] = -1
        with self.assertRaises(trainer.FullFieldTrainingError):
            trainer.validate_args(argparse.Namespace(**base))


if __name__ == "__main__":
    unittest.main()
