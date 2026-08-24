from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from tools import materialize_vae as materializer  # noqa: E402


class DynamicGeometryTests(unittest.TestCase):
    def test_frozen_dynamic_bucket_examples(self) -> None:
        self.assertEqual(
            materializer.source_aspect_bucket(704, 896),
            (432, 544),
        )
        self.assertEqual(
            materializer.source_aspect_bucket(736, 704),
            (496, 480),
        )
        for height, width in ((704, 896), (736, 704), (704, 960), (1080, 1920)):
            bucket_h, bucket_w = materializer.source_aspect_bucket(height, width)
            self.assertEqual(bucket_h % 16, 0)
            self.assertEqual(bucket_w % 16, 0)
            self.assertLessEqual(bucket_h * bucket_w, materializer.DEFAULT_MAX_PIXELS)

    def test_target_crop_aligns_to_source_and_keeps_small_mismatch(self) -> None:
        crop, retention = materializer.target_crop_to_source_aspect(
            848, 1072, 704, 896
        )
        top, left, bottom, right = crop
        cropped_ratio = (right - left) / (bottom - top)
        source_ratio = 896 / 704
        self.assertLess(abs(cropped_ratio / source_ratio - 1.0), 0.002)
        self.assertGreater(retention, 0.98)

    def test_portrait_crop_is_centered(self) -> None:
        crop, retention = materializer.target_crop_to_source_aspect(
            976, 928, 736, 704
        )
        top, left, bottom, right = crop
        self.assertLessEqual(abs(top - (976 - bottom)), 1)
        self.assertLessEqual(abs(left - (928 - right)), 1)
        self.assertGreater(retention, 0.98)

    def test_invalid_geometry_fails_closed(self) -> None:
        with self.assertRaises(materializer.VaeMaterializationError):
            materializer.source_aspect_bucket(0, 896)
        with self.assertRaises(materializer.VaeMaterializationError):
            materializer.source_aspect_bucket(704, 896, stride=0)
        with self.assertRaises(materializer.VaeMaterializationError):
            materializer.target_crop_to_source_aspect(0, 1, 1, 1)


class ReceiptTests(unittest.TestCase):
    def test_raw_job_done_is_required_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parquet = root / "raw.parquet"
            preview = root / "preview.jsonl"
            receipt_path = root / "raw.receipt.json"
            done_path = root / "job_done.json"
            parquet.write_bytes(b"raw-parquet-placeholder")
            preview.write_bytes(b"{}\n")
            receipt = {
                "schema_version": materializer.RAW_RECEIPT_FORMAT,
                "preview_only": True,
                "training_authorized": False,
                "training_use_forbidden": True,
                "production_claim_forbidden": True,
                "sample_count": 1,
                "strict_selection_rows": 1,
                "non_strict_selection_rows": 0,
                "experimental_inclusion_policy": (
                    materializer.raw_builder.STRICT_INCLUSION_POLICY
                ),
                "broader_natural_release_inclusion_acknowledged": False,
                "parquet_path": str(parquet),
                "parquet_sha256": materializer.file_sha256(parquet),
                "source_preview_manifest": str(preview),
                "source_preview_manifest_sha256": materializer.file_sha256(preview),
                "sample_ids": ["clip001"],
            }
            receipt["receipt_digest"] = materializer.object_sha256(receipt)
            receipt_path.write_text(
                json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
            )
            done = {
                "schema_version": materializer.raw_builder.JOB_DONE_FORMAT,
                "complete": True,
                "sample_count": 1,
                "strict_selection_rows": 1,
                "non_strict_selection_rows": 0,
                "experimental_inclusion_policy": (
                    materializer.raw_builder.STRICT_INCLUSION_POLICY
                ),
                "raw_parquet_sha256": materializer.file_sha256(parquet),
                "raw_receipt_sha256": materializer.file_sha256(receipt_path),
                "preview_manifest_sha256": materializer.file_sha256(preview),
            }
            done["job_done_digest"] = materializer.object_sha256(done)
            done_path.write_text(
                json.dumps(done, sort_keys=True) + "\n", encoding="utf-8"
            )
            validated = materializer._validate_raw_job_done(
                parquet, receipt_path, done_path
            )
            self.assertEqual(validated["sample_count"], 1)
            parquet.write_bytes(b"tampered")
            with self.assertRaisesRegex(
                materializer.VaeMaterializationError,
                "raw parquet hash differs|job-done artifact binding differs",
            ):
                materializer._validate_raw_job_done(
                    parquet, receipt_path, done_path
                )

    def test_existing_sample_receipt_is_hash_and_digest_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard = root / "sample.parquet"
            receipt_path = root / "sample.json"
            shard.write_bytes(b"parquet-placeholder")
            receipt = {
                "schema_version": materializer.SAMPLE_RECEIPT_FORMAT,
                "parquet_sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
            }
            receipt["receipt_digest"] = materializer.object_sha256(receipt)
            receipt_path.write_text(
                json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
            )
            self.assertTrue(
                materializer._existing_sample_is_valid(shard, receipt_path)
            )
            shard.write_bytes(b"tampered")
            with self.assertRaisesRegex(
                materializer.VaeMaterializationError, "existing sample receipt differs"
            ):
                materializer._existing_sample_is_valid(shard, receipt_path)


if __name__ == "__main__":
    unittest.main()
