from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import motion_residual as motion  # noqa: E402
from tools import build_strict_routing as builder  # noqa: E402


def _json_file(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


class FrozenReleaseFixture:
    def __init__(self, root: Path):
        self.root = root
        self.shards = root / "shards"
        self.receipts = root / "receipts"
        self.index = root / "dataset_index.jsonl"
        self.summary = root / "dataset_summary.json"
        self.output = root / "routing" / "strict-359.jsonl"
        self.shards.mkdir()
        self.receipts.mkdir()
        self.index_rows: list[dict[str, object]] = []
        self.summary_value: dict[str, object] = {}
        self._build()

    @staticmethod
    def digest(value: object) -> str:
        return builder.object_sha256(value)

    def _receipt_value(self, iid: str, *, strict: bool, shard: Path) -> dict[str, object]:
        shard_sha = hashlib.sha256(shard.read_bytes()).hexdigest()
        materialized_digest = hashlib.sha256(f"row:{iid}".encode()).hexdigest()
        receipt: dict[str, object] = {
            "schema_version": builder.SAMPLE_RECEIPT_SCHEMA,
            "complete": True,
            "iid": iid,
            "preview_only": True,
            "training_authorized": False,
            "training_use_forbidden": True,
            "experimental_training_acknowledged": True,
            "production_claim_forbidden": True,
            "experimental_inclusion_policy": builder.EXPECTED_INCLUSION_POLICY,
            "strict_selection_gates_all_true": strict,
            "selection_gates_json": json.dumps(
                {"single_actor": True, "motion_valid": strict},
                sort_keys=True,
                separators=(",", ":"),
            ),
            "frame_count": builder.EXPECTED_FRAME_COUNT,
            "fps": builder.EXPECTED_FPS,
            "latent_frame_count": builder.EXPECTED_LATENT_FRAME_COUNT,
            "bucket_hw": [480, 496],
            "posterior_parameters_shape": [32, 21, 60, 62],
            "shared_i0_exact": True,
            "raw_renderer_row_digest": hashlib.sha256(
                f"raw:{iid}".encode()
            ).hexdigest(),
            "materialized_row_digest": materialized_digest,
            "source_latent_blob_sha256": hashlib.sha256(
                f"source:{iid}".encode()
            ).hexdigest(),
            "target_latent_blob_sha256": hashlib.sha256(
                f"target:{iid}".encode()
            ).hexdigest(),
            "parquet_path": str(shard),
            "parquet_sha256": shard_sha,
            "vae_identity": {"checkpoint": "fixture"},
        }
        receipt["receipt_digest"] = self.digest(receipt)
        return receipt

    def _build(self) -> None:
        for ordinal in range(builder.EXPECTED_ROWS):
            iid = f"iid{ordinal:04d}"
            strict = ordinal < builder.EXPECTED_STRICT_ROWS
            shard = self.shards / f"{iid}.parquet"
            shard.write_bytes(f"fixture-shard:{iid}\n".encode())
            receipt_path = self.receipts / f"{iid}.json"
            receipt = self._receipt_value(iid, strict=strict, shard=shard)
            _json_file(receipt_path, receipt)
            self.index_rows.append(
                {
                    "schema_version": builder.INDEX_ROW_SCHEMA,
                    "iid": iid,
                    "parquet_path": str(shard),
                    "parquet_sha256": receipt["parquet_sha256"],
                    "materialized_row_digest": receipt["materialized_row_digest"],
                    "bucket_hw": receipt["bucket_hw"],
                    "posterior_parameters_shape": receipt[
                        "posterior_parameters_shape"
                    ],
                    "sample_receipt_path": str(receipt_path),
                    "sample_receipt_sha256": hashlib.sha256(
                        receipt_path.read_bytes()
                    ).hexdigest(),
                    "preview_only": True,
                    "production_claim_forbidden": True,
                }
            )
        self._write_index()
        self.summary_value = {
            "schema_version": builder.SUMMARY_SCHEMA,
            "complete": True,
            "preview_only": True,
            "training_authorized": False,
            "training_use_forbidden": True,
            "experimental_training_acknowledged": True,
            "production_claim_forbidden": True,
            "scientific_claim_authorized": False,
            "experimental_inclusion_policy": builder.EXPECTED_INCLUSION_POLICY,
            "raw_strict_selection_rows": builder.EXPECTED_STRICT_ROWS,
            "raw_non_strict_selection_rows": builder.EXPECTED_NON_STRICT_ROWS,
            "materialized_strict_selection_rows": builder.EXPECTED_STRICT_ROWS,
            "materialized_non_strict_selection_rows": (
                builder.EXPECTED_NON_STRICT_ROWS
            ),
            "raw_receipt_path": str(self.root / "raw.receipt.json"),
            "raw_receipt_sha256": "a" * 64,
            "raw_job_done_path": str(self.root / "raw.done.json"),
            "raw_job_done_sha256": "b" * 64,
            "expected_sample_count": builder.EXPECTED_ROWS,
            "materialized_sample_count": builder.EXPECTED_ROWS,
            "missing_sample_count": 0,
            "missing_sample_ids": [],
            "frame_count": builder.EXPECTED_FRAME_COUNT,
            "fps": builder.EXPECTED_FPS,
            "latent_frame_count": builder.EXPECTED_LATENT_FRAME_COUNT,
            "bucket_counts": {"480x496": builder.EXPECTED_ROWS},
            "vae_identity_digest": "c" * 64,
            "shards_directory": str(self.shards),
            "index_path": str(self.index),
            "index_sha256": hashlib.sha256(self.index.read_bytes()).hexdigest(),
        }
        self._write_summary()

    def _write_index(self) -> None:
        self.index.write_bytes(
            b"".join(
                builder.canonical_json_bytes(row) + b"\n" for row in self.index_rows
            )
        )

    def _write_summary(self) -> None:
        self.summary_value.pop("summary_digest", None)
        self.summary_value["index_sha256"] = hashlib.sha256(
            self.index.read_bytes()
        ).hexdigest()
        self.summary_value["summary_digest"] = self.digest(self.summary_value)
        _json_file(self.summary, self.summary_value)

    @property
    def summary_sha256(self) -> str:
        return hashlib.sha256(self.summary.read_bytes()).hexdigest()

    def receipt(self, iid: str) -> tuple[Path, dict[str, object], dict[str, object]]:
        ordinal = int(iid[3:])
        row = self.index_rows[ordinal]
        path = Path(str(row["sample_receipt_path"]))
        return path, json.loads(path.read_text(encoding="utf-8")), row

    def rebind_receipt(
        self, iid: str, receipt: dict[str, object], *, recompute_digest: bool
    ) -> None:
        path, _old, row = self.receipt(iid)
        if recompute_digest:
            receipt.pop("receipt_digest", None)
            receipt["receipt_digest"] = self.digest(receipt)
        _json_file(path, receipt)
        row["sample_receipt_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self._write_index()
        self._write_summary()

    def build(self) -> dict[str, object]:
        return builder.build_strict_routing(
            dataset_summary=self.summary,
            expected_dataset_summary_sha256=self.summary_sha256,
            output_jsonl=self.output,
        )


class StrictRoutingBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.release = FrozenReleaseFixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_builds_complete_motion_only_reject_routes_and_bound_sidecars(self) -> None:
        result = self.release.build()
        routing_lines = self.release.output.read_text(encoding="utf-8").splitlines()
        routes = [json.loads(line) for line in routing_lines]
        self.assertEqual(len(routes), builder.EXPECTED_ROWS)
        self.assertEqual(
            sum(row["tier"] == "motion_only" for row in routes),
            builder.EXPECTED_STRICT_ROWS,
        )
        self.assertEqual(
            sum(row["tier"] == "reject" for row in routes),
            builder.EXPECTED_NON_STRICT_ROWS,
        )
        self.assertEqual({row["full_target_weight"] for row in routes}, {0.0})
        self.assertEqual([row["iid"] for row in routes], sorted(row["iid"] for row in routes))

        receipt_path = Path(f"{self.release.output}.receipt.json")
        hash_path = Path(f"{self.release.output}.sha256")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        candidate = dict(receipt)
        declared_digest = candidate.pop("receipt_digest")
        self.assertEqual(builder.object_sha256(candidate), declared_digest)
        self.assertEqual(receipt["dataset_summary_sha256"], self.release.summary_sha256)
        self.assertEqual(receipt["dataset_index_sha256"], self.release.summary_value["index_sha256"])
        self.assertEqual(receipt["strict_motion_only_count"], 359)
        self.assertEqual(receipt["non_strict_reject_count"], 285)
        self.assertEqual(receipt["full_pair_count"], 0)

        sidecar = hash_path.read_text(encoding="ascii").splitlines()
        self.assertEqual(
            sidecar[0],
            f"{hashlib.sha256(self.release.output.read_bytes()).hexdigest()}  {self.release.output.name}",
        )
        self.assertEqual(
            sidecar[1],
            f"{hashlib.sha256(receipt_path.read_bytes()).hexdigest()}  {receipt_path.name}",
        )

        router = motion.ReviewRouter.load(self.release.output, default_tier="reject")
        self.assertEqual(
            router.receipt()["explicit_route_counts"],
            {"full_pair": 0, "motion_only": 359, "reject": 285},
        )
        self.assertEqual(
            router.digest,
            result["review_router_digest_required_default_reject"],
        )
        self.assertEqual(router.route("not-in-release").tier, "reject")

    def test_requires_caller_pinned_summary_hash_and_internal_summary_digest(self) -> None:
        with self.assertRaisesRegex(
            builder.StrictRoutingError, "caller-pinned hash"
        ):
            builder.build_strict_routing(
                dataset_summary=self.release.summary,
                expected_dataset_summary_sha256="0" * 64,
                output_jsonl=self.release.output,
            )
        self.assertFalse(self.release.output.exists())

        summary = json.loads(self.release.summary.read_text(encoding="utf-8"))
        summary["bucket_counts"] = {"480x496": 643, "496x480": 1}
        # Deliberately retain the old summary_digest, but pin the new file bytes.
        _json_file(self.release.summary, summary)
        with self.assertRaisesRegex(builder.StrictRoutingError, "summary digest"):
            self.release.build()
        self.assertFalse(self.release.output.exists())

    def test_rejects_index_and_sample_receipt_hash_mismatches_without_output(self) -> None:
        self.release.index.write_bytes(self.release.index.read_bytes() + b" ")
        with self.assertRaisesRegex(builder.StrictRoutingError, "index file hash"):
            self.release.build()
        self.assertFalse(self.release.output.exists())

        # Restore the bound index, then alter one receipt without rebinding it.
        self.release._write_index()
        self.release._write_summary()
        path, _receipt, _row = self.release.receipt("iid0000")
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(builder.StrictRoutingError, "receipt file hash"):
            self.release.build()
        self.assertFalse(self.release.output.exists())

    def test_rejects_sample_receipt_digest_even_if_index_and_summary_are_rehashed(self) -> None:
        _path, receipt, _row = self.release.receipt("iid0000")
        receipt["target_latent_blob_sha256"] = "d" * 64
        self.release.rebind_receipt(
            "iid0000", receipt, recompute_digest=False
        )
        with self.assertRaisesRegex(builder.StrictRoutingError, "receipt digest"):
            self.release.build()
        self.assertFalse(self.release.output.exists())

    def test_rejects_receipt_and_shard_path_rebinding(self) -> None:
        original, _receipt, row = self.release.receipt("iid0000")
        alias = self.release.receipts / "alias.json"
        shutil.copyfile(original, alias)
        row["sample_receipt_path"] = str(alias)
        row["sample_receipt_sha256"] = hashlib.sha256(alias.read_bytes()).hexdigest()
        self.release._write_index()
        self.release._write_summary()
        with self.assertRaisesRegex(builder.StrictRoutingError, "path is not dataset-bound"):
            self.release.build()
        self.assertFalse(self.release.output.exists())

    def test_rejects_358_286_receipt_cohort_even_when_every_digest_is_valid(self) -> None:
        _path, receipt, _row = self.release.receipt("iid0358")
        receipt["strict_selection_gates_all_true"] = False
        receipt["selection_gates_json"] = json.dumps(
            {"single_actor": True, "motion_valid": False},
            sort_keys=True,
            separators=(",", ":"),
        )
        self.release.rebind_receipt("iid0358", receipt, recompute_digest=True)
        with self.assertRaisesRegex(
            builder.StrictRoutingError, "strict=358 non_strict=286"
        ):
            self.release.build()
        self.assertFalse(self.release.output.exists())

    def test_create_only_sidecar_preflight_leaves_no_partial_artifact(self) -> None:
        self.release.output.parent.mkdir(parents=True)
        hash_path = Path(f"{self.release.output}.sha256")
        hash_path.write_text("preexisting\n", encoding="utf-8")
        with self.assertRaisesRegex(builder.StrictRoutingError, "create-only output exists"):
            self.release.build()
        self.assertFalse(self.release.output.exists())
        self.assertFalse(Path(f"{self.release.output}.receipt.json").exists())
        self.assertEqual(hash_path.read_text(encoding="utf-8"), "preexisting\n")


if __name__ == "__main__":
    unittest.main()
