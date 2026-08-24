from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from tools import build_renderer_dataset as builder  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _preview_fixture(root: Path, *, iid: str = "clip001") -> tuple[Path, dict]:
    strict = builder._load_strict_preview_module()
    source = root / "source.mp4"
    target = root / "target.mp4"
    shared_i0 = root / "conditioning_frame0_float32.npy"
    source.write_bytes(b"source-video")
    target.write_bytes(b"target-video")
    shared_i0.write_bytes(b"lossless-frame-zero")
    generated = {
        "schema_version": strict.WAN_GENERATED_FORMAT,
        "iid": iid,
        "source_video": str(source),
        "source_video_sha256": _sha(source),
        "target_preview_mp4": str(target),
        "target_preview_mp4_sha256": _sha(target),
        "conditioning_frame0_float32": str(shared_i0),
        "conditioning_frame0_float32_sha256": _sha(shared_i0),
    }
    generated_path = root / "generated_manifest.jsonl"
    generated_path.write_bytes(strict.canonical_json_bytes(generated) + b"\n")
    instruction = "Have the actor gradually crouch while preserving the scene."
    generation_instruction = "Starting at frame zero, make the actor crouch."
    row = {
        "schema_version": strict.PREVIEW_ROW_FORMAT,
        "iid": iid,
        "group_id": "parent-group-001",
        "family": "crouch",
        "source_video_path": str(source),
        "source_video_sha256": _sha(source),
        "target_video_path": str(target),
        "target_video_sha256": _sha(target),
        "edit_instruction": instruction,
        "edit_instruction_sha256": strict.text_sha256(instruction),
        "instruction_source": "natural",
        "generation_instruction": generation_instruction,
        "generation_instruction_sha256": strict.text_sha256(
            generation_instruction
        ),
        "source_census": {"iid": iid},
        "target_plan": {"iid": iid},
        "selection_gates": {
            "single_dynamic_actor": True,
            "source_camera_locked_off": True,
            "target_camera_locked_off": True,
        },
        "preview_only": True,
        "training_authorized": False,
        "training_use_forbidden": True,
        "production_eligible": False,
        "post_video_acceptance": "pending",
        "provenance": {
            "wan_generated_manifest_path": str(generated_path),
            "wan_generated_manifest_sha256": _sha(generated_path),
        },
    }
    row["row_digest"] = strict.object_sha256(row)
    manifest = root / "preview.jsonl"
    manifest.write_bytes(strict.canonical_json_bytes(row) + b"\n")
    return manifest, row


def _bind_natural_release(manifest: Path, row: dict) -> dict:
    strict = builder._load_strict_preview_module()
    qwen_path = manifest.parent / "qwen_passed.jsonl"
    qwen_path.write_bytes(b"{}\n")
    natural_result_path = manifest.parent / "natural_result.json"
    natural_result_path.write_bytes(b"{}\n")
    natural_receipt_path = manifest.parent / "natural_receipt.json"
    natural_receipt_path.write_bytes(b"{}\n")
    natural_instruction_path = manifest.parent / "natural_edit_instruction.txt"
    natural_instruction_path.write_text(
        row["edit_instruction"] + "\n", encoding="utf-8"
    )
    release_row = {
        "schema_version": "motive-goku-natural-motion-dataset-row-v1",
        "iid": row["iid"],
        "label_status": (
            "structured_plan_semantic_audit_passed_video_audit_pending"
        ),
        "natural_edit_instruction": row["edit_instruction"],
        "natural_edit_instruction_sha256": row["edit_instruction_sha256"],
        "source_passed_path": str(qwen_path),
        "source_passed_sha256": _sha(qwen_path),
        "result_path": str(natural_result_path),
        "result_sha256": _sha(natural_result_path),
        "semantic_audit": {
            "effective_verdict": "pass",
            "model_reported_diagnostics": {"confidence": "high"},
        },
    }
    release_manifest = manifest.parent / "natural_edit_instruction_manifest.jsonl"
    release_manifest.write_bytes(builder.canonical_json_bytes(release_row) + b"\n")
    summary = {
        "schema_version": "motive-goku-natural-motion-verify-summary-v1",
        "dataset_manifest_path": str(release_manifest),
        "dataset_manifest_sha256": _sha(release_manifest),
        "expected_rows": 1,
        "terminal_rows": 1,
        "ok_rows": 1,
        "error_rows": 0,
    }
    summary["summary_digest"] = builder.object_sha256(summary)
    summary_path = manifest.parent / "verification_summary.json"
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    row = copy.deepcopy(row)
    row["selection_gates"]["single_dynamic_actor"] = False
    row["provenance"].update(
        {
            "natural_release_summary_path": str(summary_path),
            "natural_release_summary_sha256": _sha(summary_path),
            "natural_release_manifest_path": str(release_manifest),
            "natural_release_manifest_sha256": _sha(release_manifest),
            "natural_release_row_file_sha256": hashlib.sha256(
                builder.canonical_json_bytes(release_row) + b"\n"
            ).hexdigest(),
            "qwen_passed_path": str(qwen_path),
            "qwen_passed_sha256": _sha(qwen_path),
            "natural_result_path": str(natural_result_path),
            "natural_result_sha256": _sha(natural_result_path),
            "natural_receipt_path": str(natural_receipt_path),
            "natural_receipt_sha256": _sha(natural_receipt_path),
            "natural_instruction_path": str(natural_instruction_path),
            "natural_instruction_file_sha256": _sha(natural_instruction_path),
        }
    )
    row["row_digest"] = strict.object_sha256(
        {key: value for key, value in row.items() if key != "row_digest"}
    )
    manifest.write_bytes(strict.canonical_json_bytes(row) + b"\n")
    return row


class BuildRendererDatasetTests(unittest.TestCase):
    def test_requires_explicit_preview_acknowledgement_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(
                builder.RendererDatasetError, "--acknowledge-preview-only"
            ):
                builder.build_dataset(
                    root / "does-not-exist.jsonl", root / "out.parquet"
                )

    def test_builds_official_message_order_and_preserves_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, upstream = _preview_fixture(root)
            rows, resolved_manifest, manifest_sha = builder.build_renderer_rows(
                manifest
            )
            self.assertEqual(resolved_manifest, manifest.resolve())
            self.assertEqual(manifest_sha, _sha(manifest))
            self.assertEqual(len(rows), 1)
            row = rows[0]
            messages = json.loads(row["inputs"])
            self.assertEqual(
                messages,
                [
                    {"has_loss": 0, "type": "video"},
                    {
                        "has_loss": 0,
                        "text": upstream["edit_instruction"],
                        "type": "text",
                    },
                    {"has_loss": 1, "type": "video_gen"},
                ],
            )
            self.assertEqual(
                row["videos"],
                [
                    {"video_path": str((root / "source.mp4").resolve())},
                    {"video_path": str((root / "target.mp4").resolve())},
                ],
            )
            self.assertEqual(row["preview_row_digest"], upstream["row_digest"])
            self.assertEqual(row["source_video_sha256"], upstream["source_video_sha256"])
            self.assertEqual(row["target_video_sha256"], upstream["target_video_sha256"])
            self.assertEqual(row["shared_i0_sha256"], _sha(root / "conditioning_frame0_float32.npy"))
            for field in builder.UPSTREAM_AUTHORIZATION_FIELDS:
                self.assertEqual(row[field], upstream[field])
            self.assertTrue(row["experimental_training_acknowledged"])
            self.assertTrue(row["production_claim_forbidden"])
            digest_candidate = dict(row)
            digest = digest_candidate.pop("renderer_row_digest")
            self.assertEqual(digest, builder.object_sha256(digest_candidate))

    def test_upstream_tampering_is_rejected_by_strict_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = _preview_fixture(root)
            parsed = json.loads(manifest.read_text(encoding="utf-8"))
            parsed["training_authorized"] = True
            # Deliberately retain the prior digest: either authorization or the
            # digest must make the shared validator reject this row.
            manifest.write_text(
                json.dumps(parsed, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                builder.RendererDatasetError, "preview manifest validation failed"
            ):
                builder.build_renderer_rows(manifest)

    def test_broader_natural_release_requires_opt_in_and_is_release_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, row = _preview_fixture(root)
            row = _bind_natural_release(manifest, row)
            with self.assertRaisesRegex(
                builder.RendererDatasetError, "selection gates are not all true"
            ):
                builder.build_renderer_rows(manifest)
            rows, _, _, release = builder._build_renderer_rows_with_release(
                manifest, allow_broader_natural_release=True
            )
            self.assertIsNotNone(release)
            self.assertEqual(release["release_rows"], 1)
            self.assertEqual(
                rows[0]["experimental_inclusion_policy"],
                builder.NATURAL_RELEASE_INCLUSION_POLICY,
            )
            self.assertFalse(rows[0]["strict_selection_gates_all_true"])
            self.assertEqual(
                json.loads(rows[0]["selection_gates_json"]),
                row["selection_gates"],
            )

    def test_receipt_is_explicitly_experimental_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, upstream = _preview_fixture(root)
            output = root / "dataset.parquet"
            receipt_path = root / "receipt.json"

            def fake_writer(rows, path):
                path.write_bytes(builder.canonical_json_bytes(list(rows)) + b"\n")

            receipt = builder.build_dataset(
                manifest,
                output,
                receipt_path=receipt_path,
                acknowledge_preview_only=True,
                parquet_writer=fake_writer,
            )
            self.assertTrue(output.is_file())
            self.assertEqual(receipt["parquet_sha256"], _sha(output))
            self.assertEqual(receipt["sample_ids"], [upstream["iid"]])
            self.assertEqual(receipt["sample_count"], 1)
            self.assertTrue(receipt["preview_only"])
            self.assertFalse(receipt["training_authorized"])
            self.assertTrue(receipt["training_use_forbidden"])
            self.assertFalse(receipt["production_eligible"])
            self.assertTrue(receipt["production_claim_forbidden"])
            self.assertFalse(receipt["scientific_claim_authorized"])
            on_disk = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(on_disk, receipt)
            digest_candidate = copy.deepcopy(receipt)
            digest = digest_candidate.pop("receipt_digest")
            self.assertEqual(digest, builder.object_sha256(digest_candidate))

    @unittest.skipUnless(
        importlib.util.find_spec("pyarrow") is not None, "pyarrow is not installed"
    )
    def test_real_pyarrow_round_trip(self) -> None:
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = _preview_fixture(root)
            rows, _, _ = builder.build_renderer_rows(manifest)
            output = root / "raw.parquet"
            builder.write_parquet(rows, output)
            table = pq.read_table(output)
            self.assertEqual(table.num_rows, 1)
            persisted = table.to_pylist()[0]
            self.assertEqual(json.loads(persisted["inputs"])[0]["type"], "video")
            self.assertTrue(persisted["production_claim_forbidden"])


if __name__ == "__main__":
    unittest.main()
