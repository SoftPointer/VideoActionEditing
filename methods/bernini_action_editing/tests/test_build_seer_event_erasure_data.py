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

from tools import build_seer_event_erasure_data as builder  # noqa: E402


class EventErasureBuilderTests(unittest.TestCase):
    def test_index_map_excludes_transition_and_terminal(self) -> None:
        index = builder.event_erasure_index_map()
        self.assertEqual(len(index), 81)
        self.assertEqual(index[:32], list(range(32)))
        self.assertEqual(index[32:], [31] * 49)
        self.assertLess(max(index), 32)

    def test_erased_rgb_frames_are_exact_prefix_then_hold(self) -> None:
        target = [index.to_bytes(2, "little") for index in range(81)]
        erased = builder.event_erased_rgb_frames(target)
        self.assertEqual(erased[:32], target[:32])
        self.assertEqual(erased[32:], [target[31]] * 49)
        with self.assertRaises(builder.SeerDatasetError):
            builder.event_erased_rgb_frames(target[:-1])

    def test_renderer_row_is_exact_standard_schema(self) -> None:
        owner_path = METHOD_ROOT / "assets" / "seer_owner_core2_v1.json"
        owner = json.loads(owner_path.read_text(encoding="utf-8"))
        wanted = owner["rows"][0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target, shared = root / "source.mp4", root / "target.mp4", root / "i0.npy"
            source.write_bytes(b"source")
            target.write_bytes(b"target")
            shared.write_bytes(b"i0")
            wanted = dict(wanted)
            wanted["target_video"] = str(target.resolve())
            wanted["target_video_sha256"] = hashlib.sha256(b"target").hexdigest()
            row = builder._renderer_row(
                wanted,
                owner_path=owner_path.resolve(),
                owner_sha=hashlib.sha256(owner_path.read_bytes()).hexdigest(),
                source=source.resolve(),
                source_sha=hashlib.sha256(b"source").hexdigest(),
                shared_i0=shared.resolve(),
                shared_sha=hashlib.sha256(b"i0").hexdigest(),
            )
        self.assertEqual(row["schema_version"], builder.raw_builder.ROW_FORMAT)
        self.assertEqual(row["iid"], wanted["iid"])
        self.assertEqual(row["experimental_inclusion_policy"], "strict_single_actor")
        self.assertTrue(row["strict_selection_gates_all_true"])
        self.assertFalse(row["training_authorized"])
        self.assertTrue(row["training_use_forbidden"])
        candidate = dict(row)
        declared = candidate.pop("renderer_row_digest")
        self.assertEqual(declared, builder.object_sha256(candidate))

    def test_authority_does_not_call_completion_method_success(self) -> None:
        self.assertTrue(builder.AUTHORITY["experimental_parameter_update_authorized"])
        self.assertFalse(builder.AUTHORITY["training_completion_is_method_success"])
        self.assertFalse(builder.AUTHORITY["production_claim_authorized"])
        self.assertTrue(builder.AUTHORITY["heldout_decoded_review_required"])

    def test_final_manifest_digest_and_vae_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shards = root / "shards"
            shards.mkdir()
            index = root / "index.jsonl"
            index.write_text("{}\n", encoding="utf-8")
            summary = root / "summary.json"
            summary_value = {
                "complete": True,
                "expected_sample_count": 2,
                "materialized_sample_count": 2,
                "missing_sample_count": 0,
                "experimental_inclusion_policy": "strict_single_actor",
                "shards_directory": str(shards.resolve()),
                "index_path": str(index.resolve()),
                "index_sha256": builder.file_sha256(index),
            }
            summary.write_text(json.dumps(summary_value), encoding="utf-8")
            raw = {
                "schema_version": builder.RAW_MANIFEST_SCHEMA,
                "owner_spec": {"path": "/o", "sha256": "a" * 64},
                "rows": [{"iid": "a"}, {"iid": "b"}],
                "raw": {},
                "routing": {"path": "/r", "sha256": "b" * 64, "row_count": 2},
                "authority": dict(builder.AUTHORITY),
            }
            raw["manifest_digest"] = builder.object_sha256(raw)
            raw_path = root / "raw.json"
            raw_path.write_bytes(builder.canonical_json_bytes(raw) + b"\n")
            output = root / "final.json"
            result = builder.finalize(
                raw_manifest_path=raw_path.resolve(),
                expected_raw_manifest_sha256=builder.file_sha256(raw_path),
                parquet_directory=shards.resolve(),
                dataset_summary=summary.resolve(),
                index_path=index.resolve(),
                output_manifest=output.resolve(),
            )
            value = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(value["schema_version"], builder.FINAL_MANIFEST_SCHEMA)
        candidate = dict(value)
        declared = candidate.pop("manifest_digest")
        self.assertEqual(declared, builder.object_sha256(candidate))
        self.assertEqual(result["row_count"], 2)


if __name__ == "__main__":
    unittest.main()
