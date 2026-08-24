from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for import_root in (TOOLS, ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import build_action_preservation_decoded_eval_release_v3 as builder
import prepare_action_preservation_decoded_eval_inputs_v3 as prepare


class Exact15R2InputPreparationTests(unittest.TestCase):
    def test_source_preprocessing_is_exact_ordered_r7_authority(self) -> None:
        value = prepare.build_source_preprocessing_authority()
        self.assertEqual(value["schema_version"], prepare.SOURCE_PREPROCESSING_SCHEMA)
        self.assertEqual(
            value["source_order"],
            [
                "7b88a1ca1f804f41",
                "841b5e0080a1441d",
                "a35b590961d24694",
                "a66e6818e4144928",
            ],
        )
        self.assertEqual(len(value["sources"]), 4)
        self.assertTrue(value["source_video_bytes_consumed_directly"])
        self.assertFalse(value["precomputed_transformed_source_artifact_used"])
        unsigned = dict(value)
        claimed = unsigned.pop("authority_digest")
        self.assertEqual(claimed, prepare.object_sha256(unsigned))

    def test_obsolete_r2_receipt_cannot_be_rebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            release = parent / "release"
            with self.assertRaisesRegex(
                builder.Exact15ReleaseBuildError, "exact15-r2 is obsolete"
            ):
                builder.build(release)
            self.assertFalse(release.exists())

    def test_create_only_and_hostile_source_authority_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            output = parent / "source-preprocessing.json"
            value = prepare.build_source_preprocessing_authority()
            prepare.write_create_only(output, value)
            self.assertEqual(json.loads(output.read_text())["authority_digest"],
                             value["authority_digest"])
            with self.assertRaisesRegex(
                prepare.Exact15R2InputPreparationError, "refusing to overwrite"
            ):
                prepare.write_create_only(output, value)

            output.chmod(0o644)
            hostile = json.loads(output.read_text())
            hostile["sources"][0]["seed"] += 1
            hostile["authority_digest"] = prepare.object_sha256(
                {key: item for key, item in hostile.items() if key != "authority_digest"}
            )
            output.write_bytes(prepare.canonical_json_bytes(hostile) + b"\n")
            output.chmod(0o444)
            with self.assertRaisesRegex(
                builder.Exact15ReleaseBuildError,
                "exact15-r2 is obsolete",
            ):
                builder.build(parent / "release")


if __name__ == "__main__":
    unittest.main()
