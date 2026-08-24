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

import build_action_preservation_decoded_eval_release_v4 as builder
import prepare_action_preservation_decoded_eval_inputs_v4 as prepare


MODEL_MANIFEST = ROOT / "audits/bernini_r13_ff4c5d4_checkpoint.sha256"


class Exact15R3InputPreparationTests(unittest.TestCase):
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

    def test_receipt_is_r7_specific_and_binds_r3(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            release = parent / "release"
            preprocessing = parent / "source-preprocessing.json"
            builder.build(release)
            prepare.write_create_only(
                preprocessing, prepare.build_source_preprocessing_authority()
            )
            receipt = prepare.build_authority_receipt(
                release_dir=release,
                model_manifest_path=MODEL_MANIFEST,
                source_preprocessing_path=preprocessing,
            )

        self.assertEqual(receipt["schema_version"], prepare.SCHEMA)
        self.assertEqual(receipt["release_generation"], builder.RELEASE_GENERATION)
        self.assertEqual(receipt["release_member_count"], 15)
        self.assertEqual(receipt["model_file_count"], 23)
        self.assertEqual(receipt["candidate_inherited_fd_count"], 26)
        self.assertEqual(
            receipt["training_authority"]["training_complete_sha256"],
            prepare.R7_TRAINING_COMPLETE_SHA256,
        )
        self.assertEqual(
            receipt["training_authority"]["exact_checkpoint_rows_digest"],
            prepare.R7_EXACT32_ROWS_DIGEST,
        )
        self.assertFalse(receipt["remote_upload_performed"])
        self.assertFalse(receipt["remote_launch_performed"])
        self.assertFalse(receipt["ptrace_authorization_used"])
        unsigned = dict(receipt)
        claimed = unsigned.pop("authority_receipt_digest")
        self.assertEqual(claimed, prepare.object_sha256(unsigned))

    def test_create_only_and_hostile_source_authority_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            output = parent / "source-preprocessing.json"
            value = prepare.build_source_preprocessing_authority()
            prepare.write_create_only(output, value)
            self.assertEqual(json.loads(output.read_text())["authority_digest"],
                             value["authority_digest"])
            with self.assertRaisesRegex(
                prepare.Exact15R3InputPreparationError, "refusing to overwrite"
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
            release = parent / "release"
            builder.build(release)
            with self.assertRaisesRegex(
                prepare.Exact15R3InputPreparationError,
                "source preprocessing authority differs",
            ):
                prepare.build_authority_receipt(
                    release_dir=release,
                    model_manifest_path=MODEL_MANIFEST,
                    source_preprocessing_path=output,
                )


if __name__ == "__main__":
    unittest.main()
