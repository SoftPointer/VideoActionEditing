from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for import_root in (TOOLS, ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import build_action_preservation_decoded_eval_release_v2 as builder
import prepare_action_preservation_decoded_eval_inputs_v2 as prepare


MODEL_MANIFEST = ROOT / "audits/bernini_r13_ff4c5d4_checkpoint.sha256"


class Exact15InputPreparationTests(unittest.TestCase):
    def test_receipt_binds_release_model_and_fd_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            release = Path(temporary).resolve() / "exact15"
            builder.build(release)
            receipt = prepare.build_authority_receipt(
                release_dir=release,
                model_manifest_path=MODEL_MANIFEST,
            )

        self.assertEqual(receipt["schema_version"], prepare.SCHEMA)
        self.assertEqual(receipt["release_member_count"], 15)
        self.assertEqual(receipt["model_file_count"], 23)
        self.assertEqual(receipt["model_directory_count"], 7)
        self.assertEqual(receipt["base_control_inherited_fd_count"], 23)
        self.assertEqual(receipt["candidate_inherited_fd_count"], 26)
        self.assertTrue(receipt["directory_fds_holder_private_cloexec"])
        self.assertTrue(
            receipt["leaf_file_fds_inherited_only_at_exact_spawn_boundaries"]
        )
        self.assertTrue(receipt["proc_self_fd_consumption_required"])
        self.assertFalse(receipt["ptrace_authorization_used"])
        self.assertFalse(receipt["remote_launch_performed"])
        unsigned = dict(receipt)
        digest = unsigned.pop("authority_receipt_digest")
        self.assertEqual(digest, prepare.object_sha256(unsigned))

    def test_cli_output_is_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            release = parent / "exact15"
            output = parent / "authority.json"
            builder.build(release)
            argv = [
                "--release-dir",
                str(release),
                "--model-manifest",
                str(MODEL_MANIFEST),
                "--output",
                str(output),
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(prepare.main(argv), 0)
            parsed = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(parsed["schema_version"], prepare.SCHEMA)
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(
                    prepare.Exact15InputPreparationError,
                    "refusing to overwrite",
                ):
                    prepare.main(argv)

    def test_runtime_trust_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            release = Path(temporary).resolve() / "exact15"
            builder.build(release)
            hostile = dict(prepare.runtime.TRUSTED_EXACT15)
            hostile.pop(next(iter(hostile)))
            with mock.patch.object(
                prepare.runtime,
                "TRUSTED_EXACT15",
                hostile,
            ):
                with self.assertRaisesRegex(
                    prepare.Exact15InputPreparationError,
                    "trust closure differs",
                ):
                    prepare.build_authority_receipt(
                        release_dir=release,
                        model_manifest_path=MODEL_MANIFEST,
                    )


if __name__ == "__main__":
    unittest.main()
