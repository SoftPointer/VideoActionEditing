from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_action_preservation_decoded_eval_release_v2 as builder


class Exact15ReleaseBuilderTests(unittest.TestCase):
    def test_exact15_build_is_deterministic_and_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            first = parent / "exact15-a"
            second = parent / "exact15-b"
            first_result = builder.build(first)
            second_result = builder.build(second)
            self.assertTrue(first_result["static_audit_go"])
            self.assertEqual(first_result["file_count"], 15)
            self.assertEqual(
                first_result["release_generation"],
                builder.RELEASE_GENERATION,
            )
            self.assertEqual(
                (first / "source.tar").read_bytes(),
                (second / "source.tar").read_bytes(),
            )
            manifest = json.loads(
                (first / "source.manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [row["path"] for row in manifest["files"]],
                list(builder.MEMBER_ORDER),
            )
            self.assertEqual(
                manifest["release_generation"], builder.RELEASE_GENERATION
            )
            self.assertEqual(
                builder.audit(first, against_workspace=True)[
                    "archive_sha256"
                ],
                first_result["archive_sha256"],
            )

    def test_workspace_component_drift_is_rejected_before_creation(self) -> None:
        relative = "infer_lora.py"
        current = builder.EXPECTED_COMPONENTS[relative]
        hostile = ("0" * 64, current[1], current[2])
        with tempfile.TemporaryDirectory() as temporary:
            release = Path(temporary).resolve() / "must-not-exist"
            with mock.patch.dict(
                builder.EXPECTED_COMPONENTS,
                {relative: hostile},
                clear=False,
            ):
                with self.assertRaisesRegex(Exception, "SHA differs"):
                    builder.build(release)
            self.assertFalse(release.exists())

    def test_captured_engine_has_literal_sha_and_size(self) -> None:
        engine_path = TOOLS / builder.LEGACY_ENGINE
        raw = engine_path.read_bytes()
        self.assertEqual(len(raw), builder.LEGACY_ENGINE_SIZE)
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(), builder.LEGACY_ENGINE_SHA256
        )
        self.assertEqual(builder._captured_engine().__name__, "_apv2_exact15_release_engine")


if __name__ == "__main__":
    unittest.main()
