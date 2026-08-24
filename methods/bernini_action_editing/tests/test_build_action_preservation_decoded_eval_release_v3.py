from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_action_preservation_decoded_eval_release_v3 as builder


class Exact15ReleaseBuilderTests(unittest.TestCase):
    def test_obsolete_r2_build_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            release = parent / "must-not-exist"
            with self.assertRaisesRegex(
                builder.Exact15ReleaseBuildError, "exact15-r2 is obsolete"
            ):
                builder.build(release)
            self.assertFalse(release.exists())

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
                with self.assertRaisesRegex(Exception, "exact15-r2 is obsolete"):
                    builder.build(release)
            self.assertFalse(release.exists())

    def test_unsorted_member_order_is_rejected_before_creation(self) -> None:
        hostile_order = tuple(reversed(builder.MEMBER_ORDER))
        self.assertNotEqual(list(hostile_order), sorted(hostile_order))
        with tempfile.TemporaryDirectory() as temporary:
            release = Path(temporary).resolve() / "must-not-exist"
            with mock.patch.object(builder, "MEMBER_ORDER", hostile_order):
                with self.assertRaisesRegex(
                    builder.Exact15ReleaseBuildError, "ASCII lexical order"
                ):
                    builder._engine()
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
