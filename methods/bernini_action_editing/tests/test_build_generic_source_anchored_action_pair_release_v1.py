from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from methods.bernini_action_editing.tools import (
    build_generic_source_anchored_action_pair_release_v1 as release,
)


class GenericActionPairReleaseTests(unittest.TestCase):
    @staticmethod
    def _dummy_root(root: Path) -> Path:
        method_root = root / "methods" / "bernini_action_editing"
        for index, (relative, mode) in enumerate(release.FILES_AND_MODES.items()):
            path = method_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"member-{index}-{relative}\n".encode("ascii"))
            path.chmod(mode)
        return method_root.resolve()

    def test_release_is_byte_deterministic_and_component_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            method_root = self._dummy_root(root)
            archive_a = root / "a.tar"
            manifest_a = root / "a.manifest.json"
            archive_b = root / "b.tar"
            manifest_b = root / "b.manifest.json"
            receipt_a = release.build(method_root, archive_a, manifest_a)
            receipt_b = release.build(method_root, archive_b, manifest_b)
            self.assertEqual(archive_a.read_bytes(), archive_b.read_bytes())
            self.assertEqual(manifest_a.read_bytes(), manifest_b.read_bytes())
            self.assertEqual(
                receipt_a["component_pins"], receipt_b["component_pins"]
            )
            manifest = release.audit(
                archive_a,
                manifest_a,
                expected_archive_sha256=receipt_a["archive_sha256"],
                expected_manifest_sha256=receipt_a["manifest_sha256"],
            )
            rows = {row["path"]: row for row in manifest["files"]}
            for label, relative in release.COMPONENT_FILES.items():
                self.assertEqual(
                    manifest["component_pins"][label], rows[relative]["sha256"]
                )

    def test_audit_rejects_archive_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            method_root = self._dummy_root(root)
            archive = root / "source.tar"
            manifest = root / "source.manifest.json"
            receipt = release.build(method_root, archive, manifest)
            raw = bytearray(archive.read_bytes())
            raw[0] ^= 1
            archive.chmod(0o600)
            archive.write_bytes(bytes(raw))
            with self.assertRaisesRegex(
                release.GenericActionReleaseError, "archive SHA-256 differs"
            ):
                release.audit(
                    archive,
                    manifest,
                    expected_archive_sha256=receipt["archive_sha256"],
                    expected_manifest_sha256=receipt["manifest_sha256"],
                )

    def test_release_excludes_dynamic_action_authority_files(self) -> None:
        self.assertNotIn(
            "assets/representation_train_manifest_v1.json",
            release.FILES_AND_MODES,
        )
        self.assertNotIn(
            "assets/action_source_pair_manifest_v1.json", release.FILES_AND_MODES
        )
        self.assertNotIn(
            "tools/generic_action_manifest_v1.py",
            release.FILES_AND_MODES,
        )
        self.assertIn(
            "train_clean_source_visual_context_stage_b_v1.py",
            release.FILES_AND_MODES,
        )


if __name__ == "__main__":
    unittest.main()
