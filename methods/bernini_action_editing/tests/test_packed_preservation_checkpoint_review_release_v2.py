#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import packed_preservation_checkpoint_review_release_v2 as release


class ReviewReleaseTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, dict[str, object]]:
        source = root / "source"
        source.mkdir()
        for relative, mode in release.FILES_AND_MODES.items():
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"sealed:{relative}\n".encode("ascii"))
            path.chmod(mode)
        built = dict(
            release.build_release(method_root=source, release_root=root / "release")
        )
        with tarfile.open(str(built["archive"]), mode="r:") as bundle:
            for member in bundle.getmembers():
                self.assertTrue(member.isfile())
                handle = bundle.extractfile(member)
                self.assertIsNotNone(handle)
                destination = root / "release" / member.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(handle.read())
        for relative, mode in release.FILES_AND_MODES.items():
            (root / "release" / release.MEMBER_ROOT / relative).chmod(mode)
        return root / "release", built

    def _validate(self, release_root: Path, built: dict[str, object]):
        return release.validate_executed_release(
            executed_file=release_root / release.MEMBER_ROOT / release.RUNNER_MEMBER,
            executed_launcher=release_root / release.MEMBER_ROOT / release.LAUNCHER_MEMBER,
            manifest=built["manifest"],
            expected_manifest_sha256=str(built["manifest_sha256"]),
            expected_archive_sha256=str(built["archive_sha256"]),
            expected_method_revision=str(built["method_revision"]),
        )

    def test_archive_manifest_and_executed_root_are_one_closure(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            release_root, built = self._fixture(Path(raw).resolve())
            receipt = self._validate(release_root, built)
            self.assertTrue(receipt["archive_members_verified"])
            self.assertTrue(receipt["executed_file_bound"])
            self.assertTrue(receipt["executed_launcher_bound"])
            manifest = json.loads(Path(str(built["manifest"])).read_text("ascii"))
            self.assertEqual(manifest["file_count"], len(release.FILES_AND_MODES))

    def test_extra_executed_file_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            release_root, built = self._fixture(Path(raw).resolve())
            (release_root / release.MEMBER_ROOT / "extra.py").write_text("x", "ascii")
            with self.assertRaises(release.ReviewReleaseError):
                self._validate(release_root, built)

    def test_wrong_executed_launcher_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            release_root, built = self._fixture(Path(raw).resolve())
            wrong = release_root / release.MEMBER_ROOT / release.RUNNER_MEMBER
            with self.assertRaises(release.ReviewReleaseError):
                release.validate_executed_release(
                    executed_file=wrong,
                    executed_launcher=wrong,
                    manifest=built["manifest"],
                    expected_manifest_sha256=str(built["manifest_sha256"]),
                    expected_archive_sha256=str(built["archive_sha256"]),
                    expected_method_revision=str(built["method_revision"]),
                )

    def test_tampered_archive_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            release_root, built = self._fixture(Path(raw).resolve())
            os.chmod(Path(str(built["archive"])), 0o600)
            with Path(str(built["archive"])).open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaises(release.ReviewReleaseError):
                self._validate(release_root, built)


class ReleaseWiringStaticTests(unittest.TestCase):
    def test_runner_validates_release_before_business_imports(self) -> None:
        text = (METHOD_ROOT / "infer_packed_preservation_checkpoint_review_v2.py").read_text("utf-8")
        gate = text.index("release_contract.validate_executed_release(")
        for statement in (
            "import clean_source_visual_context_checkpoint_review_contract_v1 as authoring",
            "import infer_native_identity_generation_canary as native",
            "import packed_preservation_checkpoint_review_v2 as review",
            "import packed_preservation_lora_v2 as core",
        ):
            self.assertLess(gate, text.index(statement))

    def test_launcher_forbids_caller_roots_and_binds_terminal_receipt(self) -> None:
        text = (METHOD_ROOT / release.LAUNCHER_MEMBER).read_text("utf-8")
        self.assertIn('"${python_bin}" -I -S -', text)
        self.assertIn("caller-supplied executable roots are forbidden", text)
        self.assertIn('PYTHONPATH="${method_root}"', text)
        self.assertNotIn('PYTHONPATH="${method_root}:${dependency_root}"', text)
        self.assertIn("--expected-training-receipt-sha256", text)
        self.assertIn("--expected-runtime-source-manifest-sha256", text)
        self.assertIn("executed bootstrap launcher differs from release launcher", text)


if __name__ == "__main__":
    unittest.main()
