from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from methods.bernini_action_editing.tools import (
    build_source_self_training_release_v1 as release,
)


METHOD_ROOT = Path(__file__).resolve().parents[1]


class SourceSelfReleaseTests(unittest.TestCase):
    def test_real_release_is_deterministic_and_exact(self) -> None:
        manifest, payloads = release.build_manifest(METHOD_ROOT)
        first = release.build_archive(manifest, payloads)
        second = release.build_archive(manifest, payloads)
        self.assertEqual(first, second)
        release.verify_archive_bytes(first, manifest)
        self.assertEqual(manifest["revision_kind"], "content-closure-sha1")
        self.assertFalse(manifest["git_commit_claimed"])
        self.assertEqual(
            [row["path"] for row in manifest["files"]],
            list(release.RELEASE_FILES),
        )
        with tarfile.open(fileobj=__import__("io").BytesIO(first), mode="r:") as archive:
            self.assertEqual(len(archive.getmembers()), len(release.RELEASE_FILES))

    def test_create_only_publication_and_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "release.tar"
            manifest_path = root / "release.manifest.json"
            result = release.build(METHOD_ROOT, archive, manifest_path)
            self.assertEqual(result["archive_sha256"], hashlib.sha256(archive.read_bytes()).hexdigest())
            value = json.loads(manifest_path.read_text(encoding="ascii"))
            self.assertEqual(value["manifest_digest"], result["manifest_digest"])
            self.assertEqual(archive.stat().st_mode & 0o777, 0o444)
            with self.assertRaises(release.ReleaseError):
                release.build(METHOD_ROOT, archive, root / "second.json")

    def test_release_path_is_closed(self) -> None:
        self.assertEqual(len(set(release.RELEASE_FILES)), len(release.RELEASE_FILES))
        for value in release.RELEASE_FILES:
            relative = Path(value)
            self.assertFalse(relative.is_absolute())
            self.assertNotIn("..", relative.parts)
            self.assertTrue((METHOD_ROOT / relative).is_file())


if __name__ == "__main__":
    unittest.main()
