from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from methods.bernini_action_editing.tools import (
    build_source_noised_carrier_stage_b_release_v1 as release,
)


METHOD_ROOT = Path(__file__).resolve().parents[1]


class SourceNoisedCarrierStageBReleaseTests(unittest.TestCase):
    def test_real_release_is_deterministic_and_exact(self) -> None:
        manifest, payloads = release.build_manifest(METHOD_ROOT)
        first = release.build_archive(manifest, payloads)
        second = release.build_archive(manifest, payloads)
        self.assertEqual(first, second)
        release.verify_archive_bytes(first, manifest)
        self.assertEqual(manifest["revision_kind"], "content-closure-sha1")
        self.assertFalse(manifest["git_commit_claimed"])
        self.assertEqual(manifest["schema_version"], release.SCHEMA_VERSION)
        self.assertEqual(manifest["release_generation"], "r4")
        self.assertEqual(len(release.BASE_RELEASE_FILES), 9)
        self.assertEqual(len(release.STAGE_B_RELEASE_FILES), 3)
        self.assertEqual(len(release.RELEASE_FILES), 12)
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


    def test_exact_r4_base_plus_stage_b_member_closure(self) -> None:
        self.assertEqual(
            release.RELEASE_FILES[:9],
            release.BASE_RELEASE_FILES,
        )
        self.assertEqual(
            release.RELEASE_FILES[9:],
            (
                "inference_sigma_strata.py",
                "source_noised_ladder_v1.py",
                "train_source_noised_carrier_strata_v1.py",
            ),
        )

    def test_hostile_manifest_member_hash_or_path_is_rejected(self) -> None:
        import copy

        manifest, payloads = release.build_manifest(METHOD_ROOT)
        archive = release.build_archive(manifest, payloads)
        hostile_hash = copy.deepcopy(manifest)
        hostile_hash["files"][0]["sha256"] = "0" * 64
        with self.assertRaises(release.ReleaseError):
            release.verify_archive_bytes(archive, hostile_hash)

        hostile_path = copy.deepcopy(manifest)
        hostile_path["files"][0]["path"] = "../escape.py"
        with self.assertRaises(release.ReleaseError):
            release.verify_archive_bytes(archive, hostile_path)

    def test_hostile_symlink_member_is_rejected_before_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "method"
            for relative in release.RELEASE_FILES:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(relative.encode("ascii"))
            victim = root / release.RELEASE_FILES[-1]
            victim.unlink()
            victim.symlink_to(root / release.RELEASE_FILES[0])
            with self.assertRaises(release.ReleaseError):
                release.build_manifest(root.resolve())


if __name__ == "__main__":
    unittest.main()
