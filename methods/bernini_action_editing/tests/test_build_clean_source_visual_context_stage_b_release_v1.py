from __future__ import annotations

import hashlib
import json
import ast
from pathlib import Path
import tarfile
import tempfile
import unittest

from methods.bernini_action_editing.tools import (
    build_clean_source_visual_context_stage_b_release_v1 as release,
)


METHOD_ROOT = Path(__file__).resolve().parents[1]


class CleanSourceVisualContextReleaseTests(unittest.TestCase):
    def test_release_is_deterministic_exact_ustar(self) -> None:
        manifest, payloads = release.build_manifest(METHOD_ROOT)
        first = release.build_archive(manifest, payloads)
        second = release.build_archive(manifest, payloads)
        self.assertEqual(first, second)
        release.verify_archive(first, manifest)
        self.assertEqual(manifest["schema_version"], release.SCHEMA_VERSION)
        self.assertEqual(manifest["file_count"], len(release.RELEASE_FILES))
        self.assertEqual(
            [row["path"] for row in manifest["files"]],
            list(release.RELEASE_FILES),
        )
        self.assertEqual(
            manifest["manifest_digest"],
            release.object_sha256(
                {key: value for key, value in manifest.items() if key != "manifest_digest"}
            ),
        )
        with tarfile.open(fileobj=__import__("io").BytesIO(first), mode="r:") as tar:
            for member in tar.getmembers():
                self.assertEqual(member.uid, 0)
                self.assertEqual(member.gid, 0)
                self.assertEqual(member.mtime, 0)
                self.assertEqual(member.mode & 0o777, 0o444)

    def test_create_only_release_binds_revision_and_file_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            archive = root / "method.tar"
            manifest_path = root / "method.manifest.json"
            result = release.build(METHOD_ROOT, archive, manifest_path)
            manifest = json.loads(manifest_path.read_text(encoding="ascii"))
            self.assertEqual(result["archive_sha256"], hashlib.sha256(archive.read_bytes()).hexdigest())
            self.assertEqual(result["content_closure_sha1"], manifest["content_closure_sha1"])
            self.assertEqual(archive.stat().st_mode & 0o777, 0o444)
            self.assertEqual(manifest_path.stat().st_mode & 0o777, 0o444)
            with self.assertRaises(release.CleanSourceVisualReleaseError):
                release.build(METHOD_ROOT, archive, root / "second.json")

    def test_required_runtime_and_pair_members_are_present(self) -> None:
        required = {
            "train_clean_source_visual_context_stage_b_v1.py",
            "clean_source_visual_context_pair_controller_v1.py",
            "clean_source_visual_context_stage_b_contract_v1.py",
            "scripts/auh_preservation_rank_cache_exec_v1.sh",
            "scripts/auh_train_clean_source_visual_context_stage_b_holder_v1.sh",
        }
        self.assertTrue(required.issubset(release.RELEASE_FILES))
        for relative in release.RELEASE_FILES:
            self.assertTrue((METHOD_ROOT / relative).is_file())
        tree = ast.parse(
            (METHOD_ROOT / "train_clean_source_visual_context_stage_b_v1.py").read_text(
                encoding="utf-8"
            )
        )
        declared = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "METHOD_RELEASE_FILES"
                for target in node.targets
            ):
                declared = ast.literal_eval(node.value)
        self.assertEqual(declared, release.RELEASE_FILES)


if __name__ == "__main__":
    unittest.main()
