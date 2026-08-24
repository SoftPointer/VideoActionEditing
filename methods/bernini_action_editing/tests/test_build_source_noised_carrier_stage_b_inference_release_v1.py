from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from methods.bernini_action_editing.tools import (
    build_source_noised_carrier_stage_b_inference_release_v1 as inference_release,
)
from methods.bernini_action_editing.tools import (
    build_source_noised_carrier_stage_b_release_v1 as training_release,
)


METHOD_ROOT = Path(__file__).resolve().parents[1]


def _write_base(root: Path) -> tuple[Path, Path]:
    manifest, payloads = training_release.build_manifest(METHOD_ROOT)
    archive_raw = training_release.build_archive(manifest, payloads)
    manifest_raw = training_release.canonical_json_bytes(manifest) + b"\n"
    archive = (root / "stage-b-r4.tar").resolve()
    manifest_path = (root / "stage-b-r4.manifest.json").resolve()
    archive.write_bytes(archive_raw)
    manifest_path.write_bytes(manifest_raw)
    return archive, manifest_path


class StageBInferenceReleaseTests(unittest.TestCase):
    def test_audited_base_and_frozen_inference_form_exact_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base_archive, base_manifest = _write_base(Path(directory))
            self.assertEqual(
                hashlib.sha256(base_archive.read_bytes()).hexdigest(),
                inference_release.BASE_ARCHIVE_SHA256,
            )
            self.assertEqual(
                hashlib.sha256(base_manifest.read_bytes()).hexdigest(),
                inference_release.BASE_MANIFEST_SHA256,
            )
            manifest, payloads = inference_release.build_manifest(
                METHOD_ROOT, base_archive, base_manifest
            )
            first = inference_release.build_archive(manifest, payloads)
            second = inference_release.build_archive(manifest, payloads)
            self.assertEqual(first, second)
            inference_release.verify_archive_bytes(first, manifest)
            self.assertEqual(manifest["file_count"], 14)
            self.assertEqual(
                [row["path"] for row in manifest["files"]],
                list(inference_release.RELEASE_FILES),
            )
            self.assertEqual(
                hashlib.sha256(payloads["source_self_runtime.py"]).hexdigest(),
                "62df125ac130697b03aaea167b17a02d7fcb9d766a72f0bef71037924114e59f",
            )
            self.assertEqual(
                hashlib.sha256(payloads["infer_source_noised_carrier_stage_b_v1.py"]).hexdigest(),
                "b21f7f85531fd7f41f1a9741894b26b564b25054da418d7989f2f7a588a6f84f",
            )

    def test_publication_is_fresh_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_archive, base_manifest = _write_base(root)
            archive = (root / "inference.tar").resolve()
            manifest = (root / "inference.manifest.json").resolve()
            result = inference_release.build(
                METHOD_ROOT, base_archive, base_manifest, archive, manifest
            )
            self.assertEqual(result["file_count"], 14)
            self.assertEqual(result["archive_sha256"], hashlib.sha256(archive.read_bytes()).hexdigest())
            self.assertEqual(archive.stat().st_mode & 0o777, 0o444)
            with self.assertRaises(inference_release.ReleaseError):
                inference_release.build(
                    METHOD_ROOT, base_archive, base_manifest, archive, (root / "second.json").resolve()
                )

    def test_synchronized_hostile_base_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_archive, base_manifest = _write_base(root)
            value = json.loads(base_manifest.read_text(encoding="ascii"))
            hostile = copy.deepcopy(value)
            hostile["files"][1]["sha256"] = "0" * 64
            hostile_path = (root / "hostile.json").resolve()
            hostile_path.write_text(
                json.dumps(hostile, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="ascii",
            )
            with self.assertRaises(inference_release.ReleaseError):
                inference_release.build_manifest(METHOD_ROOT, base_archive, hostile_path)


if __name__ == "__main__":
    unittest.main()
