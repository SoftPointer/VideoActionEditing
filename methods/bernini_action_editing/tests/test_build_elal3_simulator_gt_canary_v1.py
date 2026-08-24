from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from tools import build_elal3_simulator_gt_canary_v1 as builder  # noqa: E402


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_gzip_json(path: Path):
    with gzip.open(str(path), "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _verify_digest(value, key):
    copied = dict(value)
    declared = copied.pop(key)
    if declared != builder.object_sha256(copied):
        raise AssertionError("invalid %s" % key)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg suite unavailable")
class ELAL3SimulatorGTBuilderTests(unittest.TestCase):
    def test_builds_exact3_row_24_video_diagnostic_packet(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output = root / "packet"
            result = builder.build_bundle(output)
            output = Path(result["output_root"])

            self.assertEqual(result["status"], builder.STATUS)
            self.assertEqual(result["row_count"], 3)
            self.assertEqual(result["media_count"], 24)
            manifest_path = output / "manifest.json"
            manifest = _load_json(manifest_path)
            _verify_digest(manifest, "manifest_digest")
            self.assertEqual(manifest["status"], "ELAL3_SIM_DIAGNOSTIC")
            self.assertEqual(manifest["row_count"], 3)
            self.assertEqual(manifest["media_count"], 24)
            self.assertEqual(manifest["c1_row_count"], 1)
            self.assertEqual(manifest["c2_row_count"], 2)
            self.assertFalse(manifest["authority"]["training_authorized"])
            self.assertTrue(manifest["authority"]["training_use_forbidden"])
            self.assertFalse(manifest["authority"]["exact160_eligible"])
            self.assertFalse(manifest["authority"]["scientific_claim_authorized"])

            expected_variants = set(builder.MEDIA_ORDER)
            video_paths = []
            for row in manifest["rows"]:
                self.assertEqual(set(row["media"]), expected_variants)
                self.assertEqual(row["negative_order"], list(builder.NEGATIVE_ORDER))
                self.assertIn(row["entity_count"], (2, 3))
                hashes = {entry["sha256"] for entry in row["media"].values()}
                self.assertEqual(len(hashes), 8)
                source_annotation = _load_gzip_json(output / row["media"]["source"]["annotation_path"])
                target_annotation = _load_gzip_json(output / row["media"]["target"]["annotation_path"])
                self.assertEqual(
                    [item["center_xy"] for item in source_annotation["frames"][0]["entities"]],
                    [item["center_xy"] for item in target_annotation["frames"][0]["entities"]],
                )
                self.assertEqual(
                    [item["center_xy"] for item in target_annotation["frames"][65]["entities"]],
                    [item["center_xy"] for item in target_annotation["frames"][80]["entities"]],
                )
                self.assertEqual(target_annotation["phase_labels"]["shape"], [21, 4])
                self.assertEqual(target_annotation["camera_transform"]["shape"], [81, 3, 3])
                self.assertEqual(
                    target_annotation["signed_tracks"]["dense_shape"],
                    [row["entity_count"], 81, builder.HEIGHT, builder.WIDTH, 2],
                )
                self.assertEqual(
                    target_annotation["visibility_confidence"]["dense_shape"],
                    [row["entity_count"], 81, builder.HEIGHT, builder.WIDTH, 2],
                )
                self.assertFalse(target_annotation["authority"]["training_authorized"])
                self.assertTrue(target_annotation["simulator_gt"])

                if row["entity_count"] == 3:
                    fractions = [
                        entity["visibility_fraction"]
                        for frame in target_annotation["frames"]
                        for entity in frame["entities"]
                    ]
                    self.assertLess(min(fractions), 1.0)

                source_signatures = {
                    item["simulator_instance_signature"]
                    for item in source_annotation["appearance"].values()
                }
                anchor_annotation = _load_gzip_json(output / row["media"]["anchor"]["annotation_path"])
                anchor_signatures = {
                    item["simulator_instance_signature"]
                    for item in anchor_annotation["appearance"].values()
                }
                self.assertTrue(anchor_annotation["appearance_disjoint_from_source"])
                self.assertTrue(source_signatures.isdisjoint(anchor_signatures))

                for variant, media in row["media"].items():
                    video = output / media["path"]
                    video_paths.append(video)
                    self.assertEqual(builder.file_sha256(video), media["sha256"])
                    self.assertEqual(media["probe"]["frame_count"], 81)
                    self.assertEqual(media["probe"]["fps_num"], 25)
                    annotation_path = output / media["annotation_path"]
                    self.assertEqual(builder.file_sha256(annotation_path), media["annotation_sha256"])
                    receipt_path = output / media["annotation_receipt_path"]
                    self.assertEqual(builder.file_sha256(receipt_path), media["annotation_receipt_sha256"])
                    receipt = _load_json(receipt_path)
                    _verify_digest(receipt, "annotation_receipt_digest")
                    self.assertEqual(receipt["media"]["sha256"], media["sha256"])
                    self.assertEqual(receipt["annotation"]["sha256"], media["annotation_sha256"])
                    self.assertFalse(receipt["authority"]["exact160_claim_authorized"])

            self.assertEqual(len(video_paths), 24)
            html_path = output / "index.html"
            html_text = html_path.read_text(encoding="utf-8")
            self.assertEqual(html_text.count("<video controls"), 24)
            self.assertIn("SIMULATOR DIAGNOSTIC ONLY", html_text)
            self.assertIn("NOT REAL TRAINING RESULTS", html_text)
            self.assertIn("exact160 CLAIMS FORBIDDEN", html_text)

            receipt_path = output / "build-receipt.json"
            receipt = _load_json(receipt_path)
            _verify_digest(receipt, "build_receipt_digest")
            self.assertEqual(receipt["manifest"]["sha256"], builder.file_sha256(manifest_path))
            self.assertEqual(receipt["review_html"]["sha256"], builder.file_sha256(html_path))
            self.assertEqual(receipt["exact_row_count"], 3)
            self.assertEqual(receipt["exact_media_count"], 24)
            self.assertFalse(receipt["authority"]["training_authorized"])
            for path in output.rglob("*"):
                self.assertFalse(path.is_symlink())
                if path.is_file():
                    self.assertEqual(stat.S_IMODE(path.stat().st_mode) & 0o222, 0)

    def test_is_byte_deterministic_and_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            first = Path(builder.build_bundle(root / "packet-a")["output_root"])
            second = Path(builder.build_bundle(root / "packet-b")["output_root"])

            first_files = {
                path.relative_to(first).as_posix(): builder.file_sha256(path)
                for path in first.rglob("*") if path.is_file()
            }
            second_files = {
                path.relative_to(second).as_posix(): builder.file_sha256(path)
                for path in second.rglob("*") if path.is_file()
            }
            self.assertEqual(first_files, second_files)
            sentinel = first / "manifest.json"
            before = builder.file_sha256(sentinel)
            with self.assertRaisesRegex(builder.ELAL3SimulatorBuilderError, "overwrite output root"):
                builder.build_bundle(first)
            self.assertEqual(builder.file_sha256(sentinel), before)

    def test_blueprint_is_closed_and_contains_only_registered_diagnostics(self):
        rows = builder._row_specs()
        self.assertEqual(len(rows), 3)
        self.assertEqual([row["entity_count"] for row in rows], [2, 3, 3])
        self.assertEqual(sum(row["gate"].startswith("C1_") for row in rows), 1)
        self.assertEqual(sum(row["gate"].startswith("C2_") for row in rows), 2)
        for row in rows:
            self.assertEqual(tuple(row["plans"]), builder.MEDIA_ORDER)
            self.assertEqual(tuple(name for name in row["plans"] if name in builder.NEGATIVE_ORDER), builder.NEGATIVE_ORDER)
            self.assertTrue(row["plans"]["anchor"]["appearance_disjoint_from_source"])
            self.assertFalse(row["plans"]["source"]["appearance_disjoint_from_source"])
            self.assertEqual(row["plans"]["source"]["phase_mode"], "none")


if __name__ == "__main__":
    unittest.main()
