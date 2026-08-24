from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from tools import build_saic_source_anchor_manifest_v1 as builder  # noqa: E402


class BuildSAICSourceAnchorManifestV1Tests(unittest.TestCase):
    def _root(self, count: int, *, include_excluded: bool = True) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name) / "samples"
        root.mkdir()
        iids = [f"{index:016x}" for index in range(count)]
        if include_excluded:
            iids += sorted(builder.STRICT_ACTION_IIDS)
        for iid in iids:
            path = root / iid / "samples" / iid / "source_video.mp4"
            path.parent.mkdir(parents=True)
            path.write_bytes(f"video-{iid}".encode("ascii"))
        return temporary, root

    @staticmethod
    def _prepare(path: Path):
        iid = path.parent.name
        # One deliberately smaller population verifies largest-bucket choice.
        bucket = [512, 480] if int(iid[:2], 16) % 17 == 16 else [480, 496]
        return object(), {
            "frame_count": 81,
            "fps": 25.0,
            "reported_fps": 25.0,
            "source_derived_bucket_hw": bucket,
        }

    def test_builds_train64_holdout16_without_leakage(self) -> None:
        temporary, root = self._root(100)
        self.addCleanup(temporary.cleanup)
        first = builder.build_manifest(root, prepare_source=self._prepare)
        second = builder.build_manifest(root, prepare_source=self._prepare)
        self.assertEqual(first, second)
        self.assertEqual(first["train_count"], 64)
        self.assertEqual(first["holdout_count"], 16)
        self.assertEqual(sum(first["selected_bucket_counts"].values()), 80)
        self.assertIn("480x496", first["selected_bucket_counts"])
        train = first["train_rows"]
        heldout = first["holdout_rows"]
        train_iids = {row["iid"] for row in train}
        heldout_iids = {row["iid"] for row in heldout}
        self.assertFalse(train_iids & heldout_iids)
        self.assertFalse((train_iids | heldout_iids) & builder.STRICT_ACTION_IIDS)
        for split, rows, iids in (
            ("train", train, train_iids),
            ("holdout", heldout, heldout_iids),
        ):
            self.assertEqual({row["split"] for row in rows}, {split})
            self.assertEqual([row["row_index"] for row in rows], list(range(len(rows))))
            for row in rows:
                self.assertNotEqual(row["iid"], row["wrong_iid"])
                self.assertIn(row["wrong_iid"], iids)
                wrong = next(item for item in rows if item["iid"] == row["wrong_iid"])
                self.assertEqual(row["dp_arm"], wrong["dp_arm"])
                self.assertEqual(row["bucket_hw"], wrong["bucket_hw"])
        unsigned = dict(first)
        declared = unsigned.pop("manifest_digest")
        self.assertEqual(declared, builder.object_sha256(unsigned))

    def test_insufficient_same_bucket_pairs_fail(self) -> None:
        temporary, root = self._root(79, include_excluded=False)
        self.addCleanup(temporary.cleanup)
        with self.assertRaisesRegex(builder.SAICSourceAnchorManifestError, "fewer than 80"):
            builder.build_manifest(
                root,
                prepare_source=lambda path: (
                    object(),
                    {
                        "frame_count": 81,
                        "fps": 25.0,
                        "reported_fps": 25.0,
                        "source_derived_bucket_hw": [480, 496],
                    },
                ),
            )

    def test_exact_metadata_and_layout_are_fail_closed(self) -> None:
        temporary, root = self._root(80, include_excluded=False)
        self.addCleanup(temporary.cleanup)
        with self.assertRaisesRegex(builder.SAICSourceAnchorManifestError, "not exact81"):
            builder.build_manifest(
                root,
                prepare_source=lambda path: (
                    object(),
                    {
                        "frame_count": 81,
                        "fps": 25.0,
                        "reported_fps": 24.0,
                        "source_derived_bucket_hw": [480, 496],
                    },
                ),
            )

    def test_atomic_output_is_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "manifest.json"
            builder.atomic_write_json(output, {"a": 1})
            before = output.read_bytes()
            with self.assertRaisesRegex(builder.SAICSourceAnchorManifestError, "fresh"):
                builder.atomic_write_json(output, {"a": 2})
            self.assertEqual(output.read_bytes(), before)
            self.assertEqual(
                hashlib.sha256(before).hexdigest(),
                hashlib.sha256(builder.canonical_json_bytes({"a": 1}) + b"\n").hexdigest(),
            )

    def test_output_is_ascii_canonical_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "x.json"
            builder.atomic_write_json(output, {"z": 1, "a": 2})
            self.assertEqual(json.loads(output.read_text("ascii")), {"a": 2, "z": 1})
            self.assertEqual(output.read_bytes(), b'{"a":2,"z":1}\n')


if __name__ == "__main__":
    unittest.main()
