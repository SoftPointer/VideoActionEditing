from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import materialize_prospective_factorial_qwen_audit_v1 as materializer  # noqa: E402
import prospective_factorial_branch_manifest_v1 as branch_manifest  # noqa: E402
import run_prospective_factorial_branch_shard_v1 as runner  # noqa: E402


MANIFEST = METHOD_ROOT / "assets" / "prospective_factorial_branch_manifest_v1.json"
MANIFEST_SHA256 = branch_manifest.file_sha256(MANIFEST)
CELLS = ["0b2fc177202e4d08:2026082511", "1367d5595ed641ae:2026082601"]


class FactorialQwenAuditMaterializerTests(unittest.TestCase):
    def _fake_release(self, root: Path) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        rows = runner._released_entries(manifest, runner._cells(CELLS), "fit")
        (root / "entries").mkdir()
        for row in rows:
            entry = root / "entries" / row["entry_id"]
            entry.mkdir()
            output = entry / "output.mp4"
            output.write_bytes((row["entry_id"] + "\n").encode("ascii"))
            unsigned = {
                "schema_version": runner.ENTRY_RECEIPT_SCHEMA,
                "manifest_sha256": MANIFEST_SHA256,
                "manifest_digest": manifest["manifest_digest"],
                "entry_id": row["entry_id"],
                "source_id": row["source_id"],
                "seed": row["seed"],
                "branch": row["branch"],
                "executor": row["executor"],
                "source_video_sha256": row["source_video_sha256"],
                "instruction_utf8_sha256": row["instruction_utf8_sha256"],
                "output_sha256": branch_manifest.file_sha256(output),
                "native_inference_receipt_digest": None,
                "training_target_authorized": False,
                "optimizer_step_authorized": False,
                "method_success_claimed": False,
            }
            receipt = {
                **unsigned,
                "receipt_digest": branch_manifest.object_sha256(unsigned),
            }
            (entry / "entry.receipt.json").write_text(
                json.dumps(receipt), encoding="utf-8"
            )

    def test_materializes_six_supported_branches_per_cell_as_plain_copies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = root / "release"
            release.mkdir()
            self._fake_release(release)
            audit = root / "audit"

            receipt = materializer.materialize(
                manifest_path=MANIFEST,
                expected_manifest_sha256=MANIFEST_SHA256,
                cells=CELLS,
                output_root=release,
                audit_root=audit,
            )

            self.assertEqual(receipt["published_entry_count"], 12)
            self.assertEqual(receipt["omitted_branches"], ["wrong_actor_or_object"])
            self.assertFalse(receipt["authority"]["training"])
            attempts = sorted((audit / "attempts").iterdir())
            self.assertEqual(len(attempts), 12)
            for attempt in attempts:
                video = attempt / "t2v.mp4"
                original = release / "entries" / attempt.name / "output.mp4"
                self.assertNotEqual(video.stat().st_ino, original.stat().st_ino)
                self.assertEqual(video.read_bytes(), original.read_bytes())
                self.assertEqual(video.stat().st_mode & 0o777, 0o444)
                self.assertFalse(video.is_symlink())
                branch = attempt.name.rsplit("-", 1)[-1]
                # Compound branch names are easiest to recover from the root receipt.
                published = next(
                    row for row in receipt["entries"] if row["entry_id"] == attempt.name
                )
                expected_name = (
                    "saic-event-generation-receipt.json"
                    if published["branch"] in materializer.ANCHOR_BRANCHES
                    else "saic-event-topup-generation-receipt.json"
                )
                self.assertTrue((attempt / expected_name).is_file(), branch)

    def test_refuses_to_overwrite_an_existing_audit_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = root / "release"
            release.mkdir()
            self._fake_release(release)
            audit = root / "audit"
            audit.mkdir()
            with self.assertRaisesRegex(
                materializer.FactorialQwenViewError, "must be fresh"
            ):
                materializer.materialize(
                    manifest_path=MANIFEST,
                    expected_manifest_sha256=MANIFEST_SHA256,
                    cells=CELLS,
                    output_root=release,
                    audit_root=audit,
                )


if __name__ == "__main__":
    unittest.main()
