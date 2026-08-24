from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_preservation_recovery_top_index_v1 as builder  # noqa: E402


class PreservationRecoveryTopIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        for entry in builder.ENTRIES:
            directory = self.root / entry["relative_dir"]
            directory.mkdir()
            (directory / "index.html").write_text(
                f"<!doctype html><title>{entry['marker']}</title><h1>{entry['marker']}</h1>",
                encoding="utf-8",
            )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_links_three_fixed_relative_reviews_and_marks_training(self) -> None:
        output = builder.build(review_root=self.root)
        self.assertEqual(output, self.root / "index.html")
        page = output.read_text(encoding="utf-8")
        self.assertEqual(page.count('class="card"'), 3)
        for expected in (
            'href="F0_checkpoint_dynamics/index.html"',
            'href="A0_native_v_axis/index.html"',
            'href="A1_source_edge_formal_grid/index.html"',
            "旧两条 real-source preservation-residual 连续 checkpoint dynamics",
            "Frozen native full-video V-axis causal probe",
            "Frozen formal schedule × block source-edge localization",
            "TRAINED",
            "FROZEN / NO TRAINING",
            "F0 是旧方法训练动态；A0 与 A1 是 frozen-model causal localization",
        ):
            self.assertIn(expected, page)
        self.assertEqual(page.count(">TRAINED<"), 1)
        self.assertEqual(page.count(">FROZEN / NO TRAINING<"), 2)

    def test_missing_fixed_child_fails_without_top_index(self) -> None:
        missing = self.root / "A1_source_edge_formal_grid" / "index.html"
        missing.unlink()
        with self.assertRaisesRegex(builder.PreservationTopIndexError, "cannot read A1"):
            builder.build(review_root=self.root)
        self.assertFalse((self.root / "index.html").exists())

    def test_wrong_child_experiment_marker_is_rejected(self) -> None:
        child = self.root / "A0_native_v_axis" / "index.html"
        child.write_text("<!doctype html><h1>unrelated experiment</h1>", encoding="utf-8")
        with self.assertRaisesRegex(builder.PreservationTopIndexError, "identity differs"):
            builder.build(review_root=self.root)
        self.assertFalse((self.root / "index.html").exists())

    def test_existing_top_index_is_never_overwritten(self) -> None:
        existing = self.root / "index.html"
        existing.write_text("keep-me", encoding="utf-8")
        with self.assertRaisesRegex(builder.PreservationTopIndexError, "fresh"):
            builder.build(review_root=self.root)
        self.assertEqual(existing.read_text(encoding="utf-8"), "keep-me")

    def test_symlink_child_is_rejected(self) -> None:
        child = self.root / "A1_source_edge_formal_grid" / "index.html"
        target = self.root / "target.html"
        target.write_text(builder.ENTRIES[2]["marker"], encoding="utf-8")
        child.unlink()
        try:
            child.symlink_to(target)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        with self.assertRaisesRegex(builder.PreservationTopIndexError, "symlink"):
            builder.build(review_root=self.root)
        self.assertFalse((self.root / "index.html").exists())


if __name__ == "__main__":
    unittest.main()
