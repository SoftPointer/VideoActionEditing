from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import prospective_factorial_branch_manifest_v1 as manifest_module  # noqa: E402


SCRIPT = METHOD_ROOT / "tools" / "build_prospective_factorial_branch_review_html_v1.py"
SPEC = importlib.util.spec_from_file_location("build_factorial_review_html", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
MANIFEST = METHOD_ROOT / "assets" / "prospective_factorial_branch_manifest_v1.json"


class ProspectiveFactorialBranchReviewHTMLTests(unittest.TestCase):
    def test_page_shows_exact_source_and_all_six_generated_comparisons(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cell = ("0b2fc177202e4d08", 2026082511)
        rows = [
            row
            for row in manifest["entries"]
            if (row["source_id"], row["seed"]) == cell
        ]

        page = MODULE._page(manifest, [cell], rows)

        self.assertEqual(page.count("<video "), 7)
        self.assertIn("noop / exact source", page)
        self.assertIn("wrong-actor-or-object", page)
        self.assertIn("authority closed", page)
        for branch in manifest_module.BRANCHES:
            entry = next(row["entry_id"] for row in rows if row["branch"] == branch)
            self.assertIn(f"entries/{entry}/output.mp4", page)

    def test_display_order_begins_with_noop_source(self) -> None:
        self.assertEqual(MODULE.DISPLAY_ORDER[0], "noop")
        self.assertEqual(set(MODULE.DISPLAY_ORDER), set(manifest_module.BRANCHES))


if __name__ == "__main__":
    unittest.main()
