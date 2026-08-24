from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from motive.prepare_action_run import _inventory


class PrepareActionRunTests(unittest.TestCase):
    def test_inventory_is_content_bound_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "input"
            root.mkdir()
            (root / "b.bin").write_bytes(b"b")
            (root / "a.bin").write_bytes(b"a")
            rows_a, digest_a = _inventory(root)
            rows_b, digest_b = _inventory(root)
            self.assertEqual(
                [row["path"] for row in rows_a],
                ["a.bin", "b.bin"],
            )
            self.assertEqual(rows_a, rows_b)
            self.assertEqual(digest_a, digest_b)

            (root / "a.bin").write_bytes(b"changed")
            rows_c, digest_c = _inventory(root)
            self.assertNotEqual(
                json.dumps(rows_c, sort_keys=True),
                json.dumps(rows_a, sort_keys=True),
            )
            self.assertNotEqual(digest_c, digest_a)


if __name__ == "__main__":
    unittest.main()
