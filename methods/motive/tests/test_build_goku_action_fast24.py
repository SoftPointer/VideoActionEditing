from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from methods.motive.scripts import build_goku_action_fast24 as fast24


class BuildGokuActionFast24Tests(unittest.TestCase):
    def _parent(self, root: Path) -> Path:
        selected = {
            iid for shard in fast24.IIDS_BY_SHARD for iid in shard
        }
        filler_count = fast24.PARENT_ROWS - len(selected)
        rows = [
            {
                "iid": iid,
                "eligible": True,
                "selected": True,
                "payload": index,
            }
            for index, iid in enumerate(sorted(selected))
        ]
        rows.extend(
            {
                "iid": f"filler-{index:03d}",
                "eligible": True,
                "selected": True,
            }
            for index in range(filler_count)
        )
        parent = root / "parent.jsonl"
        parent.write_bytes(
            b"".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
                + b"\n"
                for row in rows
            )
        )
        return parent

    def test_frozen_real_parent_builds_exact_balanced_pool(self) -> None:
        parent = Path("/private/tmp/goku_action_parent123_20260731.jsonl")
        if not parent.is_file():
            self.skipTest("read-only real parent fixture is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fast24.jsonl"
            result = fast24.build_fast_pool(parent, output)
            self.assertEqual(result["rows"], 24)
            self.assertEqual(result["shard_counts"], [3] * 8)
            self.assertFalse(result["generation_authorized"])
            self.assertEqual(len(output.read_bytes().splitlines()), 24)

    def test_rejects_wrong_parent_and_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = self._parent(root)
            output = root / "fast24.jsonl"
            with self.assertRaisesRegex(fast24.FastPoolError, "SHA-256"):
                fast24.build_fast_pool(parent, output)
            output.write_text("occupied\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                fast24.build_fast_pool(parent, output)

    def test_declared_iids_are_unique_and_match_hash_shards(self) -> None:
        flattened = [iid for shard in fast24.IIDS_BY_SHARD for iid in shard]
        self.assertEqual(len(flattened), 24)
        self.assertEqual(len(set(flattened)), 24)
        for shard_index, iids in enumerate(fast24.IIDS_BY_SHARD):
            self.assertEqual(len(iids), 3)
            for iid in iids:
                self.assertEqual(fast24._iid_shard(iid), shard_index)


if __name__ == "__main__":
    unittest.main()
