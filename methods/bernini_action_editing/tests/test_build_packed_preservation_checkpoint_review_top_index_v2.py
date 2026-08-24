#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import build_packed_preservation_checkpoint_review_top_index_v2 as top


class TopIndexTests(unittest.TestCase):
    def _root(self, parent: Path) -> Path:
        root = parent / "review"
        root.mkdir()
        for scope, (label, parameters) in top.LANES.items():
            lane = root / scope
            lane.mkdir()
            (lane / "index.html").write_text(f"<html>{label}</html>", "utf-8")
            evidence = {
                "complete": True,
                "trajectory_label": label,
                "trainable_parameters": parameters,
                "real_source_video_count": 64,
                "logical_training_record_count": 640,
                "optimizer_update_count": 80,
                "global_batch_size": 8,
                "training_histogram": {"noop": 256, "cube": 128, "speed": 128, "tube": 128},
                "training_authority": {"lora_scope": scope},
                "quality_claimed": False,
                "manual_review_pending": True,
                "evidence_digest": scope,
            }
            (lane / "evidence.json").write_bytes(top._canonical(evidence) + b"\n")
        return root

    def test_builds_two_explicit_relative_lane_links(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = self._root(Path(raw).resolve())
            result = top.build_top_index(root)
            page = Path(result["index"]).read_text("utf-8")
            self.assertIn('href="all-attention/index.html"', page)
            self.assertIn('href="self-attention/index.html"', page)
            self.assertIn("All-attention main", page)
            self.assertIn("Self-attention control", page)
            self.assertIn("188,946,432", page)
            self.assertIn("94,574,592", page)
            self.assertIn("same 64 real source videos", page)
            self.assertIn("No automatic ranking or candidate selection", page)

    def test_rejects_scope_mislabeled_as_other_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = self._root(Path(raw).resolve())
            path = root / "self-attention" / "evidence.json"
            value = json.loads(path.read_text("ascii"))
            value["training_authority"]["lora_scope"] = "all-attention"
            path.write_bytes(top._canonical(value) + b"\n")
            with self.assertRaisesRegex(top.TopIndexError, "self-attention evidence"):
                top.build_top_index(root)


if __name__ == "__main__":
    unittest.main()
