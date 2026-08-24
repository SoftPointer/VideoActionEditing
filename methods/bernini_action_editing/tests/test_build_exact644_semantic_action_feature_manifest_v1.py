from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from methods.bernini_action_editing import (
    build_exact644_semantic_action_feature_manifest_v1 as builder,
)


def _fake_release() -> dict:
    rows = []
    for index in range(builder.ROW_COUNT):
        iid = f"{index:016x}"
        rows.append(
            {
                "iid": iid,
                "group_id": hashlib.sha256(f"group:{iid}".encode()).hexdigest(),
                "family": f"family-{index % 28}",
                "instruction_sha256": hashlib.sha256(
                    f"instruction:{iid}".encode()
                ).hexdigest(),
                "strict_selection_gates_all_true": index < 359,
                "media_binding": {
                    "frame_count": 81,
                    "fps": 25.0,
                    "source_video_path": f"/data/{iid}/source.mp4",
                    "source_video_sha256": hashlib.sha256(
                        f"source:{iid}".encode()
                    ).hexdigest(),
                    "action_anchor_video_path": f"/data/{iid}/anchor.mp4",
                    "action_anchor_video_sha256": hashlib.sha256(
                        f"anchor:{iid}".encode()
                    ).hexdigest(),
                },
            }
        )
    return {
        "schema_version": builder.INPUT_SCHEMA,
        "manifest_digest": builder.INPUT_MANIFEST_DIGEST,
        "row_count": builder.ROW_COUNT,
        "paired_ground_truth_claimed": False,
        "rows": rows,
    }


class Exact644FeatureManifestTest(unittest.TestCase):
    def _write(self, value: dict) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "source.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def _build(self, value: dict) -> dict:
        path = self._write(value)
        with mock.patch.object(builder, "INPUT_FILE_SHA256", builder.file_sha256(path)):
            return builder.build_manifest(path)

    def test_exact1288_role_closed_population(self) -> None:
        result = self._build(_fake_release())
        self.assertEqual(result["counts"]["total"], 1288)
        self.assertEqual(result["counts"]["unique_base_clips"], 644)
        self.assertFalse(result["formal_training_authorized"])
        self.assertFalse(result["paired_ground_truth_claimed"])
        self.assertEqual(len(result["items"]), 1288)
        self.assertEqual(
            {row["group"] for row in result["items"]},
            {"exact644_source", "exact644_action_anchor"},
        )
        for row in result["items"]:
            self.assertFalse(row["metadata"]["paired_ground_truth_claimed"])

    def test_duplicate_media_path_fails(self) -> None:
        source = _fake_release()
        source["rows"][1]["media_binding"]["source_video_path"] = source["rows"][0][
            "media_binding"
        ]["source_video_path"]
        with self.assertRaisesRegex(ValueError, "unique absolute"):
            self._build(source)

    def test_paired_claim_fails(self) -> None:
        source = _fake_release()
        source["paired_ground_truth_claimed"] = True
        with self.assertRaisesRegex(ValueError, "must not claim paired"):
            self._build(source)

    def test_exact81_contract_fails_closed(self) -> None:
        source = _fake_release()
        source["rows"][0]["media_binding"]["frame_count"] = 80
        with self.assertRaisesRegex(ValueError, "exact81"):
            self._build(source)


if __name__ == "__main__":
    unittest.main()
