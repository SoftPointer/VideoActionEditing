from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import repair_expanded_action_bank_flow_manifest_v1 as repair


class RepairManifestTests(unittest.TestCase):
    def test_repair_requires_and_records_matching_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            flow = root / "replacement.safetensors"
            flow.write_bytes(b"replacement-flow")
            flow.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "latent_hw": [84, 44],
                        "source_sha256": "a" * 64,
                        "anchor_sha256": "b" * 64,
                    }
                ),
                encoding="utf-8",
            )
            manifest = {
                "schema_version": repair.SCHEMA,
                "rows": [
                    {
                        "iid": "row",
                        "source_video_sha256": "a" * 64,
                        "anchor_video_sha256": "b" * 64,
                        "flow_bundle": "/old/flow.safetensors",
                        "flow_bundle_sha256": "c" * 64,
                        "latents": {"shape": [1, 16, 21, 84, 44]},
                    }
                ],
            }
            manifest["manifest_digest"] = repair.object_sha256(manifest)
            source = root / "manifest.json"
            source.write_text(repair.canonical(manifest) + "\n", encoding="utf-8")
            output = root / "repaired.json"
            argv = [
                "repair",
                "--input-manifest",
                str(source),
                "--iid",
                "row",
                "--flow-bundle",
                str(flow),
                "--output",
                str(output),
            ]
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(repair.main(), 0)
            value = json.loads(output.read_text(encoding="utf-8"))
            stored = value.pop("manifest_digest")
            self.assertEqual(repair.object_sha256(value), stored)
            self.assertEqual(value["rows"][0]["flow_bundle"], str(flow.resolve()))
            self.assertEqual(value["flow_geometry_repairs"][0]["iid"], "row")


if __name__ == "__main__":
    unittest.main()
