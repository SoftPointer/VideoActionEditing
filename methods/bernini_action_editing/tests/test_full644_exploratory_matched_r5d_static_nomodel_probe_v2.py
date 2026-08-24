from __future__ import annotations

from pathlib import Path
import sys
import unittest


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import full644_exploratory_matched_r5d_static_nomodel_probe_v2 as probe


class StaticPublicationBindingV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.identity = {
            "device": 48,
            "inode": 7,
            "uid": 2012,
            "gid": 2000,
            "mode": 0o100444,
            "nlink": 1,
            "rdev": 0,
            "size": 5423,
            "blocks": 11,
            "mtime_ns": 13,
            "ctime_ns": 17,
        }
        self.input_path = Path("/release/launch/root_launch_input_auh_r5d.json")
        self.receipt_path = Path("/release/launch/root_launch_receipt_auh_r5d.json")
        self.receipt = {
            "launch_input": {
                "path": str(self.input_path),
                "sha256": "a" * 64,
                "identity": dict(self.identity),
            },
            "receipt_path": str(self.receipt_path),
            "payload_mode": 0o444,
            "named_payload_execution_forbidden": True,
            "remote_execution_authorized_by_this_receipt": False,
        }

    def validate(self, receipt: dict[str, object]) -> None:
        probe.validate_launch_publication_binding(
            receipt,
            input_path=self.input_path,
            input_sha256="a" * 64,
            input_identity=self.identity,
            receipt_path=self.receipt_path,
        )

    def test_accepts_frozen_launcher_row_without_duplicate_size(self) -> None:
        self.validate(self.receipt)

    def test_rejects_duplicate_top_level_size_and_identity_tamper(self) -> None:
        extra = dict(self.receipt)
        extra["launch_input"] = dict(self.receipt["launch_input"], size=5423)
        with self.assertRaises(probe.R5DStaticProbeError):
            self.validate(extra)

        tampered = dict(self.receipt)
        identity = dict(self.identity)
        identity["size"] += 1
        tampered["launch_input"] = dict(
            self.receipt["launch_input"], identity=identity
        )
        with self.assertRaises(probe.R5DStaticProbeError):
            self.validate(tampered)


if __name__ == "__main__":
    unittest.main()
