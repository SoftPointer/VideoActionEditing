from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import full644_exploratory_matched_r5d_static_nomodel_probe_v3 as probe


class StaticIdentityRowV3Tests(unittest.TestCase):
    def test_replays_actual_launcher_three_field_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "identity.bin"
            raw = b"frozen launcher identity\n"
            path.write_bytes(raw)
            info = path.stat()
            row = {
                "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "identity": probe._identity(info),
            }
            probe.replay_identity_row(row)

            extra = dict(row, size=len(raw))
            with self.assertRaises(probe.R5DStaticProbeError):
                probe.replay_identity_row(extra)

            tampered = dict(row)
            identity = dict(row["identity"])
            identity["size"] += 1
            tampered["identity"] = identity
            with self.assertRaises(probe.R5DStaticProbeError):
                probe.replay_identity_row(tampered)


if __name__ == "__main__":
    unittest.main()
