from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from tools import author_appearance_identity_orbit_bundle as author


ASSET = METHOD_ROOT / "assets" / "appearance_identity_orbit_portrait2_review_v1.json"
PROTOCOL = (
    METHOD_ROOT / "assets" / "appearance_identity_orbit_full81_qualification_v1.json"
)
EXPECTED_PROTOCOL_SHA256 = (
    "6adfdff2830a4c7f1aaf2f0244c234a608509f6d3f1b0b3a2c0bca9b84982c09"
)


class AppearanceIdentityOrbitAuthoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.decisions = json.loads(ASSET.read_text(encoding="ascii"))

    def test_checked_in_review_is_closed_and_binds_protocol(self) -> None:
        qualifier, protocol_sha, rows = author._validate_decisions(self.decisions)
        self.assertEqual(qualifier, "codex-full81-grid-review-20260808-v1")
        self.assertEqual(protocol_sha, EXPECTED_PROTOCOL_SHA256)
        self.assertEqual([row["iid"] for row in rows], [
            "0014a41e55e44670",
            "00435ad621c44fac",
        ])
        self.assertEqual(
            hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(), EXPECTED_PROTOCOL_SHA256
        )
        self.assertEqual(
            [(row["variant_a"]["native_arm"], row["variant_b"]["native_arm"]) for row in rows],
            [("rv2v", "rv2v"), ("rv2v", "r2v")],
        )
        self.assertTrue(author.RECEIPT_SCHEMA.endswith("-v2"))
        self.assertEqual(author.orbit.SPEC_SCHEMA.rsplit("-", 1)[-1], "v3")
        self.assertEqual(
            author.orbit.reference_encoding_contract()["reference_rgb_indices"],
            [0, 27, 53, 80],
        )
        self.assertEqual(
            author.orbit.reference_encoding_contract()[
                "independent_vae_encode_calls_per_row"
            ],
            15,
        )

    def test_false_gate_fails_closed(self) -> None:
        value = copy.deepcopy(self.decisions)
        value["rows"][0]["qualification_gates"]["variant_a"][
            "motion_phase_and_order_preserved"
        ] = False
        with self.assertRaisesRegex(author.OrbitAuthoringError, "missing/false/extra"):
            author._validate_decisions(value)

    def test_extra_field_fails_closed(self) -> None:
        value = copy.deepcopy(self.decisions)
        value["rows"][0]["variant_a"]["prompt"] = "unbound mutable text"
        with self.assertRaisesRegex(author.OrbitAuthoringError, "field closure differs"):
            author._validate_decisions(value)

    def test_downstream_result_visibility_is_forbidden(self) -> None:
        value = copy.deepcopy(self.decisions)
        value["downstream_training_results_seen"] = True
        with self.assertRaisesRegex(author.OrbitAuthoringError, "blind"):
            author._validate_decisions(value)


if __name__ == "__main__":
    unittest.main()
