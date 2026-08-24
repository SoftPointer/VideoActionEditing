from __future__ import annotations

import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
PAYLOADS = {
    "P1": (
        ASSETS / "mev840_action_only_p1_event_order_v1.txt",
        "2a334405d892434b8855d1a652c577c6caedf9bf63e1e0698ee4cd1973dd994b",
    ),
    "P2": (
        ASSETS / "mev840_action_only_p2_relation_contact_v1.txt",
        "e22733bd003e77b0a914ce8a3a15f3b850285f7d721a04a8a51d81c1920e3f34",
    ),
}
FORBIDDEN = (
    "target",
    "source",
    "frame",
    "pixel",
    "rgb",
    "mask",
    "feature",
    "embedding",
    "latent",
    "gaussian",
    "query",
    "key",
    "value",
    "color",
    "colour",
    "material",
    "texture",
    "shape",
    "glass",
    "plastic",
    "red",
    "green",
    "blue",
    "white",
    "black",
    "left",
    "right",
    "top",
    "bottom",
)


class ActionOnlyPromptPayloadTests(unittest.TestCase):
    def test_exact_bytes_and_no_appearance_or_spatial_leakage(self):
        for label, (path, expected) in PAYLOADS.items():
            raw = path.read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), expected, label)
            self.assertTrue(raw.endswith(b"\n"), label)
            self.assertEqual(raw.count(b"\n"), 1, label)
            text = raw.decode("ascii").lower()
            for token in FORBIDDEN:
                self.assertNotIn(token, text, f"{label}:{token}")

    def test_payloads_encode_complementary_action_relations(self):
        p1 = PAYLOADS["P1"][0].read_text(encoding="ascii")
        p2 = PAYLOADS["P2"][0].read_text(encoding="ascii")
        self.assertIn("first turns", p1)
        self.assertIn("before she releases", p1)
        self.assertIn("contact before ending", p2)
        self.assertIn("distance increases", p2)
        self.assertIn("exactly one bottle", p2)
        self.assertIn("same bottle throughout", p2)
        self.assertIn("do not duplicate or replace it", p2)
        self.assertNotEqual(p1, p2)


if __name__ == "__main__":
    unittest.main()
