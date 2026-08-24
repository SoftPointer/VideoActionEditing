from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import pair_v5_t2v_calibration_bank_spec as contract  # noqa: E402


V1_PATH = METHOD_ROOT / "assets/pair_v5_t2v_calibration_core4_bank_v1.json"
V2_PATH = METHOD_ROOT / "assets/pair_v5_t2v_calibration_core4_bank_v2.json"
V2_RAW_SHA256 = "a18387b383fb11f19279c67694089754ff84b51e939e7a92b51a7e35a0743a95"
EXPECTED_CHANGED_CAPTION_HASHES = {
    ("7b88a1ca1f804f41", "shuffle"): (
        "36a55e91d1172b569101aa9f18484621249b20ea6636550da6a218bf24b1d842"
    ),
    ("841b5e0080a1441d", "shuffle"): (
        "b825a33980bc16db2fef46b3a0f07c36b2513e75f54104e3f1a4f8a4e08ff726"
    ),
    ("a35b590961d24694", "wrong_object"): (
        "7f3c10187f7a95a84870e5425d5d2cb320f9a0b02527f287b5a070cce08a84fc"
    ),
    ("a66e6818e4144928", "incomplete"): (
        "30c30afcf17c30951f39a64989289aad05a99cabcdaf14c635dbf67b5e7d2ee5"
    ),
    ("a66e6818e4144928", "shuffle"): (
        "debbf6ee20aa669ed45a955f3c9134b52db433a2f8cba9ab5b15c177510d763b"
    ),
}


def _raw_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> tuple[dict, str]:
    digest = _raw_sha256(path)
    value, actual = contract.load_sealed_spec(path.resolve(), digest)
    return value, actual


def _rows(value: dict) -> list[dict]:
    return [row for group in value["groups"] for row in group["candidates"]]


def _iid(row: dict) -> str:
    prefix = "cell-"
    suffix = f"-s{row['seed']}"
    cell = row["calibration_group_id"]
    if not cell.startswith(prefix) or not cell.endswith(suffix):
        raise AssertionError(f"unexpected calibration group: {cell}")
    return cell[len(prefix) : -len(suffix)]


def _by_coordinate(value: dict) -> dict[tuple[str, str], dict]:
    return {(_iid(row), row["semantic_branch"]): row for row in _rows(value)}


class PairV5Core4V2BankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.v1, _ = _load(V1_PATH)
        cls.v2, cls.v2_digest = _load(V2_PATH)
        cls.v1_rows = _by_coordinate(cls.v1)
        cls.v2_rows = _by_coordinate(cls.v2)

    def test_v2_is_complete_validator_compatible_core4(self) -> None:
        self.assertEqual(self.v2_digest, V2_RAW_SHA256)
        self.assertEqual(
            V2_PATH.read_bytes(),
            contract.canonical_json_bytes(json.loads(V2_PATH.read_bytes())) + b"\n",
        )
        self.assertEqual(self.v2["schema_version"], contract.SCHEMA_VERSION_V2)
        self.assertEqual(len(_rows(self.v2)), 40)
        self.assertEqual([len(group["candidates"]) for group in self.v2["groups"]], [20, 20])
        self.assertEqual(set(self.v2_rows), set(self.v1_rows))
        self.assertEqual(
            {row["analysis_split"] for row in _rows(self.v2)},
            {"fit", "confirmation"},
        )
        self.assertEqual(
            {
                (row["analysis_split"], row["action_family_id"])
                for row in _rows(self.v2)
            },
            {
                ("fit", "dog-sit-facing-camera"),
                ("confirmation", "dog-sit-facing-camera"),
                ("fit", "human-rise-to-stand"),
                ("confirmation", "human-rise-to-stand"),
            },
        )
        self.assertTrue(
            all(
                row["candidate_id"].startswith("pair5-t2v-core4-v2-")
                for row in _rows(self.v2)
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            manifest = contract.materialize_plan(
                spec_path=V2_PATH.resolve(),
                expected_sha256=V2_RAW_SHA256,
                output_dir=Path(tmp) / "v2-plan",
            )
            self.assertEqual(len(manifest["candidate_records"]), 40)
            first = manifest["candidate_records"][0]
            envelope = contract.load_candidate_envelope(
                first["path"], V2_RAW_SHA256
            )
            self.assertEqual(envelope["root_spec_raw_sha256"], V2_RAW_SHA256)

    def test_v2_change_closure_is_exactly_five_captions_and_all_ids(self) -> None:
        self.assertEqual(len(EXPECTED_CHANGED_CAPTION_HASHES), 5)
        self.assertEqual(
            {key for key in self.v2_rows if self.v2_rows[key]["full_t2v_caption"] != self.v1_rows[key]["full_t2v_caption"]},
            set(EXPECTED_CHANGED_CAPTION_HASHES),
        )
        for key, old in self.v1_rows.items():
            new = self.v2_rows[key]
            self.assertEqual(
                new["candidate_id"], old["candidate_id"].replace("core4-v1", "core4-v2")
            )
            excluded = {"candidate_id"}
            if key in EXPECTED_CHANGED_CAPTION_HASHES:
                excluded.update(
                    {"full_t2v_caption", "full_t2v_caption_utf8_sha256"}
                )
            self.assertEqual(
                {field: value for field, value in new.items() if field not in excluded},
                {field: value for field, value in old.items() if field not in excluded},
            )
            actual_caption_digest = hashlib.sha256(
                new["full_t2v_caption"].encode("utf-8")
            ).hexdigest()
            self.assertEqual(
                new["full_t2v_caption_utf8_sha256"], actual_caption_digest
            )
            if key in EXPECTED_CHANGED_CAPTION_HASHES:
                self.assertEqual(
                    actual_caption_digest, EXPECTED_CHANGED_CAPTION_HASHES[key]
                )

        normalized_v2 = json.loads(json.dumps(self.v2))
        normalized_v2["schema_version"] = contract.SCHEMA_VERSION
        for group in normalized_v2["groups"]:
            for row in group["candidates"]:
                key = (_iid(row), row["semantic_branch"])
                row["candidate_id"] = row["candidate_id"].replace(
                    "core4-v2", "core4-v1"
                )
                if key in EXPECTED_CHANGED_CAPTION_HASHES:
                    row["full_t2v_caption"] = self.v1_rows[key][
                        "full_t2v_caption"
                    ]
                    row["full_t2v_caption_utf8_sha256"] = self.v1_rows[key][
                        "full_t2v_caption_utf8_sha256"
                    ]
        self.assertEqual(normalized_v2, self.v1)

    def test_two_dog_shuffle_captions_forbid_sitting_geometry(self) -> None:
        rows = [
            row
            for row in _rows(self.v2)
            if row["action_family_id"] == "dog-sit-facing-camera"
            and row["semantic_branch"] == "shuffle"
        ]
        self.assertEqual({row["analysis_split"] for row in rows}, {"fit", "confirmation"})
        self.assertEqual(len(rows), 2)
        required_phrases = (
            "fully upright",
            "all four weight-bearing legs in every frame",
            "lateral side steps",
            "turns its head",
            "hips continuously high",
            "rear knees remain extended rather than folding",
            "never crouches, lies down, or sits",
            "last frame",
            "high-hipped four-legged standing posture",
        )
        forbidden_old_phrases = (
            "lowers briefly",
            "crouches without completing",
        )
        for row in rows:
            caption = row["full_t2v_caption"].lower()
            for phrase in required_phrases:
                self.assertIn(phrase, caption)
            for phrase in forbidden_old_phrases:
                self.assertNotIn(phrase, caption)

    def test_a66_incomplete_locks_full_body_deep_half_squat(self) -> None:
        row = self.v2_rows[("a66e6818e4144928", "incomplete")]
        caption = row["full_t2v_caption"].lower()
        for phrase in (
            "locked-off wide full-body",
            "entire body stays inside the frame",
            "both feet and both knees clearly visible in every frame",
            "rises only into a deep half-squat",
            "both knees visibly bent",
            "hips kept low",
            "torso still forward-inclined",
            "never straightens either leg or her torso",
            "never reaches an upright stand",
            "final frame",
        ):
            self.assertIn(phrase, caption)
        self.assertEqual(
            row["full_t2v_caption_utf8_sha256"],
            EXPECTED_CHANGED_CAPTION_HASHES[("a66e6818e4144928", "incomplete")],
        )
        self.assertEqual(
            self.v2_rows[("a35b590961d24694", "incomplete")][
                "full_t2v_caption_utf8_sha256"
            ],
            self.v1_rows[("a35b590961d24694", "incomplete")][
                "full_t2v_caption_utf8_sha256"
            ],
        )

    def test_a66_shuffle_stays_low_while_upper_body_moves(self) -> None:
        row = self.v2_rows[("a66e6818e4144928", "shuffle")]
        caption = row["full_t2v_caption"].lower()
        for phrase in (
            "locked-off wide full-body",
            "both feet and both knees clearly visible in every frame",
            "stays in a low deep crouch throughout",
            "turns her head left and right",
            "twists her upper torso from side to side",
            "moves both hands",
            "hips remain continuously low",
            "both knees remain deeply bent",
            "never raises her hips",
            "never straightens either leg",
            "never enters an upright standing posture",
            "final frame",
        ):
            self.assertIn(phrase, caption)
        self.assertEqual(
            row["full_t2v_caption_utf8_sha256"],
            EXPECTED_CHANGED_CAPTION_HASHES[("a66e6818e4144928", "shuffle")],
        )
        self.assertEqual(
            self.v2_rows[("a35b590961d24694", "shuffle")][
                "full_t2v_caption_utf8_sha256"
            ],
            self.v1_rows[("a35b590961d24694", "shuffle")][
                "full_t2v_caption_utf8_sha256"
            ],
        )

    def test_a35_wrong_object_only_moves_box_at_floor_level(self) -> None:
        row = self.v2_rows[("a35b590961d24694", "wrong_object")]
        caption = row["full_t2v_caption"].lower()
        for phrase in (
            "locked-off wide full-body",
            "both knees, both feet, and her low hips clearly visible in every frame",
            "one-knee kneeling pose and leg geometry completely unchanged",
            "one knee grounded",
            "other knee bent",
            "hips continuously low",
            "one hand only at floor level",
            "gently push a small box sideways and rotate it in place",
            "never lifts the box",
            "never raises her hips",
            "never extends either leg",
            "never rises, and never stands",
            "final frame",
        ):
            self.assertIn(phrase, caption)
        self.assertEqual(
            row["full_t2v_caption_utf8_sha256"],
            EXPECTED_CHANGED_CAPTION_HASHES[
                ("a35b590961d24694", "wrong_object")
            ],
        )
        self.assertEqual(
            self.v2_rows[("a66e6818e4144928", "wrong_object")][
                "full_t2v_caption_utf8_sha256"
            ],
            self.v1_rows[("a66e6818e4144928", "wrong_object")][
                "full_t2v_caption_utf8_sha256"
            ],
        )


if __name__ == "__main__":
    unittest.main()
