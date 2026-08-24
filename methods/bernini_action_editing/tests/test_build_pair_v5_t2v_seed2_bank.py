from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import pair_v5_t2v_calibration_bank_spec as contract  # noqa: E402
from tools import build_pair_v5_t2v_seed2_bank as topup  # noqa: E402


SOURCE = METHOD_ROOT / "assets" / "pair_v5_t2v_calibration_core4_bank_v2.json"


class PairV5T2VSeed2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = json.loads(SOURCE.read_text(encoding="utf-8"))

    def _reserve_like_source(self) -> dict:
        result = copy.deepcopy(self.source)
        profile = topup.REPLICATION_PROFILES["reserve4-v1"]
        core_seeds = sorted(topup.SEED_MAP)
        reserve_seeds = sorted(profile.seed_map)
        seed_rewrite = dict(zip(core_seeds, reserve_seeds))
        for group in result["groups"]:
            for row in group["candidates"]:
                old_seed = row["seed"]
                new_seed = seed_rewrite[old_seed]
                row["seed"] = new_seed
                row["candidate_id"] = (
                    profile.source_id_prefix
                    + row["candidate_id"][len(topup.SOURCE_ID_PREFIX) :]
                )
                old_suffix = f"-s{old_seed}"
                self.assertTrue(row["calibration_group_id"].endswith(old_suffix))
                row["calibration_group_id"] = (
                    row["calibration_group_id"][: -len(old_suffix)]
                    + f"-s{new_seed}"
                )
        contract.validate_root_spec(result)
        return result

    def test_pinned_source_hash_and_deterministic_derivation(self) -> None:
        self.assertEqual(topup.file_sha256(SOURCE), topup.SOURCE_SPEC_SHA256)
        first = topup.derive_seed2_spec(self.source)
        second = topup.derive_seed2_spec(self.source)
        self.assertEqual(first, second)
        contract.validate_root_spec(first)
        rows = [row for group in first["groups"] for row in group["candidates"]]
        self.assertEqual(len(rows), 40)
        self.assertEqual({row["seed"] for row in rows}, set(topup.SEED_MAP.values()))
        self.assertTrue(
            all(row["candidate_id"].startswith("pair5-t2v-core4-seed2-") for row in rows)
        )
        self.assertTrue(
            all(row["calibration_group_id"].endswith(f"-s{row['seed']}") for row in rows)
        )

    def test_only_three_candidate_fields_change(self) -> None:
        derived = topup.derive_seed2_spec(self.source)
        mutable = {"candidate_id", "calibration_group_id", "seed"}
        for old_group, new_group in zip(self.source["groups"], derived["groups"]):
            self.assertEqual(old_group["group_id"], new_group["group_id"])
            self.assertEqual(old_group["visible_gpus"], new_group["visible_gpus"])
            for old, new in zip(old_group["candidates"], new_group["candidates"]):
                self.assertEqual(
                    {key: value for key, value in old.items() if key not in mutable},
                    {key: value for key, value in new.items() if key not in mutable},
                )

    def test_reserve4_profile_changes_only_seed_bearing_fields(self) -> None:
        source = self._reserve_like_source()
        profile = topup.REPLICATION_PROFILES["reserve4-v1"]
        first = topup.derive_seed2_spec(source, "reserve4-v1")
        second = topup.derive_seed2_spec(source, "reserve4-v1")
        self.assertEqual(first, second)
        rows = [row for group in first["groups"] for row in group["candidates"]]
        self.assertEqual(len(rows), 40)
        self.assertEqual({row["seed"] for row in rows}, set(profile.seed_map.values()))
        self.assertTrue(
            all(row["candidate_id"].startswith(profile.topup_id_prefix) for row in rows)
        )
        mutable = {"candidate_id", "calibration_group_id", "seed"}
        for old_group, new_group in zip(source["groups"], first["groups"]):
            for old, new in zip(old_group["candidates"], new_group["candidates"]):
                self.assertEqual(
                    {key: value for key, value in old.items() if key not in mutable},
                    {key: value for key, value in new.items() if key not in mutable},
                )

    def test_unregistered_seed_and_authority_fail_closed(self) -> None:
        changed = copy.deepcopy(self.source)
        changed["groups"][0]["candidates"][0]["seed"] = 7
        with self.assertRaises(topup.PairV5T2VSeed2Error):
            topup.derive_seed2_spec(changed)
        with self.assertRaises(topup.PairV5T2VSeed2Error):
            topup._load_source(SOURCE.resolve(), "0" * 64)

    def test_cli_writes_once_and_replays_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve() / "seed2.json"
            result = topup.main(
                [
                    "--source-spec",
                    str(SOURCE.resolve()),
                    "--expected-source-sha256",
                    topup.SOURCE_SPEC_SHA256,
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(result, 0)
            self.assertTrue(output.is_file())
            contract.validate_root_spec(json.loads(output.read_text(encoding="utf-8")))
            with self.assertRaises(topup.PairV5T2VSeed2Error):
                topup.main(
                    [
                        "--source-spec",
                        str(SOURCE.resolve()),
                        "--expected-source-sha256",
                        topup.SOURCE_SPEC_SHA256,
                        "--output",
                        str(output),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
