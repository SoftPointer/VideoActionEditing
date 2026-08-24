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

import build_saic_reversible_source_set_v1 as source_set  # noqa: E402
import saic_pure_t2v_event_bank_v1 as v1  # noqa: E402
import saic_pure_t2v_event_bank_topup_v2 as topup  # noqa: E402


EXPECTED_V2_RAW_SHA256 = (
    "d693d0784530f007888e2825d15db3db808fdf4f1d111b5d080d968c894ff145"
)
EXPECTED_V2_CONTENT_SHA256 = (
    "af2dfc387a96ade19518c5bb5313d9485683510cdbd80a4f63b1cb0746683065"
)


class SAICPureT2VEventBankTopupV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = json.loads(v1.ASSET_PATH.read_text(encoding="ascii"))
        self.spec = json.loads(topup.ASSET_PATH.read_text(encoding="ascii"))
        self.source = source_set.load_manifest()

    def test_v1_asset_is_immutable_before_and_after_v2_authoring(self) -> None:
        before = v1.file_sha256(v1.ASSET_PATH)
        self.assertEqual(before, topup.BASE_V1_SPEC_RAW_SHA256)
        authored = topup.author_spec()
        self.assertEqual(authored, self.spec)
        self.assertEqual(v1.file_sha256(v1.ASSET_PATH), before)
        self.assertEqual(v1.object_sha256(self.base), topup.BASE_V1_SPEC_CONTENT_SHA256)

    def test_checked_asset_is_exactly_sixty_non_authorizing_topups(self) -> None:
        summary = topup.validate_spec(self.spec)
        self.assertEqual(topup.file_sha256(topup.ASSET_PATH), EXPECTED_V2_RAW_SHA256)
        self.assertEqual(summary["spec_content_sha256"], EXPECTED_V2_CONTENT_SHA256)
        self.assertEqual(summary["candidate_count"], 60)
        self.assertEqual(summary["row_count"], 8)
        self.assertEqual(summary["seed_cell_count"], 20)
        self.assertTrue(summary["top_up_only"])
        self.assertFalse(summary["event_verified"])
        self.assertFalse(summary["optimizer_authorized"])
        self.assertEqual(self.spec["branch_order"], list(topup.BRANCH_ORDER))
        self.assertEqual(
            self.spec["merged_branch_order"], list(topup.MERGED_BRANCH_ORDER)
        )
        serialized = topup.canonical_json_bytes(self.spec)
        for row in self.source["rows"]:
            self.assertNotIn(row["source_video"].encode("ascii"), serialized)
        self.assertFalse(self.spec["semantic_input_closure"]["real_source_rgb_read"])
        self.assertFalse(
            self.spec["semantic_input_closure"]["real_source_latent_read_or_created"]
        )
        self.assertFalse(
            self.spec["semantic_input_closure"]["real_source_noise_read_or_created"]
        )
        self.assertFalse(self.spec["semantic_input_closure"]["target_video"])
        self.assertFalse(self.spec["semantic_input_closure"]["motion_donor"])

    def test_v1_and_v2_merge_into_twenty_exact_six_branch_cells(self) -> None:
        merged = topup.merge_six_branch_cells(self.base, self.spec)
        self.assertEqual(len(merged), 20)
        for (iid, seed), rows in merged.items():
            self.assertEqual(len(rows), 6)
            self.assertEqual(
                [candidate["branch"] for candidate in rows],
                list(topup.MERGED_BRANCH_ORDER),
            )
            base_forward = rows[0]
            for candidate in rows[3:]:
                self.assertEqual(candidate["iid"], iid)
                self.assertEqual(candidate["seed"], seed)
                self.assertEqual(
                    candidate["identity_scene_caption"],
                    base_forward["identity_scene_caption"],
                )
                self.assertEqual(
                    candidate["branch_start_state_caption"],
                    base_forward["branch_start_state_caption"],
                )
                self.assertEqual(
                    candidate["source_geometry_hw"], base_forward["source_geometry_hw"]
                )
                self.assertFalse(candidate["event_verified"])
                self.assertFalse(candidate["seed_selection_authorized"])
                self.assertFalse(candidate["training_target_authorized"])
                self.assertFalse(candidate["optimizer_authorized"])

    def test_fit_has_two_and_confirmation_has_three_seeds_without_selection(self) -> None:
        cells: dict[str, set[int]] = {}
        candidates = [
            candidate
            for group in self.spec["groups"]
            for candidate in group["candidates"]
        ]
        for candidate in candidates:
            cells.setdefault(candidate["iid"], set()).add(candidate["seed"])
        split_by_iid = {row["iid"]: row["analysis_split"] for row in self.source["rows"]}
        self.assertEqual(len(cells), 8)
        for iid, seeds in cells.items():
            self.assertEqual(len(seeds), 2 if split_by_iid[iid] == "fit" else 3)
        self.assertEqual(
            {candidate["branch"] for candidate in candidates}, set(topup.BRANCH_ORDER)
        )
        self.assertTrue(all("saic-topup-v2-" in c["candidate_id"] for c in candidates))

    def test_mutation_or_added_authority_fails_closed(self) -> None:
        attacks = []
        changed = copy.deepcopy(self.spec)
        changed["groups"][0]["candidates"][0]["event_verified"] = True
        attacks.append(changed)
        changed = copy.deepcopy(self.spec)
        changed["groups"][0]["candidates"][0]["branch_instruction"] += " Changed."
        attacks.append(changed)
        changed = copy.deepcopy(self.spec)
        changed["groups"][0]["candidates"][0]["extra"] = "open schema"
        attacks.append(changed)
        changed = copy.deepcopy(self.spec)
        changed["groups"][1]["candidates"].pop()
        attacks.append(changed)
        for index, attack in enumerate(attacks):
            with self.subTest(index=index), self.assertRaises(
                topup.SAICPureT2VEventBankTopupError
            ):
                topup.validate_spec(attack)

    def test_plan_reuses_black_proxy_contract_without_real_source_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source_path = root / "source.json"
            source_path.write_bytes(source_set.ASSET_PATH.read_bytes())
            base_path = root / "base-v1.json"
            base_path.write_bytes(v1.ASSET_PATH.read_bytes())
            spec_path = root / "topup-v2.json"
            spec_path.write_bytes(topup.ASSET_PATH.read_bytes())
            proxy_root = root / "proxies"
            proxy_root.mkdir()
            geometries = sorted(
                {
                    tuple(candidate["source_geometry_hw"])
                    for group in self.spec["groups"]
                    for candidate in group["candidates"]
                }
            )
            records = []
            for height, width in geometries:
                path = proxy_root / f"black-h{height}-w{width}.mp4"
                path.write_bytes(f"black-{height}-{width}".encode("ascii"))
                records.append(
                    {
                        "height": height,
                        "width": width,
                        "path": str(path),
                        "sha256": topup.file_sha256(path),
                        "probe": {},
                        "source_media_read": False,
                    }
                )
            unsigned = {
                "schema_version": topup.PROXY_RECEIPT_SCHEMA_VERSION,
                "geometry_proxy_contract": topup.GEOMETRY_PROXY_CONTRACT,
                "ffmpeg_path": "/compute/static/ffmpeg",
                "ffmpeg_version_line": "ffmpeg static test",
                "ffprobe_path": "/archive/tools/ffprobe_pyav_saic.py",
                "records": records,
                "source_media_paths_opened": [],
                "source_media_bytes_read": 0,
            }
            receipt = {**unsigned, "receipt_digest": topup.object_sha256(unsigned)}
            receipt_path = proxy_root / "geometry-proxy-receipt.json"
            receipt_path.write_bytes(topup.canonical_json_bytes(receipt) + b"\n")
            plan_root = root / "plan"
            manifest = topup.materialize_plan(
                spec_path=spec_path,
                expected_spec_raw_sha256=EXPECTED_V2_RAW_SHA256,
                source_manifest_path=source_path,
                base_v1_spec_path=base_path,
                proxy_receipt_path=receipt_path,
                output_dir=plan_root,
            )
            self.assertEqual(manifest["candidate_count"], 60)
            self.assertTrue(manifest["top_up_only"])
            envelopes = list((plan_root / "sp4-a").glob("*.json")) + list(
                (plan_root / "sp4-b").glob("*.json")
            )
            self.assertEqual(len(envelopes), 60)
            self.assertTrue(
                all((path.stat().st_mode & 0o777) == 0o444 for path in envelopes)
            )
            payload = b"".join(path.read_bytes() for path in envelopes)
            for row in self.source["rows"]:
                self.assertNotIn(row["source_video"].encode("ascii"), payload)
            loaded = topup.load_candidate_envelope(
                envelopes[0], expected_root_spec_sha256=EXPECTED_V2_RAW_SHA256
            )
            self.assertFalse(loaded["geometry_proxy"]["source_media_read"])


if __name__ == "__main__":
    unittest.main()
