from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import build_saic_reversible_source_set_v1 as source_set  # noqa: E402
import saic_pure_t2v_event_bank_v1 as bank  # noqa: E402


EXPECTED_RAW_SHA256 = (
    "623a7ed8a2ce2d327247c541b59aa2d39f1fbfe4a480f7351d042c7ef7a47927"
)
EXPECTED_CONTENT_SHA256 = (
    "3920d5c121b75c6bbf984c24440c9773dfb49006778c61a671ae50963bb5456a"
)


class SAICPureT2VEventBankV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = json.loads(bank.ASSET_PATH.read_text(encoding="ascii"))
        self.source = source_set.load_manifest()

    def test_checked_asset_is_complete_text_only_and_non_authorizing(self) -> None:
        summary = bank.validate_spec(self.spec)
        self.assertEqual(bank.file_sha256(bank.ASSET_PATH), EXPECTED_RAW_SHA256)
        self.assertEqual(summary["spec_content_sha256"], EXPECTED_CONTENT_SHA256)
        self.assertEqual(summary["candidate_count"], 60)
        self.assertEqual(summary["row_count"], 8)
        self.assertEqual(summary["seed_cell_count"], 20)
        self.assertFalse(summary["event_verified"])
        self.assertFalse(summary["optimizer_authorized"])
        self.assertEqual(
            [(group["group_id"], group["actor_family"], group["visible_gpus"]) for group in self.spec["groups"]],
            [
                ("sp4-a", "dog", [0, 1, 2, 3]),
                ("sp4-b", "human", [4, 5, 6, 7]),
            ],
        )
        serialized = bank.canonical_json_bytes(self.spec)
        for row in self.source["rows"]:
            self.assertNotIn(row["source_video"].encode("ascii"), serialized)
        self.assertFalse(self.spec["semantic_input_closure"]["real_source_rgb_read"])
        self.assertFalse(
            self.spec["semantic_input_closure"]["real_source_video_path_present"]
        )

    def test_every_registered_seed_has_exact_forward_reverse_noop_triplet(self) -> None:
        candidates = [
            candidate
            for group in self.spec["groups"]
            for candidate in group["candidates"]
        ]
        by_cell: dict[tuple[str, int], list[dict[str, object]]] = {}
        for candidate in candidates:
            by_cell.setdefault((candidate["iid"], candidate["seed"]), []).append(candidate)
        self.assertEqual(len(by_cell), 20)
        self.assertTrue(all(len(rows) == 3 for rows in by_cell.values()))
        for (_, _), rows in by_cell.items():
            self.assertEqual([row["branch"] for row in rows], list(bank.BRANCH_ORDER))
            self.assertTrue(rows[0]["branch_start_state_caption"].startswith("At frame 0"))
            self.assertTrue(rows[1]["branch_start_state_caption"].startswith("At frame 0"))
            self.assertEqual(
                rows[0]["branch_start_state_caption"],
                rows[2]["branch_start_state_caption"],
            )
            self.assertNotEqual(
                rows[0]["branch_start_state_caption"],
                rows[1]["branch_start_state_caption"],
            )
            self.assertEqual(len({row["full_t2v_caption_utf8_sha256"] for row in rows}), 3)
            self.assertTrue(all(row["event_verified"] is False for row in rows))
            self.assertTrue(all(row["optimizer_authorized"] is False for row in rows))

    def test_authorization_or_text_mutation_fails_closed(self) -> None:
        attacks = []
        changed = copy.deepcopy(self.spec)
        changed["artifact_authority"]["optimizer_update_authorized"] = True
        attacks.append(changed)
        changed = copy.deepcopy(self.spec)
        changed["groups"][0]["candidates"][0]["event_verified"] = True
        attacks.append(changed)
        changed = copy.deepcopy(self.spec)
        changed["groups"][0]["candidates"][0]["full_t2v_caption"] += " Changed."
        attacks.append(changed)
        changed = copy.deepcopy(self.spec)
        changed["groups"][0]["candidates"].pop()
        attacks.append(changed)
        for index, attack in enumerate(attacks):
            with self.subTest(index=index), self.assertRaises(
                bank.SAICPureT2VEventBankError
            ):
                bank.validate_spec(attack)

    def test_plan_binds_black_proxies_without_source_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source-manifest.json"
            source_path.write_bytes(source_set.ASSET_PATH.read_bytes())
            spec_path = root / "event-spec.json"
            spec_path.write_bytes(bank.ASSET_PATH.read_bytes())
            proxy_root = root / "proxies"
            proxy_root.mkdir()
            records = []
            geometries = sorted(
                {
                    tuple(candidate["source_geometry_hw"])
                    for group in self.spec["groups"]
                    for candidate in group["candidates"]
                }
            )
            for height, width in geometries:
                path = proxy_root / f"black-h{height}-w{width}.mp4"
                path.write_bytes(f"black-{height}-{width}".encode("ascii"))
                records.append(
                    {
                        "height": height,
                        "width": width,
                        "path": str(path),
                        "sha256": bank.file_sha256(path),
                        "probe": {},
                        "source_media_read": False,
                    }
                )
            unsigned = {
                "schema_version": bank.PROXY_RECEIPT_SCHEMA_VERSION,
                "geometry_proxy_contract": bank.GEOMETRY_PROXY_CONTRACT,
                "ffmpeg_path": "/usr/bin/ffmpeg",
                "ffmpeg_version_line": "ffmpeg test",
                "ffprobe_path": "/usr/bin/ffprobe",
                "records": records,
                "source_media_paths_opened": [],
                "source_media_bytes_read": 0,
            }
            receipt = {**unsigned, "receipt_digest": bank.object_sha256(unsigned)}
            receipt_path = proxy_root / "geometry-proxy-receipt.json"
            receipt_path.write_bytes(bank.canonical_json_bytes(receipt) + b"\n")
            plan_root = root / "plan"
            result = bank.materialize_plan(
                spec_path=spec_path,
                expected_spec_raw_sha256=EXPECTED_RAW_SHA256,
                source_manifest_path=source_path,
                proxy_receipt_path=receipt_path,
                output_dir=plan_root,
            )
            self.assertEqual(result["candidate_count"], 60)
            envelopes = sorted((plan_root / "sp4-a").glob("*.json")) + sorted(
                (plan_root / "sp4-b").glob("*.json")
            )
            self.assertEqual(len(envelopes), 60)
            payload = b"".join(path.read_bytes() for path in envelopes)
            for row in self.source["rows"]:
                self.assertNotIn(row["source_video"].encode("ascii"), payload)
            loaded = bank.load_candidate_envelope(
                envelopes[0], expected_root_spec_sha256=EXPECTED_RAW_SHA256
            )
            self.assertFalse(loaded["geometry_proxy"]["source_media_read"])

    def test_asset_builder_is_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "event-spec.json"
            result = bank.build_asset(
                source_manifest_path=source_set.ASSET_PATH, output_path=output
            )
            self.assertEqual(result["output_raw_sha256"], EXPECTED_RAW_SHA256)
            with self.assertRaises(bank.SAICPureT2VEventBankError):
                bank.build_asset(
                    source_manifest_path=source_set.ASSET_PATH, output_path=output
                )


if __name__ == "__main__":
    unittest.main()
