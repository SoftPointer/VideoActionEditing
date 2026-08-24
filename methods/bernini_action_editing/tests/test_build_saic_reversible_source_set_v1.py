from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import build_saic_reversible_source_set_v1 as builder


EXPECTED_CONTENT_SHA256 = (
    "9c2a3d6841951ea0ed050dc230630a1176460e25a979ec199eab575ad22f3c6f"
)


class SAICReversibleSourceSetV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = builder.load_manifest()

    def _reject(self, mutated: dict[str, object]) -> None:
        with self.assertRaises(builder.SAICReversibleSourceSetError):
            builder.validate_manifest(mutated)

    def test_checked_asset_is_exactly_balanced_and_non_authorizing(self) -> None:
        receipt = builder.validate_manifest(self.manifest)
        self.assertEqual(receipt["manifest_content_sha256"], EXPECTED_CONTENT_SHA256)
        self.assertEqual(receipt["row_count"], 8)
        self.assertEqual(receipt["fit_row_count"], 4)
        self.assertEqual(receipt["confirmation_row_count"], 4)
        self.assertTrue(receipt["source_manifest_ready"])
        self.assertFalse(receipt["terminal_events_verified"])
        self.assertFalse(receipt["optimizer_updates_authorized"])
        self.assertFalse(receipt["bound_files_verified"])

        rows = self.manifest["rows"]
        self.assertEqual(
            Counter((row["analysis_split"], row["actor_family"]) for row in rows),
            Counter(builder.EXPECTED_COUNTS),
        )
        self.assertEqual(
            tuple(
                (row["analysis_split"], row["actor_family"], row["iid"])
                for row in rows
            ),
            builder.EXPECTED_ROW_ORDER,
        )
        for row in rows:
            self.assertEqual(row["media_probe"]["nb_frames"], 81)
            self.assertEqual(row["media_probe"]["nb_read_frames"], 81)
            self.assertEqual(row["media_probe"]["avg_frame_rate"], "25/1")
            self.assertEqual(
                row["five_point_timeline"]["frame_indices"], [0, 20, 40, 60, 80]
            )
            self.assertTrue(row["five_point_timeline"]["initial_state_observed"])
            self.assertFalse(
                row["five_point_timeline"]["full81_independent_human_audit"]
            )
            terminal = row["terminal_state_contract"]
            self.assertFalse(terminal["terminal_event_verified"])
            self.assertIsNone(terminal["pure_t2v_event_receipt"])
            self.assertIsNone(terminal["decoded_terminal_media_path"])
            self.assertIsNone(terminal["decoded_terminal_media_sha256"])
            self.assertFalse(row["optimizer_eligible"])

    def test_builder_writes_canonical_create_only_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "manifest.json"
            receipt = builder.build_manifest(output_path=output)
            expected_payload = builder.canonical_json_bytes(self.manifest) + b"\n"
            self.assertEqual(output.read_bytes(), expected_payload)
            self.assertEqual(
                receipt["output_file_sha256"],
                hashlib.sha256(expected_payload).hexdigest(),
            )
            self.assertFalse(receipt["optimizer_updates_authorized"])
            with self.assertRaises(builder.SAICReversibleSourceSetError):
                builder.build_manifest(output_path=output)

    def test_terminal_flags_receipts_and_media_cannot_self_authorize(self) -> None:
        attacks = []
        changed = copy.deepcopy(self.manifest)
        changed["scientific_status"]["terminal_events_verified"] = True
        attacks.append(changed)
        changed = copy.deepcopy(self.manifest)
        changed["scientific_status"]["optimizer_updates_authorized"] = True
        attacks.append(changed)
        changed = copy.deepcopy(self.manifest)
        changed["rows"][0]["optimizer_eligible"] = True
        attacks.append(changed)
        changed = copy.deepcopy(self.manifest)
        changed["rows"][0]["terminal_state_contract"]["terminal_event_verified"] = True
        attacks.append(changed)
        changed = copy.deepcopy(self.manifest)
        changed["rows"][0]["terminal_state_contract"]["pure_t2v_event_receipt"] = {
            "claimed": True
        }
        attacks.append(changed)
        changed = copy.deepcopy(self.manifest)
        changed["rows"][0]["terminal_state_contract"][
            "decoded_terminal_media_path"
        ] = "/tmp/fake.mp4"
        attacks.append(changed)
        changed = copy.deepcopy(self.manifest)
        changed["rows"][0]["terminal_state_contract"][
            "decoded_terminal_media_sha256"
        ] = "a" * 64
        attacks.append(changed)
        for index, attack in enumerate(attacks):
            with self.subTest(index=index):
                self._reject(attack)

    def test_exact81_timeline_and_q0_state_are_fail_closed(self) -> None:
        mutations = []
        changed = copy.deepcopy(self.manifest)
        changed["rows"][0]["media_probe"]["nb_read_frames"] = 80
        mutations.append(changed)
        changed = copy.deepcopy(self.manifest)
        changed["rows"][0]["media_probe"]["avg_frame_rate"] = "24/1"
        mutations.append(changed)
        changed = copy.deepcopy(self.manifest)
        changed["rows"][0]["five_point_timeline"]["frame_indices"][-1] = 79
        mutations.append(changed)
        changed = copy.deepcopy(self.manifest)
        changed["rows"][0]["five_point_timeline"]["state_labels"][-1] = (
            "dog_stable_sit"
        )
        mutations.append(changed)
        changed = copy.deepcopy(self.manifest)
        changed["rows"][0]["five_point_timeline"][
            "body_state_held_at_all_reviewed_frames"
        ] = False
        mutations.append(changed)
        changed = copy.deepcopy(self.manifest)
        changed["rows"][0]["source_census"]["camera_motion_class"] = "pan"
        mutations.append(changed)
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                self._reject(mutation)

    def test_paths_hashes_splits_groups_and_seeds_are_closed(self) -> None:
        mutations = []
        changed = copy.deepcopy(self.manifest)
        changed["rows"][0]["source_video"] = changed["rows"][1]["source_video"]
        mutations.append(changed)
        changed = copy.deepcopy(self.manifest)
        changed["rows"][0]["source_video_sha256"] = "A" * 64
        mutations.append(changed)
        changed = copy.deepcopy(self.manifest)
        changed["rows"][4]["analysis_split"] = "fit"
        mutations.append(changed)
        changed = copy.deepcopy(self.manifest)
        changed["rows"][1]["actor_group_id"] = changed["rows"][0]["actor_group_id"]
        mutations.append(changed)
        changed = copy.deepcopy(self.manifest)
        changed["rows"][4]["rollout_seeds"][0] = changed["rows"][0][
            "rollout_seeds"
        ][0]
        mutations.append(changed)
        changed = copy.deepcopy(self.manifest)
        changed["rows"][4]["confirmation_exposure_scan"][
            "auh_bernini_experiment_receipt_matches"
        ] = 1
        mutations.append(changed)
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                self._reject(mutation)

    def test_action_inverse_noop_text_and_strict_human_type_are_closed(self) -> None:
        mutations = []
        changed = copy.deepcopy(self.manifest)
        changed["rows"][0]["forward_instruction"] = changed["rows"][0][
            "noop_instruction"
        ]
        mutations.append(changed)
        changed = copy.deepcopy(self.manifest)
        changed["rows"][0]["noop_instruction"] = changed["rows"][0][
            "noop_instruction"
        ].replace("never", "might")
        mutations.append(changed)
        changed = copy.deepcopy(self.manifest)
        changed["rows"][2]["initial_state_type"] = "human_low_support"
        mutations.append(changed)
        changed = copy.deepcopy(self.manifest)
        changed["rows"][2]["inverse_instruction"] = "x" * 120
        mutations.append(changed)
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                self._reject(mutation)

    def test_optional_bound_file_verification_rechecks_all_media_and_qwen_rows(self) -> None:
        rows_by_source = {Path(row["source_video"]): row for row in self.manifest["rows"]}
        rows_by_census = {
            Path(row["source_census"]["path"]): row for row in self.manifest["rows"]
        }

        def fake_sha(path: str | Path) -> str:
            candidate = Path(path)
            if candidate in rows_by_source:
                return rows_by_source[candidate]["source_video_sha256"]
            return rows_by_census[candidate]["source_census"]["file_sha256"]

        def fake_probe(path: Path) -> dict[str, object]:
            return copy.deepcopy(rows_by_source[path]["media_probe"])

        def fake_load(path: str | Path) -> dict[str, object]:
            row = rows_by_census[Path(path)]
            binding = row["source_census"]
            return {
                "iid": row["iid"],
                "record_digest": binding["record_digest"],
                "visual_input_digest": binding["visual_input_digest"],
                "source_census": {
                    "confidence": "high",
                    "all_dynamic_subjects_enumerated": True,
                    "crowd_or_unresolved_motion": False,
                    "camera": {"motion_class": "locked_off"},
                },
            }

        with mock.patch.object(Path, "is_file", return_value=True), mock.patch.object(
            Path, "is_symlink", return_value=False
        ), mock.patch.object(builder, "file_sha256", side_effect=fake_sha) as sha, mock.patch.object(
            builder, "_ffprobe_video", side_effect=fake_probe
        ) as probe, mock.patch.object(builder, "load_manifest", side_effect=fake_load) as load:
            receipt = builder.validate_manifest(self.manifest, verify_bound_files=True)

        self.assertTrue(receipt["bound_files_verified"])
        self.assertEqual(sha.call_count, 16)
        self.assertEqual(probe.call_count, 8)
        self.assertEqual(load.call_count, 8)

    def test_duplicate_json_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"schema_version":1,"schema_version":2}\n', encoding="ascii")
            with self.assertRaises(builder.SAICReversibleSourceSetError):
                builder.load_manifest(path)


if __name__ == "__main__":
    unittest.main()
