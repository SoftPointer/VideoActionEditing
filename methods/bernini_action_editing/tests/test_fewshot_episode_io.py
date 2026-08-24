from __future__ import annotations

from copy import deepcopy
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

import fewshot_episode_io as episode_io  # noqa: E402


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_text(value: str) -> str:
    return _sha_bytes(value.encode("utf-8"))


class EpisodeFixture:
    IIDS = (
        "1111111111111111",
        "2222222222222222",
        "3333333333333333",
    )
    SOURCE_ACTIONS = ("standing_still", "walking_right", "standing_still")

    def __init__(self, root: Path) -> None:
        self.root = root
        self.config_path = root / "epmc.json"
        self.preview_path = root / "preview.jsonl"
        self.vae_path = root / "vae-index.jsonl"
        self.artifact_root = root / "artifacts"
        self.artifact_root.mkdir(parents=True)
        self.preview_rows: list[dict[str, object]] = []
        self.vae_rows: list[dict[str, object]] = []
        self.config_rows: list[dict[str, object]] = []

        roles: tuple[tuple[str, int | None], ...] = (
            ("support", 1),
            ("support", 2),
            ("heldout", None),
        )
        for ordinal, (iid, source_action, role_spec) in enumerate(
            zip(self.IIDS, self.SOURCE_ACTIONS, roles), 1
        ):
            role, support_index = role_spec
            self._add_row(
                iid=iid,
                ordinal=ordinal,
                role=role,
                support_index=support_index,
                source_action=source_action,
            )

        self.config: dict[str, object] = {
            "schema_version": episode_io.CONFIG_SCHEMA,
            "purpose": "experimental_engineering_canary_only",
            "production_claim_forbidden": True,
            "scientific_claim_authorized": False,
            "upstream_authorization": {
                "post_video_acceptance": "pending",
                "preview_only": True,
                "training_authorized": False,
                "training_use_forbidden": True,
                "user_requested_experimental_training_ack_required": True,
            },
            "inference_contract": {
                "external_inputs": ["source_video", "edit_instruction"],
                "target_available": False,
                "support_available": False,
                "external_mask_flow_pose_track_trajectory": False,
            },
            "dataset_binding": {
                "preview_manifest_sha256": "0" * 64,
                "vae_index_sha256": "0" * 64,
            },
            "micro_program": {
                "target_action_signature": episode_io.EXPECTED_TARGET_ACTION,
                "entity_type": episode_io.EXPECTED_ENTITY_TYPE,
                "num_rgb_frames": 81,
                "fps": 25,
                "latent_phases": 21,
                "bucket_hw": [480, 496],
                "posterior_parameters_shape": [1, 32, 21, 60, 62],
                "patch_grid_thw": [21, 30, 31],
            },
            "split_seed": 20260807,
            "rows": self.config_rows,
            "manual_contact_sheet_review": {
                "date": "2026-08-07",
                "frames": [0, 20, 40, 60, 80],
                "verdict": (
                    "eligible_for_representation_canary_not_dataset_acceptance"
                ),
                "observations": ["fixture review remains experimental"],
            },
        }
        self.write_inputs()

    def _write_artifact(self, name: str, payload: bytes) -> tuple[Path, str]:
        path = self.artifact_root / name
        path.write_bytes(payload)
        return path, _sha_bytes(payload)

    def _add_row(
        self,
        *,
        iid: str,
        ordinal: int,
        role: str,
        support_index: int | None,
        source_action: str,
    ) -> None:
        source_path, source_sha = self._write_artifact(
            f"{iid}-source.mp4", f"source-video-{iid}".encode()
        )
        target_path, target_sha = self._write_artifact(
            f"{iid}-target.mp4", f"target-video-{iid}".encode()
        )
        parquet_path, parquet_sha = self._write_artifact(
            f"{iid}.parquet", f"vae-parquet-{iid}".encode()
        )
        receipt_path, receipt_sha = self._write_artifact(
            f"{iid}.receipt.json", f'{{"iid":"{iid}"}}\n'.encode()
        )
        group_id = _sha_text(f"group-{ordinal}")
        instruction = f"Have animal {ordinal} sit and turn its head."
        generation = f"Generate exactly 81 frames for animal {ordinal}."
        subject_id = "subject_01"
        preview: dict[str, object] = {
            "schema_version": episode_io.PREVIEW_ROW_SCHEMA,
            "iid": iid,
            "group_id": group_id,
            "family": "sit_down",
            "source_video_path": str(source_path),
            "source_video_sha256": source_sha,
            "target_video_path": str(target_path),
            "target_video_sha256": target_sha,
            "edit_instruction": instruction,
            "edit_instruction_sha256": _sha_text(instruction),
            "instruction_source": "natural",
            "generation_instruction": generation,
            "generation_instruction_sha256": _sha_text(generation),
            "source_census": {
                "iid": iid,
                "dynamic_subjects": [
                    {
                        "subject_id": subject_id,
                        "dynamic": True,
                        "entity_type": "animal",
                        "source_action_signature": source_action,
                    }
                ],
                "camera": {"motion_class": "locked_off"},
                "confidence": "high",
            },
            "target_plan": {
                "iid": iid,
                "dynamic_subject_targets": [
                    {
                        "subject_id": subject_id,
                        "substantive_change": True,
                        "target_action_signature": "sit_and_turn_head",
                    }
                ],
                "camera_target": {
                    "motion_class": "locked_off",
                    "relation": "preserve_static",
                },
                "confidence": "high",
            },
            "selection_gates": {
                "single_dynamic_actor": True,
                "source_camera_locked_off": True,
                "target_camera_locked_off": True,
                "target_camera_preserve_static": True,
                "source_census_high_confidence": True,
                "target_plan_high_confidence": True,
            },
            "preview_only": True,
            "training_authorized": False,
            "training_use_forbidden": True,
            "production_eligible": False,
            "post_video_acceptance": "pending",
            "provenance": {"fixture": True},
        }
        self.rebind_preview_row(preview)
        vae: dict[str, object] = {
            "schema_version": episode_io.VAE_INDEX_ROW_SCHEMA,
            "iid": iid,
            "parquet_path": str(parquet_path),
            "parquet_sha256": parquet_sha,
            "materialized_row_digest": _sha_text(f"materialized-{iid}"),
            "bucket_hw": [480, 496],
            "posterior_parameters_shape": [1, 32, 21, 60, 62],
            "sample_receipt_path": str(receipt_path),
            "sample_receipt_sha256": receipt_sha,
            "preview_only": True,
            "production_claim_forbidden": True,
        }
        config_row: dict[str, object] = {
            "role": role,
            "iid": iid,
            "group_id": group_id,
            "source_video_sha256": source_sha,
            "target_video_sha256": target_sha,
            "edit_instruction_sha256": _sha_text(instruction),
            "vae_parquet_sha256": parquet_sha,
            "source_action_signature": source_action,
        }
        if support_index is not None:
            config_row["support_index"] = support_index
        self.preview_rows.append(preview)
        self.vae_rows.append(vae)
        self.config_rows.append(config_row)

    @staticmethod
    def rebind_preview_row(row: dict[str, object]) -> None:
        row.pop("row_digest", None)
        row["row_digest"] = episode_io.object_sha256(row)

    def write_config(self) -> None:
        self.config_path.write_text(
            json.dumps(self.config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def write_preview(self, *, rebind_config: bool = True) -> None:
        payload = b"".join(
            episode_io.canonical_json_bytes(row) + b"\n"
            for row in sorted(self.preview_rows, key=lambda item: str(item["iid"]))
        )
        self.preview_path.write_bytes(payload)
        if rebind_config:
            self.config["dataset_binding"]["preview_manifest_sha256"] = _sha_bytes(
                payload
            )

    def write_vae(self, *, rebind_config: bool = True) -> None:
        payload = b"".join(
            episode_io.canonical_json_bytes(row) + b"\n"
            for row in sorted(self.vae_rows, key=lambda item: str(item["iid"]))
        )
        self.vae_path.write_bytes(payload)
        if rebind_config:
            self.config["dataset_binding"]["vae_index_sha256"] = _sha_bytes(payload)

    def write_inputs(self) -> None:
        self.write_preview()
        self.write_vae()
        self.write_config()

    @property
    def config_sha256(self) -> str:
        return _sha_bytes(self.config_path.read_bytes())

    def load(self, *, acknowledgement: object = True) -> episode_io.AuditedFewShotEpisode:
        return episode_io.load_epmc_k2_canary(
            self.config_path,
            self.preview_path,
            self.vae_path,
            experimental_training_acknowledged=acknowledgement,
            expected_config_sha256=self.config_sha256,
        )


class FewShotEpisodeIOTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = EpisodeFixture(self.root)
        self.probe = mock.patch.object(
            episode_io,
            "_probe_video_metadata",
            return_value=episode_io.VideoMetadata(81, 25, 1),
        )
        self.mock_probe = self.probe.start()

    def tearDown(self) -> None:
        self.probe.stop()
        self.temporary.cleanup()

    def test_reference_config_pin_matches_checked_in_config(self) -> None:
        config = METHOD_ROOT / "configs" / "epmc_sit_turn_head_k2_v1.json"
        self.assertEqual(
            _sha_bytes(config.read_bytes()), episode_io.REFERENCE_CONFIG_SHA256
        )

    def test_success_parses_two_supports_one_heldout_and_audits_artifacts(self) -> None:
        loaded = self.fixture.load()
        self.assertEqual([row.support_index for row in loaded.supports], [1, 2])
        self.assertEqual(
            [row.iid for row in loaded.rows], list(EpisodeFixture.IIDS)
        )
        self.assertEqual(loaded.heldout.role, "heldout")
        self.assertIsNone(loaded.heldout.support_index)
        self.assertEqual(self.mock_probe.call_count, 6)
        for row in loaded.rows:
            self.assertEqual(row.source_video_metadata.frame_count, 81)
            self.assertEqual(row.target_video_metadata.fps, 25)
            self.assertEqual(row.posterior_parameters_shape, (1, 32, 21, 60, 62))

        receipt = loaded.audit_receipt()
        digest = receipt.pop("audit_digest")
        self.assertEqual(digest, episode_io.object_sha256(receipt))
        self.assertTrue(receipt["experimental_training_acknowledged"])
        self.assertTrue(receipt["preview_only"])
        self.assertFalse(receipt["training_authorized"])
        self.assertTrue(receipt["training_use_forbidden"])
        self.assertFalse(receipt["scientific_claim_authorized"])

    def test_experimental_acknowledgement_is_literal_and_mandatory(self) -> None:
        for value in (False, None, 1, "true"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    episode_io.FewShotEpisodeIOError,
                    "experimental_training_acknowledged=True",
                ):
                    self.fixture.load(acknowledgement=value)
        self.assertEqual(self.mock_probe.call_count, 0)

    def test_config_pin_schema_and_exact_81_frame_contract_fail_closed(self) -> None:
        with self.assertRaisesRegex(episode_io.FewShotEpisodeIOError, "pinned"):
            episode_io.load_epmc_k2_canary(
                self.fixture.config_path,
                self.fixture.preview_path,
                self.fixture.vae_path,
                experimental_training_acknowledged=True,
                expected_config_sha256="0" * 64,
            )

        self.fixture.config["micro_program"]["num_rgb_frames"] = 80
        self.fixture.write_config()
        with self.assertRaisesRegex(episode_io.FewShotEpisodeIOError, "81-frame"):
            self.fixture.load()

        self.fixture.config["micro_program"]["num_rgb_frames"] = 81
        self.fixture.config["unknown"] = True
        self.fixture.write_config()
        with self.assertRaisesRegex(episode_io.FewShotEpisodeIOError, "fields differ"):
            self.fixture.load()

    def test_config_and_preview_safety_states_cannot_be_reauthorized(self) -> None:
        self.fixture.config["upstream_authorization"]["training_authorized"] = True
        self.fixture.write_config()
        with self.assertRaisesRegex(
            episode_io.FewShotEpisodeIOError, "authorization state"
        ):
            self.fixture.load()

        self.fixture = EpisodeFixture(self.root / "second")
        self.fixture.preview_rows[0]["training_use_forbidden"] = False
        self.fixture.rebind_preview_row(self.fixture.preview_rows[0])
        self.fixture.write_inputs()
        with self.assertRaisesRegex(episode_io.FewShotEpisodeIOError, "safety state"):
            self.fixture.load()

    def test_manifest_and_index_top_level_hashes_are_authoritative(self) -> None:
        self.fixture.preview_path.write_bytes(
            self.fixture.preview_path.read_bytes() + b"\n"
        )
        with self.assertRaisesRegex(episode_io.FewShotEpisodeIOError, "pinned SHA-256"):
            self.fixture.load()

        self.fixture = EpisodeFixture(self.root / "second")
        self.fixture.vae_path.write_bytes(self.fixture.vae_path.read_bytes() + b"\n")
        with self.assertRaisesRegex(episode_io.FewShotEpisodeIOError, "pinned SHA-256"):
            self.fixture.load()

    def test_preview_row_digest_and_config_identity_are_both_enforced(self) -> None:
        self.fixture.preview_rows[0]["family"] = "tampered"
        self.fixture.write_inputs()
        with self.assertRaisesRegex(episode_io.FewShotEpisodeIOError, "row digest"):
            self.fixture.load()

        self.fixture = EpisodeFixture(self.root / "second")
        self.fixture.preview_rows[0]["group_id"] = _sha_text("different-group")
        self.fixture.rebind_preview_row(self.fixture.preview_rows[0])
        self.fixture.write_inputs()
        with self.assertRaisesRegex(episode_io.FewShotEpisodeIOError, "group_id differs"):
            self.fixture.load()

    def test_exact_iid_join_duplicate_and_missing_rows_fail_closed(self) -> None:
        self.fixture.preview_rows.append(deepcopy(self.fixture.preview_rows[0]))
        self.fixture.write_inputs()
        with self.assertRaisesRegex(episode_io.FewShotEpisodeIOError, "duplicate"):
            self.fixture.load()

        self.fixture = EpisodeFixture(self.root / "second")
        self.fixture.preview_rows = self.fixture.preview_rows[1:]
        self.fixture.write_inputs()
        with self.assertRaisesRegex(episode_io.FewShotEpisodeIOError, "incomplete"):
            self.fixture.load()

    def test_iid_source_hash_and_group_identity_must_each_be_disjoint(self) -> None:
        for field in ("iid", "source_video_sha256", "group_id"):
            with self.subTest(field=field):
                fixture = EpisodeFixture(self.root / field)
                fixture.config_rows[1][field] = fixture.config_rows[0][field]
                fixture.write_config()
                with self.assertRaisesRegex(
                    episode_io.FewShotEpisodeIOError, f"{field}.*disjoint"
                ):
                    fixture.load()

    def test_roles_support_indices_and_latent_shape_are_exact(self) -> None:
        self.fixture.config_rows[1]["support_index"] = 1
        self.fixture.write_config()
        with self.assertRaisesRegex(episode_io.FewShotEpisodeIOError, "support"):
            self.fixture.load()

        self.fixture = EpisodeFixture(self.root / "second")
        self.fixture.vae_rows[0]["posterior_parameters_shape"] = [1, 32, 20, 60, 62]
        self.fixture.write_inputs()
        with self.assertRaisesRegex(episode_io.FewShotEpisodeIOError, "posterior"):
            self.fixture.load()

    def test_selected_video_parquet_and_receipt_bytes_are_rehashed(self) -> None:
        selected = self.fixture.preview_rows[0]
        Path(str(selected["source_video_path"])).write_bytes(b"changed")
        with self.assertRaisesRegex(episode_io.FewShotEpisodeIOError, "SHA-256"):
            self.fixture.load()

        self.fixture = EpisodeFixture(self.root / "second")
        Path(str(self.fixture.vae_rows[0]["parquet_path"])).write_bytes(b"changed")
        with self.assertRaisesRegex(episode_io.FewShotEpisodeIOError, "SHA-256"):
            self.fixture.load()

        self.fixture = EpisodeFixture(self.root / "third")
        Path(str(self.fixture.vae_rows[0]["sample_receipt_path"])).write_bytes(
            b"changed"
        )
        with self.assertRaisesRegex(episode_io.FewShotEpisodeIOError, "SHA-256"):
            self.fixture.load()

    def test_actual_video_frame_count_and_fps_are_exact(self) -> None:
        self.mock_probe.return_value = episode_io.VideoMetadata(80, 25, 1)
        with self.assertRaisesRegex(episode_io.FewShotEpisodeIOError, "81 frames"):
            self.fixture.load()

        self.mock_probe.return_value = episode_io.VideoMetadata(81, 24, 1)
        with self.assertRaisesRegex(episode_io.FewShotEpisodeIOError, "25 FPS"):
            self.fixture.load()

    def test_duplicate_json_keys_are_rejected_before_any_artifact_use(self) -> None:
        payload = self.fixture.config_path.read_text(encoding="utf-8")
        payload = payload.replace(
            '  "schema_version":',
            '  "schema_version": "duplicate",\n  "schema_version":',
            1,
        )
        self.fixture.config_path.write_text(payload, encoding="utf-8")
        with self.assertRaisesRegex(episode_io.FewShotEpisodeIOError, "duplicate JSON"):
            episode_io.load_epmc_k2_canary(
                self.fixture.config_path,
                self.fixture.preview_path,
                self.fixture.vae_path,
                experimental_training_acknowledged=True,
                expected_config_sha256=_sha_bytes(payload.encode("utf-8")),
            )
        self.assertEqual(self.mock_probe.call_count, 0)


if __name__ == "__main__":
    unittest.main()
