from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from action.dataset import (  # noqa: E402
    ACTION_MANIFEST_FORMAT,
    ACTION_PAYLOAD_FIELDS,
    ACTION_PAYLOAD_FORMAT,
    ACTION_PROVENANCE_FORMAT,
    ACTION_TRAINING_RELEASE_FORMAT,
    ACTION_TRAINING_RELEASE_VERIFICATION_FORMAT,
    ActionDatasetError,
    ActionLatentDataset,
    action_tensor_sha256,
    collate_action_latents,
    validate_action_payload,
)
from action.config import load_action_config  # noqa: E402
from pact.dataset import (  # noqa: E402
    ENCODER_CONTRACT_FORMAT,
    UMT5_EMBEDDING_DIM,
    UMT5_MAX_SEQUENCE_LENGTH,
    UMT5_PADDING_POLICY,
    UMT5_SEGMENT_ORDER,
    VLM_EMBEDDING_DIM,
    VLM_FEATURE_TENSOR,
    VLM_TOKEN_SELECTION,
    WAN21_VAE_CHANNEL_MEAN,
    WAN21_VAE_CHANNEL_STD,
    WAN21_VAE_INPUT_PIXEL_RANGE,
    WAN21_VAE_POSTERIOR_MODE,
    WAN21_VAE_STRIDE,
    encoder_contract_sha256,
)


def encoder_contract(*, vlm_digest: str = "3" * 64) -> dict:
    return {
        "format": ENCODER_CONTRACT_FORMAT,
        "vae": {
            "checkpoint_sha256": "1" * 64,
            "preprocessing_contract_sha256": "4" * 64,
            "input_pixel_range": list(WAN21_VAE_INPUT_PIXEL_RANGE),
            "posterior_mode": WAN21_VAE_POSTERIOR_MODE,
            "channel_mean": list(WAN21_VAE_CHANNEL_MEAN),
            "channel_std": list(WAN21_VAE_CHANNEL_STD),
            "stride": list(WAN21_VAE_STRIDE),
        },
        "umt5": {
            "checkpoint_manifest_sha256": "2" * 64,
            "preprocessing_contract_sha256": "5" * 64,
            "embedding_dim": UMT5_EMBEDDING_DIM,
            "max_sequence_length_per_segment": UMT5_MAX_SEQUENCE_LENGTH,
            "segment_order": list(UMT5_SEGMENT_ORDER),
            "padding_policy": UMT5_PADDING_POLICY,
        },
        "vlm": {
            "checkpoint_manifest_sha256": vlm_digest,
            "feature_extraction_contract_sha256": "6" * 64,
            "embedding_dim": VLM_EMBEDDING_DIM,
            "feature_tensor": VLM_FEATURE_TENSOR,
            "token_selection": VLM_TOKEN_SELECTION,
        },
    }


def payload(
    *,
    sample_id: str = "sample-001",
    task_type: str = "action_edit",
    preview_only: bool = False,
    motion_tokens: int = 4,
) -> dict:
    return {
        "format": ACTION_PAYLOAD_FORMAT,
        "sample_id": sample_id,
        "encoder_contract": encoder_contract(),
        "source_latent": torch.randn(16, 2, 3, 4),
        "target_latent": torch.randn(16, 2, 3, 4),
        "text_context": torch.randn(7, 4096),
        "source_vlm_context": torch.randn(5, 2048),
        "target_motion_tokens": torch.randn(motion_tokens, 2048),
        "task_type": task_type,
        "preview_only": preview_only,
    }


def write_dataset(
    root: Path,
    value: dict,
    *,
    manifest_preview: bool | None = None,
    manifest_task_type: str | None = None,
    preprocessing: dict | None = None,
) -> Path:
    payload_path = root / "payload.pt"
    torch.save(value, payload_path)
    digest = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    effective_preview = (
        value["preview_only"] if manifest_preview is None else manifest_preview
    )
    provenance = {
        "schema_version": ACTION_PROVENANCE_FORMAT,
        "sample_id": value["sample_id"],
        "parent_id": value["sample_id"],
        "split_group": value["sample_id"],
        "direction": "forward",
        "task_type": (
            value["task_type"] if manifest_task_type is None else manifest_task_type
        ),
        "preview_only": effective_preview,
        "training_authorized": not effective_preview,
        "training_use_forbidden": effective_preview,
        "production_eligible": not effective_preview,
        "post_video_acceptance": "pending" if effective_preview else "accepted",
        "preview_join": {
            "manifest_path": "test://preview",
            "manifest_sha256": "1" * 64,
            "row_digest": "2" * 64,
            "row_file_sha256": "3" * 64,
            "upstream_provenance_sha256": "4" * 64,
        },
        "media": {
            "source_video_path": "test://source",
            "source_video_sha256": "5" * 64,
            "target_video_path": "test://target",
            "target_video_sha256": "6" * 64,
            "shared_i0_path": "test://i0",
            "shared_i0_sha256": "7" * 64,
            "preprocessing": (
                {"test_only": True} if preprocessing is None else preprocessing
            ),
        },
        "conditioning": {
            "instruction": "test instruction",
            "instruction_sha256": hashlib.sha256(b"test instruction").hexdigest(),
            "instruction_source": "test",
            "generation_instruction_sha256": "8" * 64,
            "target_caption": "test caption",
            "target_caption_sha256": hashlib.sha256(b"test caption").hexdigest(),
            "target_caption_origin": "test",
            "motion_text": "test motion",
            "motion_text_sha256": hashlib.sha256(b"test motion").hexdigest(),
            "motion_teacher_visual_input": "target_video_only",
            "motion_teacher_feature_input": "canonical_motion_text_only",
            "motion_pool": "test",
            "target_motion_tokens_usage": "planner_loss_only",
        },
        "encoder": {
            "contract": value["encoder_contract"],
            "contract_sha256": encoder_contract_sha256(value["encoder_contract"]),
            "checkpoint_identities": {"test_only": True},
        },
        "tensor_sha256": {
            field: action_tensor_sha256(value[field])
            for field in (
                "source_latent",
                "target_latent",
                "text_context",
                "source_vlm_context",
                "target_motion_tokens",
            )
        },
        "payload": {"path": payload_path.name, "sha256": digest},
    }
    if not effective_preview:
        release_path = root / "signed_release.json"
        release_path.write_text(
            json.dumps({"sample_id": value["sample_id"], "signed": True}) + "\n",
            encoding="utf-8",
        )
        release_digest = hashlib.sha256(release_path.read_bytes()).hexdigest()
        release_row = {
            "sample_id": value["sample_id"],
            "training_authorized": True,
            "training_use_forbidden": False,
            "production_eligible": True,
            "post_video_acceptance": "accepted",
        }
        verification_path = root / "signed_release_verification.json"
        verification_path.write_text(
            json.dumps(
                {
                    "schema_version": ACTION_TRAINING_RELEASE_VERIFICATION_FORMAT,
                    "status": "verified",
                    "sample_id": value["sample_id"],
                    "release_sha256": release_digest,
                    "release_row": release_row,
                },
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        provenance["training_release"] = {
            "schema_version": ACTION_TRAINING_RELEASE_FORMAT,
            "release_path": release_path.name,
            "release_sha256": release_digest,
            "verification_receipt_path": verification_path.name,
            "verification_receipt_sha256": hashlib.sha256(
                verification_path.read_bytes()
            ).hexdigest(),
            "sample_row_sha256": hashlib.sha256(
                json.dumps(
                    release_row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest(),
            "verification_status": "verified",
        }
    provenance_path = root / "provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, sort_keys=True) + "\n", encoding="utf-8"
    )
    provenance_digest = hashlib.sha256(provenance_path.read_bytes()).hexdigest()
    row = {
        "format": ACTION_MANIFEST_FORMAT,
        "sample_id": value["sample_id"],
        "payload_path": payload_path.name,
        "payload_sha256": digest,
        "provenance_path": provenance_path.name,
        "provenance_sha256": provenance_digest,
        "task_type": (
            value["task_type"] if manifest_task_type is None else manifest_task_type
        ),
        "preview_only": (
            value["preview_only"]
            if manifest_preview is None
            else manifest_preview
        ),
    }
    manifest = root / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return manifest


class ActionDatasetTest(unittest.TestCase):
    @staticmethod
    def _full_81_preprocessing() -> dict:
        return {
            "temporal_mode": "full_81_25fps",
            "spatial_profile": "full_480p",
            "frame_indices": list(range(81)),
            "source_frame_count": 81,
            "target_frame_count": 81,
            "materialized_frame_count": 81,
            "source_fps": 25.0,
            "target_fps": 25.0,
            "materialized_fps": 25.0,
            "sampling_policy": "all_frames_in_order_no_temporal_subsampling",
            "temporal_subsampled": False,
            "bucket_hw": [480, 832],
        }

    def test_payload_is_closed_and_has_only_full_training_fields(self) -> None:
        value = payload()
        self.assertEqual(set(value), set(ACTION_PAYLOAD_FIELDS))
        checked = validate_action_payload(value, expected_motion_tokens=4)
        self.assertEqual(checked["source_latent"].shape, (16, 2, 3, 4))
        self.assertEqual(checked["target_latent"].shape, (16, 2, 3, 4))
        self.assertEqual(checked["target_motion_tokens"].shape, (4, 2048))

        unknown = payload()
        unknown["legacy_spatial_control"] = torch.zeros(1)
        with self.assertRaisesRegex(ActionDatasetError, "unknown"):
            validate_action_payload(unknown)

        missing = payload()
        del missing["target_latent"]
        with self.assertRaisesRegex(ActionDatasetError, "missing"):
            validate_action_payload(missing)

    def test_shapes_dimensions_task_and_preview_types_are_strict(self) -> None:
        cases = []
        wrong_shape = payload()
        wrong_shape["target_latent"] = torch.randn(16, 3, 3, 4)
        cases.append(("shape", "shapes differ", wrong_shape))

        wrong_text = payload()
        wrong_text["text_context"] = torch.randn(7, 1024)
        cases.append(("text", "4096", wrong_text))

        wrong_vlm = payload()
        wrong_vlm["source_vlm_context"] = torch.randn(5, 1024)
        cases.append(("vlm", "2048", wrong_vlm))

        wrong_tokens = payload(motion_tokens=3)
        cases.append(("tokens", "must be 4", wrong_tokens))

        wrong_task = payload(task_type="global_action")
        cases.append(("task", "task_type", wrong_task))

        wrong_preview = payload()
        wrong_preview["preview_only"] = 0
        cases.append(("preview", "must be bool", wrong_preview))

        for label, message, value in cases:
            with self.subTest(label=label), self.assertRaisesRegex(
                ActionDatasetError, message
            ):
                validate_action_payload(value, expected_motion_tokens=4)

    def test_digest_bound_dataset_and_collation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_dataset(root, payload())
            dataset = ActionLatentDataset(manifest, expected_motion_tokens=4)
            sample = dataset[0]
            self.assertEqual(sample["sample_id"], "sample-001")
            self.assertEqual(sample["task_type"], "action_edit")
            batch = collate_action_latents([sample, sample])
            self.assertEqual(batch["source_latent"].shape, (2, 16, 2, 3, 4))
            self.assertEqual(batch["target_latent"].shape, (2, 16, 2, 3, 4))
            self.assertEqual(batch["target_motion_tokens"].shape, (2, 4, 2048))
            self.assertEqual(len(batch["text_context"]), 2)
            self.assertEqual(len(batch["source_vlm_context"]), 2)
            self.assertEqual(len(batch["provenance_sha256"]), 2)
            self.assertTrue(torch.equal(batch["preview_only"], torch.zeros(2, dtype=torch.bool)))

            mixed = dict(sample)
            mixed["encoder_contract"] = encoder_contract(vlm_digest="7" * 64)
            from pact.dataset import encoder_contract_sha256

            mixed["encoder_contract_sha256"] = encoder_contract_sha256(
                mixed["encoder_contract"]
            )
            with self.assertRaisesRegex(ActionDatasetError, "mixed"):
                collate_action_latents([sample, mixed])

    def test_preview_and_tampering_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_dataset(root, payload(preview_only=True))
            with self.assertRaisesRegex(ActionDatasetError, "preview-only"):
                ActionLatentDataset(manifest)
            inspection = ActionLatentDataset(manifest, allow_preview=True)
            self.assertTrue(inspection[0]["preview_only"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_dataset(root, payload())
            row = json.loads(manifest.read_text(encoding="utf-8"))
            row["payload_sha256"] = "0" * 64
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ActionDatasetError, "digest differs"):
                ActionLatentDataset(manifest)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_dataset(
                root,
                payload(task_type="action_edit"),
                manifest_task_type="native_replay",
            )
            dataset = ActionLatentDataset(manifest)
            with self.assertRaisesRegex(ActionDatasetError, "differs from manifest"):
                dataset[0]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_dataset(root, payload())
            provenance_path = root / "provenance.json"
            provenance_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ActionDatasetError, "provenance digest differs"):
                ActionLatentDataset(manifest)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_dataset(root, payload())
            provenance_path = root / "provenance.json"
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance["tensor_sha256"]["source_latent"] = "0" * 64
            provenance_path.write_text(
                json.dumps(provenance, sort_keys=True) + "\n", encoding="utf-8"
            )
            row = json.loads(manifest.read_text(encoding="utf-8"))
            row["provenance_sha256"] = hashlib.sha256(
                provenance_path.read_bytes()
            ).hexdigest()
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            dataset = ActionLatentDataset(manifest)
            with self.assertRaisesRegex(ActionDatasetError, "source_latent"):
                dataset[0]

    def test_81_profile_rejects_41_sampling_wrong_fps_and_wrong_indices(self) -> None:
        data = load_action_config(ROOT / "configs" / "marp_1_3b.json").data
        cases = []

        smoke = self._full_81_preprocessing()
        smoke.update(
            {
                "temporal_mode": "smoke_41_12p5fps",
                "frame_indices": list(range(0, 81, 2)),
                "materialized_frame_count": 41,
                "materialized_fps": 12.5,
                "sampling_policy": "explicit_stride_2_smoke_ablation_only",
                "temporal_subsampled": True,
            }
        )
        cases.append(("41-mode", smoke, "temporal_mode"))

        wrong_fps = self._full_81_preprocessing()
        wrong_fps["materialized_fps"] = 12.5
        cases.append(("fps", wrong_fps, "materialized_fps"))

        wrong_indices = self._full_81_preprocessing()
        wrong_indices["frame_indices"] = list(range(80, -1, -1))
        cases.append(("indices", wrong_indices, "frame_indices"))

        for label, preprocessing, message in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = write_dataset(
                    root, payload(), preprocessing=preprocessing
                )
                with self.assertRaisesRegex(ActionDatasetError, message):
                    ActionLatentDataset(
                        manifest,
                        expected_data_config=data,
                    )

    def test_nonpreview_requires_authorization_and_bound_signed_release(self) -> None:
        def rewrite_provenance(root: Path, mutate) -> Path:
            manifest = root / "manifest.jsonl"
            provenance_path = root / "provenance.json"
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            mutate(provenance)
            provenance_path.write_text(
                json.dumps(provenance, sort_keys=True) + "\n", encoding="utf-8"
            )
            row = json.loads(manifest.read_text(encoding="utf-8"))
            row["provenance_sha256"] = hashlib.sha256(
                provenance_path.read_bytes()
            ).hexdigest()
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            return manifest

        cases = (
            (
                "unauthorized",
                lambda provenance: provenance.__setitem__(
                    "training_authorized", False
                ),
                "non-preview provenance",
            ),
            (
                "forbidden",
                lambda provenance: provenance.__setitem__(
                    "training_use_forbidden", True
                ),
                "non-preview provenance",
            ),
            (
                "not-production",
                lambda provenance: provenance.__setitem__(
                    "production_eligible", False
                ),
                "non-preview provenance",
            ),
            (
                "not-accepted",
                lambda provenance: provenance.__setitem__(
                    "post_video_acceptance", "pending"
                ),
                "non-preview provenance",
            ),
            (
                "missing-release",
                lambda provenance: provenance.pop("training_release"),
                "missing",
            ),
            (
                "wrong-release-row-digest",
                lambda provenance: provenance["training_release"].__setitem__(
                    "sample_row_sha256", "0" * 64
                ),
                "sample_row_sha256 differs",
            ),
        )
        for label, mutate, message in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_dataset(root, payload())
                manifest = rewrite_provenance(root, mutate)
                with self.assertRaisesRegex(ActionDatasetError, message):
                    ActionLatentDataset(manifest)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_dataset(root, payload())
            (root / "signed_release.json").write_text(
                "changed after materialization\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ActionDatasetError, "release_sha256 differs"):
                ActionLatentDataset(manifest)

        for label, mutate_receipt, message in (
            (
                "receipt-other-sample",
                lambda receipt: receipt.__setitem__("sample_id", "other-sample"),
                "receipt sample_id differs",
            ),
            (
                "receipt-unauthorized-row",
                lambda receipt: receipt["release_row"].__setitem__(
                    "production_eligible", False
                ),
                "does not authorize sample",
            ),
            (
                "receipt-wrong-release",
                lambda receipt: receipt.__setitem__("release_sha256", "0" * 64),
                "receipt release_sha256 differs",
            ),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = write_dataset(root, payload())
                receipt_path = root / "signed_release_verification.json"
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                mutate_receipt(receipt)
                receipt_path.write_text(
                    json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
                )
                provenance_path = root / "provenance.json"
                provenance = json.loads(
                    provenance_path.read_text(encoding="utf-8")
                )
                provenance["training_release"]["verification_receipt_sha256"] = (
                    hashlib.sha256(receipt_path.read_bytes()).hexdigest()
                )
                provenance_path.write_text(
                    json.dumps(provenance, sort_keys=True) + "\n", encoding="utf-8"
                )
                row = json.loads(manifest.read_text(encoding="utf-8"))
                row["provenance_sha256"] = hashlib.sha256(
                    provenance_path.read_bytes()
                ).hexdigest()
                manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ActionDatasetError, message):
                    ActionLatentDataset(manifest)


if __name__ == "__main__":
    unittest.main()
