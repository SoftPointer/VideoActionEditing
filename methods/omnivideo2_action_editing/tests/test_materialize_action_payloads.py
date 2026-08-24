from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from action.dataset import ActionLatentDataset, validate_action_payload  # noqa: E402
from tools import materialize_action_payloads as materializer  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contract() -> dict:
    return {
        "format": "pact-omnivideo2-offline-encoder-contract-v1",
        "vae": {
            "checkpoint_sha256": "1" * 64,
            "preprocessing_contract_sha256": "2" * 64,
            "input_pixel_range": [-1.0, 1.0],
            "posterior_mode": "mean",
            "channel_mean": [
                -0.7571,
                -0.7089,
                -0.9113,
                0.1075,
                -0.1745,
                0.9653,
                -0.1517,
                1.5508,
                0.4134,
                -0.0715,
                0.5517,
                -0.3632,
                -0.1922,
                -0.9497,
                0.2503,
                -0.2921,
            ],
            "channel_std": [
                2.8184,
                1.4541,
                2.3275,
                2.6558,
                1.2196,
                1.7708,
                2.6052,
                2.0743,
                3.2687,
                2.1526,
                2.8652,
                1.5579,
                1.6382,
                1.1253,
                2.8251,
                1.9160,
            ],
            "stride": [4, 8, 8],
        },
        "umt5": {
            "checkpoint_manifest_sha256": "3" * 64,
            "preprocessing_contract_sha256": "4" * 64,
            "embedding_dim": 4096,
            "max_sequence_length_per_segment": 512,
            "segment_order": ["target_caption", "edit_instruction"],
            "padding_policy": "slice_to_attention_mask_length",
        },
        "vlm": {
            "checkpoint_manifest_sha256": "5" * 64,
            "feature_extraction_contract_sha256": "6" * 64,
            "embedding_dim": 2048,
            "feature_tensor": "vlm_last_hidden_states",
            "token_selection": "attention_mask_then_drop_system_prefix",
        },
    }


class FakeMediaStage:
    def __init__(
        self,
        *,
        temporal_mode: str = materializer.DEFAULT_TEMPORAL_MODE,
        spatial_profile: str = materializer.DEFAULT_SPATIAL_PROFILE,
    ) -> None:
        self.seen = []
        self.temporal_sampling = materializer.temporal_sampling_contract(
            temporal_mode
        )
        self.spatial_sampling = materializer.spatial_profile_contract(spatial_profile)

    def prepare(self, item: materializer.ValidatedPreviewRow) -> materializer.PreparedMedia:
        self.seen.append(item.row["iid"])
        temporal = self.temporal_sampling
        spatial = self.spatial_sampling
        bucket = spatial.landscape_bucket_hw
        shared = torch.zeros(
            (3, temporal.materialized_frame_count, 1, 1), dtype=torch.float32
        ).expand(
            3,
            temporal.materialized_frame_count,
            bucket[0],
            bucket[1],
        )
        return materializer.PreparedMedia(
            source_video=shared,
            target_video=shared,
            source_qwen_path=str(item.source_video),
            target_qwen_path=str(item.target_video),
            metadata={
                "temporal_mode": temporal.mode,
                "spatial_profile": spatial.profile,
                "frame_indices": list(temporal.frame_indices),
                "source_frame_count": temporal.source_frame_count,
                "target_frame_count": temporal.source_frame_count,
                "materialized_frame_count": temporal.materialized_frame_count,
                "source_fps": 25.0,
                "target_fps": 25.0,
                "materialized_fps": temporal.materialized_fps,
                "sampling_policy": temporal.sampling_policy,
                "temporal_subsampled": temporal.temporal_subsampled,
                "bucket_hw": list(bucket),
                "source_crop_tlbr": [48, 0, 657, 1056],
                "target_crop_tlbr": [48, 0, 721, 1168],
                "shared_i0_crop_tlbr": [48, 0, 721, 1168],
                "shared_frame0_exact": True,
            },
        )


class FakeEncoderStage:
    contract = _contract()
    checkpoint_identities = {
        "qwen": {"manifest_sha256": "5" * 64},
        "umt5": {"manifest_sha256": "3" * 64},
        "vae": {"manifest_sha256": "1" * 64},
    }

    def __init__(self) -> None:
        self.instructions = []

    def encode(
        self,
        item: materializer.ValidatedPreviewRow,
        media: materializer.PreparedMedia,
    ) -> materializer.EncodedSample:
        self.instructions.append(item.row["edit_instruction"])
        source = torch.zeros((16, 2, 4, 4), dtype=torch.float32)
        target = torch.ones_like(source)
        return materializer.EncodedSample(
            source_latent=source,
            target_latent=target,
            text_context=torch.zeros((3, 4096), dtype=torch.float32),
            source_vlm_context=torch.zeros((5, 2048), dtype=torch.float32),
            target_motion_tokens=torch.arange(
                4 * 2048, dtype=torch.float32
            ).reshape(4, 2048),
            target_caption="A person crouches while the camera remains static.",
            motion_text=(
                "SUBJECT_MOTION: standing to crouching\n"
                "CAMERA_MOTION: locked off\n"
                "TIMING: gradual change, then a held endpoint"
            ),
            metadata={
                "target_caption_origin": (
                    "official_qwen_source_caption_plus_instruction_prediction"
                ),
                "source_vlm_origin": "official_qwen_source_video_plus_instruction",
                "motion_teacher_visual_input": "target_video_only",
                "motion_teacher_feature_input": "canonical_motion_text_only",
                "target_motion_tokens_usage": "planner_loss_only",
                "motion_pool": "deterministic_integer_bins_mean",
            },
        )


def _preview_fixture(root: Path, *, iid: str = "clip001") -> tuple[Path, dict]:
    source = root / "source.mp4"
    target = root / "target.mp4"
    frame0 = root / "conditioning_frame0_float32.npy"
    source.write_bytes(b"source-video")
    target.write_bytes(b"target-video")
    frame0.write_bytes(b"lossless-i0")
    generated = {
        "schema_version": materializer.WAN_GENERATED_FORMAT,
        "iid": iid,
        "source_video": str(source),
        "source_video_sha256": _sha(source),
        "target_preview_mp4": str(target),
        "target_preview_mp4_sha256": _sha(target),
        "conditioning_frame0_float32": str(frame0),
        "conditioning_frame0_float32_sha256": _sha(frame0),
    }
    generated_path = root / "generated_manifest.jsonl"
    generated_path.write_bytes(materializer.canonical_json_bytes(generated) + b"\n")
    instruction = "Have the actor gradually crouch while preserving the scene."
    generation_instruction = "Starting at frame zero, make the actor crouch."
    row = {
        "schema_version": materializer.PREVIEW_ROW_FORMAT,
        "iid": iid,
        "group_id": "parent-group-001",
        "family": "crouch",
        "source_video_path": str(source),
        "source_video_sha256": _sha(source),
        "target_video_path": str(target),
        "target_video_sha256": _sha(target),
        "edit_instruction": instruction,
        "edit_instruction_sha256": materializer.text_sha256(instruction),
        "instruction_source": "natural",
        "generation_instruction": generation_instruction,
        "generation_instruction_sha256": materializer.text_sha256(
            generation_instruction
        ),
        "source_census": {"iid": iid},
        "target_plan": {"iid": iid},
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
        "provenance": {
            "wan_generated_manifest_path": str(generated_path),
            "wan_generated_manifest_sha256": _sha(generated_path),
        },
    }
    row["row_digest"] = materializer.object_sha256(row)
    manifest = root / "preview.jsonl"
    manifest.write_bytes(materializer.canonical_json_bytes(row) + b"\n")
    return manifest, row


class MaterializeActionPayloadsTests(unittest.TestCase):
    def test_temporal_and_spatial_contract_is_deterministic(self) -> None:
        full = materializer.temporal_sampling_contract()
        self.assertEqual(full.mode, materializer.TEMPORAL_MODE_FULL_81)
        self.assertEqual(full.frame_indices, tuple(range(81)))
        self.assertEqual(full.materialized_frame_count, 81)
        self.assertEqual(full.materialized_fps, 25.0)
        self.assertFalse(full.temporal_subsampled)
        self.assertEqual(materializer.FRAME_INDICES, tuple(range(81)))
        self.assertEqual(materializer.MATERIALIZED_FRAME_COUNT, 81)
        self.assertEqual(materializer.MATERIALIZED_FPS, 25.0)
        smoke = materializer.temporal_sampling_contract(
            materializer.TEMPORAL_MODE_SMOKE_41
        )
        self.assertEqual(smoke.frame_indices, tuple(range(0, 81, 2)))
        self.assertEqual(materializer.temporal_indices_81_to_41(), smoke.frame_indices)
        self.assertEqual(smoke.materialized_frame_count, 41)
        self.assertEqual(smoke.materialized_fps, 12.5)
        self.assertTrue(smoke.temporal_subsampled)
        self.assertEqual(materializer.choose_bucket(704, 1056), (480, 832))
        self.assertEqual(materializer.choose_bucket(1056, 704), (832, 480))
        self.assertEqual(
            materializer.choose_bucket(
                704,
                1056,
                spatial_profile=materializer.SPATIAL_PROFILE_MOTION_384P,
            ),
            (384, 640),
        )
        motion_stage = materializer.DecordOmniMediaStage(
            spatial_profile=materializer.SPATIAL_PROFILE_MOTION_384P
        )
        self.assertEqual(
            motion_stage.temporal_sampling.mode,
            materializer.TEMPORAL_MODE_FULL_81,
        )
        self.assertEqual(
            materializer.center_crop_box(704, 1056, (480, 832)),
            (48, 0, 657, 1056),
        )
        self.assertAlmostEqual(
            materializer.crop_retention(704, 1056, (480, 832)),
            (609 * 1056) / (704 * 1056),
        )
        with self.assertRaisesRegex(
            materializer.MaterializationError, "min_crop_retention"
        ):
            materializer.DecordOmniMediaStage(min_crop_retention=0.0)
        with self.assertRaisesRegex(materializer.MaterializationError, "square"):
            materializer.choose_bucket(512, 512)
        with self.assertRaisesRegex(materializer.MaterializationError, "temporal mode"):
            materializer.temporal_sampling_contract("implicit-downsample")
        with self.assertRaisesRegex(materializer.MaterializationError, "spatial profile"):
            materializer.spatial_profile_contract("unknown")
        args = materializer.parse_args(
            [
                "--preview-manifest",
                "preview.jsonl",
                "--output-dir",
                "output",
                "--qwen-checkpoint",
                "qwen",
                "--vae-checkpoint",
                "vae.pth",
                "--umt5-checkpoint",
                "umt5.pth",
                "--umt5-tokenizer",
                "umt5-tokenizer",
                "--spatial-profile",
                materializer.SPATIAL_PROFILE_MOTION_384P,
            ]
        )
        self.assertEqual(args.temporal_mode, materializer.TEMPORAL_MODE_FULL_81)
        self.assertEqual(args.spatial_profile, materializer.SPATIAL_PROFILE_MOTION_384P)

    def test_motion_pool_uses_fixed_integer_mean_bins(self) -> None:
        tokens = torch.arange(8, dtype=torch.float32).reshape(8, 1).repeat(1, 2048)
        pooled = materializer.deterministic_motion_pool(tokens, 4)
        self.assertEqual(tuple(pooled.shape), (4, 2048))
        self.assertTrue(
            torch.equal(pooled[:, 0], torch.tensor([0.5, 2.5, 4.5, 6.5]))
        )

    def test_motion_record_allows_only_an_optional_markdown_fence(self) -> None:
        record = (
            "SUBJECT_MOTION: the subject lowers both arms\n"
            "CAMERA_MOTION: locked off\n"
            "TIMING: lowers, then holds"
        )
        self.assertEqual(
            materializer.canonical_motion_record(f"```text\n{record}\n```"),
            record,
        )
        with self.assertRaisesRegex(
            materializer.MaterializationError, "exactly the required three lines"
        ):
            materializer.canonical_motion_record("Here is the result:\n" + record)

    def test_preview_requires_explicit_exploration_acknowledgement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _row = _preview_fixture(root)
            with self.assertRaisesRegex(
                materializer.MaterializationError,
                "allow_preview_exploration",
            ):
                materializer.materialize_action_payloads(
                    preview_manifest=manifest,
                    output_dir=root / "output",
                    media_stage=FakeMediaStage(),
                    encoder_stage=FakeEncoderStage(),
                )

    def test_explicit_sample_selection_fails_if_iid_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _row = _preview_fixture(root)
            with self.assertRaisesRegex(
                materializer.MaterializationError, "absent from preview manifest"
            ):
                materializer.materialize_action_payloads(
                    preview_manifest=manifest,
                    output_dir=root / "output",
                    allow_preview_exploration=True,
                    sample_ids=("not-present",),
                    media_stage=FakeMediaStage(),
                    encoder_stage=FakeEncoderStage(),
                )

    def test_payload_manifest_and_provenance_are_create_only_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, input_row = _preview_fixture(root)
            media_stage = FakeMediaStage()
            encoder_stage = FakeEncoderStage()
            output = root / "output"
            receipt = materializer.materialize_action_payloads(
                preview_manifest=manifest,
                output_dir=output,
                allow_preview_exploration=True,
                media_stage=media_stage,
                encoder_stage=encoder_stage,
            )
            self.assertTrue(receipt["preview_only"])
            self.assertFalse(receipt["training_authorized"])
            self.assertFalse(receipt["scientific_claim_authorized"])
            self.assertEqual(receipt["target_motion_tokens_usage"], "planner_loss_only")
            self.assertEqual(
                receipt["temporal_mode"], materializer.TEMPORAL_MODE_FULL_81
            )
            self.assertEqual(receipt["temporal_indices"], list(range(81)))
            self.assertEqual(receipt["source_frame_count"], 81)
            self.assertEqual(receipt["materialized_frame_count"], 81)
            self.assertEqual(receipt["materialized_fps"], 25.0)
            self.assertFalse(receipt["temporal_subsampled"])
            self.assertEqual(
                receipt["spatial_profile"], materializer.SPATIAL_PROFILE_FULL_480P
            )
            row = json.loads((output / "manifest.jsonl").read_text())
            self.assertEqual(set(row), materializer.MATERIALIZED_MANIFEST_FIELDS)
            self.assertTrue(row["preview_only"])
            payload_path = output / "payloads" / row["payload_path"]
            self.assertEqual(row["payload_sha256"], _sha(payload_path))
            payload = torch.load(payload_path, map_location="cpu", weights_only=True)
            checked = validate_action_payload(
                payload,
                expected_motion_tokens=4,
                allowed_task_types=("action_edit",),
            )
            self.assertEqual(checked["task_type"], "action_edit")
            self.assertTrue(checked["preview_only"])
            forbidden = {
                "source_component_mask",
                "target_component_mask",
                "track_record",
                "erased_source",
            }
            self.assertFalse(forbidden & set(checked))
            dataset = ActionLatentDataset(
                output / "manifest.jsonl",
                payload_root=output / "payloads",
                expected_motion_tokens=4,
                allowed_task_types=("action_edit",),
                allow_preview=True,
            )
            self.assertEqual(dataset[0]["sample_id"], input_row["iid"])

            provenance_path = output / row["provenance_path"]
            self.assertEqual(row["provenance_sha256"], _sha(provenance_path))
            provenance = json.loads(provenance_path.read_text())
            self.assertEqual(provenance["split_group"], input_row["group_id"])
            self.assertEqual(provenance["direction"], "forward")
            self.assertEqual(
                provenance["preview_join"]["row_digest"], input_row["row_digest"]
            )
            self.assertEqual(
                provenance["preview_join"]["row_file_sha256"],
                hashlib.sha256(
                    materializer.canonical_json_bytes(input_row) + b"\n"
                ).hexdigest(),
            )
            self.assertEqual(
                provenance["conditioning"]["instruction_sha256"],
                input_row["edit_instruction_sha256"],
            )
            self.assertNotEqual(
                provenance["conditioning"]["target_caption"],
                input_row["edit_instruction"],
            )
            self.assertEqual(
                provenance["conditioning"]["motion_teacher_feature_input"],
                "canonical_motion_text_only",
            )
            self.assertEqual(
                provenance["conditioning"]["target_motion_tokens_usage"],
                "planner_loss_only",
            )
            preprocessing = provenance["media"]["preprocessing"]
            self.assertEqual(preprocessing["frame_indices"], list(range(81)))
            self.assertEqual(preprocessing["materialized_frame_count"], 81)
            self.assertEqual(preprocessing["materialized_fps"], 25.0)
            self.assertFalse(preprocessing["temporal_subsampled"])
            self.assertEqual(
                preprocessing["spatial_profile"],
                materializer.SPATIAL_PROFILE_FULL_480P,
            )
            self.assertEqual(media_stage.seen, [input_row["iid"]])
            self.assertEqual(encoder_stage.instructions, [input_row["edit_instruction"]])
            with self.assertRaisesRegex(
                materializer.MaterializationError, "create-only output"
            ):
                materializer.materialize_action_payloads(
                    preview_manifest=manifest,
                    output_dir=output,
                    allow_preview_exploration=True,
                    media_stage=FakeMediaStage(),
                    encoder_stage=FakeEncoderStage(),
                )

    def test_smoke_temporal_and_motion_spatial_profiles_are_explicit_and_audited(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _row = _preview_fixture(root)

            smoke_mode = materializer.TEMPORAL_MODE_SMOKE_41
            smoke_output = root / "smoke-output"
            smoke_receipt = materializer.materialize_action_payloads(
                preview_manifest=manifest,
                output_dir=smoke_output,
                allow_preview_exploration=True,
                temporal_mode=smoke_mode,
                media_stage=FakeMediaStage(temporal_mode=smoke_mode),
                encoder_stage=FakeEncoderStage(),
            )
            self.assertEqual(smoke_receipt["temporal_mode"], smoke_mode)
            self.assertEqual(smoke_receipt["temporal_indices"], list(range(0, 81, 2)))
            self.assertEqual(smoke_receipt["materialized_frame_count"], 41)
            self.assertEqual(smoke_receipt["materialized_fps"], 12.5)
            self.assertTrue(smoke_receipt["temporal_subsampled"])
            smoke_row = json.loads((smoke_output / "manifest.jsonl").read_text())
            smoke_provenance = json.loads(
                (smoke_output / smoke_row["provenance_path"]).read_text()
            )
            self.assertEqual(
                smoke_provenance["media"]["preprocessing"]["sampling_policy"],
                "explicit_stride_2_smoke_ablation_only",
            )

            motion_profile = materializer.SPATIAL_PROFILE_MOTION_384P
            motion_output = root / "motion-output"
            motion_receipt = materializer.materialize_action_payloads(
                preview_manifest=manifest,
                output_dir=motion_output,
                allow_preview_exploration=True,
                spatial_profile=motion_profile,
                media_stage=FakeMediaStage(spatial_profile=motion_profile),
                encoder_stage=FakeEncoderStage(),
            )
            self.assertEqual(motion_receipt["spatial_profile"], motion_profile)
            self.assertEqual(motion_receipt["landscape_bucket_hw"], [384, 640])
            self.assertEqual(motion_receipt["portrait_bucket_hw"], [640, 384])
            self.assertEqual(
                motion_receipt["temporal_mode"], materializer.TEMPORAL_MODE_FULL_81
            )
            self.assertEqual(motion_receipt["materialized_frame_count"], 81)
            self.assertFalse(motion_receipt["temporal_subsampled"])

            with self.assertRaisesRegex(
                materializer.MaterializationError,
                "media sampling metadata mismatch",
            ):
                materializer.materialize_action_payloads(
                    preview_manifest=manifest,
                    output_dir=root / "implicit-smoke-is-forbidden",
                    allow_preview_exploration=True,
                    media_stage=FakeMediaStage(temporal_mode=smoke_mode),
                    encoder_stage=FakeEncoderStage(),
                )

    def test_tampered_preview_row_fails_before_stages_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, row = _preview_fixture(root)
            tampered = copy.deepcopy(row)
            tampered["edit_instruction"] += " Tampered."
            manifest.write_bytes(materializer.canonical_json_bytes(tampered) + b"\n")
            media_stage = FakeMediaStage()
            with self.assertRaisesRegex(
                materializer.MaterializationError, "row digest mismatch"
            ):
                materializer.materialize_action_payloads(
                    preview_manifest=manifest,
                    output_dir=root / "output",
                    allow_preview_exploration=True,
                    media_stage=media_stage,
                    encoder_stage=FakeEncoderStage(),
                )
            self.assertEqual(media_stage.seen, [])

    def test_custom_media_stage_cannot_forge_geometry_in_metadata(self) -> None:
        class WrongGeometryStage(FakeMediaStage):
            def prepare(self, item):
                media = super().prepare(item)
                media.source_video = torch.zeros(
                    (3, self.temporal_sampling.materialized_frame_count, 2, 2),
                    dtype=torch.float32,
                )
                return media

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _row = _preview_fixture(root)
            with self.assertRaisesRegex(
                materializer.MaterializationError,
                "prepared source tensor geometry differs",
            ):
                materializer.materialize_action_payloads(
                    preview_manifest=manifest,
                    output_dir=root / "output",
                    allow_preview_exploration=True,
                    media_stage=WrongGeometryStage(),
                    encoder_stage=FakeEncoderStage(),
                )


if __name__ == "__main__":
    unittest.main()
