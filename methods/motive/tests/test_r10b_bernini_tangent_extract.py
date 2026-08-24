from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - local lightweight environments
    torch = None

try:
    import PIL
except ImportError:  # pragma: no cover - local lightweight environments
    PIL = None

from motive import r10b_bernini_tangent_extract as bernini_extract
from motive.r10b_bernini_tangent_extract import (
    R10BBerniniExtractError,
    _build_run_contract,
    _checkpoint_manifest,
    _checkpoint_revision_metadata,
    _git_blob_digest,
    _load_paired_video_frames,
    _load_fixed_tokenizer,
    _noise_for_mode,
    _pack_source_and_target,
    _resize_frame,
    _resize_transform,
    _scheduler_point,
    _track_saliency_for_row,
    _unpack_target_prediction,
    _validate_run_contract,
    _validate_tokenizer_provenance,
)


class R10BBerniniCheckpointBindingTests(unittest.TestCase):
    @staticmethod
    def _minimal_checkpoint(root: Path) -> Path:
        for component in (
            "scheduler",
            "text_encoder",
            "tokenizer",
            "transformer",
            "vae",
        ):
            (root / component).mkdir()
        (root / "config.json").write_bytes(b"{}")
        return root / "config.json"

    def test_checkpoint_manifest_binds_file_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._minimal_checkpoint(root)
            expected = {
                "config.json": (config.stat().st_size, _git_blob_digest(config))
            }
            with mock.patch.object(
                bernini_extract,
                "EXPECTED_CHECKPOINT_FILES",
                expected,
            ):
                manifest = _checkpoint_manifest(root)
                self.assertEqual(manifest["file_count"], 1)
                config.write_bytes(b"[]")
                with self.assertRaisesRegex(
                    R10BBerniniExtractError,
                    "checkpoint content differs",
                ):
                    _checkpoint_manifest(root)

    def test_revision_metadata_binds_every_expected_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._minimal_checkpoint(root)
            identity = _git_blob_digest(config)
            expected = {"config.json": (config.stat().st_size, identity)}
            metadata = root / ".cache/huggingface/download/config.json.metadata"
            metadata.parent.mkdir(parents=True)
            metadata.write_text(
                f"{bernini_extract.EXPECTED_REPO_REVISION}\n{identity}\n0\n",
                encoding="utf-8",
            )
            with mock.patch.object(
                bernini_extract,
                "EXPECTED_CHECKPOINT_FILES",
                expected,
            ):
                result = _checkpoint_revision_metadata(root)
                self.assertEqual(result["metadata_files"], 1)
                metadata.write_text(
                    f"{bernini_extract.EXPECTED_REPO_REVISION}\n"
                    f"{'0' * 40}\n0\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    R10BBerniniExtractError,
                    "checkpoint metadata differs",
                ):
                    _checkpoint_revision_metadata(root)


class R10BBerniniTokenizerContractTests(unittest.TestCase):
    def test_loader_explicitly_enables_corrected_regex(self) -> None:
        class FakeAutoTokenizer:
            observed_args = None
            observed_kwargs = None

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                cls.observed_args = args
                cls.observed_kwargs = kwargs
                return object()

        model_path = Path("/fixed/bernini")
        _load_fixed_tokenizer(FakeAutoTokenizer, model_path)
        self.assertEqual(FakeAutoTokenizer.observed_args, (str(model_path),))
        self.assertEqual(
            FakeAutoTokenizer.observed_kwargs,
            {
                "subfolder": "tokenizer",
                "local_files_only": True,
                "fix_mistral_regex": True,
            },
        )

    @staticmethod
    def _valid_provenance():
        contract = bernini_extract.TOKENIZER_CONTRACT
        version = "5.5.4"
        summary = {
            "runtime": {"transformers_version": version},
            "measurement": {
                "prompt_conditioning": {
                    "mode": bernini_extract.PROMPT_MODE,
                    "tokenizer": {
                        **contract,
                        "contract_sha256": bernini_extract.object_digest(
                            contract
                        ),
                        "transformers_version": version,
                    },
                }
            },
        }
        rows = [
            {
                "prompt_conditioning": {
                    "tokenizer_fix_mistral_regex": True,
                    "tokenizer_contract_sha256": (
                        bernini_extract.object_digest(contract)
                    ),
                }
            }
        ]
        return summary, rows

    def test_validator_accepts_fixed_tokenizer_contract(self) -> None:
        summary, rows = self._valid_provenance()
        _validate_tokenizer_provenance(summary, rows)

    def test_validator_rejects_silent_legacy_fallback(self) -> None:
        summary, rows = self._valid_provenance()
        rows[0]["prompt_conditioning"][
            "tokenizer_fix_mistral_regex"
        ] = False
        with self.assertRaisesRegex(
            R10BBerniniExtractError,
            "row tokenizer provenance differs",
        ):
            _validate_tokenizer_provenance(summary, rows)

    def test_contract_records_reason_and_fail_closed_compatibility(self) -> None:
        contract = bernini_extract.TOKENIZER_CONTRACT
        self.assertIs(contract["fix_mistral_regex"], True)
        self.assertIn("different token IDs", contract["rationale"])
        self.assertIn("fail", contract["version_compatibility"])


class R10BBerniniTrackBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        frames, tracks = 4, 3
        source = np.zeros((2, frames, tracks, 2), dtype=np.float32)
        source[:, :, :, 0] = [0.2, 0.5, 0.8]
        source[:, :, :, 1] = [0.3, 0.5, 0.7]
        target = source.copy()
        target[0, :, 0, 0] += [0.00, 0.02, 0.06, 0.12]
        target[1, :, 1, 1] += [0.00, 0.04, 0.12, 0.24]

        visibility = np.ones((2, frames, tracks), dtype=np.float32)
        self.cache = {
            "input_indices": np.asarray([11, 29], dtype=np.int64),
            "source_stabilized_tracks": source,
            "target_stabilized_tracks": target,
            "source_visibility": visibility,
            "target_visibility": visibility,
        }

    def test_each_row_uses_its_own_cache_index(self) -> None:
        row_zero = {"iid": "zero", "track_cache_index": 0, "track_input_index": 11}
        row_one = {"iid": "one", "track_cache_index": 1, "track_input_index": 29}
        mask_zero, metrics_zero = _track_saliency_for_row(
            row_zero,
            self.cache,
        )
        mask_one, metrics_one = _track_saliency_for_row(
            row_one,
            self.cache,
        )
        self.assertEqual(mask_zero.shape, mask_one.shape)
        self.assertGreater(
            metrics_one["track_delta_energy"],
            metrics_zero["track_delta_energy"],
        )
        self.assertFalse((mask_zero == mask_one).all())

    def test_cache_input_binding_mismatch_fails_closed(self) -> None:
        row = {"iid": "bad", "track_cache_index": 0, "track_input_index": 29}
        with self.assertRaisesRegex(
            R10BBerniniExtractError,
            "track input/cache binding differs",
        ):
            _track_saliency_for_row(row, self.cache)


class R10BBerniniPilotModeContractTests(unittest.TestCase):
    @staticmethod
    def _contract(**overrides):
        values = {
            "artifact_kind": "controlled_retrieval_pilot",
            "scheduler_class": "FakeScheduler",
            "scheduler_steps": 50,
            "scheduler_index": 25,
            "scheduler_timestep": 500.0,
            "scheduler_sigma": 0.55,
            "noise_mode": "iid_spatiotemporal",
            "diffusion_noise_seed": 260108853,
            "resize_mode": "aspect_preserving_center_crop",
            "width": 256,
            "height": 256,
            "num_frames": 17,
        }
        values.update(overrides)
        return _build_run_contract(**values)

    @staticmethod
    def _bound_artifact(contract):
        digest = bernini_extract.object_digest(contract)
        resize = contract["resize"]
        measurement = {
            "run_contract": contract,
            "run_contract_sha256": digest,
            "noise_mode": contract["noise"]["mode"],
            "diffusion_noise_seed": contract["noise"]["seed"],
            "scheduler_class": contract["scheduler"]["class"],
            "scheduler_steps": contract["scheduler"]["steps"],
            "scheduler_index": contract["scheduler"]["index"],
            "scheduler_timestep": contract["scheduler"]["timestep"],
            "scheduler_sigma": contract["scheduler"]["sigma"],
            "resize_policy": {
                **resize,
                "technical_smoke_only": (
                    resize["mode"] == "exact_technical"
                ),
                "scientific_retrieval_promotion_eligible": False,
            },
        }
        summary = {
            "artifact_kind": contract["artifact_kind"],
            "measurement": measurement,
            "runtime": {
                "width": resize["width"],
                "height": resize["height"],
                "num_frames": resize["num_frames"],
            },
        }
        done = {
            "artifact_kind": contract["artifact_kind"],
            "run_contract_sha256": digest,
        }
        spatial_transform = {
            "input_width": resize["width"],
            "input_height": resize["height"],
            **_resize_transform(
                resize["width"],
                resize["height"],
                output_width=resize["width"],
                output_height=resize["height"],
                mode=resize["mode"],
            ),
        }
        source_transforms = [dict(spatial_transform) for _ in range(17)]
        target_transforms = [dict(spatial_transform) for _ in range(17)]
        media = {
            "resize_mode": resize["mode"],
            "source_decoded_frame_count": 17,
            "target_decoded_frame_count": 17,
            "shared_sampling_frame_count": 17,
            "selected_frame_indices": list(range(17)),
            "source_target_frame_indices_identical": True,
            "output_width": resize["width"],
            "output_height": resize["height"],
            "source_spatial_transforms": source_transforms,
            "target_spatial_transforms": target_transforms,
            "source_spatial_transforms_sha256": (
                bernini_extract.object_digest(source_transforms)
            ),
            "target_spatial_transforms_sha256": (
                bernini_extract.object_digest(target_transforms)
            ),
        }
        rows = [
            {
                "artifact_kind": contract["artifact_kind"],
                "run_contract_sha256": digest,
                "media_preprocessing": media,
            }
        ]
        return summary, rows, done

    def test_scheduler_uses_requested_index_and_rejects_bounds(self) -> None:
        class FakeScheduler:
            def set_timesteps(self, steps, device=None):
                self.timesteps = [1000.0 - index for index in range(steps)]
                self.sigmas = [0.99 - index / (steps + 2) for index in range(steps)]

        scheduler = FakeScheduler()
        timestep, sigma, index = _scheduler_point(
            scheduler,
            steps=50,
            index=17,
            device="cpu",
        )
        self.assertEqual(index, 17)
        self.assertEqual(timestep, scheduler.timesteps[17])
        self.assertEqual(sigma, scheduler.sigmas[17])
        with self.assertRaisesRegex(
            R10BBerniniExtractError,
            "out of range",
        ):
            _scheduler_point(
                scheduler,
                steps=50,
                index=50,
                device="cpu",
            )

    def test_cli_defaults_are_explicit_engineering_smoke_settings(self) -> None:
        with mock.patch(
            "sys.argv",
            [
                "r10b_bernini_tangent_extract",
                "--validate-only",
                "--output-dir",
                "artifact",
            ],
        ):
            args = bernini_extract._parse_args()
        self.assertEqual(args.artifact_kind, "engineering_smoke")
        self.assertEqual(args.scheduler_steps, 50)
        self.assertEqual(args.scheduler_index, 25)
        self.assertEqual(args.noise_mode, "temporal_broadcast")
        self.assertEqual(args.resize_mode, "exact_technical")

    def test_controlled_pilot_requires_aspect_preserving_crop(self) -> None:
        with self.assertRaisesRegex(
            R10BBerniniExtractError,
            "requires aspect_preserving_center_crop",
        ):
            self._contract(resize_mode="exact_technical")

    def test_validator_binds_every_run_setting(self) -> None:
        contract = self._contract()
        summary, rows, done = self._bound_artifact(contract)
        self.assertEqual(
            _validate_run_contract(summary, rows, done),
            contract,
        )
        summary["measurement"]["scheduler_index"] = 24
        with self.assertRaisesRegex(
            R10BBerniniExtractError,
            "flat measurement/run-contract",
        ):
            _validate_run_contract(summary, rows, done)

    @unittest.skipUnless(PIL is not None, "Pillow is required")
    def test_resize_geometry_is_aspect_preserving_center_crop(self) -> None:
        transform = _resize_transform(
            8,
            4,
            output_width=4,
            output_height=4,
            mode="aspect_preserving_center_crop",
        )
        self.assertEqual(
            transform,
            {
                "resized_width": 8,
                "resized_height": 4,
                "crop_left": 2,
                "crop_top": 0,
                "crop_right": 6,
                "crop_bottom": 4,
            },
        )
        frame = np.zeros((4, 8, 3), dtype=np.uint8)
        resized, recorded = _resize_frame(
            frame,
            width=4,
            height=4,
            mode="aspect_preserving_center_crop",
        )
        self.assertEqual(resized.size, (4, 4))
        self.assertEqual(recorded["input_width"], 8)
        self.assertEqual(recorded["crop_left"], 2)

    @unittest.skipUnless(PIL is not None, "Pillow is required")
    def test_pair_loader_uses_one_absolute_index_list(self) -> None:
        source = [
            np.full((4, 8, 3), index, dtype=np.uint8)
            for index in range(9)
        ]
        target = [
            np.full((4, 8, 3), 100 + index, dtype=np.uint8)
            for index in range(10)
        ]
        with mock.patch.object(
            bernini_extract,
            "_decode_video_frames",
            side_effect=[source, target],
        ):
            source_frames, target_frames, metadata = (
                _load_paired_video_frames(
                    Path("source.mp4"),
                    Path("target.mp4"),
                    width=4,
                    height=4,
                    num_frames=5,
                    resize_mode="aspect_preserving_center_crop",
                )
            )
        self.assertEqual(metadata["selected_frame_indices"], [0, 2, 4, 6, 8])
        self.assertEqual(metadata["shared_sampling_frame_count"], 9)
        self.assertIs(
            metadata["source_target_frame_indices_identical"],
            True,
        )
        self.assertEqual(len(source_frames), len(target_frames))


@unittest.skipUnless(torch is not None, "PyTorch is required")
class R10BBerniniPackingTests(unittest.TestCase):
    def test_iid_noise_is_seeded_full_shape_and_not_broadcast(self) -> None:
        reference = torch.zeros((1, 2, 4, 3, 3), dtype=torch.float32)
        first = _noise_for_mode(
            reference,
            seed=12345,
            mode="iid_spatiotemporal",
        )
        second = _noise_for_mode(
            reference,
            seed=12345,
            mode="iid_spatiotemporal",
        )
        different = _noise_for_mode(
            reference,
            seed=12346,
            mode="iid_spatiotemporal",
        )
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual(first.shape, reference.shape)
        self.assertFalse(torch.equal(first, different))
        self.assertFalse(torch.equal(first[:, :, 0], first[:, :, 1]))

    def test_temporal_broadcast_mode_reuses_spatial_draw(self) -> None:
        reference = torch.zeros((1, 2, 4, 3, 3), dtype=torch.float32)
        noise = _noise_for_mode(
            reference,
            seed=12345,
            mode="temporal_broadcast",
        )
        torch.testing.assert_close(
            noise[:, :, 0],
            noise[:, :, 1],
            rtol=0,
            atol=0,
        )

    def test_unpack_target_prediction_inverts_official_patch_order(self) -> None:
        spatial = torch.arange(
            1 * 2 * 2 * 4 * 6,
            dtype=torch.float32,
        ).reshape(1, 2, 2, 4, 6)
        packed = (
            spatial.reshape(1, 2, 2, 1, 2, 2, 3, 2)
            .permute(0, 2, 4, 6, 3, 5, 7, 1)
            .reshape(1, 2 * 2 * 3, 1 * 2 * 2 * 2)
        )
        restored = _unpack_target_prediction(
            packed,
            spatial,
            patch_size=(1, 2, 2),
        )
        torch.testing.assert_close(restored, spatial)

    def test_pack_uses_source_id_one_then_target_id_zero(self) -> None:
        class FakeTransformer:
            dtype = torch.float32

            def __init__(self) -> None:
                self.source_ids = []

            def patch_vae_latent(self, values, source_id=None):
                self.source_ids.append(source_id)
                tokens = torch.full(
                    (1, 3, 4),
                    float(source_id),
                    dtype=values.dtype,
                    device=values.device,
                )
                rope = torch.full(
                    (1, 1, 3, 2),
                    complex(float(source_id), 0.0),
                    dtype=torch.complex128,
                    device=values.device,
                )
                return tokens, rope

        transformer = FakeTransformer()
        reference = torch.zeros((1, 2, 2, 4, 4))
        hidden, rotary, mask, lengths = _pack_source_and_target(
            transformer,
            reference,
            reference,
        )
        self.assertEqual(transformer.source_ids, [1, 0])
        self.assertEqual(tuple(hidden.shape), (1, 6, 4))
        self.assertEqual(tuple(rotary.shape), (1, 1, 6, 2))
        self.assertEqual(mask.tolist(), [False, False, False, True, True, True])
        self.assertEqual(lengths, [6])


if __name__ == "__main__":
    unittest.main()
