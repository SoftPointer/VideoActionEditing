#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import materialize_saic_frame0_latent_v1 as frame0  # noqa: E402


class _Config:
    z_dim = 16


class _VAE:
    config = _Config()
    training = False

    def parameters(self):
        return ()


class Frame0MaterializerContractTest(unittest.TestCase):
    def test_registered_coordinate_is_one_frame_and_non_authoritative(self) -> None:
        self.assertEqual(frame0.LATENT_FRAME_COUNT, 1)
        self.assertEqual(frame0.TENSOR_KEY, "reference_frame0_latent")
        self.assertEqual(
            frame0.ARTIFACT_METADATA,
            {
                "schema_version": "bernini-saic-frame0-latent-artifact-v1",
                "coordinate": "bernini_source_rgb_frame0_vae_latent",
                "frame_contract": "source_rgb_index0_latent1",
                "artifact_role": "saic_common_visual_i0_reference_coordinate",
                "source": "sealed_exact81_source_rgb_frame0_wan_vae_mode",
                "posterior": "mode",
                "sampling": "false",
                "authority": "false",
            },
        )
        receipt = frame0.build_receipt(
            artifact={"x": 1},
            sealed_inputs={"x": 2},
            preprocessing={"x": 3},
            model_closure={"x": 4},
            encoding={"x": 5},
            runtime={"x": 6},
        )
        self.assertEqual(receipt["schema_version"], frame0.RECEIPT_SCHEMA)
        self.assertEqual(receipt["method"], frame0.METHOD)
        self.assertTrue(receipt["authority"])
        self.assertTrue(all(value is False for value in receipt["authority"].values()))
        unsigned = dict(receipt)
        declared = unsigned.pop("receipt_digest")
        self.assertEqual(frame0.object_sha256(unsigned), declared)

    def test_encoder_slices_exact_rgb_index_zero_once(self) -> None:
        import torch

        source = torch.arange(
            1 * 3 * 81 * 16 * 24, dtype=torch.float32
        ).reshape(1, 3, 81, 16, 24).contiguous()
        original = source.clone()
        calls = []

        def encoder(vae, pixels):
            self.assertIsInstance(vae, _VAE)
            calls.append(pixels.detach().clone())
            self.assertEqual(tuple(pixels.shape), (1, 3, 1, 16, 24))
            self.assertTrue(torch.equal(pixels, original[:, :, 0:1]))
            return torch.ones((1, 16, 1, 2, 3), dtype=torch.float32)

        latent, state = frame0.encode_source_frame0_once(
            _VAE(), source, encoder=encoder
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(tuple(latent.shape), (1, 16, 1, 2, 3))
        self.assertTrue(torch.equal(source, original))
        self.assertEqual(state["full_source_vae_encode_count"], 0)
        self.assertEqual(state["source_frame0_vae_encode_count"], 1)
        self.assertEqual(state["total_vae_encode_count"], 1)
        self.assertEqual(state["source_rgb_indices"], [0])
        self.assertIs(state["temporal_video_latent_slice_used"], False)
        self.assertEqual(
            state["source_frame0_pixels_raw_sha256"],
            state["source_frame0_pixels_after_sha256"],
        )

    def test_encoder_rejects_full_temporal_latent_and_input_mutation(self) -> None:
        import torch

        source = torch.zeros((1, 3, 81, 16, 24), dtype=torch.float32)

        def wrong_temporal(_vae, _pixels):
            return torch.zeros((1, 16, 21, 2, 3), dtype=torch.float32)

        with self.assertRaises(frame0.Frame0LatentMaterializationError):
            frame0.encode_source_frame0_once(_VAE(), source, encoder=wrong_temporal)

        def mutate(_vae, pixels):
            pixels.add_(1)
            return torch.zeros((1, 16, 1, 2, 3), dtype=torch.float32)

        with self.assertRaises(frame0.Frame0LatentMaterializationError):
            frame0.encode_source_frame0_once(_VAE(), source, encoder=mutate)

    def test_cli_requires_job132387_frame0_digest(self) -> None:
        parser = frame0.build_parser()
        options = {action.dest for action in parser._actions if action.required}
        self.assertIn("expected_reference_frame0_tensor_raw_sha256", options)
        self.assertIn("materialize_saic_frame0_latent_v1.py", frame0.RUNTIME_METHOD_FILES)
        self.assertNotIn(
            "materialize_saic_reference_frame0_latent_v1.py",
            frame0.RUNTIME_METHOD_FILES,
        )
        source = inspect.getsource(frame0.materialize_reference_frame0_latent)
        self.assertIn("job132387_frame0_tensor_raw_sha256_match", source)
        self.assertNotIn(
            "source-frame0 latent differs from the Job132387 I0 coordinate",
            source,
        )


if __name__ == "__main__":
    unittest.main()
