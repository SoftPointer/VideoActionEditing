from __future__ import annotations

import ast
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:  # Static/contract tests still run on CPU-only dev hosts.
    torch = None  # type: ignore[assignment]


METHOD_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = METHOD_ROOT / "materialize_dclr_source_condition.py"

import sys

if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import materialize_dclr_source_condition as source_only


class _Config(dict):
    def __getattr__(self, name: str):
        return self[name]


class _FakeVAE:
    def __init__(self, latent: torch.Tensor):
        self.config = _Config(
            {
                "_class_name": "AutoencoderKLWan",
                "z_dim": 16,
                "temperal_downsample": [False, True, True],
                "latents_mean": [float(index) for index in range(16)],
                "latents_std": [float(index + 1) for index in range(16)],
            }
        )
        self._latent = latent

    def encode(self, pixels: torch.Tensor):
        self.encoded_pixels = pixels
        return SimpleNamespace(
            latent_dist=SimpleNamespace(mode=lambda: self._latent)
        )


class SourceConditionMaterializerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SCRIPT_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_cli_exposes_only_one_raw_source_not_target_or_parquet(self) -> None:
        parser = source_only.build_parser()
        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        expected = {
            "-h",
            "--help",
            "--iid",
            "--source-video",
            "--expected-source-sha256",
            "--expected-bucket-hw",
            "--checkpoint",
            "--checkpoint-content-manifest",
            "--expected-checkpoint-tree-sha256",
            "--expected-checkpoint-content-manifest-sha256",
            "--expected-vae-config-sha256",
            "--output-dir",
            "--method-source-revision",
            "--method-source-archive-sha256",
        }
        self.assertEqual(option_strings, expected)
        for forbidden in (
            "--target",
            "--edited",
            "--parquet",
            "--posterior",
            "--source-column",
        ):
            self.assertFalse(any(forbidden in value for value in option_strings))
        self.assertNotIn("video_vae_latents", self.source)

    def test_implementation_does_not_enumerate_source_directory(self) -> None:
        forbidden_attributes = {"glob", "rglob", "iterdir", "listdir", "walk"}
        seen = {
            node.func.attr
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertTrue(forbidden_attributes.isdisjoint(seen))

    def test_pinned_vae_config_validation_is_fail_closed(self) -> None:
        value = {
            "_class_name": "AutoencoderKLWan",
            "z_dim": 16,
            "temperal_downsample": [False, True, True],
            "latents_mean": [0.0] * 16,
            "latents_std": [1.0] * 16,
        }
        result = source_only.validate_vae_config(value)
        self.assertEqual(result["class_name"], "AutoencoderKLWan")
        self.assertEqual(result["z_dim"], 16)
        for key, replacement in (
            ("_class_name", "OtherVAE"),
            ("z_dim", 32),
            ("temperal_downsample", [True, True, True]),
            ("latents_std", [1.0] * 15),
        ):
            malformed = dict(value)
            malformed[key] = replacement
            with self.subTest(key=key), self.assertRaises(
                source_only.SourceConditionMaterializationError
            ):
                source_only.validate_vae_config(malformed)
        malformed = dict(value)
        malformed["latents_std"] = [1.0] * 15 + [0.0]
        with self.assertRaises(source_only.SourceConditionMaterializationError):
            source_only.validate_vae_config(malformed)

    @unittest.skipIf(torch is None, "torch is unavailable on this static-test host")
    def test_normalization_uses_source_posterior_mode_channel_statistics(self) -> None:
        assert torch is not None
        channels = torch.arange(16, dtype=torch.float32).view(1, 16, 1, 1, 1)
        latent = channels.expand(1, 16, 21, 2, 2).contiguous() + 2.0
        vae = _FakeVAE(latent)
        pixels = torch.zeros((1, 3, 81, 16, 16), dtype=torch.float32)
        normalized = source_only.normalized_wan_vae_mode(vae, pixels)
        expected = torch.empty_like(latent)
        for channel in range(16):
            expected[:, channel] = 2.0 / float(channel + 1)
        self.assertTrue(torch.equal(vae.encoded_pixels, pixels))
        self.assertTrue(torch.allclose(normalized, expected, atol=0.0, rtol=0.0))
        self.assertEqual(normalized.dtype, torch.float32)
        self.assertFalse(normalized.requires_grad)

    def test_receipt_matches_runtime_source_only_schema_exactly(self) -> None:
        contract = {
            "iid": "7594dcea796540fc",
            "source": Path("/abs/source.mp4"),
            "source_sha256": "1" * 64,
            "expected_bucket_hw": (496, 480),
            "checkpoint_tree_sha256": source_only.EXPECTED_CHECKPOINT_TREE_SHA256,
            "checkpoint_manifest_sha256": source_only.EXPECTED_CHECKPOINT_MANIFEST_SHA256,
            "vae_config_sha256": source_only.EXPECTED_VAE_CONFIG_SHA256,
            "method_source_revision": "2" * 40,
            "method_source_archive_sha256": "3" * 64,
        }
        artifact = {
            "path": "/abs/source.normalized-clean-latent.safetensors",
            "sha256": "4" * 64,
            "tensor_key": "normalized_clean_latent",
            "shape": [1, 16, 21, 62, 60],
            "artifact_role": "source_video_condition",
            "coordinate": "bernini_normalized_clean_vae_latent",
            "frame_contract": "exact81_latent21",
            "metadata": dict(source_only.ARTIFACT_METADATA),
            "tensor_identity": {
                "shape": [1, 16, 21, 62, 60],
                "dtype": "torch.float32",
                "finite": True,
            },
            "stored_dtype": "torch.float32",
            "source_video_vae_encode_before_any_decode": True,
            "mp4_decode_reencode_used": False,
            "roundtrip_tensor_exact": True,
        }
        receipt = source_only.build_receipt(
            contract=contract,
            source_metadata={"source_derived_bucket_hw": [496, 480]},
            checkpoint=Path("/abs/checkpoint"),
            checkpoint_identity={"every_file_sha256_verified": True},
            vae_config_identity={"class_name": "AutoencoderKLWan"},
            artifact=artifact,
            runtime={"device_count": 1},
        )
        self.assertEqual(
            receipt["schema_version"],
            "bernini-source-only-vae-materialization-v1",
        )
        self.assertIs(receipt["source_only"], True)
        self.assertEqual(
            receipt["access_audit"]["source_columns_accessed"],
            ["iid", "source_video", "source_video_sha256"],
        )
        self.assertEqual(receipt["access_audit"]["target_columns_accessed"], [])
        self.assertIs(receipt["access_audit"]["target_media_accessed"], False)
        self.assertIs(receipt["access_audit"]["paired_target_accessed"], False)
        self.assertEqual(receipt["input"]["source_iid"], contract["iid"])
        self.assertEqual(
            receipt["input"]["source_video_sha256"], contract["source_sha256"]
        )
        self.assertEqual(receipt["source_condition_artifact"], artifact)
        unsigned = dict(receipt)
        digest = unsigned.pop("receipt_digest")
        self.assertEqual(digest, source_only.legacy.object_sha256(unsigned))

    def test_artifact_metadata_is_runtime_loader_compatible(self) -> None:
        self.assertEqual(
            source_only.ARTIFACT_METADATA,
            {
                "coordinate": "bernini_normalized_clean_vae_latent",
                "frame_contract": "exact81_latent21",
                "artifact_role": "source_video_condition",
                "source": "source_video_vae_encode_before_any_decode",
            },
        )
        self.assertEqual(source_only.ARTIFACT_NAME, "source.normalized-clean-latent.safetensors")
        self.assertEqual(source_only.LATENT_PHASES, 21)
        self.assertEqual(source_only.VAE_CHANNELS, 16)


if __name__ == "__main__":
    unittest.main()
