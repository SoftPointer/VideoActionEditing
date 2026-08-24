from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from tools import materialize_ramp_motion_analogy_vae as ramp  # noqa: E402

try:
    import numpy as np
    import torch
except ImportError:  # Dependency-light local host still runs contract tests.
    np = None
    torch = None


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fixture_row(root: Path, *, kind: str = "reverse", parameter: float = 0.0):
    source = root / "source.mp4"
    donor = root / "donor.mp4"
    source.write_bytes(b"source-A")
    donor.write_bytes(b"donor-B")
    row = {
        "schema_version": ramp.MANIFEST_ROW_FORMAT,
        "row_id": "example-001",
        "source_video_path": str(source.resolve()),
        "source_video_sha256": ramp.file_sha256(source),
        "donor_video_path": str(donor.resolve()),
        "donor_video_sha256": ramp.file_sha256(donor),
        "program_kind": kind,
        "program_parameter": parameter,
        "program_parameter_hex": float(parameter).hex(),
    }
    row["manifest_row_digest"] = ramp.object_sha256(row)
    return row, source, donor


class ManifestContractTests(unittest.TestCase):
    def test_hash_bound_manifest_preserves_exact_parameter_hex(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row, _, _ = _fixture_row(root, kind="speed_up", parameter=0.5)
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            loaded = ramp.load_manifest(
                manifest.resolve(), expected_sha256=ramp.file_sha256(manifest)
            )
            self.assertEqual(len(loaded.rows), 1)
            parsed = loaded.rows[0]
            self.assertEqual(parsed.row_id, "example-001")
            self.assertEqual(parsed.program_parameter_hex, float(0.5).hex())
            self.assertEqual(parsed.manifest_row_digest, row["manifest_row_digest"])

    def test_rejects_target_surface_duplicate_keys_and_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row, source, _ = _fixture_row(root)
            with self.subTest("external target"):
                bad = dict(row)
                bad.pop("manifest_row_digest")
                bad["target_video_path"] = "/forbidden/target.mp4"
                with self.assertRaisesRegex(
                    ramp.RampVaeMaterializationError,
                    "forbidden/unrecognized fields",
                ):
                    ramp.validate_manifest_row(bad)
            with self.subTest("parameter hex"):
                bad = dict(row)
                bad.pop("manifest_row_digest")
                bad["program_parameter_hex"] = float(1.0).hex()
                with self.assertRaisesRegex(
                    ramp.RampVaeMaterializationError, "parameter_hex differs"
                ):
                    ramp.validate_manifest_row(bad)
            with self.subTest("media tamper"):
                source.write_bytes(b"tampered")
                with self.assertRaisesRegex(
                    ramp.RampVaeMaterializationError, "source A SHA-256 differs"
                ):
                    ramp.validate_manifest_row(row)
            with self.subTest("duplicate key"):
                manifest = root / "duplicate.jsonl"
                text = json.dumps(row)
                text = text[:-1] + ',"row_id":"second"}\n'
                manifest.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(
                    ramp.RampVaeMaterializationError, "duplicate JSON object key"
                ):
                    ramp.load_manifest(manifest.resolve())

    def test_requires_distinct_paths_and_sha_identities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row, source, _ = _fixture_row(root)
            same_path = dict(row)
            same_path.pop("manifest_row_digest")
            same_path["donor_video_path"] = str(source.resolve())
            same_path["donor_video_sha256"] = ramp.file_sha256(source)
            with self.assertRaisesRegex(
                ramp.RampVaeMaterializationError, "paths must differ"
            ):
                ramp.validate_manifest_row(same_path)

            donor_copy = root / "copy.mp4"
            donor_copy.write_bytes(source.read_bytes())
            same_hash = dict(row)
            same_hash.pop("manifest_row_digest")
            same_hash["donor_video_path"] = str(donor_copy.resolve())
            same_hash["donor_video_sha256"] = ramp.file_sha256(donor_copy)
            with self.assertRaisesRegex(
                ramp.RampVaeMaterializationError, "identities must differ"
            ):
                ramp.validate_manifest_row(same_hash)

    def test_cli_has_no_target_or_privileged_geometry_input(self) -> None:
        option_strings = {
            option
            for action in ramp.build_parser()._actions
            for option in action.option_strings
        }
        for forbidden in (
            "--target",
            "--mask",
            "--flow",
            "--pose",
            "--track",
            "--box",
            "--trajectory",
            "--edited-first-frame",
        ):
            self.assertNotIn(forbidden, option_strings)
        parameters = set(inspect.signature(ramp.prepare_motion_analogy_rgb).parameters)
        self.assertEqual(parameters, {"row", "max_pixels", "stride"})


class PinnedVaeIdentityTests(unittest.TestCase):
    def test_verifies_every_manifest_bound_vae_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            vae = checkpoint / "vae"
            vae.mkdir(parents=True)
            config = vae / "config.json"
            weights = vae / "diffusion_pytorch_model.safetensors"
            config.write_bytes(b"config")
            weights.write_bytes(b"weights")
            manifest = root / "checkpoint.sha256"
            manifest.write_text(
                f"{ramp.file_sha256(config)}  ./vae/config.json\n"
                f"{ramp.file_sha256(weights)}  ./vae/diffusion_pytorch_model.safetensors\n",
                encoding="utf-8",
            )
            identity = ramp.validate_pinned_vae_checkpoint(
                checkpoint.resolve(),
                manifest.resolve(),
                expected_manifest_sha256=ramp.file_sha256(manifest),
                expected_vae_config_sha256=ramp.file_sha256(config),
            )
            self.assertTrue(identity["every_vae_file_sha256_verified"])
            self.assertEqual(
                identity["posterior_representation"],
                "latent_dist.parameters_fp32",
            )
            weights.write_bytes(b"changed")
            with self.assertRaisesRegex(
                ramp.RampVaeMaterializationError, "file hash differs"
            ):
                ramp.validate_pinned_vae_checkpoint(
                    checkpoint.resolve(),
                    manifest.resolve(),
                    expected_manifest_sha256=ramp.file_sha256(manifest),
                    expected_vae_config_sha256=ramp.file_sha256(config),
                )


@unittest.skipIf(torch is None or np is None, "torch/numpy are unavailable")
class RgbAndParquetMaterializationTests(unittest.TestCase):
    @staticmethod
    def _decoded_videos():
        assert np is not None
        timeline = np.arange(ramp.FRAME_COUNT, dtype=np.uint8)
        source = np.broadcast_to(
            timeline[:, None, None, None], (ramp.FRAME_COUNT, 8, 12, 3)
        ).copy()
        donor = np.broadcast_to(
            (timeline * 2)[:, None, None, None],
            (ramp.FRAME_COUNT, 10, 10, 3),
        ).copy()
        return source, donor

    def test_rgb_builder_runs_before_vae_and_does_not_share_i0(self) -> None:
        assert torch is not None
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, _, _ = _fixture_row(root)
            row = ramp.validate_manifest_row(raw)
            source, donor = self._decoded_videos()
            decoded = [
                (source, ramp.FPS, (8, 12)),
                (donor, ramp.FPS, (10, 10)),
            ]
            with mock.patch.object(
                ramp.base, "_decode_exact_video", side_effect=decoded
            ):
                example, media = ramp.prepare_motion_analogy_rgb(
                    row, max_pixels=96, stride=4
                )
            self.assertEqual(
                tuple(example.source_identity_video.shape), (1, 3, 81, 8, 12)
            )
            self.assertTrue(
                torch.equal(
                    example.regression_target_video,
                    example.source_identity_video.flip(2),
                )
            )
            self.assertTrue(
                torch.equal(
                    example.motion_donor_after_video,
                    example.motion_donor_before_video.flip(2),
                )
            )
            self.assertFalse(media["shared_i0_used"])
            self.assertFalse(media["direct_21_phase_permutation_authorized"])
            self.assertTrue(
                torch.equal(
                    example.regression_target_video[:, :, 0],
                    example.source_identity_video[:, :, 80],
                )
            )

    def test_four_independent_role_order_and_create_only_receipt(self) -> None:
        try:
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow is unavailable")
        assert torch is not None

        class FakeEncoder:
            def __init__(self):
                self.calls = []
                self.identity = {
                    "vae_identity_digest": "9" * 64,
                    "posterior_representation": "latent_dist.parameters_fp32",
                    "every_vae_file_sha256_verified": True,
                }

            def encode(self, video):
                self.calls.append(video.detach().clone())
                blob = f"posterior-call-{len(self.calls)}".encode("ascii")
                return blob, {
                    "posterior_parameters_shape": [1, 32, 21, 1, 1],
                    "posterior_parameters_dtype": "torch.float32",
                    "posterior_parameters_tensor_sha256": _sha(blob),
                }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, _, _ = _fixture_row(root)
            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
            manifest = ramp.load_manifest(manifest_path.resolve())
            source, donor = self._decoded_videos()
            decoded = [
                (source, ramp.FPS, (8, 12)),
                (donor, ramp.FPS, (10, 10)),
            ]
            encoder = FakeEncoder()
            output = root / "output"
            with mock.patch.object(
                ramp.base, "_decode_exact_video", side_effect=decoded
            ):
                result = ramp.materialize_one(
                    manifest.rows[0],
                    manifest=manifest,
                    encoder=encoder,
                    output_root=output,
                    max_pixels=96,
                    stride=4,
                )
            self.assertEqual(len(encoder.calls), 4)
            self.assertTrue(torch.equal(encoder.calls[2], encoder.calls[1].flip(1)))
            self.assertTrue(torch.equal(encoder.calls[3], encoder.calls[0].flip(1)))
            persisted = pq.read_table(result["parquet_path"]).to_pylist()[0]
            self.assertEqual(
                [persisted[field] for field in ramp.ROLE_TO_BLOB_FIELD.values()],
                [
                    b"posterior-call-1",
                    b"posterior-call-2",
                    b"posterior-call-3",
                    b"posterior-call-4",
                ],
            )
            receipt = json.loads(Path(result["receipt_path"]).read_text())
            unsigned = dict(receipt)
            digest = unsigned.pop("receipt_digest")
            self.assertEqual(digest, ramp.object_sha256(unsigned))
            self.assertFalse(receipt["shared_i0_used"])
            self.assertFalse(receipt["training_authorized"])
            self.assertFalse(receipt["action_training_authorized"])
            self.assertFalse(receipt["direct_21_phase_permutation_authorized"])
            with self.assertRaisesRegex(
                ramp.RampVaeMaterializationError, "create-only"
            ):
                ramp.materialize_one(
                    manifest.rows[0],
                    manifest=manifest,
                    encoder=encoder,
                    output_root=output,
                    max_pixels=96,
                    stride=4,
                )


if __name__ == "__main__":
    unittest.main()
