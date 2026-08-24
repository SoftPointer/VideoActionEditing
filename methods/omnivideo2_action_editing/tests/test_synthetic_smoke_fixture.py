from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pact.dataset import (  # noqa: E402
    ENCODER_CONTRACT_FORMAT,
    AtomicLatentDataset,
    encoder_contract_sha256,
)
from pact.manifest import (  # noqa: E402
    canonical_json_bytes,
    file_sha256,
    load_jsonl,
    verify_post_generation_release,
)
from pact.training import build_edit_support, validate_training_config  # noqa: E402
from tools.build_synthetic_smoke_fixture import (  # noqa: E402
    FIXTURE_SCHEMA,
    GLOBAL_ROW_SCHEMA,
    SYNTHETIC_UMT5_MANIFEST_SHA256,
    SYNTHETIC_UMT5_PREPROCESSING_SHA256,
    SYNTHETIC_VAE_CHECKPOINT_SHA256,
    SYNTHETIC_VAE_PREPROCESSING_SHA256,
    SYNTHETIC_VLM_FEATURE_CONTRACT_SHA256,
    SYNTHETIC_VLM_MANIFEST_SHA256,
    SyntheticFixtureError,
    build_synthetic_smoke_fixture,
)


class SyntheticSmokeFixtureTest(unittest.TestCase):
    def test_builds_real_signed_and_bound_one_step_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fixture"
            summary = build_synthetic_smoke_fixture(output, seed=17)

            self.assertEqual(summary["authorized_atoms"], 1)
            self.assertEqual(summary["latent_shape"], [16, 3, 8, 8])
            self.assertTrue(summary["ephemeral_private_key_destroyed"])
            self.assertFalse(
                (output / "synthetic_trust" / "synthetic_release_ed25519").exists()
            )

            marker_path = output / "SYNTHETIC_INTEGRATION_ONLY.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            self.assertEqual(marker["schema_version"], FIXTURE_SCHEMA)
            self.assertTrue(marker["production_training_forbidden"])
            self.assertFalse(marker["contains_real_video"])
            self.assertEqual(
                marker_path.read_bytes(), canonical_json_bytes(marker) + b"\n"
            )

            verified = verify_post_generation_release(
                global_manifest_path=output
                / "signed_release"
                / "global_manifest.jsonl",
                release_receipt_path=output
                / "signed_release"
                / "post_generation_release.json",
                public_key_path=output
                / "synthetic_trust"
                / "synthetic_release_ed25519.pub",
                expected_signer_fingerprint=summary[
                    "release_signer_fingerprint"
                ],
                row_schema_version=GLOBAL_ROW_SCHEMA,
            )
            self.assertEqual(verified.release_id, summary["release_id"])

            atoms = load_jsonl(output / "atomic" / "atomic_manifest.jsonl")
            self.assertEqual(len(atoms), 1)
            self.assertTrue(atoms[0]["training_authorized"])
            self.assertEqual(
                atoms[0]["post_generation_release"]["receipt_sha256"],
                file_sha256(
                    output
                    / "signed_release"
                    / "post_generation_release.json"
                ),
            )

            dataset = AtomicLatentDataset(
                output / "training" / "training_manifest.jsonl",
                payload_root=output / "payloads",
            )
            sample = dataset[0]
            self.assertEqual(tuple(sample["source_latent"].shape), (16, 3, 8, 8))
            self.assertEqual(tuple(sample["text_context"].shape), (2, 4096))
            self.assertEqual(tuple(sample["vlm_context"].shape), (3, 2048))
            contract = sample["encoder_contract"]
            self.assertEqual(contract["format"], ENCODER_CONTRACT_FORMAT)
            self.assertEqual(
                contract["vae"]["checkpoint_sha256"],
                SYNTHETIC_VAE_CHECKPOINT_SHA256,
            )
            self.assertEqual(
                contract["vae"]["preprocessing_contract_sha256"],
                SYNTHETIC_VAE_PREPROCESSING_SHA256,
            )
            self.assertEqual(
                contract["umt5"]["checkpoint_manifest_sha256"],
                SYNTHETIC_UMT5_MANIFEST_SHA256,
            )
            self.assertEqual(
                contract["umt5"]["preprocessing_contract_sha256"],
                SYNTHETIC_UMT5_PREPROCESSING_SHA256,
            )
            self.assertEqual(
                contract["vlm"]["checkpoint_manifest_sha256"],
                SYNTHETIC_VLM_MANIFEST_SHA256,
            )
            self.assertEqual(
                contract["vlm"]["feature_extraction_contract_sha256"],
                SYNTHETIC_VLM_FEATURE_CONTRACT_SHA256,
            )
            self.assertEqual(
                sample["encoder_contract_sha256"],
                encoder_contract_sha256(contract),
            )
            self.assertEqual(
                summary["encoder_contract_sha256"],
                sample["encoder_contract_sha256"],
            )
            edit_mask, _ = build_edit_support(
                sample["source_component_mask"].unsqueeze(0),
                sample["target_component_mask"].unsqueeze(0),
                dilation_radius=(0, 1, 1),
                feather_radius=(0, 1, 1),
            )
            self.assertGreater(float(edit_mask.sum()), 0.0)
            self.assertGreater(float((1.0 - edit_mask).sum()), 0.0)

            config = validate_training_config(
                json.loads(
                    (
                        output
                        / "configs"
                        / "pact_1_3b_one_step_smoke.json"
                    ).read_text(encoding="utf-8")
                )
            )
            self.assertEqual(
                config["training"],
                {
                    "epochs": 1,
                    "batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "num_workers": 0,
                    "max_steps": 1,
                    "checkpoint_every": 1,
                    "log_every": 1,
                },
            )
            done = json.loads((output / "done.json").read_text(encoding="utf-8"))
            self.assertTrue(done["complete"])
            self.assertEqual(done["summary_sha256"], file_sha256(output / "summary.json"))

    def test_output_is_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fixture"
            build_synthetic_smoke_fixture(output)
            before = (output / "done.json").read_bytes()

            with self.assertRaisesRegex(
                SyntheticFixtureError, "already exists; refusing overwrite"
            ):
                build_synthetic_smoke_fixture(output)

            self.assertEqual((output / "done.json").read_bytes(), before)

    def test_same_seed_reproduces_all_training_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = [root / "fixture_a", root / "fixture_b"]
            for output in outputs:
                build_synthetic_smoke_fixture(output, seed=1234)
            samples = [
                AtomicLatentDataset(
                    output / "training" / "training_manifest.jsonl",
                    payload_root=output / "payloads",
                )[0]
                for output in outputs
            ]
            for field in (
                "source_latent",
                "global_target_latent",
                "source_component_mask",
                "target_component_mask",
                "text_context",
                "vlm_context",
            ):
                self.assertTrue(torch.equal(samples[0][field], samples[1][field]))
            self.assertEqual(
                samples[0]["encoder_contract"], samples[1]["encoder_contract"]
            )
            self.assertEqual(
                samples[0]["encoder_contract_sha256"],
                samples[1]["encoder_contract_sha256"],
            )

    def test_rejects_invalid_seed_without_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fixture"
            with self.assertRaisesRegex(SyntheticFixtureError, "seed"):
                build_synthetic_smoke_fixture(output, seed=-1)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
