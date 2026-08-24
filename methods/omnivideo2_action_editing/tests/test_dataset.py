from __future__ import annotations

import copy
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

from pact.dataset import (  # noqa: E402
    ENCODER_CONTRACT_FORMAT,
    PAYLOAD_FORMAT,
    PAYLOAD_PROVENANCE_BINDINGS,
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
    AtomicLatentDataset,
    DatasetContractError,
    collate_atomic_latents,
    encoder_contract_sha256,
    validate_encoder_contract,
    validate_precomputed_payload,
)
from tests.test_manifest import authorized_atom_fixture  # noqa: E402


def encoder_contract(
    *,
    vae_digest: str = "1" * 64,
    umt5_digest: str = "2" * 64,
    vlm_digest: str = "3" * 64,
    vae_preprocessing_digest: str = "4" * 64,
    umt5_preprocessing_digest: str = "5" * 64,
    vlm_feature_contract_digest: str = "6" * 64,
) -> dict:
    return {
        "format": ENCODER_CONTRACT_FORMAT,
        "vae": {
            "checkpoint_sha256": vae_digest,
            "preprocessing_contract_sha256": vae_preprocessing_digest,
            "input_pixel_range": list(WAN21_VAE_INPUT_PIXEL_RANGE),
            "posterior_mode": WAN21_VAE_POSTERIOR_MODE,
            "channel_mean": list(WAN21_VAE_CHANNEL_MEAN),
            "channel_std": list(WAN21_VAE_CHANNEL_STD),
            "stride": list(WAN21_VAE_STRIDE),
        },
        "umt5": {
            "checkpoint_manifest_sha256": umt5_digest,
            "preprocessing_contract_sha256": umt5_preprocessing_digest,
            "embedding_dim": UMT5_EMBEDDING_DIM,
            "max_sequence_length_per_segment": UMT5_MAX_SEQUENCE_LENGTH,
            "segment_order": list(UMT5_SEGMENT_ORDER),
            "padding_policy": UMT5_PADDING_POLICY,
        },
        "vlm": {
            "checkpoint_manifest_sha256": vlm_digest,
            "feature_extraction_contract_sha256": vlm_feature_contract_digest,
            "embedding_dim": VLM_EMBEDDING_DIM,
            "feature_tensor": VLM_FEATURE_TENSOR,
            "token_selection": VLM_TOKEN_SELECTION,
        },
    }


def payload(atom: dict, atom_id: str | None = None) -> dict:
    shape = (16, 3, 4, 5)
    value = {
        "format": PAYLOAD_FORMAT,
        "atom_id": atom["atom_id"] if atom_id is None else atom_id,
        "encoder_contract": encoder_contract(),
        "source_latent": torch.randn(shape),
        "global_target_latent": torch.randn(shape),
        "source_component_mask": torch.zeros(1, *shape[1:]),
        "target_component_mask": torch.ones(1, *shape[1:]),
        "text_context": torch.randn(7, 4096),
        "vlm_context": torch.randn(9, 2048),
    }
    for payload_field, atomic_field in PAYLOAD_PROVENANCE_BINDINGS:
        value[payload_field] = atom[atomic_field]
    return value


class DatasetTest(unittest.TestCase):
    def test_payload_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            atom = authorized_atom_fixture(Path(directory))[0]
            checked = validate_precomputed_payload(payload(atom))
            self.assertEqual(checked["source_latent"].shape, (16, 3, 4, 5))
            bad = payload(atom)
            bad["text_context"] = torch.randn(2, 1024)
            with self.assertRaisesRegex(DatasetContractError, "4096"):
                validate_precomputed_payload(bad)

            legacy = payload(atom)
            for key in (
                "source_video_sha256",
                "global_counterfactual_target_video_sha256",
                "source_component_mask_sha256",
                "target_component_mask_sha256",
                "track_record_sha256",
            ):
                del legacy[key]
            with self.assertRaisesRegex(DatasetContractError, "payload fields missing"):
                validate_precomputed_payload(legacy)

            no_encoder_contract = payload(atom)
            del no_encoder_contract["encoder_contract"]
            with self.assertRaisesRegex(
                DatasetContractError, "encoder_contract"
            ):
                validate_precomputed_payload(no_encoder_contract)

            wrong_channels = payload(atom)
            wrong_channels["source_latent"] = torch.randn(8, 3, 4, 5)
            wrong_channels["global_target_latent"] = torch.randn(8, 3, 4, 5)
            with self.assertRaisesRegex(DatasetContractError, "16 channels"):
                validate_precomputed_payload(wrong_channels)

    def test_encoder_contract_is_closed_exact_and_canonically_hashed(self) -> None:
        expected = encoder_contract()
        checked = validate_encoder_contract(expected)
        self.assertEqual(checked, expected)

        reordered = {key: expected[key] for key in reversed(tuple(expected))}
        self.assertEqual(
            encoder_contract_sha256(reordered),
            encoder_contract_sha256(expected),
        )

        cases = []
        unknown = copy.deepcopy(expected)
        unknown["vae"]["unknown"] = "forbidden"
        cases.append(("unknown", "fields differ", unknown))

        sampled = copy.deepcopy(expected)
        sampled["vae"]["posterior_mode"] = "sample"
        cases.append(("posterior", "posterior_mode", sampled))

        integer_pixels = copy.deepcopy(expected)
        integer_pixels["vae"]["input_pixel_range"] = [-1, 1]
        cases.append(("pixel-type", "exact float", integer_pixels))

        wrong_mean = copy.deepcopy(expected)
        wrong_mean["vae"]["channel_mean"][0] += 0.0001
        cases.append(("mean", "exact float constant", wrong_mean))

        tuple_stride = copy.deepcopy(expected)
        tuple_stride["vae"]["stride"] = (4, 8, 8)
        cases.append(("stride-type", "JSON list", tuple_stride))

        float_dim = copy.deepcopy(expected)
        float_dim["umt5"]["embedding_dim"] = 4096.0
        cases.append(("dimension-type", "exact integer", float_dim))

        wrong_padding = copy.deepcopy(expected)
        wrong_padding["umt5"]["padding_policy"] = "zero_padding"
        cases.append(("padding", "padding_policy", wrong_padding))

        wrong_segment_order = copy.deepcopy(expected)
        wrong_segment_order["umt5"]["segment_order"].reverse()
        cases.append(("segment-order", "must equal", wrong_segment_order))

        uppercase_digest = copy.deepcopy(expected)
        uppercase_digest["vlm"]["checkpoint_manifest_sha256"] = "A" * 64
        cases.append(("digest", "lowercase SHA-256", uppercase_digest))

        for label, message, value in cases:
            with self.subTest(label=label), self.assertRaisesRegex(
                DatasetContractError, message
            ):
                validate_encoder_contract(value)

    def test_load_digest_and_collate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = authorized_atom_fixture(root)[0]
            payload_path = root / "sample.pt"
            torch.save(payload(row), payload_path)
            digest = hashlib.sha256(payload_path.read_bytes()).hexdigest()
            row["latent_payload_path"] = payload_path.name
            row["latent_payload_sha256"] = digest
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            dataset = AtomicLatentDataset(manifest)
            sample = dataset[0]
            self.assertEqual(sample["atom_id"], row["atom_id"])
            self.assertEqual(sample["encoder_contract"], encoder_contract())
            self.assertEqual(
                sample["encoder_contract_sha256"],
                encoder_contract_sha256(encoder_contract()),
            )
            batch = collate_atomic_latents([sample, sample])
            self.assertEqual(batch["source_latent"].shape, (2, 16, 3, 4, 5))
            self.assertEqual(len(batch["text_context"]), 2)
            self.assertEqual(batch["encoder_contract"], encoder_contract())
            self.assertEqual(
                batch["encoder_contract_sha256"],
                sample["encoder_contract_sha256"],
            )

            mixed = dict(sample)
            mixed_contract = encoder_contract(vlm_digest="4" * 64)
            mixed["encoder_contract"] = mixed_contract
            mixed["encoder_contract_sha256"] = encoder_contract_sha256(
                mixed_contract
            )
            with self.assertRaisesRegex(
                DatasetContractError, "mixed encoder_contract"
            ):
                collate_atomic_latents([sample, mixed])

            forged = dict(sample)
            forged["encoder_contract_sha256"] = "f" * 64
            with self.assertRaisesRegex(
                DatasetContractError, "differs from canonical"
            ):
                collate_atomic_latents([forged])

    def test_preview_and_digest_tampering_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = authorized_atom_fixture(root)[0]
            payload_path = root / "sample.pt"
            torch.save(payload(row), payload_path)
            row["training_authorized"] = False
            row["training_use_forbidden"] = True
            row["parent_preview_only"] = True
            release = row.pop("post_generation_release")
            row["latent_payload_path"] = payload_path.name
            row["latent_payload_sha256"] = "0" * 64
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(DatasetContractError, "not authorized"):
                AtomicLatentDataset(manifest)
            row["training_authorized"] = True
            row["training_use_forbidden"] = False
            row["parent_preview_only"] = False
            row["post_generation_release"] = release
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(DatasetContractError, "digest differs"):
                AtomicLatentDataset(manifest)

    def test_semantic_binding_and_runtime_digest_are_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = authorized_atom_fixture(root)[0]
            payload_path = root / "sample.pt"
            wrong = payload(row)
            wrong["parent_row_sha256"] = "f" * 64
            torch.save(wrong, payload_path)
            row["latent_payload_path"] = payload_path.name
            row["latent_payload_sha256"] = hashlib.sha256(
                payload_path.read_bytes()
            ).hexdigest()
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            dataset = AtomicLatentDataset(manifest)
            with self.assertRaisesRegex(DatasetContractError, "parent_row_sha256"):
                dataset[0]

            torch.save(payload(row), payload_path)
            row["latent_payload_sha256"] = hashlib.sha256(
                payload_path.read_bytes()
            ).hexdigest()
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            dataset = AtomicLatentDataset(manifest)
            torch.save(payload(row, atom_id="another_atom"), payload_path)
            with self.assertRaisesRegex(DatasetContractError, "at load time"):
                dataset[0]

    def test_video_mask_and_track_provenance_swaps_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = authorized_atom_fixture(root)[0]
            payload_path = root / "sample.pt"
            row["latent_payload_path"] = payload_path.name
            manifest = root / "manifest.jsonl"

            cases = []
            swapped_video = payload(row)
            swapped_video["source_video_sha256"], swapped_video[
                "global_counterfactual_target_video_sha256"
            ] = (
                swapped_video["global_counterfactual_target_video_sha256"],
                swapped_video["source_video_sha256"],
            )
            cases.append(("video", "source_video_sha256", swapped_video))

            swapped_masks = payload(row)
            swapped_masks["source_component_mask_sha256"], swapped_masks[
                "target_component_mask_sha256"
            ] = (
                swapped_masks["target_component_mask_sha256"],
                swapped_masks["source_component_mask_sha256"],
            )
            cases.append(("mask", "source_component_mask_sha256", swapped_masks))

            copied_track = payload(row)
            copied_track["track_record_sha256"] = copied_track[
                "parent_row_sha256"
            ]
            cases.append(("track", "track_record_sha256", copied_track))

            for label, expected_field, wrong_payload in cases:
                with self.subTest(label=label):
                    torch.save(wrong_payload, payload_path)
                    row["latent_payload_sha256"] = hashlib.sha256(
                        payload_path.read_bytes()
                    ).hexdigest()
                    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
                    dataset = AtomicLatentDataset(manifest)
                    with self.assertRaisesRegex(
                        DatasetContractError, expected_field
                    ):
                        dataset[0]

    def test_payload_paths_cannot_escape_the_declared_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_root = root / "manifest"
            manifest_root.mkdir()
            row = authorized_atom_fixture(root)[0]
            payload_path = root / "outside.pt"
            torch.save(payload(row), payload_path)
            row["latent_payload_path"] = "../outside.pt"
            row["latent_payload_sha256"] = hashlib.sha256(
                payload_path.read_bytes()
            ).hexdigest()
            manifest = manifest_root / "manifest.jsonl"
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(DatasetContractError, "relative"):
                AtomicLatentDataset(manifest)


if __name__ == "__main__":
    unittest.main()
