from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import tarfile
import types
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import torch

import materialize_qmosaic_editor_runtime_v1 as subject
import self_imagined_native_rv2v_hidden_vjp_v1 as core
import source_self_native_ref_contrastive_v3 as schedule_contract

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )
except ImportError:  # pragma: no cover - AUH dependency
    serialization = None
    Ed25519PrivateKey = None

try:
    import safetensors  # noqa: F401
except ImportError:  # pragma: no cover - local dependency
    safetensors = None


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


class QmosaicEditorRuntimeMaterializerTests(unittest.TestCase):
    def test_module_metadata_covers_parameters_and_buffers_once(self) -> None:
        class Fixture(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.left = torch.nn.Parameter(
                    torch.ones(2), requires_grad=False
                )
                self.right = torch.nn.Parameter(
                    torch.zeros(3), requires_grad=False
                )
                self.register_buffer("scale", torch.tensor([2.0]))

        receipt = subject._module_metadata_receipt(
            Fixture(), component="two-parameter-one-buffer-fixture"
        )
        self.assertEqual(receipt["state_entry_count"], 3)
        self.assertEqual(receipt["trainable_parameter_tensors"], 0)
        self.assertEqual(receipt["trainable_parameter_elements"], 0)
        self.assertTrue(receipt["all_parameters_frozen"])

    def test_t5_receipt_plainifies_auh_nested_mappingproxy_provenance(self) -> None:
        class Fixture(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.ones(2), requires_grad=False
                )
                self.register_buffer("scale", torch.tensor([2.0]))

        metadata = subject._module_metadata_receipt(
            Fixture(), component="auh-umt5-fixture"
        )
        identity = types.MappingProxyType(
            {
                "all_rank_exact": True,
                "identity": types.MappingProxyType(
                    {
                        "shape": (1, 512, 4096),
                        "dtype": "torch.float32",
                        "tensor_sha256": _digest("condition"),
                    }
                ),
            }
        )
        unsigned = {
            "schema_version": "qmosaic-pinned-umt5-receipt-v1",
            "checkpoint_content_receipt_digest": _digest("checkpoint"),
            "loaded_component": metadata,
            "action_condition_tensor_sha256": _digest("action"),
            "noop_condition_tensor_sha256": _digest("noop"),
            "negative_condition_tensor_sha256": _digest("negative"),
            "condition_identities": {
                role: identity for role in ("action", "noop", "negative")
            },
            "all_conditions_rank0_broadcast_and_world4_exact": True,
        }
        receipt = subject._seal_text_encoder_receipt(unsigned)

        self.assertIs(type(metadata), types.MappingProxyType)
        self.assertIs(type(receipt), dict)
        self.assertIs(type(receipt["loaded_component"]), dict)
        self.assertIs(type(receipt["condition_identities"]["action"]), dict)
        self.assertIs(
            type(receipt["condition_identities"]["action"]["identity"]), dict
        )
        self.assertIs(
            type(
                receipt["condition_identities"]["action"]["identity"]["shape"]
            ),
            list,
        )
        sealed = dict(receipt)
        declared = sealed.pop("digest")
        self.assertEqual(declared, subject.object_sha256(sealed))

    def test_plain_provenance_boundary_rejects_arbitrary_mapping_types(self) -> None:
        class ForeignMapping(dict):
            pass

        with self.assertRaisesRegex(
            subject.QmosaicRuntimeMaterializationError,
            "unsupported provenance type",
        ):
            subject._plain_provenance_tree(
                {"foreign": ForeignMapping(value=1)}, label="test provenance"
            )
        with self.assertRaisesRegex(
            subject.QmosaicRuntimeMaterializationError,
            "not canonical finite ASCII JSON",
        ):
            subject.canonical_json_bytes(
                {"still_strict": types.MappingProxyType({"value": 1})}
            )

    def test_materializer_bytes_are_bound_to_exact_archive_member(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            archive = root / "source.tar"
            source = Path(subject.__file__).resolve().read_bytes()
            with tarfile.open(archive, "w") as handle:
                info = tarfile.TarInfo(subject.MATERIALIZER_ARCHIVE_MEMBER)
                info.size = len(source)
                handle.addfile(info, io.BytesIO(source))
            observed = subject._validate_materializer_archive_member(
                archive.resolve(),
                expected_file_sha256=hashlib.sha256(source).hexdigest(),
            )
            self.assertEqual(observed, hashlib.sha256(source).hexdigest())
            with self.assertRaisesRegex(
                subject.QmosaicRuntimeMaterializationError, "differs"
            ):
                subject._validate_materializer_archive_member(
                    archive.resolve(), expected_file_sha256="0" * 64
                )

    def _runtime_code_receipt(self) -> dict:
        source = Path(subject.__file__).resolve()
        return dict(
            subject._build_runtime_code_receipt(
                {role: source for role in subject.RUNTIME_CODE_ROLES}
            )
        )

    def _schedule_receipts(self) -> tuple[dict, dict]:
        class Scheduler:
            timesteps = torch.tensor(
                schedule_contract.NATIVE_UNIPC40_TIMESTEPS,
                dtype=torch.float32,
            )
            sigmas = torch.tensor(
                (*schedule_contract.NATIVE_UNIPC40_SIGMAS, 0.0),
                dtype=torch.float32,
            )

        return (
            dict(schedule_contract.native_unipc40_schedule_receipt()),
            dict(
                subject._capture_live_exact40_schedule(
                    Scheduler(), schedule_contract=schedule_contract
                )
            ),
        )

    def _keys(self, root: Path, *, stem: str = "authority") -> tuple[Path, Path, str]:
        private = Ed25519PrivateKey.generate()
        private_path = root / f"{stem}.private.pem"
        public_path = root / f"{stem}.public.pem"
        private_path.write_bytes(
            private.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        os.chmod(private_path, 0o600)
        public_path.write_bytes(
            private.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
        return private_path, public_path, subject.file_sha256(public_path)

    def _products(
        self, root: Path, *, corrupt_noise: bool = False
    ) -> subject.RuntimeProducts:
        source_video = root / "original-source.mp4"
        source_video.write_bytes(b"method-owned exact81 source fixture\x00")
        shape = (1, 16, subject.LATENT_PHASES, 2, 2)
        noise = subject._official_cpu_gaussian(shape, query_seed=1017)
        if corrupt_noise:
            noise = noise.clone()
            noise.reshape(-1)[0] += 1.0
        tensors = {
            "source_latent": torch.full(shape, 0.125, dtype=torch.float32),
            "image_reference_0": torch.full(
                (1, 16, 1, 2, 2), 0.25, dtype=torch.float32
            ),
            "image_reference_1": torch.full(
                (1, 16, 1, 2, 2), 0.5, dtype=torch.float32
            ),
            "image_reference_2": torch.full(
                (1, 16, 1, 2, 2), 0.75, dtype=torch.float32
            ),
            "image_reference_3": torch.full(
                (1, 16, 1, 2, 2), 1.0, dtype=torch.float32
            ),
            "clean_latent": torch.full(shape, -0.125, dtype=torch.float32),
            "official_initial_noise": noise,
            "action_condition": torch.zeros(subject.TEXT_SHAPE, dtype=torch.float32),
            "noop_condition": torch.ones(subject.TEXT_SHAPE, dtype=torch.float32),
            "timestep": torch.tensor(
                [float(subject.NATIVE_TIMESTEP)], dtype=torch.float32
            ),
        }
        owner_digest = _digest("owner-packet")
        checkpoint_digest = _digest("checkpoint-packet")
        clean_sha = subject.tensor_sha256(
            tensors["clean_latent"], label="fixture clean"
        )
        noise_sha = subject.tensor_sha256(
            tensors["official_initial_noise"], label="fixture noise"
        )
        runtime_code_receipt = self._runtime_code_receipt()
        exact40_schedule, live_exact40_schedule = self._schedule_receipts()
        tokenizer_value = {
            "schema_version": "qmosaic-pinned-tokenizer-receipt-v1",
            "checkpoint_content_receipt_digest": checkpoint_digest,
        }
        text_encoder_value = {
            "schema_version": "qmosaic-pinned-umt5-receipt-v1",
            "checkpoint_content_receipt_digest": checkpoint_digest,
        }
        return subject._seal_runtime_products(
            cell_id="dog",
            owner_query_seed=17,
            editor_noise_seed=1017,
            source_iid="source-iid",
            source_video_path=source_video.resolve(),
            source_video_sha256=subject.file_sha256(source_video.resolve()),
            action_prompt="The dog sits and holds the pose.",
            noop_prompt="The dog remains standing.",
            owner_packet_receipt_digest=owner_digest,
            checkpoint_content_receipt_digest=checkpoint_digest,
            owner_authority={
                "owner_packet_receipt_digest": owner_digest,
                "registry_file_sha256": _digest("registry"),
                "owner_quotient_revalidated_live_before_and_after": True,
            },
            checkpoint_authority={
                "checkpoint_content_receipt_digest": checkpoint_digest,
                "checkpoint_rehashed_before_and_after": True,
                "runtime_code_receipt_digest": runtime_code_receipt["digest"],
                "native_unipc40_schedule_digest": (
                    subject.NATIVE_UNIPC40_SCHEDULE_DIGEST
                ),
            },
            source_provenance={
                "source_video_sha256": subject.file_sha256(source_video.resolve()),
                "registry_file_sha256": _digest("registry"),
                "selected_before_materialization": True,
            },
            vae_receipt={
                "schema_version": "qmosaic-method-owned-wan-vae-encoding-v1",
                "checkpoint_content_receipt_digest": checkpoint_digest,
                "reference_indices": list(subject.REFERENCE_INDICES),
                "references_from_full_video_latent_slice": False,
                "no_external_latent_consumed": True,
            },
            tokenizer_receipt={
                **tokenizer_value,
                "digest": subject.object_sha256(tokenizer_value),
            },
            text_encoder_receipt={
                **text_encoder_value,
                "digest": subject.object_sha256(text_encoder_value),
            },
            runtime_code_receipt=runtime_code_receipt,
            native_endpoint_receipt={
                "schema_version": "qmosaic-frozen-native-rv2v-base-endpoint-v1",
                "clean_latent_semantics": subject.CLEAN_LATENT_SEMANTICS,
                "generated_inside_same_invocation": True,
                "adapter_or_lora_loaded": False,
                "frozen_model": True,
                "owner_query_seed": 17,
                "editor_noise_seed": 1017,
                "owner_editor_noise_seed_shared": False,
                "num_inference_steps": subject.NUM_INFERENCE_STEPS,
                "frame_count": subject.FRAME_COUNT,
                "latent_phases": subject.LATENT_PHASES,
                "guidance_mode": "rv2v",
                "full_source_video_count": 1,
                "source_reference_count": 4,
                "reference_indices": list(subject.REFERENCE_INDICES),
                "exact40_schedule": exact40_schedule,
                "live_exact40_schedule": live_exact40_schedule,
                "clean_latent_tensor_sha256": clean_sha,
                "official_initial_noise_tensor_sha256": noise_sha,
                "official_cpu_gaussian_independently_regenerated_exact": True,
                "no_external_clean_latent_consumed": True,
            },
            tensors=tensors,
        )

    def _publish(self, root: Path) -> tuple[Path, dict, Path, str]:
        private_path, public_path, public_sha = self._keys(root)
        authority = subject._load_signing_authority(
            private_key_path=private_path.resolve(),
            public_key_path=public_path.resolve(),
            expected_public_key_file_sha256=public_sha,
        )
        output = root / "published"
        source_archive = root / "method-source-archive.tar"
        materializer_bytes = Path(subject.__file__).resolve().read_bytes()
        with tarfile.open(source_archive, "w") as handle:
            info = tarfile.TarInfo(subject.MATERIALIZER_ARCHIVE_MEMBER)
            info.size = len(materializer_bytes)
            handle.addfile(info, io.BytesIO(materializer_bytes))
        publication = dict(
            subject._publish_runtime_products(
                products=self._products(root),
                output_dir=output.resolve(),
                authority=authority,
                method_source_revision="a" * 40,
                method_source_archive=source_archive.resolve(),
                method_source_archive_sha256=subject.file_sha256(
                    source_archive.resolve()
                ),
                materializer_source_sha256=subject.file_sha256(
                    Path(subject.__file__).resolve()
                ),
            )
        )
        return output, publication, public_path.resolve(), public_sha

    def test_vendor_condition_is_bf16_only_at_call_edge(self) -> None:
        signed = torch.randn(1, 3, 4, dtype=torch.float32)
        signed_before = signed.clone()
        transformer = types.SimpleNamespace(
            patch_embedding=torch.nn.Linear(
                4, 4, bias=False, dtype=torch.bfloat16
            )
        )
        vendor = subject._vendor_condition_bf16(
            signed, transformer=transformer, label="test condition"
        )
        self.assertEqual(vendor.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(signed, signed_before))
        self.assertEqual(signed.dtype, torch.float32)

    def test_condition_consensus_row_is_canonical_json_ready(self) -> None:
        class Native:
            @staticmethod
            def _all_rank_tensor_identity(value, *, label, world_size):
                self.assertIsInstance(value, torch.Tensor)
                self.assertEqual(label, "qmosaic_action_condition")
                self.assertEqual(world_size, subject.WORLD_SIZE)
                return {
                    "schema_version": "test-world4-tensor-identity-v1",
                    "all_rank_exact": True,
                    "tensor_sha256": subject.tensor_sha256(
                        value, label="test action condition"
                    ),
                }

        row = subject._json_ready_tensor_consensus(
            torch.ones(1, dtype=torch.float32),
            label="qmosaic_action_condition",
            native=Native(),
        )
        self.assertIs(type(row), dict)
        digest = subject.object_sha256(
            {"condition_identities": {"action": row}}
        )
        self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_terminal_rank_consensus_is_canonical_json_ready(self) -> None:
        consensus = types.MappingProxyType(
            {
                "schema_version": "qmosaic-terminal-consensus-test-v1",
                "nested": types.MappingProxyType(
                    {"shape": (1, 16, 21, 60, 62), "all_rank_exact": True}
                ),
            }
        )

        with self.assertRaisesRegex(
            subject.QmosaicRuntimeMaterializationError,
            "not canonical finite ASCII JSON",
        ):
            subject.canonical_json_bytes(consensus)
        encoded = subject._canonical_rank_consensus_bytes(consensus)
        decoded = json.loads(encoded.decode("ascii"))
        self.assertEqual(decoded["nested"]["shape"], [1, 16, 21, 60, 62])
        self.assertTrue(decoded["nested"]["all_rank_exact"])

    @unittest.skipIf(
        Ed25519PrivateKey is None or safetensors is None,
        "cryptography/safetensors unavailable",
    )
    def test_packet_satisfies_core_payload_loader_directly(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            output, publication, public_path, public_sha = self._publish(root)
            postflight = subject.validate_published_files(
                output_root=output,
                expected_materialization_receipt_file_sha256=publication[
                    "materialization_receipt_file_sha256"
                ],
                expected_runtime_receipt_file_sha256=publication[
                    "runtime_receipt_file_sha256"
                ],
                public_key_path=public_path,
                expected_public_key_file_sha256=public_sha,
            )
            self.assertTrue(postflight["all_artifacts_reopened"])
            receipt, receipt_path, key_path, artifact_root, tensors = (
                core._verify_editor_runtime_input_payload(
                    receipt_path=output / "editor-runtime-input.json",
                    expected_receipt_file_sha256=publication[
                        "runtime_receipt_file_sha256"
                    ],
                    public_key_path=public_path,
                    expected_public_key_file_sha256=public_sha,
                    artifact_root=output,
                )
            )
            self.assertEqual(receipt["schema_version"], subject.RUNTIME_INPUT_SCHEMA)
            self.assertEqual(receipt_path, output / "editor-runtime-input.json")
            self.assertEqual(key_path, public_path)
            self.assertEqual(artifact_root, output)
            self.assertEqual(tuple(tensors), subject.RUNTIME_TENSOR_KEYS)

    @unittest.skipIf(
        Ed25519PrivateKey is None or safetensors is None,
        "cryptography/safetensors unavailable",
    )
    def test_materializer_digest_is_computed_and_covered_by_signature(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            output, publication, public_path, public_sha = self._publish(root)
            material = json.loads(
                (output / "materialization-receipt.json").read_text("ascii")
            )
            runtime = json.loads(
                (output / "editor-runtime-input.json").read_text("ascii")
            )
            unsigned = dict(material)
            declared = unsigned.pop("receipt_digest")
            self.assertEqual(declared, subject.object_sha256(unsigned))
            self.assertEqual(
                runtime["runtime_tensor_artifact"]["materializer_receipt_digest"],
                declared,
            )
            subject._verify_runtime_signature(
                runtime,
                public_key_path=public_path,
                expected_public_key_file_sha256=public_sha,
            )
            runtime["runtime_tensor_artifact"]["materializer_receipt_digest"] = (
                "0" * 64
            )
            with self.assertRaisesRegex(
                subject.QmosaicRuntimeMaterializationError,
                "seal|signature",
            ):
                subject._verify_runtime_signature(
                    runtime,
                    public_key_path=public_path,
                    expected_public_key_file_sha256=public_sha,
                )
            self.assertEqual(
                publication["materialization_receipt_digest"], declared
            )

    def test_non_official_cpu_gaussian_is_rejected_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            with self.assertRaisesRegex(
                subject.QmosaicRuntimeMaterializationError,
                "official CPU-generator Gaussian",
            ):
                subject._validate_runtime_products(
                    self._products(root, corrupt_noise=True)
                )

    @unittest.skipIf(Ed25519PrivateKey is None, "cryptography Ed25519 unavailable")
    def test_private_key_cannot_sign_for_a_different_public_key(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            private_path, _public_path, _public_sha = self._keys(root, stem="one")
            _private_two, public_two, public_two_sha = self._keys(root, stem="two")
            with self.assertRaisesRegex(
                subject.QmosaicRuntimeMaterializationError,
                "do not match",
            ):
                subject._load_signing_authority(
                    private_key_path=private_path.resolve(),
                    public_key_path=public_two.resolve(),
                    expected_public_key_file_sha256=public_two_sha,
                )

    def test_parser_exposes_no_clean_latent_or_materializer_digest_input(self) -> None:
        parser = subject.build_parser()
        destinations = {action.dest for action in parser._actions}
        self.assertNotIn("clean_latent", destinations)
        self.assertNotIn("clean_latent_mode", destinations)
        self.assertNotIn("source_latent", destinations)
        self.assertNotIn("reference_latents", destinations)
        self.assertNotIn("action_condition", destinations)
        self.assertNotIn("noop_condition", destinations)
        self.assertNotIn("official_initial_noise", destinations)
        self.assertNotIn("materializer_receipt_digest", destinations)

    def test_tensor_digest_matches_core_loader_digest(self) -> None:
        value = torch.arange(48, dtype=torch.float32).reshape(1, 3, 4, 4)
        self.assertEqual(
            subject.tensor_sha256(value, label="subject"),
            core.tensor_sha256(value, label="core"),
        )

    def test_runtime_code_rows_cannot_be_replaced_by_a_self_signed_hex(self) -> None:
        receipt = json.loads(json.dumps(self._runtime_code_receipt()))
        receipt["files"][1]["file_sha256"] = "0" * 64
        unsigned = dict(receipt)
        unsigned.pop("digest")
        receipt["digest"] = subject.object_sha256(unsigned)
        with self.assertRaisesRegex(
            subject.QmosaicRuntimeMaterializationError,
            "bytes changed",
        ):
            subject._validate_runtime_code_receipt(receipt)


if __name__ == "__main__":
    unittest.main()
