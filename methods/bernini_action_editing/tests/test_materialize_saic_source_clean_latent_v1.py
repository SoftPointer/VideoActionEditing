#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

try:
    import safetensors  # noqa: F401

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import build_saic_reversible_source_set_v1 as source_set  # noqa: E402
import materialize_saic_source_clean_latent_v1 as materializer  # noqa: E402


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class FakeVAE:
    def __init__(self) -> None:
        self.config = SimpleNamespace(z_dim=16)
        self.training = False

    def parameters(self):
        return iter(())


def receipt_sections(encoding: dict[str, object]):
    zero = "0" * 64
    sealed = {
        "accepted_roles": list(materializer.ACCEPTED_INPUT_ROLES),
        "forbidden_roles": list(materializer.FORBIDDEN_INPUT_ROLES),
        "source_manifest_path": "/sealed/source.json",
        "source_manifest_raw_sha256": zero,
        "source_manifest_content_sha256": zero,
        "source_manifest_schema_version": source_set.SCHEMA_VERSION,
        "source_manifest_dataset_id": source_set.DATASET_ID,
        "source_manifest_bound_files_verified": False,
        "row_id": "fit-dog-0000000000000000",
        "iid": "0000000000000000",
        "analysis_split": "fit",
        "actor_family": "dog",
        "source_video_path": "/sealed/source.mp4",
        "source_video_sha256": zero,
        "source_video_rehashed_after_encode": True,
        "source_manifest_terminal_events_verified": False,
        "optimizer_authorized": False,
    }
    preprocessing = {
        "decoded_from_private_byte_snapshot": True,
        "frame_count": 81,
        "fps": 25,
        "reported_fps": 25.0,
        "source_input_hw": [16, 24],
        "source_derived_bucket_hw": [16, 24],
        "max_pixels": 245760,
        "stride": 16,
        "temporal_policy": "all_integer_frames_0_through_80_no_subsampling",
        "spatial_policy": "sqrt_max_pixels_then_floor_each_dimension_to_stride",
        "resize": "torchvision_bicubic_antialias_true",
        "external_shared_i0": False,
        "source_pixels_shape": [1, 3, 81, 16, 24],
        "source_pixels_dtype": "torch.float32",
        "source_pixels_raw_sha256": zero,
    }
    provenance = {
        "revision": "1" * 40,
        "scratch_archive_path": "/sealed/method.tar",
        "durable_archive_path": "/durable/method.tar",
        "archive_sha256": zero,
        "archive_safe_scoped_duplicate_free_link_free": True,
        "revision_label_matches_archive_comment": True,
        "git_revision_verified_by_runner": False,
        "runtime_source_sha256": {"methods/bernini_action_editing/x.py": zero},
        "runtime_source_index_sha256": zero,
        "bytecode_policy": {"dont_write_bytecode": True},
    }
    model = {
        "checkpoint_path": "/sealed/checkpoint",
        "checkpoint_tree_sha256": zero,
        "checkpoint_content_manifest_audit": {
            "manifest_path": "/sealed/checkpoint.sha256",
            "manifest_sha256_computed": zero,
            "manifest_sha256_expected": zero,
            "verified_file_count": 23,
            "every_file_sha256_verified": True,
            "verified_entries_digest": zero,
        },
        "bernini_revision": "1" * 40,
        "veomni_revision": "2" * 40,
        "bernini_inference_files": {"bernini/pipeline.py": zero},
        "bernini_inference_files_index_sha256": zero,
        "method_source_revision": "1" * 40,
        "method_source_archive_sha256": zero,
        "runtime_source_index_sha256": zero,
        "method_provenance": provenance,
    }
    runtime = {
        "device_requested": "cuda:0",
        "world_size": 1,
        "distributed_initialized": False,
        "python_version": "3.10.0",
        "torch_version": "2.4.0",
        "hip_version": "6.2",
        "diffusers_version": "0.35.0",
        "safetensors_version": "0.4.5",
    }
    return sealed, preprocessing, model, encoding, runtime


class CliAndSourceContractTests(unittest.TestCase):
    def test_cli_exposes_no_edit_or_target_input(self) -> None:
        destinations = {
            action.dest for action in materializer.build_parser()._actions
        }
        self.assertFalse(
            {
                "instruction",
                "event_bank",
                "target_video",
                "mask",
                "pose",
                "flow",
                "reference_frame",
                "adapter",
                "branch",
                "rollout_seed",
            }.intersection(destinations)
        )
        self.assertTrue(
            {
                "source_manifest",
                "row_id",
                "expected_source_video_sha256",
                "checkpoint",
                "output",
            }.issubset(destinations)
        )

    def test_cli_pins_model_and_explicit_device(self) -> None:
        args = argparse.Namespace(
            expected_bernini_commit=materializer.legacy.trainer.BERNINI_OFFICIAL_COMMIT,
            expected_veomni_commit=materializer.legacy.trainer.VEOMNI_TESTED_COMMIT,
            method_source_revision="1" * 40,
            expected_source_manifest_sha256="2" * 64,
            expected_source_video_sha256="3" * 64,
            expected_checkpoint_tree_sha256=materializer.legacy.trainer.CHECKPOINT_TREE_SHA256,
            method_source_archive_sha256="4" * 64,
            row_id="fit-dog-abc",
            device="cuda:0",
        )
        materializer.validate_cli(args)
        args.device = "cuda"
        with self.assertRaises(materializer.SourceCleanLatentMaterializationError):
            materializer.validate_cli(args)

    def test_real_sealed_manifest_selects_one_non_authoritative_row(self) -> None:
        manifest = source_set.load_manifest(source_set.ASSET_PATH)
        row = manifest["rows"][0]
        selected, sealed = materializer.load_sealed_source_row(
            source_set.ASSET_PATH.resolve(),
            expected_raw_sha256=materializer.file_sha256(source_set.ASSET_PATH),
            row_id=row["row_id"],
            expected_source_video_sha256=row["source_video_sha256"],
        )
        self.assertEqual(selected["iid"], row["iid"])
        self.assertIs(sealed["optimizer_authorized"], False)
        self.assertIs(
            sealed["source_manifest_terminal_events_verified"], False
        )
        self.assertEqual(
            sealed["accepted_roles"], list(materializer.ACCEPTED_INPUT_ROLES)
        )
        self.assertNotIn("target_video", sealed["accepted_roles"])
        self.assertIn("target_video", sealed["forbidden_roles"])

    def test_output_is_create_only_safetensors_plus_fixed_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "source.safetensors"
            artifact, receipt = materializer.resolve_output(output)
            self.assertEqual(artifact, output.resolve(strict=False))
            self.assertEqual(receipt.name, "source.safetensors.receipt.json")
            output.write_bytes(b"occupied")
            with self.assertRaises(
                materializer.SourceCleanLatentMaterializationError
            ):
                materializer.resolve_output(output)


class DecodeAndEncodingTests(unittest.TestCase):
    def test_private_exact81_decode_and_source_identity(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "source.mp4"
            source.write_bytes(b"private-source-bytes")
            observed = {}

            def prepare(snapshot: Path):
                observed["snapshot"] = snapshot
                self.assertNotEqual(snapshot, source)
                self.assertEqual(snapshot.read_bytes(), source.read_bytes())
                tensor = torch.zeros((1, 3, 81, 16, 32), dtype=torch.float32)
                return tensor, {
                    "frame_count": 81,
                    "fps": 25.0,
                    "reported_fps": 25.0,
                    "source_input_hw": [16, 32],
                    "source_derived_bucket_hw": [16, 32],
                    "max_pixels": 245760,
                    "stride": 16,
                    "temporal_policy": "all_integer_frames_0_through_80_no_subsampling",
                    "spatial_policy": "sqrt_max_pixels_then_floor_each_dimension_to_stride",
                    "resize": "torchvision_bicubic_antialias_true",
                    "external_shared_i0": False,
                }

            tensor, metadata, digest = materializer.prepare_private_exact_source(
                source.resolve(), prepare_fn=prepare
            )
            self.assertEqual(tuple(tensor.shape), (1, 3, 81, 16, 32))
            self.assertTrue(metadata["decoded_from_private_byte_snapshot"])
            self.assertEqual(digest, sha_bytes(b"private-source-bytes"))
            self.assertFalse(observed["snapshot"].exists())

    def test_full_source_encoder_is_called_exactly_once(self) -> None:
        pixels = torch.zeros((1, 3, 81, 16, 24), dtype=torch.float32)
        calls = []

        def encoder(vae, value):
            calls.append((vae, value))
            return torch.ones((1, 16, 21, 2, 3), dtype=torch.float32)

        latent, receipt = materializer.encode_full_source_once(
            FakeVAE(), pixels, encoder=encoder
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(tuple(latent.shape), (1, 16, 21, 2, 3))
        self.assertEqual(receipt["full_source_vae_encode_count"], 1)
        self.assertEqual(receipt["total_vae_encode_count"], 1)
        self.assertIs(receipt["encoded_in_runner"], False)
        self.assertEqual(receipt["posterior_statistic"], "latent_dist.mode")
        self.assertIs(receipt["sampling"], False)

    def test_encoder_source_contains_one_dynamic_invocation(self) -> None:
        tree = ast.parse(inspect.getsource(materializer.encode_full_source_once))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "encoder"
        ]
        self.assertEqual(len(calls), 1)

    def test_source_tensor_mutation_fails_after_one_call(self) -> None:
        pixels = torch.zeros((1, 3, 81, 16, 24), dtype=torch.float32)
        calls = []

        def encoder(_vae, value):
            calls.append(1)
            value.add_(1.0)
            return torch.zeros((1, 16, 21, 2, 3), dtype=torch.float32)

        with self.assertRaises(materializer.SourceCleanLatentMaterializationError):
            materializer.encode_full_source_once(
                FakeVAE(), pixels, encoder=encoder
            )
        self.assertEqual(len(calls), 1)


class ClosureTests(unittest.TestCase):
    def test_checkpoint_manifest_hashes_complete_file_set(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            checkpoint = base / "checkpoint"
            checkpoint.mkdir()
            payload = b"checkpoint-content"
            (checkpoint / "a.bin").write_bytes(payload)
            manifest = base / "checkpoint.sha256"
            manifest.write_text(
                f"{sha_bytes(payload)}  ./a.bin\n", encoding="utf-8"
            )
            identity = materializer.validate_checkpoint_content(
                checkpoint,
                manifest,
                expected_manifest_sha256=materializer.file_sha256(manifest),
                expected_file_count=1,
            )
            self.assertTrue(identity["every_file_sha256_verified"])
            self.assertEqual(identity["verified_file_count"], 1)
            (checkpoint / "a.bin").write_bytes(b"mutated")
            with self.assertRaises(
                materializer.SourceCleanLatentMaterializationError
            ):
                materializer.validate_checkpoint_content(
                    checkpoint,
                    manifest,
                    expected_manifest_sha256=materializer.file_sha256(manifest),
                    expected_file_count=1,
                )

    def test_method_archive_binds_live_runtime_and_durable_copy(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            runtime = base / "runtime"
            runtime.mkdir()
            live = runtime / "unit.py"
            live.write_bytes(b"VALUE = 1\n")
            revision = "a" * 40
            scratch = base / "method.tar"
            with tarfile.open(
                scratch,
                "w",
                format=tarfile.PAX_FORMAT,
                pax_headers={"comment": revision},
            ) as handle:
                info = tarfile.TarInfo(
                    "methods/bernini_action_editing/unit.py"
                )
                info.size = len(live.read_bytes())
                handle.addfile(info, io.BytesIO(live.read_bytes()))
            durable = base / "durable.tar"
            shutil.copyfile(scratch, durable)
            args = SimpleNamespace(
                method_source_revision=revision,
                method_source_archive=str(scratch.resolve()),
                durable_method_source_archive=str(durable.resolve()),
                method_source_archive_sha256=materializer.file_sha256(scratch),
            )
            with mock.patch.object(materializer, "METHOD_ROOT", runtime), mock.patch.object(
                materializer, "RUNTIME_METHOD_FILES", ("unit.py",)
            ), mock.patch.object(
                materializer,
                "RUNTIME_ARCHIVE_MEMBERS",
                ("methods/bernini_action_editing/unit.py",),
            ), mock.patch.object(
                materializer, "_bytecode_policy", return_value={"sealed": True}
            ):
                identity = materializer.validate_method_provenance(args)
                self.assertEqual(identity["revision"], revision)
                self.assertEqual(
                    identity["archive_sha256"], args.method_source_archive_sha256
                )
                live.write_bytes(b"VALUE = 2\n")
                with self.assertRaises(
                    materializer.SourceCleanLatentMaterializationError
                ):
                    materializer.validate_method_provenance(args)

    def test_callable_identity_binds_module_signature_path_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            bernini_root = Path(root)
            package = bernini_root / "bernini"
            package.mkdir()
            pipeline = package / "pipeline.py"
            source = (
                "import torch\n"
                "def _vae_encode(vae, x: torch.Tensor) -> torch.Tensor:\n"
                "    return x\n"
            )
            pipeline.write_text(source, encoding="utf-8")
            namespace = {"__name__": "bernini.pipeline"}
            # The pinned vendor pipeline does not enable postponed annotations;
            # do not inherit this test module's ``__future__`` flags.
            exec(
                compile(source, str(pipeline), "exec", dont_inherit=True),
                namespace,
            )
            identity = materializer.validate_encoder_callable(
                namespace["_vae_encode"],
                bernini_root,
                expected_pipeline_sha256=materializer.file_sha256(pipeline),
            )
            self.assertEqual(
                identity["encoder_symbol"], "bernini.pipeline._vae_encode"
            )
            self.assertEqual(
                identity["callable_signature"],
                "(vae, x: torch.Tensor) -> torch.Tensor",
            )


@unittest.skipUnless(HAS_SAFETENSORS, "safetensors is required")
class PublicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.latent = torch.arange(
            1 * 16 * 21 * 2 * 3, dtype=torch.float32
        ).reshape(1, 16, 21, 2, 3)
        _, self.encoding = materializer.encode_full_source_once(
            FakeVAE(),
            torch.zeros((1, 3, 81, 16, 24), dtype=torch.float32),
            encoder=lambda _vae, _pixels: self.latent.clone(),
        )
        self.encoding = {
            "encoder_symbol": "bernini.pipeline._vae_encode",
            "callable_module": "bernini.pipeline",
            "callable_name": "_vae_encode",
            "callable_qualname": "_vae_encode",
            "callable_signature": "(vae, x: torch.Tensor) -> torch.Tensor",
            **self.encoding,
        }

    def test_bundle_is_create_only_0444_canonical_and_false_authority(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "source.safetensors"
            receipt_path = Path(root) / "source.safetensors.receipt.json"
            sections = receipt_sections(self.encoding)
            result = materializer.publish_materialization_bundle(
                self.latent,
                output,
                receipt_path,
                sealed_inputs=sections[0],
                preprocessing=sections[1],
                model_closure=sections[2],
                encoding=sections[3],
                runtime=sections[4],
                transaction_token="unit-transaction",
            )
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o444)
            self.assertEqual(stat.S_IMODE(receipt_path.stat().st_mode), 0o444)
            receipt = json.loads(receipt_path.read_text(encoding="ascii"))
            self.assertEqual(
                set(receipt),
                {
                    "schema_version",
                    "method",
                    "artifact",
                    "sealed_inputs",
                    "preprocessing",
                    "model_closure",
                    "encoding",
                    "runtime",
                    "authority",
                    "receipt_digest",
                },
            )
            self.assertEqual(
                set(receipt["artifact"]),
                {
                    "schema_version",
                    "path",
                    "file_sha256",
                    "size_bytes",
                    "mode",
                    "tensor_key",
                    "tensor_raw_sha256",
                    "shape",
                    "dtype",
                    "metadata",
                },
            )
            self.assertEqual(receipt["artifact"]["mode"], "0444")
            self.assertEqual(receipt["artifact"]["dtype"], "torch.float32")
            self.assertEqual(
                receipt["artifact"]["metadata"], materializer.ARTIFACT_METADATA
            )
            self.assertTrue(
                all(type(value) is bool and value is False for value in receipt["authority"].values())
            )
            unsigned = dict(receipt)
            declared = unsigned.pop("receipt_digest")
            self.assertEqual(materializer.object_sha256(unsigned), declared)
            self.assertEqual(
                receipt_path.read_bytes(),
                materializer.canonical_json_bytes(receipt) + b"\n",
            )
            self.assertTrue(
                result["terminal_verification"]["canonical_receipt_verified"]
            )
            with self.assertRaises(
                materializer.SourceCleanLatentMaterializationError
            ):
                materializer.publish_materialization_bundle(
                    self.latent,
                    output,
                    receipt_path,
                    sealed_inputs=sections[0],
                    preprocessing=sections[1],
                    model_closure=sections[2],
                    encoding=sections[3],
                    runtime=sections[4],
                    transaction_token="second-transaction",
                )

    def test_receipt_failure_removes_only_owned_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "source.safetensors"
            receipt_path = Path(root) / "source.safetensors.receipt.json"
            sections = receipt_sections(self.encoding)
            with mock.patch.object(
                materializer,
                "_publish_receipt_owned",
                side_effect=RuntimeError("injected receipt failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    materializer.publish_materialization_bundle(
                        self.latent,
                        output,
                        receipt_path,
                        sealed_inputs=sections[0],
                        preprocessing=sections[1],
                        model_closure=sections[2],
                        encoding=sections[3],
                        runtime=sections[4],
                        transaction_token="cleanup-transaction",
                    )
            self.assertFalse(output.exists())
            self.assertFalse(receipt_path.exists())

    def test_owned_cleanup_refuses_substituted_inode(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "owned.bin"
            path.write_bytes(b"owned")
            identity = materializer.artifact_identity(path)
            path.unlink()
            path.write_bytes(b"attacker replacement")
            self.assertFalse(materializer.unlink_owned_artifact(path, identity))
            self.assertEqual(path.read_bytes(), b"attacker replacement")


if __name__ == "__main__":
    unittest.main()
