#!/usr/bin/env python3

from __future__ import annotations

import inspect
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import materialize_starc_core4_hidden_v1 as materializer  # noqa: E402


try:
    import torch
except ImportError:  # pragma: no cover - dependency-light contract host
    torch = None

try:
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError:  # pragma: no cover - dependency-light contract host
    safe_open = None
    save_file = None


class GeometryAndReceiptContractTests(unittest.TestCase):
    EXPECTED_DIGESTS = {
        (30, 31): (
            "5a75404b60cadddb29ac7473fc4596d7ebfcd306acfb3fa1a6bc6575a228a246",
            "be43863f6a000fb00083798610e3993200c24e5fd94dcb2ef7d4e3858618dde7",
            "4a8330c77079671f6515bda07acc21f0d060176c4c07d2609ad2553acf657561",
        ),
        (32, 29): (
            "260d47275c7d407512ff4fca9fa20d2223eaa29b6e4d151b7495e51721980df4",
            "9fdee154009d0d4283716a4e93abe4df2dde5241065040eaf05bd2c9a9f2fa64",
            "be52cac4d90f0a5a70368d25fef2fb1edb4d346fb10598329f5bb7e8e7285ede",
        ),
        (34, 27): (
            "f48f9577ec829cc67bd5f9da09721bebccec7e6c92b18f5322e25ab76f19192a",
            "d05582d93963ae8de876171526f00671b7fbe0ca27841b1ab4c32b196afbc911",
            "9cc6e96d5909542189ca43ea2ff54efda6a44b302483890629b82d2ecad7f7ba",
        ),
    }

    def test_core4_is_three_geometries_not_one_global_shape(self) -> None:
        self.assertEqual(
            materializer.CORE4_LATENT_SHAPES,
            (
                (1, 16, 21, 60, 62),
                (1, 16, 21, 64, 58),
                (1, 16, 21, 68, 54),
            ),
        )
        self.assertEqual(
            materializer.CORE4_PATCH_GRIDS,
            ((30, 31), (32, 29), (34, 27)),
        )
        source = Path(materializer.__file__).read_text(encoding="utf-8")
        self.assertNotIn("LATENT_SHAPE = (1, 16, 21, 60, 62)", source)
        self.assertNotIn("PATCH_POSITIONS = PATCH_HEIGHT * PATCH_WIDTH", source)

    def test_all_three_counter_hash_sketch_digest_triplets_are_pinned(self) -> None:
        for grid, expected in self.EXPECTED_DIGESTS.items():
            with self.subTest(grid=grid):
                actual = materializer.spatial_sketch_digests(
                    patch_height=grid[0], patch_width=grid[1]
                )
                self.assertEqual(actual, expected)
                self.assertEqual(
                    materializer.SKETCH_DIGESTS_BY_PATCH_POSITIONS[grid[0] * grid[1]],
                    expected,
                )
                raw = materializer.reconstruct_spatial_sketch_bytes(
                    patch_height=grid[0], patch_width=grid[1]
                )
                self.assertEqual(len(raw), 16 * grid[0] * grid[1] * 4)

    @unittest.skipIf(torch is None, "Torch is unavailable")
    def test_generalized_ulysses_layout_covers_every_phase_and_only_final_padding(self) -> None:
        expected_local_and_padding = {
            (30, 31): (4883, 2),
            (32, 29): (4872, 0),
            (34, 27): (4820, 2),
        }
        for grid, (expected_local, expected_padding) in expected_local_and_padding.items():
            with self.subTest(grid=grid):
                layouts = [
                    materializer.build_starc_local_layout(
                        rank, patch_height=grid[0], patch_width=grid[1]
                    )
                    for rank in range(4)
                ]
                patch_positions = grid[0] * grid[1]
                self.assertEqual({row.local_sequence_length for row in layouts}, {expected_local})
                self.assertEqual(sum(row.padding_tokens_excluded for row in layouts), expected_padding)
                self.assertEqual(
                    sum(row.target_tokens_selected for row in layouts),
                    21 * patch_positions,
                )
                phase_totals = [
                    sum(int(row.phase_token_count[index]) for row in layouts)
                    for index in range(21)
                ]
                self.assertEqual(phase_totals, [patch_positions] * 21)
                self.assertTrue(all(row.padding_tokens_excluded == 0 for row in layouts[:-1]))

    def test_output_and_forward_counts_remain_geometry_independent(self) -> None:
        self.assertEqual(materializer.ARMS_PER_CELL, 13)
        self.assertEqual(materializer.CORE4_ARM_COUNT, 52)
        self.assertEqual(materializer.MODEL_FORWARDS_TOTAL, 104)
        self.assertEqual(
            materializer.TENSOR_KEY,
            "sketched_action_minus_noop_hidden_residual",
        )
        self.assertEqual(
            materializer._ARTIFACT_FIELDS,
            {
                "path",
                "file_sha256",
                "tensor_key",
                "tensor_shape",
                "tensor_dtype",
                "tensor_sha256",
                "detached_finite_fp32",
            },
        )
        self.assertIn("spatial_sketch_binding", materializer._ARM_FIELDS)
        self.assertIn("spatial_sketch_bindings_by_episode", materializer._GROUP_FIELDS)
        self.assertIn("spatial_sketch_bindings_by_episode", materializer._MASTER_FIELDS)

    def test_same_state_model_boundary_has_no_label_source_or_editor_argument(self) -> None:
        names = set(inspect.signature(materializer.forward_same_state_hidden_pair).parameters)
        self.assertEqual(
            names,
            {
                "diffusion",
                "transformer",
                "observer",
                "x_sigma",
                "action_condition",
                "noop_condition",
                "arm_key",
                "dist_module",
                "group",
            },
        )
        forbidden = {
            "event_label",
            "source_video",
            "source_latent",
            "target_video",
            "target_latent",
            "donor",
            "mask",
            "flow",
            "pose",
            "track",
            "trajectory",
            "optimizer",
            "editor",
        }
        self.assertFalse(names & forbidden)
        source = inspect.getsource(materializer.forward_same_state_hidden_pair)
        self.assertIn("transformer.patch_vae_latent", source)
        self.assertIn("diffusion.shared_step", source)
        self.assertIn("all_reduce_block15_sketch", source)
        self.assertIn("block0_input_and_attn1_exact_parity", source)

    def test_parser_requires_explicit_critic_only_denials(self) -> None:
        parser = materializer.build_parser()
        args = parser.parse_args(
            [
                "materialize-group",
                "--root-spec", "/x/spec.json",
                "--expected-root-spec-sha256", "1" * 64,
                "--bank-output-dir", "/x/bank",
                "--bank-receipt", "/x/bank/receipt.json",
                "--expected-bank-receipt-sha256", "2" * 64,
                "--detached-label-manifest", "/x/labels.json",
                "--expected-detached-label-manifest-sha256", "3" * 64,
                "--critic-use-sidecar", "/x/use.json",
                "--expected-critic-use-sidecar-sha256", "4" * 64,
                "--critic-use-authority-evidence", "/x/evidence.md",
                "--group-id", "sp4-a",
                "--bernini-root", "/x/bernini",
                "--veomni-root", "/x/veomni",
                "--checkpoint", "/x/checkpoint",
                "--checkpoint-content-manifest", "/x/checkpoint.json",
                "--output-root", "/x/out",
                "--expected-bernini-commit", "5" * 40,
                "--expected-veomni-commit", "6" * 40,
                "--method-source-revision", "7" * 40,
                "--method-source-archive-sha256", "8" * 64,
                "--formal-d541801-source-revision", "f" * 40,
                "--formal-d541801-source-archive-sha256", "0" * 64,
                "--expected-materializer-source-sha256", "9" * 64,
                "--expected-formal-d541801-scorer-source-sha256", "1" * 64,
                "--expected-temporal-scorer-source-sha256", "a" * 64,
                "--expected-temporal-contract-source-sha256", "b" * 64,
                "--expected-fitq-observer-source-sha256", "c" * 64,
                "--expected-dataset-contract-source-sha256", "d" * 64,
                "--expected-label-author-source-sha256", "e" * 64,
            ]
        )
        self.assertFalse(args.ack_generated_t2v_hidden_critic_only)
        self.assertFalse(args.ack_no_generated_media_editor_use)
        self.assertFalse(args.ack_no_optimizer_or_editor_update)

    def test_authority_evidence_and_sidecar_are_exactly_bound(self) -> None:
        evidence = (
            METHOD_ROOT.parents[1]
            / "md"
            / "action_editing"
            / "bernini_starc_core4_critic_use_authority_20260808.md"
        )
        self.assertEqual(
            materializer.file_sha256(evidence),
            materializer.REQUIRED_CRITIC_USE_EVIDENCE_SHA256,
        )
        sidecar = materializer.make_required_critic_use_sidecar(
            authority_evidence=evidence
        )
        self.assertEqual(
            sidecar["bank_receipt_digest"],
            materializer.REQUIRED_BANK_RECEIPT_DIGEST,
        )
        self.assertFalse(
            sidecar["authorized_use"]["generated_rgb_or_latent_may_train_editor"]
        )

        sidecar_path = (
            METHOD_ROOT / "assets" / "starc_core4_critic_only_use_v1.json"
        ).resolve()
        expected_sidecar_sha256 = (
            "a71854673f64e027bd673cf4c74673bcd7de74dca6f5b7b3b2c429467055f215"
        )
        loaded, loaded_path, observed_sha256 = (
            materializer.load_required_critic_use_sidecar(
                sidecar_path,
                expected_sha256=expected_sidecar_sha256,
            )
        )
        self.assertEqual(loaded_path, sidecar_path)
        self.assertEqual(observed_sha256, expected_sidecar_sha256)
        self.assertEqual(loaded, sidecar)
        self.assertEqual(
            loaded["receipt_digest"],
            "20c937abd401849b11e2001ae466ae616f1832265d58c955a1a8e6247f980afa",
        )

    def test_materializer_uses_the_sealed_d541801_v3_label_schema(self) -> None:
        self.assertEqual(
            materializer.label_author.LABEL_MANIFEST_SCHEMA,
            "bernini-pair-v5-core4-v2-detached-event-label-manifest-d541801-v3",
        )
        self.assertEqual(
            Path(materializer.label_author.__file__).name,
            "author_pair_v5_core4_event_labels_d541801_v3.py",
        )
        self.assertEqual(
            materializer.label_author.REQUIRED_SCORER_SOURCE_SHA256,
            materializer.REQUIRED_D541801_SCORER_SOURCE_SHA256,
        )


@unittest.skipIf(
    torch is None or safe_open is None or save_file is None,
    "Torch/safetensors are unavailable",
)
class HistoricalCleanLatentAuthenticationTests(unittest.TestCase):
    class FrozenIdentityFixture:
        class PairV5T2VEnergyScoringError(RuntimeError):
            pass

        @staticmethod
        def native_tensor_value_identity(value):
            import hashlib

            cpu = value.detach().to(device="cpu").contiguous().clone()
            raw = cpu.view(torch.uint8).reshape(-1).numpy().tobytes()
            metadata = {
                "shape": [int(item) for item in cpu.shape],
                "dtype": str(cpu.dtype),
                "numel": int(cpu.numel()),
                "byte_count": len(raw),
            }
            return {
                **metadata,
                "raw_value_sha256": hashlib.sha256(raw).hexdigest(),
                "content_sha256": hashlib.sha256(
                    materializer.canonical_json_bytes(metadata) + b"\x00" + raw
                ).hexdigest(),
            }

        @classmethod
        def verify_native_tensor_value_identity(cls, value, artifact, *, label):
            identity = cls.native_tensor_value_identity(value)
            expected = {
                "shape": artifact.get("shape"),
                "dtype": artifact.get("stored_dtype"),
                "raw_value_sha256": artifact.get("raw_value_sha256"),
                "content_sha256": artifact.get("content_sha256"),
            }
            observed = {
                key: identity[key]
                for key in (
                    "shape",
                    "dtype",
                    "raw_value_sha256",
                    "content_sha256",
                )
            }
            if observed != expected:
                raise cls.PairV5T2VEnergyScoringError(
                    f"{label} actual tensor value differs from native receipt"
                )
            return identity

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.path = self.root / "clean.safetensors"
        self.tensor = torch.arange(
            math.prod((1, 16, 21, 60, 62)), dtype=torch.float32
        ).reshape(1, 16, 21, 60, 62).contiguous()
        save_file(
            {"normalized_clean_latent": self.tensor},
            str(self.path),
            metadata={
                "coordinate": "bernini_normalized_clean_vae_latent",
                "frame_contract": "exact81_latent21",
                "artifact_role": "native_sampler_proposal",
                "source": "native_sampler_before_vae_decode",
            },
        )
        with safe_open(str(self.path), framework="pt", device="cpu") as opened:
            self.loaded = opened.get_tensor("normalized_clean_latent").contiguous()
        self.artifact = {
            "artifact_role": "native_sampler_proposal",
            "coordinate": "bernini_normalized_clean_vae_latent",
            "mp4_decode_reencode_used": False,
            "native_sampler_before_vae_decode": True,
            "origin": "native_sampler_before_vae_decode",
            "path": str(self.path),
            "roundtrip_byte_exact_fp32": True,
            "sampler_return_dtype": "torch.float32",
            "sha256": materializer.file_sha256(self.path),
            "shape": list(self.loaded.shape),
            "source_video_vae_encode_before_any_decode": False,
            "stored_dtype": "torch.float32",
            "tensor_key": "normalized_clean_latent",
        }
        self.frozen = self.FrozenIdentityFixture

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def verify(self, artifact: dict) -> dict:
        return materializer.verify_authenticated_native_clean_tensor_identity(
            self.loaded,
            artifact,
            label="fixture clean latent",
            frozen=self.frozen,
        )

    def test_historical_container_contract_passes_and_binds_observed_hashes(self) -> None:
        binding = self.verify(dict(self.artifact))
        checked = materializer.validate_clean_latent_authentication_binding(binding)
        self.assertEqual(
            checked["authenticated_container_sha256"], self.artifact["sha256"]
        )
        self.assertFalse(checked["recorded_value_hashes_present"])
        self.assertTrue(checked["historical_native_receipt_value_hashes_absent"])
        self.assertFalse(checked["native_receipt_value_hashes_synthesized"])
        self.assertFalse(checked["producer_time_value_digest_claimed_by_materializer"])
        self.assertTrue(
            checked["observed_value_hashes_recomputed_after_authenticated_reopen"]
        )
        self.assertEqual(
            checked["value_identity_observation_time"],
            "materializer_authenticated_reopen",
        )

    def test_partial_value_identity_and_historical_mismatches_fail_closed(self) -> None:
        for field in ("raw_value_sha256", "content_sha256"):
            with self.subTest(partial=field):
                with self.assertRaisesRegex(
                    materializer.STARCMaterializationError,
                    "partial native value identity",
                ):
                    self.verify({**self.artifact, field: "d" * 64})
        changes = {
            "sha256": "0" * 64,
            "tensor_key": "other",
            "coordinate": "other",
            "artifact_role": "source_video_condition",
            "roundtrip_byte_exact_fp32": False,
        }
        for field, replacement in changes.items():
            with self.subTest(mismatch=field):
                with self.assertRaises(materializer.STARCMaterializationError):
                    self.verify({**self.artifact, field: replacement})

    def test_full_value_identity_is_still_strict(self) -> None:
        identity = self.frozen.native_tensor_value_identity(self.loaded)
        current = {
            **self.artifact,
            "raw_value_sha256": identity["raw_value_sha256"],
            "content_sha256": identity["content_sha256"],
        }
        checked = self.verify(current)
        self.assertTrue(checked["recorded_value_hashes_present"])
        self.assertTrue(checked["strict_recorded_value_identity_verified"])
        with self.assertRaises(materializer.STARCMaterializationError):
            self.verify({**current, "raw_value_sha256": "e" * 64})

    def test_reopened_value_and_safetensors_metadata_mutations_are_rejected(self) -> None:
        changed_value = self.loaded.clone()
        changed_value.reshape(-1)[0] += 1.0
        with self.assertRaisesRegex(
            materializer.STARCMaterializationError,
            "loaded value differs",
        ):
            materializer.verify_authenticated_native_clean_tensor_identity(
                changed_value,
                dict(self.artifact),
                label="mutated clean latent",
                frozen=self.frozen,
            )
        metadata_path = self.root / "wrong-metadata.safetensors"
        save_file(
            {"normalized_clean_latent": self.loaded},
            str(metadata_path),
            metadata={
                "coordinate": "wrong-coordinate",
                "frame_contract": "exact81_latent21",
                "artifact_role": "native_sampler_proposal",
                "source": "native_sampler_before_vae_decode",
            },
        )
        with self.assertRaisesRegex(
            materializer.STARCMaterializationError,
            "safetensors metadata differs",
        ):
            self.verify(
                {
                    **self.artifact,
                    "path": str(metadata_path),
                    "sha256": materializer.file_sha256(metadata_path),
                }
            )


if torch is not None:

    class _MockAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.to_out = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])


    class _MockBlock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn1 = _MockAttention()
            self.attn2 = _MockAttention()

        def forward(self, hidden_states):
            hidden_states = self.attn1.to_out[0](hidden_states)
            hidden_states = self.attn2.to_out[0](hidden_states)
            return hidden_states


    class _MockTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList([_MockBlock() for _ in range(30)])
            self.proj_out = torch.nn.Identity()
            self.config = SimpleNamespace(
                num_layers=30,
                num_attention_heads=12,
                attention_head_dim=128,
            )

        def patch_vae_latent(self, value=None, source_id=None):
            return value, source_id

        def forward(self, hidden_states):
            for block in self.blocks:
                hidden_states = block(hidden_states)
            return self.proj_out(hidden_states)


@unittest.skipIf(torch is None, "Torch is unavailable")
class TorchSketchAndHookTests(unittest.TestCase):
    def test_all_three_torch_sketches_match_pure_python_and_are_full_rank(self) -> None:
        for patch_height, patch_width in materializer.CORE4_PATCH_GRIDS:
            with self.subTest(grid=(patch_height, patch_width)):
                matrix = materializer.fixed_spatial_sketch(
                    patch_height=patch_height, patch_width=patch_width
                )
                binding = materializer.sketch_binding(
                    matrix,
                    patch_height=patch_height,
                    patch_width=patch_width,
                )
                self.assertEqual(
                    (
                        binding["matrix_raw_bytes_sha256"],
                        binding["matrix_value_sha256"],
                        binding["critic_tensor_sha256"],
                    ),
                    GeometryAndReceiptContractTests.EXPECTED_DIGESTS[
                        (patch_height, patch_width)
                    ],
                )
                self.assertEqual(int(torch.linalg.matrix_rank(matrix)), 16)

    def test_temporal_transforms_preserve_each_native_geometry(self) -> None:
        for shape in materializer.CORE4_LATENT_SHAPES:
            clean = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
            for transform in ("chronological", "reverse", "freeze_first", "phase_shuffle"):
                with self.subTest(shape=shape, transform=transform):
                    result = materializer.apply_temporal_transform(clean, transform)
                    self.assertEqual(tuple(result.shape), shape)
                    self.assertFalse(result.requires_grad)
                    self.assertTrue(torch.isfinite(result).all())

    def test_narrow_hook_materializes_dynamic_rank_local_shape_without_mutation(self) -> None:
        transformer = _MockTransformer()
        for patch_height, patch_width in materializer.CORE4_PATCH_GRIDS:
            with self.subTest(grid=(patch_height, patch_width)):
                layout = materializer.build_starc_local_layout(
                    3, patch_height=patch_height, patch_width=patch_width
                )
                matrix = materializer.fixed_spatial_sketch(
                    patch_height=patch_height, patch_width=patch_width
                )
                value = torch.ones(
                    1, layout.local_sequence_length, 1536, dtype=torch.float32
                )
                baseline = value.clone()
                observed = materializer.Block15SpatialSketchObserver(
                    transformer,
                    sp_rank=3,
                    patch_height=patch_height,
                    patch_width=patch_width,
                    spatial_sketch=matrix,
                )
                with observed:
                    with observed.capture("test") as holder:
                        output = transformer(value)
                self.assertEqual(len(holder), 1)
                capture = holder[0]
                self.assertEqual(
                    tuple(capture.sketch.shape), (1, 21, 16, 1536)
                )
                self.assertEqual(capture.layout.patch_positions, patch_height * patch_width)
                self.assertEqual(capture.hook_call_counts, {
                    "block.00.input": 1,
                    "block.00.attn1": 1,
                    "block.15.output": 1,
                })
                self.assertTrue(torch.equal(output, baseline))
                self.assertEqual(observed.trainable_parameters, ())


if __name__ == "__main__":
    unittest.main()
