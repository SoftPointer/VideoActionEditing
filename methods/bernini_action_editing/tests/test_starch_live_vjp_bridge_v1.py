from __future__ import annotations

import hashlib
import io
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import tarfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(METHOD_ROOT))

import starch_live_vjp_bridge_v1 as bridge  # noqa: E402

try:
    import torch  # noqa: E402
except ImportError:  # pragma: no cover - dependency-light local hosts
    torch = None


EXPECTED_GEOMETRIES = {
    (1, 16, 21, 60, 62): {
        "grid": (30, 31),
        "patches": 930,
        "tokens": 19_530,
        "local": 4_883,
        "padding": 2,
        "digest": "4a8330c77079671f6515bda07acc21f0d060176c4c07d2609ad2553acf657561",
    },
    (1, 16, 21, 64, 58): {
        "grid": (32, 29),
        "patches": 928,
        "tokens": 19_488,
        "local": 4_872,
        "padding": 0,
        "digest": "be52cac4d90f0a5a70368d25fef2fb1edb4d346fb10598329f5bb7e8e7285ede",
    },
    (1, 16, 21, 68, 54): {
        "grid": (34, 27),
        "patches": 918,
        "tokens": 19_278,
        "local": 4_820,
        "padding": 2,
        "digest": "9cc6e96d5909542189ca43ea2ff54efda6a44b302483890629b82d2ecad7f7ba",
    },
}


class LiveVJPStaticContractTests(unittest.TestCase):
    def test_composite_v2_schema_and_dynamic_sketch_match_runner_exactly(self) -> None:
        import run_starc_core4_critic_pilot_v1 as runner

        self.assertEqual(
            bridge.COMPOSITE_SCHEMA_VERSION, runner.LIVE_VJP_BINDING_SCHEMA
        )
        self.assertEqual(
            bridge.LIVE_VJP_SP4_IMPLEMENTATION,
            runner.LIVE_VJP_SP4_IMPLEMENTATION,
        )
        for shape in EXPECTED_GEOMETRIES:
            geometry = bridge.LatentPatchGeometry(shape)
            self.assertEqual(
                bridge.geometry_spatial_sketch_binding(geometry),
                runner.reconstruct_geometry_spatial_sketch_binding(
                    geometry.patch_rows, geometry.patch_columns
                ),
            )

    def test_executing_bridge_is_bound_to_git_archive_member_and_revision(self) -> None:
        revision = "7" * 40
        source = METHOD_ROOT / "starch_live_vjp_bridge_v1.py"
        payload = source.read_bytes()
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root).resolve()
            archive = root / "bridge.tar"
            with tarfile.open(
                archive,
                mode="w",
                format=tarfile.PAX_FORMAT,
                pax_headers={"comment": revision},
            ) as handle:
                info = tarfile.TarInfo(bridge.LIVE_VJP_BRIDGE_ARCHIVE_MEMBER)
                info.size = len(payload)
                info.mode = 0o644
                info.mtime = 0
                handle.addfile(info, io.BytesIO(payload))
            archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
            binding = bridge.authenticate_bridge_source_archive(
                archive,
                expected_source_archive_sha256=archive_sha,
                source_git_revision=revision,
            )
            self.assertEqual(binding.source_path, str(source))
            self.assertEqual(
                binding.source_archive_bridge_member_sha256,
                hashlib.sha256(payload).hexdigest(),
            )
            with self.assertRaisesRegex(
                bridge.STARCLiveVJPContractError, "revision differs"
            ):
                bridge.authenticate_bridge_source_archive(
                    archive,
                    expected_source_archive_sha256=archive_sha,
                    source_git_revision="8" * 40,
                )

    def test_checkpoint_content_authentication_hashes_every_noncache_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root).resolve()
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "nested").mkdir()
            payloads = {
                "config.json": b"{}\n",
                "nested/model.bin": b"weights\n",
            }
            lines = []
            for relative, payload in sorted(payloads.items()):
                (checkpoint / relative).write_bytes(payload)
                lines.append(
                    f"{hashlib.sha256(payload).hexdigest()}  ./{relative}"
                )
            manifest = root / "checkpoint.sha256"
            manifest.write_text("\n".join(lines) + "\n", encoding="ascii")
            manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
            tree_sha = hashlib.sha256(b"test checkpoint tree").hexdigest()
            with mock.patch.object(
                bridge,
                "BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256",
                manifest_sha,
            ), mock.patch.object(
                bridge, "BERNINI_CHECKPOINT_CONTENT_FILE_COUNT", len(payloads)
            ), mock.patch.object(
                bridge, "BERNINI_CHECKPOINT_TREE_SHA256", tree_sha
            ):
                binding = bridge.authenticate_frozen_bernini_checkpoint_content(
                    checkpoint,
                    manifest,
                    expected_checkpoint_tree_sha256=tree_sha,
                    expected_checkpoint_content_manifest_sha256=manifest_sha,
                )
                self.assertEqual(
                    binding.checkpoint_content_verified_file_count, len(payloads)
                )
                (checkpoint / "nested/model.bin").write_bytes(b"tampered\n")
                with self.assertRaisesRegex(
                    bridge.STARCLiveVJPContractError, "content hash differs"
                ):
                    bridge.authenticate_frozen_bernini_checkpoint_content(
                        checkpoint,
                        manifest,
                        expected_checkpoint_tree_sha256=tree_sha,
                        expected_checkpoint_content_manifest_sha256=manifest_sha,
                    )

    def test_authenticated_builder_emits_runner_exact_v2_not_mechanism_schema(self) -> None:
        import run_starc_core4_critic_pilot_v1 as runner

        instruction = "the dog turns its head right"
        geometry = bridge.LatentPatchGeometry((1, 16, 21, 60, 62))
        clean_sha = hashlib.sha256(b"clean").hexdigest()
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root).resolve()

            candidate_unsigned = {
                "schema_version": bridge.CANDIDATE_BINDING_SCHEMA,
                "candidate_id": "current-full644-dog",
                "source_video_sha256": hashlib.sha256(b"source").hexdigest(),
                "instruction_sha256": hashlib.sha256(
                    instruction.encode("utf-8")
                ).hexdigest(),
                "current_clean_latent_tensor_sha256": clean_sha,
                "latent_shape": list(geometry.latent_shape),
                "patch_order": "phase_major_then_patch_row_major",
                "external_inference_inputs": ["source_video", "instruction"],
                "auxiliary_spatial_inputs": [],
            }
            candidate_receipt = {
                **candidate_unsigned,
                "receipt_digest": bridge.object_sha256(candidate_unsigned),
            }
            candidate_path = root / "candidate.json"
            candidate_path.write_text(
                json.dumps(candidate_receipt, sort_keys=True, separators=(",", ":")),
                encoding="ascii",
            )
            candidate = bridge.authenticate_current_candidate_manifest(
                candidate_path,
                expected_manifest_sha256=hashlib.sha256(
                    candidate_path.read_bytes()
                ).hexdigest(),
                instruction=instruction,
            )

            critic_path = root / "critic.safetensors"
            critic_path.write_bytes(b"frozen critic head")
            critic_sha = hashlib.sha256(critic_path.read_bytes()).hexdigest()
            config_unsigned = {"schema_version": bridge.CRITIC_CONFIG_SCHEMA}
            config_receipt = {
                **config_unsigned,
                "receipt_digest": bridge.object_sha256(config_unsigned),
            }
            config_path = root / "critic-config.json"
            config_path.write_text(
                json.dumps(config_receipt, sort_keys=True, separators=(",", ":")),
                encoding="ascii",
            )
            state_digest = hashlib.sha256(b"critic-state").hexdigest()
            checkpoint_unsigned = {
                "schema_version": bridge.CRITIC_CHECKPOINT_SCHEMA,
                "checkpoint_path": str(critic_path),
                "checkpoint_file_sha256": critic_sha,
                "checkpoint_state_content_digest": state_digest,
                "config_receipt_digest": config_receipt["receipt_digest"],
            }
            checkpoint_receipt = {
                **checkpoint_unsigned,
                "receipt_digest": bridge.object_sha256(checkpoint_unsigned),
            }
            checkpoint_receipt_path = root / "critic-checkpoint.json"
            checkpoint_receipt_path.write_text(
                json.dumps(
                    checkpoint_receipt, sort_keys=True, separators=(",", ":")
                ),
                encoding="ascii",
            )
            artifact = bridge.VerifiedCriticArtifact(
                checkpoint_path=str(critic_path),
                checkpoint_file_sha256=critic_sha,
                manifest_path=str(checkpoint_receipt_path),
                manifest_file_sha256=hashlib.sha256(
                    checkpoint_receipt_path.read_bytes()
                ).hexdigest(),
                manifest_receipt_digest=checkpoint_receipt["receipt_digest"],
                config_manifest_path=str(config_path),
                config_manifest_file_sha256=hashlib.sha256(
                    config_path.read_bytes()
                ).hexdigest(),
                config_manifest_receipt_digest=config_receipt["receipt_digest"],
                checkpoint_state_content_digest=state_digest,
                checkpoint_tensor_count=1,
                excluded_state_keys=bridge.NON_HEAD_CRITIC_STATE_KEYS,
                runtime_class=(
                    "latent_temporal_event_critic.FrozenHiddenTemporalEventCritic"
                ),
                verified=True,
            )

            evidence_rows = tuple(
                {
                    "rank": index,
                    "shape": [1, 21, 16, 1536],
                    "action_digest": hashlib.sha256(
                        f"action-{index}".encode("ascii")
                    ).hexdigest(),
                    "noop_digest": hashlib.sha256(
                        f"noop-{index}".encode("ascii")
                    ).hexdigest(),
                    "norm": 0.5 + index,
                    "finite_nonzero": True,
                    "action_is_exact_negative_noop": True,
                }
                for index in range(4)
            )
            evidence_digest = bridge.object_sha256(
                {
                    "schema_version": "bernini-starc-all-rank-hidden-vjp-v2",
                    "ordered_rank_evidence": list(evidence_rows),
                }
            )
            evidence = bridge.AllRankHiddenBackwardEvidence(
                ordered_rank_evidence=evidence_rows,
                evidence_digest=evidence_digest,
            )
            proof = bridge.STARCLiveVJPProof(
                gradient=SimpleNamespace(
                    shape=geometry.latent_shape, dtype="torch.float32"
                ),
                critic_score=0.25,
                gradient_norm=0.125,
                minimum_norm=1.0e-12,
                geometry=geometry,
                shard=bridge.make_sp4_contiguous_shard(geometry, 0),
                candidate=candidate,
                critic_artifact=artifact,
                sketch_digest=bridge.fixed_spatial_sketch_digest(
                    geometry.patch_positions
                ),
                sketch_coordinates=16,
                sketch_seed=bridge.SPATIAL_SKETCH_SEED,
                clean_latent_value_digest=clean_sha,
                collective_backend=bridge.LIVE_VJP_SP4_IMPLEMENTATION,
                real_sp4_autograd_collective=True,
                replica_contract_digest=hashlib.sha256(
                    b"replica-contract"
                ).hexdigest(),
                replica_consensus_observed=True,
                replicated_score_consensus_digest=hashlib.sha256(
                    b"score"
                ).hexdigest(),
                all_rank_hidden_backward_digest=evidence_digest,
                production_runtime_dimensions=True,
                hook_call_order=("action", "noop"),
                x_sigma_value_digest=hashlib.sha256(b"x-sigma").hexdigest(),
                action_condition_value_digest=hashlib.sha256(
                    b"action condition"
                ).hexdigest(),
                noop_condition_value_digest=hashlib.sha256(
                    b"noop condition"
                ).hexdigest(),
                all_rank_hidden_backward_evidence=evidence,
                instruction_text=instruction,
            )

            source = METHOD_ROOT / "starch_live_vjp_bridge_v1.py"
            source_payload = source.read_bytes()
            source_revision = "6" * 40
            source_archive = root / "bridge.tar"
            with tarfile.open(
                source_archive,
                mode="w",
                format=tarfile.PAX_FORMAT,
                pax_headers={"comment": source_revision},
            ) as handle:
                info = tarfile.TarInfo(bridge.LIVE_VJP_BRIDGE_ARCHIVE_MEMBER)
                info.size = len(source_payload)
                info.mode = 0o644
                info.mtime = 0
                handle.addfile(info, io.BytesIO(source_payload))
            source_archive_sha = hashlib.sha256(
                source_archive.read_bytes()
            ).hexdigest()

            checkpoint_root = root / "bernini"
            checkpoint_root.mkdir()
            checkpoint_payload = b"Bernini weights"
            (checkpoint_root / "model.bin").write_bytes(checkpoint_payload)
            content_manifest = root / "bernini.sha256"
            content_manifest.write_text(
                f"{hashlib.sha256(checkpoint_payload).hexdigest()}  ./model.bin\n",
                encoding="ascii",
            )
            content_manifest_sha = hashlib.sha256(
                content_manifest.read_bytes()
            ).hexdigest()
            tree_sha = hashlib.sha256(b"Bernini tree").hexdigest()

            master = root / "master.json"
            master.write_text("{}\n", encoding="ascii")
            master_sha = hashlib.sha256(master.read_bytes()).hexdigest()
            graph = SimpleNamespace(
                master_path=master,
                master_file_sha256=master_sha,
                master_receipt_digest=hashlib.sha256(b"master receipt").hexdigest(),
                content_digest=hashlib.sha256(b"population").hexdigest(),
            )
            gradient_sha = hashlib.sha256(b"gradient").hexdigest()
            with mock.patch.object(
                bridge,
                "BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256",
                content_manifest_sha,
            ), mock.patch.object(
                bridge, "BERNINI_CHECKPOINT_CONTENT_FILE_COUNT", 1
            ), mock.patch.object(
                bridge, "BERNINI_CHECKPOINT_TREE_SHA256", tree_sha
            ), mock.patch.object(
                bridge, "_tensor_value_digest", return_value=gradient_sha
            ), mock.patch.object(
                runner.StarcMaterializerAdapter, "load", return_value=graph
            ):
                receipt = bridge.build_authenticated_composite_receipt(
                    proof,
                    materializer_master=master,
                    expected_materializer_master_sha256=master_sha,
                    bridge_source_archive=source_archive,
                    expected_bridge_source_archive_sha256=source_archive_sha,
                    bridge_source_git_revision=source_revision,
                    checkpoint_root=checkpoint_root,
                    checkpoint_content_manifest=content_manifest,
                    expected_checkpoint_tree_sha256=tree_sha,
                    expected_checkpoint_content_manifest_sha256=(
                        content_manifest_sha
                    ),
                )
            self.assertEqual(
                receipt["schema_version"], runner.LIVE_VJP_BINDING_SCHEMA
            )
            self.assertNotEqual(receipt["schema_version"], bridge.SCHEMA_VERSION)
            self.assertEqual(
                receipt["current_rv2v_clean_latent"][
                    "candidate_manifest_receipt_digest"
                ],
                candidate.manifest_receipt_digest,
            )
            collective = receipt["sp4_differentiable_collective_proof"]
            self.assertEqual(
                collective["rank_local_hidden_global_shape"],
                [1, 21, 930, 1536],
            )
            self.assertEqual(
                collective["autograd_collective_tensor_shape"],
                [1, 21, 16, 1536],
            )
            self.assertFalse(receipt["editor_optimizer_authorized"])
            self.assertFalse(receipt["scientific_critic_claim_authorized"])

    def test_all_three_authenticated_full644_geometries_are_derived(self) -> None:
        self.assertEqual(
            set(bridge.SUPPORTED_FULL644_LATENT_SHAPES), set(EXPECTED_GEOMETRIES)
        )
        for shape, expected in EXPECTED_GEOMETRIES.items():
            geometry = bridge.LatentPatchGeometry(shape)
            self.assertTrue(geometry.is_supported_full644)
            self.assertEqual(
                (geometry.patch_rows, geometry.patch_columns), expected["grid"]
            )
            self.assertEqual(geometry.patch_positions, expected["patches"])
            self.assertEqual(geometry.global_tokens, expected["tokens"])
            self.assertEqual(geometry.local_tokens, expected["local"])
            self.assertEqual(geometry.padding_tokens, expected["padding"])
            self.assertEqual(
                bridge.fixed_spatial_sketch_digest(geometry.patch_positions),
                expected["digest"],
            )
            self.assertEqual(
                geometry.token_coordinate(geometry.global_tokens - 1),
                (20, geometry.patch_rows - 1, geometry.patch_columns - 1),
            )

    def test_core4_60x62_rank3_mapping_excludes_exact_two_padding_tokens(self) -> None:
        geometry = bridge.LatentPatchGeometry((1, 16, 21, 60, 62))
        shards = [bridge.make_sp4_contiguous_shard(geometry, rank) for rank in range(4)]
        self.assertEqual(
            [(row.global_start, row.global_valid_stop) for row in shards],
            [(0, 4883), (4883, 9766), (9766, 14649), (14649, 19530)],
        )
        self.assertEqual(shards[3].padded_stop, 19532)
        self.assertEqual(shards[3].valid_tokens, 4881)
        self.assertEqual(shards[3].padding_tokens, 2)

    def test_other_geometry_padding_is_exact_not_assumed_from_p930(self) -> None:
        p928 = bridge.LatentPatchGeometry((1, 16, 21, 64, 58))
        p918 = bridge.LatentPatchGeometry((1, 16, 21, 68, 54))
        self.assertEqual(
            [bridge.make_sp4_contiguous_shard(p928, rank).padding_tokens for rank in range(4)],
            [0, 0, 0, 0],
        )
        self.assertEqual(
            [bridge.make_sp4_contiguous_shard(p918, rank).padding_tokens for rank in range(4)],
            [0, 0, 0, 2],
        )

    def test_only_autograd_aware_distributed_all_reduce_is_present(self) -> None:
        source = (METHOD_ROOT / "starch_live_vjp_bridge_v1.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("torch.distributed.all_reduce(", source)
        self.assertNotIn("distributed.all_reduce(", source)
        self.assertIn("distributed_nn_functional.all_reduce(", source)
        self.assertIn("score.sum() / float(SP_SIZE)", source)
        self.assertIn("reduce_replicated_input_vjp", source)

    def test_public_contract_has_no_mask_track_pose_flow_or_optimizer(self) -> None:
        self.assertEqual(
            bridge.EXTERNAL_INFERENCE_INPUTS, ("source_video", "instruction")
        )
        self.assertIn("mask", bridge.FORBIDDEN_AUXILIARY_INPUTS)
        public_parameters = set(
            inspect.signature(bridge.STARCLiveVJPBridgeV1.__init__).parameters
        ) | set(
            inspect.signature(
                bridge.STARCLiveVJPBridgeV1.prove_current_clean_latent_vjp
            ).parameters
        )
        self.assertTrue(
            set(bridge.FORBIDDEN_AUXILIARY_INPUTS).isdisjoint(public_parameters)
        )
        source = (METHOD_ROOT / "starch_live_vjp_bridge_v1.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("optimizer.step", source)
        self.assertNotIn("backward()", source)

    def test_proof_never_authorizes_scientific_claim_or_editor_optimizer(self) -> None:
        instruction = "the actor turns around"
        candidate = bridge.mechanism_only_candidate_binding(
            (1, 16, 21, 60, 62), instruction=instruction
        )
        geometry = candidate.geometry
        proof = bridge.STARCLiveVJPProof(
            gradient=SimpleNamespace(shape=geometry.latent_shape),
            critic_score=0.1,
            gradient_norm=1.0,
            minimum_norm=1.0e-12,
            geometry=geometry,
            shard=bridge.make_sp4_contiguous_shard(geometry, 0),
            candidate=candidate,
            critic_artifact=None,
            sketch_digest=bridge.fixed_spatial_sketch_digest(
                geometry.patch_positions
            ),
            sketch_coordinates=16,
            sketch_seed=bridge.SPATIAL_SKETCH_SEED,
            clean_latent_value_digest="a" * 64,
            collective_backend="test-only",
            real_sp4_autograd_collective=False,
            replica_contract_digest="b" * 64,
            replica_consensus_observed=False,
            replicated_score_consensus_digest=None,
            all_rank_hidden_backward_digest=None,
            production_runtime_dimensions=False,
            hook_call_order=("action", "noop"),
        )
        receipt = proof.receipt()
        self.assertTrue(receipt["mechanism_vjp_nonzero_finite"])
        self.assertFalse(receipt["critic_artifact_verified"])
        self.assertFalse(receipt["scientific_claim_authorized"])
        self.assertFalse(receipt["action_editing_success_claim_authorized"])
        self.assertFalse(receipt["editor_optimizer_authorized"])
        self.assertIn(
            "hash_bound_real_critic_checkpoint_manifest_missing",
            receipt["claim_blockers"],
        )
        self.assertEqual(receipt["auxiliary_spatial_inputs"], [])
        with self.assertRaisesRegex(
            bridge.STARCLiveVJPContractError, "mechanism-only or incomplete"
        ):
            bridge.build_authenticated_composite_receipt(
                proof,
                materializer_master="/unreachable/materializer.json",
                expected_materializer_master_sha256="1" * 64,
                bridge_source_archive="/unreachable/bridge.tar",
                expected_bridge_source_archive_sha256="2" * 64,
                bridge_source_git_revision="3" * 40,
                checkpoint_root="/unreachable/checkpoint",
                checkpoint_content_manifest="/unreachable/checkpoint.sha256",
            )
        with self.assertRaisesRegex(
            bridge.STARCLiveVJPContractError, "only SP rank 0"
        ):
            bridge.write_authenticated_composite_receipt(
                "/unreachable/composite.json", receipt
            )

    def test_candidate_manifest_hash_geometry_and_public_inputs_are_bound(self) -> None:
        instruction = "the dog jumps over the box"
        unsigned = {
            "schema_version": bridge.CANDIDATE_BINDING_SCHEMA,
            "candidate_id": "full644-cell-0007",
            "source_video_sha256": "1" * 64,
            "instruction_sha256": hashlib.sha256(
                instruction.encode("utf-8")
            ).hexdigest(),
            "current_clean_latent_tensor_sha256": "2" * 64,
            "latent_shape": [1, 16, 21, 64, 58],
            "patch_order": "phase_major_then_patch_row_major",
            "external_inference_inputs": ["source_video", "instruction"],
            "auxiliary_spatial_inputs": [],
        }
        manifest = {**unsigned, "receipt_digest": bridge.object_sha256(unsigned)}
        payload = json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root).resolve()
            path = root / "candidate.json"
            path.write_bytes(payload)
            binding = bridge.authenticate_current_candidate_manifest(
                path,
                expected_manifest_sha256=hashlib.sha256(payload).hexdigest(),
                instruction=instruction,
            )
        self.assertTrue(binding.authenticated)
        self.assertEqual(binding.geometry.patch_positions, 928)
        self.assertEqual(binding.source_video_sha256, "1" * 64)
        self.assertEqual(binding.blockers(), ())

    def test_real_critic_verifier_matches_geometry_neutral_head_checkpoint(self) -> None:
        import run_starc_core4_critic_pilot_v1 as pilot

        self.assertEqual(
            bridge.GEOMETRY_NEUTRAL_CRITIC_CONFIG,
            pilot.GEOMETRY_NEUTRAL_CRITIC_CONFIG,
        )
        self.assertEqual(
            bridge.NON_HEAD_CRITIC_STATE_KEYS, pilot.NON_HEAD_STATE_KEYS
        )
        signature = inspect.signature(bridge.verify_frozen_starc_critic_artifact)
        self.assertIn("config_manifest_path", signature.parameters)
        self.assertIn("expected_config_manifest_sha256", signature.parameters)
        source = inspect.getsource(bridge.verify_frozen_starc_critic_artifact)
        self.assertIn("geometry_neutral_pre_sketched_critic_head_only", source)
        self.assertIn("forward_sketched_residual_only", source)
        self.assertIn("if name not in NON_HEAD_CRITIC_STATE_KEYS", source)


if torch is not None:

    class _ToyConfig:
        patch_size = (1, 2, 2)


    class _ToyBlock(torch.nn.Module):
        def __init__(self, *, detach: bool = False) -> None:
            super().__init__()
            self.detach = detach

        def forward(self, value):
            result = value + 0.031 * torch.tanh(value)
            return result.detach() if self.detach else result


    class _ToyTransformer(torch.nn.Module):
        dtype = torch.float32
        config = _ToyConfig()

        def __init__(self, *, detach_block15: bool = False) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList(
                [_ToyBlock(detach=detach_block15 and index == 15) for index in range(30)]
            )
            self.patch_calls = 0

        def patch_vae_latent(self, value, *, source_id):
            if source_id != 0:
                raise AssertionError("toy received non-target source_id")
            self.patch_calls += 1
            batch, channels, phases, height, width = map(int, value.shape)
            packed = (
                value.reshape(
                    batch,
                    channels,
                    phases,
                    height // 2,
                    2,
                    width // 2,
                    2,
                )
                .permute(0, 2, 3, 5, 1, 4, 6)
                .reshape(batch, phases * (height // 2) * (width // 2), -1)
            )
            count = int(packed.shape[1])
            real = torch.arange(count * 2, dtype=torch.float64).reshape(
                1, 1, count, 2
            )
            rotary = torch.complex(real, torch.zeros_like(real))
            return packed, rotary


    class _ToyDiffusion(torch.nn.Module):
        def __init__(self, transformer: _ToyTransformer, *, rank: int = 0) -> None:
            super().__init__()
            self.transformer = transformer
            self.transformer_2 = None
            self.rank = rank
            self.calls = []

        def shared_step(self, **kwargs):
            tokens = kwargs["noisy_latents"]
            total = int(tokens.shape[1])
            local = (total + 3) // 4
            start = self.rank * local
            stop = min(start + local, total)
            hidden = tokens[:, start:stop]
            if int(hidden.shape[1]) < local:
                hidden = torch.nn.functional.pad(
                    hidden, (0, 0, 0, local - int(hidden.shape[1]))
                )
            condition = kwargs["cond_embeds"].float().mean()
            hidden = hidden * (1.0 + 0.25 * condition)
            for block_module in self.transformer.blocks:
                hidden = block_module(hidden)
            self.calls.append(
                {
                    "tokens_id": id(tokens),
                    "condition": float(condition.detach().item()),
                    "local_shape": tuple(hidden.shape),
                }
            )
            patch_dimension = int(tokens.shape[2])
            return tokens[:, :, :patch_dimension] * (1.0 + 0.01 * condition)


    class _ToyCritic(torch.nn.Module):
        def forward_sketched_residual(self, value, *, require_input_grad):
            score = value.square().mean(dim=(1, 2, 3)) + 0.01 * value.mean(
                dim=(1, 2, 3)
            )
            if require_input_grad and (not score.requires_grad or score.grad_fn is None):
                raise RuntimeError("toy score detached")
            return SimpleNamespace(score=score)


    class _RankLocalTestCollective:
        real_sp4_autograd_collective = False
        backend = "single_process_rank_local_test_oracle"

        @staticmethod
        def assert_replica_consensus(digest, *, rank):
            del digest, rank
            return False

        @staticmethod
        def assert_replicated_score_consensus(score, *, rank):
            del score, rank
            return None

        @staticmethod
        def assert_all_rank_hidden_backward(action_gradient, noop_gradient, *, rank):
            del action_gradient, noop_gradient, rank
            return None

        def globalize_sketch(self, local_sketch, *, geometry, shard, role):
            del geometry
            return bridge.GlobalSketchAssembly(
                tensor=local_sketch,
                backend=self.backend,
                real_sp4_autograd_collective=False,
                rank=shard.rank,
                role=role,
            )

        @staticmethod
        def reduce_replicated_input_vjp(local_vjp, *, rank):
            del rank
            return local_vjp


    def _toy_bridge(*, detach_block15: bool = False):
        instruction = "the actor completes the motion"
        candidate = bridge.mechanism_only_candidate_binding(
            (1, 2, 3, 4, 6), instruction=instruction
        )
        dimensions = bridge.BerniniRuntimeDimensions(
            block_count=30,
            block_index=15,
            hidden_size=8,
            text_tokens=4,
            text_width=6,
            rotary_width=2,
            spatial_sketch_coordinates=2,
        )
        transformer = _ToyTransformer(detach_block15=detach_block15).eval()
        diffusion = _ToyDiffusion(transformer).eval()
        critic = _ToyCritic().eval()
        action = torch.tensor(0.4, dtype=torch.float32).reshape(1, 1, 1).expand(
            1, 4, 6
        )
        noop = torch.tensor(-0.2, dtype=torch.float32).reshape(1, 1, 1).expand(
            1, 4, 6
        )
        instance = bridge.STARCLiveVJPBridgeV1(
            diffusion=diffusion,
            transformer=transformer,
            critic=critic,
            candidate=candidate,
            instruction=instruction,
            action_condition=action,
            noop_condition=noop,
            sp_rank=0,
            collective=_RankLocalTestCollective(),
            dimensions=dimensions,
        )
        return instance, transformer, diffusion


@unittest.skipIf(torch is None, "torch is unavailable")
class LiveVJPTensorMechanismTests(unittest.TestCase):
    def test_geometry_neutral_head_safetensors_artifact_verifies_exactly(self) -> None:
        try:
            from safetensors.torch import save_file
            import latent_temporal_event_critic as critic_core
        except ImportError:
            self.skipTest("safetensors or STARC critic runtime is unavailable")

        config = critic_core.CriticConfig(**bridge.GEOMETRY_NEUTRAL_CRITIC_CONFIG)
        critic = critic_core.FrozenHiddenTemporalEventCritic(
            torch.eye(16, dtype=torch.float32), config=config
        ).eval()
        for parameter in critic.parameters():
            parameter.requires_grad_(False)
        state = {
            name: tensor.detach().cpu().contiguous().clone()
            for name, tensor in critic.state_dict().items()
            if name not in bridge.NON_HEAD_CRITIC_STATE_KEYS
        }
        state_digest, state_count = bridge._critic_state_content_digest(
            critic, excluded_keys=bridge.NON_HEAD_CRITIC_STATE_KEYS
        )
        config_unsigned = {
            "schema_version": bridge.CRITIC_CONFIG_SCHEMA,
            "critic_config": dict(bridge.GEOMETRY_NEUTRAL_CRITIC_CONFIG),
            "critic_config_content_digest": bridge.object_sha256(
                bridge.GEOMETRY_NEUTRAL_CRITIC_CONFIG
            ),
            "pre_sketched_head_contract": {
                "entrypoint": "forward_sketched_residual_only",
                "geometry_neutral_after_fixed_sketch": True,
                "constructor_spatial_buffer": "inert_16x16_identity_never_consumed",
                "constructor_spatial_buffer_checkpointed": False,
                "full_hidden_forward_authorized": False,
                "geometry_specific_sketches_authenticated_by_materializer": True,
            },
            "nuisance_basis_used": False,
            "core4_scientific_claim_authorized": False,
            "editor_optimizer_present_or_authorized": False,
        }
        config_manifest = {
            **config_unsigned,
            "receipt_digest": bridge.object_sha256(config_unsigned),
        }
        config_payload = json.dumps(
            config_manifest, sort_keys=True, separators=(",", ":")
        ).encode("ascii")

        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root).resolve()
            config_path = root / "config.json"
            config_path.write_bytes(config_payload)
            checkpoint_path = root / "critic.safetensors"
            metadata = {
                "schema_version": bridge.CRITIC_CHECKPOINT_SCHEMA,
                "config_receipt_digest": config_manifest["receipt_digest"],
                "checkpoint_state_content_digest": state_digest,
                "optimizer_step": "200",
                "selection": "final_step_200_only",
            }
            save_file(state, str(checkpoint_path), metadata=metadata)
            checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            checkpoint_unsigned = {
                "schema_version": bridge.CRITIC_CHECKPOINT_SCHEMA,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_file_sha256": checkpoint_sha,
                "checkpoint_state_content_digest": state_digest,
                "checkpoint_tensor_count": state_count,
                "checkpoint_scope": "geometry_neutral_pre_sketched_critic_head_only",
                "excluded_constructor_buffer_keys": list(
                    bridge.NON_HEAD_CRITIC_STATE_KEYS
                ),
                "config_receipt_digest": config_manifest["receipt_digest"],
                "optimizer_step": 200,
                "only_final_checkpoint_saved": True,
                "best_checkpoint_saved": False,
                "confirmation_sample_seen_before_checkpoint_save": False,
                "state_tensor_byte_parity_after_fresh_load": True,
                "fit_score_parity_after_fresh_load": True,
                "critic_frozen_after_reload": True,
                "editor_checkpoint_or_parameter_present": False,
                "editor_optimizer_authorized": False,
            }
            checkpoint_manifest = {
                **checkpoint_unsigned,
                "receipt_digest": bridge.object_sha256(checkpoint_unsigned),
            }
            checkpoint_payload = json.dumps(
                checkpoint_manifest, sort_keys=True, separators=(",", ":")
            ).encode("ascii")
            checkpoint_manifest_path = root / "checkpoint.json"
            checkpoint_manifest_path.write_bytes(checkpoint_payload)
            artifact = bridge.verify_frozen_starc_critic_artifact(
                critic,
                checkpoint_path=checkpoint_path,
                expected_checkpoint_sha256=checkpoint_sha,
                manifest_path=checkpoint_manifest_path,
                expected_manifest_sha256=hashlib.sha256(
                    checkpoint_payload
                ).hexdigest(),
                config_manifest_path=config_path,
                expected_config_manifest_sha256=hashlib.sha256(
                    config_payload
                ).hexdigest(),
            )
        self.assertTrue(artifact.verified)
        self.assertEqual(
            artifact.excluded_state_keys, bridge.NON_HEAD_CRITIC_STATE_KEYS
        )
        self.assertEqual(artifact.checkpoint_state_content_digest, state_digest)

    def test_actual_geometry_mapping_reconstructs_exact_global_order(self) -> None:
        for shape in EXPECTED_GEOMETRIES:
            geometry = bridge.LatentPatchGeometry(shape)
            full = torch.arange(
                geometry.global_tokens, dtype=torch.float32
            ).reshape(1, geometry.global_tokens, 1)
            padded = torch.nn.functional.pad(
                full, (0, 0, 0, geometry.padding_tokens), value=-999.0
            )
            shards = list(padded.split(geometry.local_tokens, dim=1))
            assembled = bridge.assemble_sp4_shards_for_test(
                shards, geometry=geometry
            )
            self.assertTrue(torch.equal(assembled, full))
            self.assertFalse(bool((assembled == -999.0).any().item()))

    def test_sp4_assembly_and_replicated_input_vjp_match_full_reference(self) -> None:
        geometry = bridge.LatentPatchGeometry((1, 2, 3, 4, 6))
        torch.manual_seed(11)
        clean_reference = torch.randn(
            1, geometry.global_tokens, 4, dtype=torch.float64, requires_grad=True
        )
        weight = torch.linspace(0.2, 1.1, geometry.global_tokens, dtype=torch.float64)

        def hidden(value):
            return torch.tanh(0.7 * value + 0.13)

        reference_hidden = hidden(clean_reference)
        reference_score = (
            reference_hidden.square() * weight[None, :, None]
        ).sum()
        reference_vjp = torch.autograd.grad(reference_score, clean_reference)[0]

        replicas = [clean_reference.detach().clone().requires_grad_(True) for _ in range(4)]
        local_shards = []
        padding_leaves = []
        for rank, replica in enumerate(replicas):
            shard = bridge.make_sp4_contiguous_shard(geometry, rank)
            local = hidden(replica)[:, shard.global_start : shard.global_valid_stop]
            if shard.padding_tokens:
                padding = torch.randn(
                    1,
                    shard.padding_tokens,
                    4,
                    dtype=torch.float64,
                    requires_grad=True,
                )
                padding_leaves.append(padding)
                local = torch.cat((local, padding), dim=1)
            local_shards.append(local)
        assembled = bridge.assemble_sp4_shards_for_test(
            local_shards, geometry=geometry
        )
        distributed_score = (assembled.square() * weight[None, :, None]).sum()
        # Four ranks evaluate the same replicated critic score.  Their score/4
        # objectives sum to one reference scalar before the all-reduce backward.
        distributed_replicated_objective = sum(
            distributed_score / 4.0 for _rank in range(4)
        )
        gradients = torch.autograd.grad(
            distributed_replicated_objective,
            tuple(replicas) + tuple(padding_leaves),
        )
        replicated_input_vjp = torch.stack(gradients[:4]).sum(dim=0)
        self.assertTrue(torch.allclose(assembled, reference_hidden, atol=0.0, rtol=0.0))
        self.assertTrue(
            torch.allclose(
                replicated_input_vjp, reference_vjp, atol=1.0e-12, rtol=1.0e-12
            )
        )
        for padding_gradient in gradients[4:]:
            self.assertTrue(torch.equal(padding_gradient, torch.zeros_like(padding_gradient)))

    def test_live_local_sketch_matches_materializer_fp32_index_add_order(self) -> None:
        geometry = bridge.LatentPatchGeometry((1, 2, 3, 4, 6))
        shard = bridge.make_sp4_contiguous_shard(geometry, 3)
        base = torch.linspace(
            -0.8,
            0.9,
            shard.local_tokens * 8,
            dtype=torch.float32,
            requires_grad=True,
        ).reshape(1, shard.local_tokens, 8)
        hidden = base.to(torch.bfloat16)
        sketch = bridge.make_fixed_spatial_sketch(
            geometry.patch_positions, coordinates=2
        )
        observed = bridge.sketch_rank_local_hidden_exact(
            hidden, geometry=geometry, shard=shard, spatial_sketch=sketch
        )

        local_index = torch.arange(shard.valid_tokens, dtype=torch.long)
        global_index = local_index + shard.global_start
        phase = torch.div(
            global_index, geometry.patch_positions, rounding_mode="floor"
        )
        patch = torch.remainder(global_index, geometry.patch_positions)
        values = hidden[0].index_select(0, local_index).float()
        expected = torch.zeros(1, geometry.phases, 2, 8, dtype=torch.float32)
        for coordinate in range(2):
            weights = sketch[coordinate].index_select(0, patch).unsqueeze(1)
            expected[0, :, coordinate, :].index_add_(
                0, phase, values * weights
            )
        self.assertTrue(torch.equal(observed, expected))
        self.assertEqual(observed.dtype, torch.float32)
        gradient = torch.autograd.grad(observed.square().sum(), base)[0]
        self.assertTrue(bool(torch.isfinite(gradient).all().item()))
        self.assertGreater(float(torch.linalg.vector_norm(gradient).item()), 0.0)

    def test_toy_live_hook_critic_scalar_has_nonzero_finite_clean_vjp(self) -> None:
        instance, transformer, diffusion = _toy_bridge()
        torch.manual_seed(17)
        clean = torch.randn(1, 2, 3, 4, 6, dtype=torch.float32, requires_grad=True)
        epsilon = torch.randn_like(clean).detach()
        proof = instance.prove_current_clean_latent_vjp(clean, epsilon)
        self.assertEqual(tuple(proof.gradient.shape), tuple(clean.shape))
        self.assertTrue(bool(torch.isfinite(proof.gradient).all().item()))
        self.assertGreater(proof.gradient_norm, 0.0)
        self.assertEqual(proof.hook_call_order, ("action", "noop"))
        self.assertEqual(transformer.patch_calls, 1)
        self.assertEqual(len(diffusion.calls), 2)
        self.assertEqual(len({row["tokens_id"] for row in diffusion.calls}), 1)
        receipt = proof.receipt()
        self.assertTrue(receipt["mechanism_vjp_nonzero_finite"])
        self.assertTrue(receipt["live_differentiable_forward_hook_observed"])
        self.assertFalse(receipt["critic_artifact_verified"])
        self.assertFalse(receipt["real_sp4_autograd_collective_observed"])
        self.assertFalse(receipt["scientific_claim_authorized"])
        self.assertEqual(receipt["spatial_sketch"]["patch_positions"], 6)
        self.assertEqual(receipt["spatial_sketch"]["coordinates"], 2)

    def test_detached_block15_hook_fails_closed_before_claim(self) -> None:
        instance, _transformer, _diffusion = _toy_bridge(detach_block15=True)
        clean = torch.randn(1, 2, 3, 4, 6, dtype=torch.float32, requires_grad=True)
        epsilon = torch.randn_like(clean).detach()
        with self.assertRaisesRegex(
            bridge.STARCLiveVJPContractError, "graph-connected"
        ):
            instance.prove_current_clean_latent_vjp(clean, epsilon)


if __name__ == "__main__":
    unittest.main()
