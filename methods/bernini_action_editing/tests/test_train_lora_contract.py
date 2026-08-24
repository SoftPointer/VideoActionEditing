from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import train_lora as trainer  # noqa: E402


class _Affine:
    weight = object()


class _AttentionModel:
    def named_modules(self):
        yield "diff_dec.transformer.blocks.0.attn1.to_q", _Affine()
        yield "diff_dec.transformer.blocks.0.attn1.to_k", _Affine()
        yield "diff_dec.transformer.blocks.0.attn1.to_v", _Affine()
        yield "diff_dec.transformer.blocks.0.attn1.to_out.0", _Affine()
        yield "diff_dec.transformer.blocks.0.attn2.to_q", _Affine()
        yield "diff_dec.transformer.blocks.0.attn2.to_k", _Affine()
        yield "diff_dec.transformer.blocks.0.attn2.to_v", _Affine()
        yield "diff_dec.transformer.blocks.0.attn2.to_out.0", _Affine()
        # A similarly named frozen text projection must never match.
        yield "t5_text_encoder.block.0.attn1.to_q", _Affine()


class TrainerContractTests(unittest.TestCase):
    @staticmethod
    def _full644_cli() -> argparse.Namespace:
        return argparse.Namespace(
            num_frames=81,
            max_steps=644,
            save_every=64,
            learning_rate=1.0e-4,
            weight_decay=0.0,
            max_grad_norm=1.0,
            seed=20260817,
            objective="reference_dpo_preservation",
            contrastive_negative_schedule="rotate",
            preference_weight=1.0,
            preference_margin=0.05,
            preference_temperature=20.0,
            dpo_beta=10.0,
            preservation_weight=0.25,
            lora_rank=64,
            lora_alpha=64,
            exploratory_full644_one_pass=True,
            allow_incomplete_dataset=False,
            allow_reward_selected_synthetic_targets=False,
            resume=None,
            full644_source_authority_receipt="/sealed/full644-authority.json",
            expected_full644_source_authority_sha256=(
                trainer.FULL644_SOURCE_AUTHORITY_SHA256
            ),
            expected_bernini_commit=trainer.BERNINI_OFFICIAL_COMMIT,
            expected_veomni_commit=trainer.VEOMNI_TESTED_COMMIT,
            expected_checkpoint_tree_sha256=trainer.CHECKPOINT_TREE_SHA256,
            method_source_revision="1" * 40,
            method_source_archive_sha256="2" * 64,
        )

    @staticmethod
    def _full644_peft_train_config(targets: list[str]) -> dict:
        return {
            "alora_invocation_tokens": None,
            "alpha_pattern": {},
            "arrow_config": None,
            "auto_mapping": None,
            "base_model_name_or_path": None,
            "bias": "none",
            "corda_config": None,
            "ensure_weight_tying": False,
            "eva_config": None,
            "exclude_modules": None,
            "fan_in_fan_out": False,
            "inference_mode": False,
            "init_lora_weights": True,
            "layer_replication": None,
            "layers_pattern": None,
            "layers_to_transform": None,
            "loftq_config": {},
            "lora_alpha": trainer.FULL644_EXPLORATORY_ALPHA,
            "lora_bias": False,
            "lora_dropout": 0.0,
            "lora_ga_config": None,
            "megatron_config": None,
            "megatron_core": "megatron.core",
            "modules_to_save": None,
            "peft_type": "LORA",
            "peft_version": trainer.FULL644_PEFT_VERSION,
            "qalora_group_size": 16,
            "r": trainer.FULL644_EXPLORATORY_RANK,
            "rank_pattern": {},
            "revision": None,
            "target_modules": set(targets),
            "target_parameters": None,
            "task_type": None,
            "trainable_token_indices": None,
            "use_bdlora": None,
            "use_dora": False,
            "use_qalora": False,
            "use_rslora": False,
        }

    def test_single_expert_full_noise_mv2v_contract(self) -> None:
        scheduler = trainer.noise_scheduler_kwargs()
        self.assertEqual(scheduler["noise_tmin"], 0.0)
        self.assertEqual(scheduler["noise_tmax"], 1.0)
        self.assertEqual(scheduler["shift_config"]["mv2v"], 5.0)
        self.assertEqual(trainer.NUM_FRAMES, 81)
        self.assertEqual(trainer.LATENT_FRAMES, 21)

    def test_renderer_config_requires_local_single_transformer(self) -> None:
        checkpoint = Path("/abs/bernini-r-1p3b")
        mapping = {
            "model_type": "bernini_renderer",
            "wan22_base": str(checkpoint),
            "skip_transformer_1": False,
            "skip_transformer_2": True,
            "use_src_id_rotary_emb": True,
            "max_sequence_length": 512,
        }
        trainer.validate_renderer_config_mapping(mapping, checkpoint)
        mapping["skip_transformer_2"] = False
        with self.assertRaisesRegex(trainer.TrainingContractError, "transformer_1"):
            trainer.validate_renderer_config_mapping(mapping, checkpoint)

    def test_attention_selection_is_wan_only_and_complete(self) -> None:
        selected = trainer.select_attention_projection_names(_AttentionModel())
        self.assertEqual(len(selected), 8)
        self.assertTrue(all(name.startswith("diff_dec.transformer") for name in selected))
        self.assertFalse(any("t5_text_encoder" in name for name in selected))

    def test_sanitize_keeps_only_instruction_and_two_vae_distributions(self) -> None:
        messages = [
            {"type": "video", "has_loss": 0},
            {"type": "text", "text": "Make the actor crouch.", "has_loss": 0},
            {"type": "video_gen", "has_loss": 1},
        ]
        clean = trainer.sanitize_preprocessed_row(
            {
                "inputs": json.dumps(messages),
                "video_vae_latents": [b"source", b"target"],
                "target_caption": "must be dropped",
                "source_video_path": "/audit/only.mp4",
            }
        )
        self.assertEqual(set(clean), {"inputs", "video_vae_latents", "source_name"})
        self.assertNotIn("target_caption", clean)
        self.assertEqual(clean["source_name"], trainer.TASK_SOURCE_NAME)

    def test_external_mask_or_swept_tube_is_rejected(self) -> None:
        messages = [
            {"type": "video", "has_loss": 0},
            {"type": "text", "text": "Turn left.", "has_loss": 0},
            {"type": "video_gen", "has_loss": 1},
        ]
        for key in ("edit_mask", "swept_tube", "trajectories"):
            with self.subTest(key=key), self.assertRaisesRegex(
                trainer.TrainingContractError, "spatial conditioning"
            ):
                trainer.sanitize_preprocessed_row(
                    {
                        "inputs": json.dumps(messages),
                        "video_vae_latents": [b"source", b"target"],
                        key: [1],
                    }
                )

    def test_single_sample_collate_preserves_latent_token_axis(self) -> None:
        try:
            import torch
        except Exception as error:  # pragma: no cover - minimal lint environment
            self.skipTest(f"torch unavailable: {error}")
        feature = {
            "input_ids": torch.ones(7, dtype=torch.long),
            "attention_mask": torch.ones(7, dtype=torch.long),
            "t5_input_lens": torch.tensor([7]),
            "input_vae_latents": torch.zeros(10, 16, 1, 2, 2),
            "input_vae_rope": torch.zeros(10, 1, 128),
            "vae_latents_mask": torch.tensor([False] * 5 + [True] * 5),
            "vae_seqlen": torch.tensor([10]),
            "timesteps": torch.tensor([500]),
            "target_velocity": torch.zeros(5, 16, 1, 2, 2),
            "target_lens": torch.tensor([5]),
            "num_tokens": torch.tensor([17]),
            "vlm_seqlen": torch.tensor([7]),
        }
        batch = trainer.collate_single_renderer_sample([feature])
        self.assertEqual(tuple(batch["input_ids"].shape), (1, 7))
        self.assertEqual(tuple(batch["vae_seqlen"].shape), (1, 1))
        self.assertEqual(tuple(batch["input_vae_latents"].shape), (10, 16, 1, 2, 2))
        trainer.validate_collated_supervision(batch)

    def test_cli_requires_frozen_hashes_and_exact_81_frames(self) -> None:
        base = dict(
            num_frames=81,
            max_steps=1,
            save_every=1,
            learning_rate=1e-4,
            weight_decay=0.0,
            max_grad_norm=1.0,
            expected_bernini_commit=trainer.BERNINI_OFFICIAL_COMMIT,
            expected_veomni_commit=trainer.VEOMNI_TESTED_COMMIT,
            expected_checkpoint_tree_sha256=trainer.CHECKPOINT_TREE_SHA256,
            method_source_revision="1" * 40,
            method_source_archive_sha256="2" * 64,
        )
        trainer.validate_cli(argparse.Namespace(**base))
        base["num_frames"] = 41
        with self.assertRaisesRegex(trainer.TrainingContractError, "81-frame"):
            trainer.validate_cli(argparse.Namespace(**base))

    def test_full644_profile_is_exact_one_pass_and_fresh_only(self) -> None:
        args = self._full644_cli()
        trainer.validate_cli(args)
        for name, hostile in (
            ("max_steps", 645),
            ("save_every", 0),
            ("lora_rank", 8),
            ("objective", "sft"),
            ("allow_incomplete_dataset", True),
            ("resume", "/old/checkpoint"),
            ("exploratory_full644_one_pass", 1),
            ("expected_full644_source_authority_sha256", "0" * 64),
        ):
            candidate = copy.copy(args)
            setattr(candidate, name, hostile)
            with self.subTest(name=name), self.assertRaises(
                trainer.TrainingContractError
            ):
                trainer.validate_cli(candidate)

        for step in range(trainer.FULL644_EXPLORATORY_STEPS):
            self.assertEqual(
                trainer.full644_one_pass_row_index(step, 644), step
            )
        for step, rows in ((-1, 644), (644, 644), (0, 643), (True, 644)):
            with self.subTest(step=step, rows=rows), self.assertRaises(
                trainer.TrainingContractError
            ):
                trainer.full644_one_pass_row_index(step, rows)

        trainer.validate_full644_trainable_parameter_count(
            trainer.FULL644_EXPLORATORY_TRAINABLE_PARAMETER_COUNT,
            profile_enabled=True,
        )
        trainer.validate_full644_trainable_parameter_count(
            1, profile_enabled=False
        )
        for value in (
            1,
            trainer.FULL644_EXPLORATORY_TRAINABLE_PARAMETER_COUNT - 1,
            trainer.FULL644_EXPLORATORY_TRAINABLE_PARAMETER_COUNT + 1,
        ):
            with self.subTest(trainable_parameter_count=value), self.assertRaises(
                trainer.TrainingContractError
            ):
                trainer.validate_full644_trainable_parameter_count(
                    value, profile_enabled=True
                )

    def test_full644_peft_construction_closes_all_38_fields(self) -> None:
        targets = [
            f"diff_dec.transformer.route_{index:03d}"
            for index in range(trainer.EXPECTED_LORA_TARGET_MODULES)
        ]
        config = self._full644_peft_train_config(targets)
        self.assertEqual(len(trainer.FULL644_PEFT_LORA_CONFIG_FIELDS), 38)
        self.assertEqual(set(config), trainer.FULL644_PEFT_LORA_CONFIG_FIELDS)
        trainer.validate_full644_peft_construction(
            config,
            peft_version=trainer.FULL644_PEFT_VERSION,
            expected_target_modules=targets,
            profile_enabled=True,
        )
        trainer.validate_full644_peft_construction(
            {"changed": True},
            peft_version="future-version",
            expected_target_modules=[],
            profile_enabled=False,
        )

        mutations = {
            "rank": ("r", 8),
            "alpha": ("lora_alpha", 8),
            "pissa": ("init_lora_weights", "pissa"),
            "loftq": ("loftq_config", {"loftq_bits": 4}),
            "lora_ga": ("lora_ga_config", {"direction": "stable"}),
            "alora": ("alora_invocation_tokens", [1]),
            "arrow": ("arrow_config", {"top_k": 1}),
            "bdlora": ("use_bdlora", {"nblocks": 4}),
            "lora_bias": ("lora_bias", True),
            "weight_tying": ("ensure_weight_tying", True),
            "task_type": ("task_type", "CAUSAL_LM"),
            "dora": ("use_dora", True),
            "rslora": ("use_rslora", True),
            "qalora_bool_alias": ("use_qalora", 0),
            "modules_to_save": ("modules_to_save", ["head"]),
            "target_parameters": ("target_parameters", ["weight"]),
            "peft_config_version": ("peft_version", "0.19.0"),
        }
        for label, (name, value) in mutations.items():
            candidate = copy.deepcopy(config)
            candidate[name] = value
            with self.subTest(label=label), self.assertRaises(
                trainer.TrainingContractError
            ):
                trainer.validate_full644_peft_construction(
                    candidate,
                    peft_version=trainer.FULL644_PEFT_VERSION,
                    expected_target_modules=targets,
                    profile_enabled=True,
                )

        for label, transform in (
            ("unknown_field", lambda value: value.__setitem__("runtime_config", {})),
            (
                "target_scope",
                lambda value: value["target_modules"].remove(targets[-1]),
            ),
        ):
            candidate = copy.deepcopy(config)
            transform(candidate)
            with self.subTest(label=label), self.assertRaises(
                trainer.TrainingContractError
            ):
                trainer.validate_full644_peft_construction(
                    candidate,
                    peft_version=trainer.FULL644_PEFT_VERSION,
                    expected_target_modules=targets,
                    profile_enabled=True,
                )

        with self.assertRaisesRegex(
            trainer.TrainingContractError, "runtime PEFT"
        ):
            trainer.validate_full644_peft_construction(
                config,
                peft_version="0.19.0",
                expected_target_modules=targets,
                profile_enabled=True,
            )

    def test_full644_source_authority_is_exact_and_non_profile_cannot_claim_it(self) -> None:
        authority = (
            METHOD_ROOT.parents[1]
            / "md/action_editing/20260814_man/evidence/"
            "stage_r64_joint_136309_r2/run_receipt.json"
        ).resolve()
        value = trainer.validate_full644_source_authority(
            authority,
            expected_sha256=trainer.FULL644_SOURCE_AUTHORITY_SHA256,
        )
        self.assertEqual(value["membership_rows"], 644)
        self.assertEqual(value["unique_group_id"], 644)
        self.assertEqual(value["unique_source_video_sha256"], 644)
        self.assertEqual(value["action_family_count"], 28)
        self.assertIs(
            value[
                "historical_receipt_user_authorization_is_not_current_launch_authority"
            ],
            True,
        )
        with self.assertRaisesRegex(
            trainer.TrainingContractError, "expected SHA"
        ):
            trainer.validate_full644_source_authority(
                authority, expected_sha256="0" * 64
            )

        ordinary = self._full644_cli()
        ordinary.exploratory_full644_one_pass = False
        ordinary.objective = "sft"
        ordinary.max_steps = 1
        ordinary.save_every = 1
        ordinary.lora_rank = 8
        ordinary.lora_alpha = 8
        with self.assertRaisesRegex(
            trainer.TrainingContractError, "require the exploratory profile"
        ):
            trainer.validate_cli(ordinary)

    def test_lora_parameters_are_broadcast_before_distributed_training(self) -> None:
        class FakeParameter:
            def __init__(self, value):
                self.data = value
                self.requires_grad = True

        class FakeDistributed:
            def __init__(self):
                self.broadcasts = []
                self.barriers = 0

            @staticmethod
            def is_available():
                return True

            @staticmethod
            def is_initialized():
                return True

            @staticmethod
            def get_world_size():
                return 4

            def broadcast(self, value, *, src):
                self.broadcasts.append((value, src))

            def barrier(self):
                self.barriers += 1

            @staticmethod
            def all_gather_object(output, value):
                output[:] = [value] * 4

        fake_dist = FakeDistributed()
        parameters = [
            ("block.0.lora_A", FakeParameter("rank0-A")),
            ("block.0.lora_B", FakeParameter("rank0-B")),
        ]
        digest = trainer.synchronize_trainable_parameters(
            parameters,
            source_rank=0,
            dist_module=fake_dist,
            digest_function=lambda _parameters: "same-digest",
        )
        self.assertEqual(
            fake_dist.broadcasts, [("rank0-A", 0), ("rank0-B", 0)]
        )
        self.assertEqual(fake_dist.barriers, 1)
        self.assertEqual(digest, "same-digest")

        with self.assertRaisesRegex(
            trainer.TrainingContractError, "outside world size"
        ):
            trainer.synchronize_trainable_parameters(
                parameters, source_rank=4, dist_module=fake_dist
            )

    def test_world4_model_construction_is_rank_serialized_and_metadata_closed(self) -> None:
        class FakeParameter:
            requires_grad = True
            shape = (2, 2)
            dtype = "torch.float32"

            @staticmethod
            def numel():
                return 4

        named = [("block.0.lora_A", FakeParameter())]
        targets = ["diff_dec.transformer.blocks.0.attn1.to_q"]

        class FakeCuda:
            def __init__(self):
                self.synchronizations = []

            def synchronize(self, device):
                self.synchronizations.append(str(device))

        fake_torch = types.SimpleNamespace(cuda=FakeCuda())

        class FakeDistributed:
            def __init__(self, *, mismatch_rank=None):
                self.sources = []
                self.barriers = 0
                self.mismatch_rank = mismatch_rank

            @staticmethod
            def is_available():
                return True

            @staticmethod
            def is_initialized():
                return True

            @staticmethod
            def get_world_size():
                return 4

            @staticmethod
            def get_rank():
                return 0

            def broadcast_object_list(self, payload, *, src):
                self.sources.append(src)
                if src:
                    payload[0] = trainer._serialized_construction_success_status(
                        active_rank=src,
                        device=f"cuda:{src}",
                        target_modules=targets,
                        named_trainable=named,
                        trainable_count=4,
                    )
                    if src == self.mismatch_rank:
                        payload[0]["target_modules_sha256"] = "0" * 64

            def barrier(self):
                self.barriers += 1

        builds = []
        trims = []

        def build():
            builds.append(0)
            return "model", targets, named, 4

        def trim(*, torch_module):
            trims.append(torch_module)

        fake_dist = FakeDistributed()
        result, statuses = trainer.world4_rank_serialized_model_construction(
            contract=trainer.DistributedContract(4, 0, 0, 4),
            device="cuda:0",
            build_function=build,
            torch_module=fake_torch,
            dist_module=fake_dist,
            trim_function=trim,
        )
        self.assertEqual(result[0], "model")
        self.assertEqual(builds, [0])
        self.assertEqual(trims, [fake_torch])
        self.assertEqual(fake_torch.cuda.synchronizations, ["cuda:0"])
        self.assertEqual(fake_dist.sources, [0, 1, 2, 3])
        self.assertEqual(fake_dist.barriers, 4)
        self.assertEqual([row["active_rank"] for row in statuses], [0, 1, 2, 3])

        with self.assertRaisesRegex(
            trainer.TrainingContractError, "metadata differs"
        ):
            trainer.world4_rank_serialized_model_construction(
                contract=trainer.DistributedContract(4, 0, 0, 4),
                device="cuda:0",
                build_function=build,
                torch_module=fake_torch,
                dist_module=FakeDistributed(mismatch_rank=2),
                trim_function=trim,
            )

    def test_world4_serialized_construction_propagates_active_rank_failure(self) -> None:
        class FakeParameter:
            requires_grad = True
            shape = (2, 2)
            dtype = "torch.float32"

            @staticmethod
            def numel():
                return 4

        named = [("block.0.lora_A", FakeParameter())]
        targets = ["diff_dec.transformer.blocks.0.attn1.to_q"]

        class FakeCuda:
            def __init__(self):
                self.synchronizations = []

            def synchronize(self, device):
                self.synchronizations.append(str(device))

        class FakeDistributed:
            def __init__(self, *, failure_rank):
                self.sources = []
                self.barriers = 0
                self.failure_rank = failure_rank

            @staticmethod
            def is_available():
                return True

            @staticmethod
            def is_initialized():
                return True

            @staticmethod
            def get_world_size():
                return 4

            @staticmethod
            def get_rank():
                return 0

            def broadcast_object_list(self, payload, *, src):
                self.sources.append(src)
                if src == self.failure_rank:
                    if src:
                        payload[0] = {
                            "schema_version": trainer._SERIALIZED_CONSTRUCTION_STATUS_SCHEMA,
                            "active_rank": src,
                            "ok": False,
                            "device": f"cuda:{src}",
                            "error_type": "RuntimeError",
                            "error_message_sha256": hashlib.sha256(
                                b"injected constructor failure"
                            ).hexdigest(),
                        }
                elif src:
                    payload[0] = trainer._serialized_construction_success_status(
                        active_rank=src,
                        device=f"cuda:{src}",
                        target_modules=targets,
                        named_trainable=named,
                        trainable_count=4,
                    )

            def barrier(self):
                self.barriers += 1

        for failure_rank in range(4):
            with self.subTest(failure_rank=failure_rank):
                fake_dist = FakeDistributed(failure_rank=failure_rank)

                def build():
                    if failure_rank == 0:
                        raise RuntimeError("injected constructor failure")
                    return "model", targets, named, 4

                with self.assertRaisesRegex(
                    trainer.TrainingContractError,
                    f"failed on rank {failure_rank}",
                ):
                    trainer.world4_rank_serialized_model_construction(
                        contract=trainer.DistributedContract(4, 0, 0, 4),
                        device="cuda:0",
                        build_function=build,
                        torch_module=types.SimpleNamespace(cuda=FakeCuda()),
                        dist_module=fake_dist,
                        trim_function=lambda **_kwargs: None,
                    )
                self.assertEqual(fake_dist.sources, list(range(failure_rank + 1)))
                self.assertEqual(fake_dist.barriers, failure_rank)

    def test_world1_construction_stays_direct_and_malloc_trim_zero_is_valid(self) -> None:
        class ForbiddenDistributed:
            @staticmethod
            def is_available():
                raise AssertionError("WORLD1 must not inspect distributed state")

        marker = object()
        result, statuses = trainer.world4_rank_serialized_model_construction(
            contract=trainer.DistributedContract(1, 0, 0, 1),
            device="cuda:0",
            build_function=lambda: marker,
            torch_module=object(),
            dist_module=ForbiddenDistributed(),
        )
        self.assertIs(result, marker)
        self.assertEqual(statuses, [])

        class FakeTrim:
            argtypes = None
            restype = None

            @staticmethod
            def __call__(_padding):
                return 0

        class FakeCtypes:
            c_size_t = int
            c_int = int

            class CDLL:
                def __new__(_cls, _name):
                    return types.SimpleNamespace(malloc_trim=FakeTrim())

        events = []
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(empty_cache=lambda: events.append("empty"))
        )
        fake_gc = types.SimpleNamespace(collect=lambda: events.append("gc"))
        trainer._trim_host_allocator_after_model_move(
            torch_module=fake_torch,
            gc_module=fake_gc,
            ctypes_module=FakeCtypes,
        )
        self.assertEqual(events, ["gc", "empty"])
        with self.assertRaisesRegex(trainer.TrainingContractError, "unavailable"):
            trainer._trim_host_allocator_after_model_move(
                torch_module=fake_torch,
                gc_module=fake_gc,
                ctypes_module=types.SimpleNamespace(
                    CDLL=lambda _name: object(), c_size_t=int, c_int=int
                ),
            )

    def test_dataset_summary_binds_index_and_shard_and_gates_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shards = root / "shards"
            shards.mkdir()
            iid = "clip001"
            shard = shards / f"{iid}.parquet"
            shard.write_bytes(b"frozen-parquet-shard")
            index_row = {
                "schema_version": trainer.VAE_DATASET_INDEX_ROW_SCHEMA,
                "iid": iid,
                "parquet_path": str(shard),
                "parquet_sha256": trainer.file_sha256(shard),
            }
            index = root / "smoke_index.jsonl"
            index.write_bytes(trainer.canonical_json_bytes(index_row) + b"\n")
            summary = {
                "schema_version": trainer.VAE_DATASET_SUMMARY_SCHEMA,
                "complete": False,
                "preview_only": True,
                "training_authorized": False,
                "training_use_forbidden": True,
                "experimental_training_acknowledged": True,
                "production_claim_forbidden": True,
                "scientific_claim_authorized": False,
                "experimental_inclusion_policy": trainer.EXPECTED_INCLUSION_POLICY,
                "raw_strict_selection_rows": trainer.EXPECTED_STRICT_ROWS,
                "raw_non_strict_selection_rows": trainer.EXPECTED_NON_STRICT_ROWS,
                "materialized_strict_selection_rows": 0,
                "materialized_non_strict_selection_rows": 1,
                "expected_sample_count": trainer.EXPECTED_DATASET_ROWS,
                "materialized_sample_count": 1,
                "missing_sample_count": trainer.EXPECTED_DATASET_ROWS - 1,
                "missing_sample_ids": [f"missing-{index}" for index in range(643)],
                "frame_count": 81,
                "fps": 25.0,
                "latent_frame_count": 21,
                "bucket_counts": {"576x416": 1},
                "shards_directory": str(shards),
                "index_path": str(index),
                "index_sha256": trainer.file_sha256(index),
            }
            summary["summary_digest"] = trainer.object_sha256(summary)
            summary_path = root / "smoke_summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            class FakeDataset:
                def __init__(self):
                    self.root = shards.resolve()
                    self.files = (shard.resolve(),)

                @staticmethod
                def __len__():
                    return 1

            dataset = FakeDataset()
            identity = trainer.validate_preprocessed_dataset_summary(
                summary_path, dataset, allow_incomplete=True
            )
            self.assertFalse(identity["complete"])
            self.assertEqual(identity["materialized_rows"], 1)
            with self.assertRaisesRegex(
                trainer.TrainingContractError, "dataset is incomplete"
            ):
                trainer.validate_preprocessed_dataset_summary(
                    summary_path, dataset, allow_incomplete=False
                )

            shard.write_bytes(b"tampered")
            with self.assertRaisesRegex(
                trainer.TrainingContractError, "indexed shard identity"
            ):
                trainer.validate_preprocessed_dataset_summary(
                    summary_path, dataset, allow_incomplete=True
                )

    def test_bound_parquet_bytes_reject_same_size_replacement_and_map_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            shard = root / "row.parquet"
            shard.write_bytes(b"exact-original-bytes")
            expected_sha = trainer.file_sha256(shard)

            class FakeRowGroup:
                num_rows = 1

            class FakeMetadata:
                num_row_groups = 1

                @staticmethod
                def row_group(_index):
                    return FakeRowGroup()

            class FakeParquetFile:
                schema_arrow = "inputs: string\nvideo_vae_latents: binary"
                metadata = FakeMetadata()

                def __init__(self, raw):
                    self.raw = raw

            pyarrow = types.ModuleType("pyarrow")
            parquet = types.ModuleType("pyarrow.parquet")
            parquet.ParquetFile = FakeParquetFile
            pyarrow.parquet = parquet

            store = object.__new__(trainer.ParquetRowStore)
            store.root = root
            store.files = (shard,)
            store._length = None
            store._groups = []
            store._ends = []
            store._schema = None
            store._layout_signature = None
            store._cached_key = None
            store._cached_rows = None
            store._expected_file_sha256 = None
            store.content_signature = None
            store.signature = None
            with mock.patch.dict(
                sys.modules,
                {"pyarrow": pyarrow, "pyarrow.parquet": parquet},
            ):
                store.bind_indexed_file_hashes({shard: expected_sha})
                self.assertEqual(len(store), 1)
                self.assertRegex(store.content_signature, r"^[0-9a-f]{64}$")
                self.assertEqual(
                    trainer.ParquetRowStore._stable_plain_file_bytes(
                        shard, expected_sha
                    ),
                    b"exact-original-bytes",
                )

                with self.assertRaisesRegex(
                    trainer.TrainingContractError, "membership"
                ):
                    other = object.__new__(trainer.ParquetRowStore)
                    other.root = root
                    other.files = (shard,)
                    other._length = None
                    other._expected_file_sha256 = None
                    other.content_signature = None
                    other.bind_indexed_file_hashes({})

                shard.write_bytes(b"hostile-replaced-xxx")
                self.assertEqual(shard.stat().st_size, len(b"exact-original-bytes"))
                with self.assertRaisesRegex(
                    trainer.TrainingContractError, "identity changed or hash differs"
                ):
                    store.revalidate_bound_files()

    def test_full644_receipt_records_peft_and_denies_historical_launch_authority(
        self,
    ) -> None:
        args = self._full644_cli()

        class FakeDataset:
            root = Path("/sealed/full644")
            signature = "legacy-signature"
            content_signature = "3" * 64

            @staticmethod
            def __len__():
                return trainer.FULL644_EXPLORATORY_STEPS

        targets = [
            f"diff_dec.transformer.route_{index:03d}"
            for index in range(trainer.EXPECTED_LORA_TARGET_MODULES)
        ]
        source_authority = {
            "sha256": trainer.FULL644_SOURCE_AUTHORITY_SHA256,
            "historical_receipt_user_authorization_is_not_current_launch_authority": True,
        }
        receipt = trainer.build_receipt(
            args=args,
            global_step=0,
            last_loss=None,
            gradient_norm=None,
            dataset=FakeDataset(),
            dataset_summary={"sha256": trainer.FULL644_DATASET_SUMMARY_SHA256},
            checkpoint=Path("/sealed/checkpoint"),
            bernini_revision=trainer.BERNINI_OFFICIAL_COMMIT,
            veomni_revision=trainer.VEOMNI_TESTED_COMMIT,
            distributed=trainer.DistributedContract(4, 0, 0, 4),
            backend="nccl/rccl",
            target_modules=targets,
            trainable_parameter_count=(
                trainer.FULL644_EXPLORATORY_TRAINABLE_PARAMETER_COUNT
            ),
            lora_initialization_digest="4" * 64,
            peft_version=trainer.FULL644_PEFT_VERSION,
            transformers_version="5.5.4",
            resumed_from=None,
            full644_source_authority=source_authority,
            terminal_dataset_reverified=False,
        )
        self.assertEqual(
            receipt["training_contract"]["peft_version"],
            trainer.FULL644_PEFT_VERSION,
        )
        self.assertIs(
            receipt["exploratory_full644"][
                "historical_source_receipt_is_not_current_launch_authority"
            ],
            True,
        )
        self.assertIs(
            receipt["exploratory_full644"]["source_authority"][
                "historical_receipt_user_authorization_is_not_current_launch_authority"
            ],
            True,
        )

    def test_checkpoint_publication_rejects_dangling_destination_symlink(self) -> None:
        fake_torch = types.ModuleType("torch")
        fake_dist = types.ModuleType("torch.distributed")
        fake_torch.distributed = fake_dist
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve()
            final = output / "checkpoint-00000644"
            final.symlink_to(output / "missing-checkpoint", target_is_directory=True)
            with mock.patch.dict(
                sys.modules,
                {"torch": fake_torch, "torch.distributed": fake_dist},
            ), self.assertRaisesRegex(
                trainer.TrainingContractError, "refusing to overwrite checkpoint"
            ):
                trainer.save_training_checkpoint(
                    model=None,
                    optimizer=None,
                    output=output,
                    global_step=644,
                    receipt={},
                    dataset_signature="sealed",
                    rank=0,
                )

    @staticmethod
    def _checkpoint_stage(root: Path) -> Path:
        source = root / ".checkpoint-00000644.tmp-test"
        adapter = source / "adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_bytes(b"{}")
        (adapter / "adapter_model.safetensors").write_bytes(b"weights")
        (source / "optimizer.pt").write_bytes(b"optimizer")
        (source / "receipt.json").write_bytes(b"receipt")
        manifest = trainer.build_checkpoint_content_manifest(
            source, global_step=644, receipt_digest="a" * 64
        )
        trainer._atomic_write_json(source / "checkpoint_manifest.json", manifest)
        return source

    def test_checkpoint_hardlink_marker_publication_succeeds_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = self._checkpoint_stage(root)
            destination = root / "checkpoint-00000644"
            trainer._atomic_rename_directory_noreplace(source, destination)
            self.assertFalse(source.exists())
            self.assertTrue(destination.is_dir())
            self.assertEqual(destination.stat().st_mode & 0o777, 0o700)
            self.assertFalse((destination / ".INCOMPLETE").exists())
            manifest = trainer._read_json(destination / "checkpoint_manifest.json")
            rebuilt = trainer.build_checkpoint_content_manifest(
                destination, global_step=644, receipt_digest="a" * 64
            )
            self.assertEqual(manifest, rebuilt)
            for path in destination.rglob("*"):
                if path.is_dir():
                    self.assertEqual(path.stat().st_mode & 0o777, 0o700)
                if path.is_file():
                    self.assertEqual(path.stat().st_nlink, 1)

    def test_checkpoint_marker_is_closed_before_nfs_namespace_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = self._checkpoint_stage(root)
            destination = root / "checkpoint-00000644"
            real_open = os.open
            real_close = os.close
            real_unlink = os.unlink
            marker_descriptors: set[int] = set()
            closed_marker_descriptors: set[int] = set()

            def tracked_open(path, flags, *args, **kwargs):
                descriptor = real_open(path, flags, *args, **kwargs)
                if path == ".INCOMPLETE":
                    marker_descriptors.add(descriptor)
                return descriptor

            def tracked_close(descriptor):
                if descriptor in marker_descriptors:
                    closed_marker_descriptors.add(descriptor)
                return real_close(descriptor)

            def nfs_silly_rename_if_open(path, *args, **kwargs):
                if path == ".INCOMPLETE" and (
                    marker_descriptors - closed_marker_descriptors
                ):
                    directory_fd = kwargs.get("dir_fd")
                    os.rename(
                        ".INCOMPLETE",
                        ".nfs-test",
                        src_dir_fd=directory_fd,
                        dst_dir_fd=directory_fd,
                    )
                    return None
                return real_unlink(path, *args, **kwargs)

            with mock.patch.object(
                trainer.os, "open", side_effect=tracked_open
            ), mock.patch.object(
                trainer.os, "close", side_effect=tracked_close
            ), mock.patch.object(
                trainer.os, "unlink", side_effect=nfs_silly_rename_if_open
            ):
                trainer._atomic_rename_directory_noreplace(source, destination)
            self.assertTrue(marker_descriptors)
            self.assertEqual(marker_descriptors, closed_marker_descriptors)
            self.assertFalse((destination / ".INCOMPLETE").exists())
            self.assertFalse((destination / ".nfs-test").exists())

    def test_checkpoint_source_files_close_before_nfs_name_consumption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = self._checkpoint_stage(root)
            destination = root / "checkpoint-00000644"
            real_open = os.open
            real_close = os.close
            real_unlink = os.unlink
            source_directory_fd = None
            source_payload_fds: set[int] = set()
            closed_source_payload_fds: set[int] = set()

            def tracked_open(path, flags, *args, **kwargs):
                nonlocal source_directory_fd
                descriptor = real_open(path, flags, *args, **kwargs)
                path_text = os.fspath(path)
                directory_fd = kwargs.get("dir_fd")
                if (
                    path_text == source.name
                    and flags & os.O_DIRECTORY
                    and directory_fd is not None
                ):
                    source_directory_fd = descriptor
                elif (
                    source_directory_fd is not None
                    and directory_fd == source_directory_fd
                    and not flags & os.O_DIRECTORY
                ):
                    source_payload_fds.add(descriptor)
                return descriptor

            def tracked_close(descriptor):
                if descriptor in source_payload_fds:
                    closed_source_payload_fds.add(descriptor)
                return real_close(descriptor)

            def nfs_silly_rename_source_if_open(path, *args, **kwargs):
                directory_fd = kwargs.get("dir_fd")
                if directory_fd == source_directory_fd and (
                    source_payload_fds - closed_source_payload_fds
                ):
                    path_text = os.fspath(path)
                    parent, basename = path_text.rsplit("/", 1) if "/" in path_text else ("", path_text)
                    nfs_name = (
                        f"{parent}/.nfs-test-{basename}"
                        if parent
                        else f".nfs-test-{basename}"
                    )
                    os.rename(
                        path_text,
                        nfs_name,
                        src_dir_fd=directory_fd,
                        dst_dir_fd=directory_fd,
                    )
                    return None
                return real_unlink(path, *args, **kwargs)

            with mock.patch.object(
                trainer.os, "open", side_effect=tracked_open
            ), mock.patch.object(
                trainer.os, "close", side_effect=tracked_close
            ), mock.patch.object(
                trainer.os, "unlink", side_effect=nfs_silly_rename_source_if_open
            ):
                trainer._atomic_rename_directory_noreplace(source, destination)
            self.assertTrue(source_payload_fds)
            self.assertEqual(source_payload_fds, closed_source_payload_fds)
            self.assertFalse(source.exists())
            self.assertFalse(
                any(path.name.startswith(".nfs-test-") for path in destination.rglob("*"))
            )
            for path in destination.rglob("*"):
                if path.is_file():
                    self.assertEqual(path.stat().st_nlink, 1)

    def test_checkpoint_hardlink_publication_preserves_existing_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = self._checkpoint_stage(root)
            destination = root / "checkpoint-00000644"
            destination.mkdir()
            sentinel = destination / "sentinel"
            sentinel.write_bytes(b"keep")
            before = (destination.stat().st_ino, sentinel.stat().st_ino, sentinel.read_bytes())
            with self.assertRaisesRegex(
                trainer.TrainingContractError, "refusing to overwrite checkpoint"
            ):
                trainer._atomic_rename_directory_noreplace(source, destination)
            self.assertTrue(source.is_dir())
            self.assertEqual(
                (destination.stat().st_ino, sentinel.stat().st_ino, sentinel.read_bytes()),
                before,
            )

    def test_checkpoint_hardlink_failure_preserves_partial_and_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = self._checkpoint_stage(root)
            destination = root / "checkpoint-00000644"
            real_link = os.link
            link_calls = 0

            def fail_second_link(*args, **kwargs):
                nonlocal link_calls
                link_calls += 1
                if link_calls == 2:
                    raise OSError("injected link failure")
                return real_link(*args, **kwargs)

            with mock.patch.object(
                trainer.os, "link", side_effect=fail_second_link
            ), self.assertRaisesRegex(
                trainer.TrainingContractError, "create-only link failed"
            ):
                trainer._atomic_rename_directory_noreplace(source, destination)
            self.assertTrue(source.is_dir())
            self.assertTrue(destination.is_dir())
            self.assertTrue((destination / ".INCOMPLETE").is_file())
            self.assertTrue((destination / "adapter/adapter_config.json").is_file())
            self.assertFalse((source / "adapter/adapter_config.json").exists())
            self.assertFalse((destination / "checkpoint_manifest.json").exists())

    def test_checkpoint_publication_rejects_nlink_and_extra_stage_hostiles(self) -> None:
        for case in ("nlink", "extra"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                source = self._checkpoint_stage(root)
                destination = root / "checkpoint-00000644"
                if case == "nlink":
                    os.link(source / "optimizer.pt", root / "hostile-hardlink")
                    pattern = "single-link regular file"
                else:
                    (source / "unmanifested.bin").write_bytes(b"extra")
                    pattern = "changed after its manifest"
                with self.assertRaisesRegex(trainer.TrainingContractError, pattern):
                    trainer._atomic_rename_directory_noreplace(source, destination)
                self.assertFalse(destination.exists())
                self.assertTrue(source.is_dir())

    def test_checkpoint_manifest_seals_exact_payload_closure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}", encoding="ascii")
            (adapter / "adapter_model.safetensors").write_bytes(b"weights")
            (root / "optimizer.pt").write_bytes(b"optimizer")
            (root / "receipt.json").write_bytes(b"receipt")
            manifest = trainer.build_checkpoint_content_manifest(
                root, global_step=644, receipt_digest="a" * 64
            )
            self.assertEqual(manifest["file_count"], 4)
            self.assertEqual(
                manifest["manifest_digest"],
                trainer.object_sha256(
                    {key: value for key, value in manifest.items() if key != "manifest_digest"}
                ),
            )
            self.assertEqual(
                {row["path"] for row in manifest["entries"]},
                {
                    "adapter/adapter_config.json",
                    "adapter/adapter_model.safetensors",
                    "optimizer.pt",
                    "receipt.json",
                },
            )


if __name__ == "__main__":
    unittest.main()
