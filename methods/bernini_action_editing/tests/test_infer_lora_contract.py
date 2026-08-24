from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_lora as inference  # noqa: E402
import train_lora as trainer  # noqa: E402


def _adapter_config() -> dict:
    return {
        "peft_type": "LORA",
        "r": 8,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "bias": "none",
        # PEFT 0.19 compacts the fully-qualified training list to these four
        # common suffixes when serializing adapter_config.json.
        "target_modules": sorted(inference.PEFT_COMPACT_TARGET_MODULES),
        "modules_to_save": None,
        "use_dora": False,
        "use_rslora": False,
    }


def _training_receipt() -> dict:
    targets = inference.expected_lora_target_modules()
    receipt = {
        "schema_version": trainer.RECEIPT_SCHEMA,
        "global_step": 1,
        "last_loss": 0.75,
        "last_preclip_gradient_norm": 0.25,
        "bernini_commit": trainer.BERNINI_OFFICIAL_COMMIT,
        "bernini_training_files_index_sha256": inference.object_sha256(
            trainer.BERNINI_PINNED_FILE_HASHES
        ),
        "veomni_commit": trainer.VEOMNI_TESTED_COMMIT,
        "method_source_revision": "1" * 40,
        "method_source_archive_sha256": "2" * 64,
        "checkpoint_tree_sha256": trainer.CHECKPOINT_TREE_SHA256,
        "training_contract": {
            "model": "Bernini-R-1.3B-Diffusers renderer-only",
            "single_expert": "transformer_1",
            "mv2v_flow_shift": 5.0,
            "num_frames": 81,
            "latent_frames": 21,
            "task_source_name": trainer.TASK_SOURCE_NAME,
            "external_spatial_mask": False,
            "external_tracking_or_swept_tube": False,
            "conditioning": ["clean_source_video_vae", "edit_instruction"],
            "target_embedding_or_caption_conditioning": False,
            "lora_rank": 8,
            "lora_alpha": 8,
            "tokenizer_fix_mistral_regex": True,
            "transformers_version": "5.5.4",
        },
        "distributed": {"world_size": 4, "ulysses_size": 4},
        "target_module_count": len(targets),
        "target_modules_sha256": inference.object_sha256(targets),
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
    }
    receipt["receipt_digest"] = inference.object_sha256(receipt)
    return receipt


def _full644_adapter_config() -> dict:
    return {
        "alora_invocation_tokens": None,
        "alpha_pattern": {},
        "arrow_config": None,
        "auto_mapping": {
            "base_model_class": "BerniniRendererModel",
            "parent_library": "bernini.models.renderer",
        },
        "base_model_name_or_path": "",
        "bias": "none",
        "corda_config": None,
        "ensure_weight_tying": False,
        "eva_config": None,
        "exclude_modules": None,
        "fan_in_fan_out": False,
        "inference_mode": True,
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
        "peft_version": "0.19.1",
        "qalora_group_size": 16,
        "r": trainer.FULL644_EXPLORATORY_RANK,
        "rank_pattern": {},
        "revision": None,
        "target_modules": sorted(inference.PEFT_COMPACT_TARGET_MODULES),
        "target_parameters": None,
        "task_type": None,
        "trainable_token_indices": None,
        "use_bdlora": None,
        "use_dora": False,
        "use_qalora": False,
        "use_rslora": False,
    }


def _full644_training_receipt() -> dict:
    receipt = _training_receipt()
    receipt.pop("receipt_digest")
    receipt.update(
        {
            "global_step": trainer.FULL644_EXPLORATORY_STEPS,
            "max_steps": trainer.FULL644_EXPLORATORY_STEPS,
            "seed": trainer.FULL644_EXPLORATORY_SEED,
            "resumed_from": None,
            "experimental_training": True,
        }
    )
    receipt["training_contract"].update(
        {
            "lora_rank": trainer.FULL644_EXPLORATORY_RANK,
            "lora_alpha": trainer.FULL644_EXPLORATORY_ALPHA,
            "objective": "reference_dpo_preservation",
            "contrastive_negative_schedule": "rotate",
            "preference_weight": 1.0,
            "preference_margin": 0.05,
            "preference_temperature": 20.0,
            "dpo_beta": 10.0,
            "preservation_weight": 0.25,
            "preservation_branch": "source_as_target_conditional_identity",
            "peft_version": inference.FULL644_PEFT_VERSION,
        }
    )
    receipt["optimizer"] = {
        "type": "AdamW",
        "learning_rate": 1.0e-4,
        "weight_decay": 0.0,
        "max_gradient_norm": 1.0,
    }
    receipt["distributed"] = {
        "world_size": 4,
        "ulysses_size": 4,
        "backend": "nccl/rccl",
        "same_sample_all_ranks": True,
        "same_seed_all_ranks": True,
        "lora_initialization_seeded_all_ranks": True,
        "lora_parameters_broadcast_from_rank": 0,
        "lora_initialization_digest": "d" * 64,
        "explicit_lora_gradient_all_reduce": True,
    }
    receipt["trainable_parameter_count"] = (
        trainer.FULL644_EXPLORATORY_TRAINABLE_PARAMETER_COUNT
    )
    content_signature = "c" * 64
    receipt["dataset"] = {
        "rows": trainer.FULL644_EXPLORATORY_STEPS,
        "content_signature": content_signature,
        "summary": {
            "sha256": trainer.FULL644_DATASET_SUMMARY_SHA256,
            "summary_digest": trainer.FULL644_DATASET_SUMMARY_DIGEST,
            "index_sha256": trainer.FULL644_DATASET_INDEX_SHA256,
            "complete": True,
            "materialized_rows": trainer.FULL644_EXPLORATORY_STEPS,
        },
    }
    receipt["exploratory_full644"] = {
        "profile": trainer.FULL644_EXPLORATORY_PROFILE,
        "historical_train_debug_rows": trainer.EXPECTED_DATASET_ROWS,
        "optimizer_rows_consumed": trainer.FULL644_EXPLORATORY_STEPS,
        "next_row_index": None,
        "row_sequence_prefix": "0..643",
        "row_sequence_sha256": inference.object_sha256(
            list(range(trainer.FULL644_EXPLORATORY_STEPS))
        ),
        "no_replacement_within_pass": True,
        "complete_one_pass": True,
        "historical_dataset_exists": True,
        "historical_optimizer_contribution_rows": trainer.EXPECTED_DATASET_ROWS,
        "runtime_data_integrity_validated": True,
        "dataset_quality_accepted_under_0817": False,
        "formal_training_dataset_authorized": False,
        "formal_heldout_contribution": 0,
        "target_scientific_qualification_complete": False,
        "matched_frozen_evaluation_required_before_claim": True,
        "resume_policy": "forbidden_for_this_profile",
        "intermediate_checkpoints_archival_only": True,
        "interrupted_run_requires_fresh_step0_restart": True,
        "dataset_summary_sha256": trainer.FULL644_DATASET_SUMMARY_SHA256,
        "dataset_summary_digest": trainer.FULL644_DATASET_SUMMARY_DIGEST,
        "dataset_index_sha256": trainer.FULL644_DATASET_INDEX_SHA256,
        "dataset_content_signature": content_signature,
        "source_authority": {
            "sha256": trainer.FULL644_SOURCE_AUTHORITY_SHA256,
            "membership_rows": 644,
            "unique_group_id": 644,
            "unique_source_video_sha256": 644,
            "action_family_count": 28,
        },
        "indexed_source_and_target_vae_shards_verified_before_training": True,
        "indexed_source_and_target_vae_shards_reverified_after_training": True,
    }
    receipt["receipt_digest"] = inference.object_sha256(receipt)
    return receipt


class InferenceContractTests(unittest.TestCase):
    def test_cli_exposes_only_source_and_instruction_as_user_conditions(self) -> None:
        parser = inference.build_parser()
        destinations = {action.dest for action in parser._actions}
        self.assertIn("source_video", destinations)
        self.assertIn("instruction", destinations)
        forbidden = {
            "target_video",
            "mask",
            "edit_mask",
            "track",
            "tube",
            "pose",
            "trajectory",
            "image",
            "images",
            "reference_video",
            "shared_i0",
        }
        self.assertTrue(destinations.isdisjoint(forbidden))

    def test_cli_requires_exactly_one_adaptation_mode(self) -> None:
        common = dict(
            instruction="Make the actor crouch.",
            num_inference_steps=40,
            seed=42,
            expected_bernini_commit=trainer.BERNINI_OFFICIAL_COMMIT,
            expected_veomni_commit=trainer.VEOMNI_TESTED_COMMIT,
            expected_checkpoint_tree_sha256=trainer.CHECKPOINT_TREE_SHA256,
            method_source_revision="1" * 40,
            method_source_archive_sha256="2" * 64,
        )
        inference.validate_cli(
            argparse.Namespace(**common, base_only=True, adapter_checkpoint=None)
        )
        inference.validate_cli(
            argparse.Namespace(
                **common,
                base_only=False,
                adapter_checkpoint="/abs/checkpoint-00000001",
            )
        )
        for base_only, adapter_checkpoint in ((False, None), (True, "/abs/adapter")):
            with self.subTest(
                base_only=base_only, adapter_checkpoint=adapter_checkpoint
            ), self.assertRaises(inference.InferenceContractError):
                inference.validate_cli(
                    argparse.Namespace(
                        **common,
                        base_only=base_only,
                        adapter_checkpoint=adapter_checkpoint,
                    )
                )

    def test_terminal_full644_rank64_adapter_has_a_separate_exact_contract(self) -> None:
        receipt = _full644_training_receipt()
        identity = inference.validate_adapter_contract(
            _full644_adapter_config(), receipt
        )
        self.assertEqual(identity["global_step"], 644)
        self.assertEqual(identity["lora_rank"], 64)
        self.assertEqual(identity["lora_alpha"], 64)
        self.assertEqual(identity["target_module_count"], 240)
        self.assertIs(identity["exploratory_full644"], True)

        for mutation in (
            "rank",
            "step",
            "row-sequence",
            "terminal-revalidation",
            "optimizer",
            "dpo-beta",
            "distributed",
            "trainable-count",
            "bool-preference-weight",
            "bool-optimizer-grad-clip",
            "bool-formal-heldout",
            "bool-broadcast-rank",
            "peft-rank-pattern",
            "peft-bool-alias",
            "peft-exclude-modules",
            "peft-init-loftq",
            "peft-init-pissa",
            "peft-lora-ga",
            "peft-alora",
            "peft-runtime-offload",
            "peft-lora-bias",
            "peft-bdlora",
            "peft-arrow",
            "peft-weight-tying",
            "peft-task-type",
            "peft-unknown-field",
        ):
            config = copy.deepcopy(_full644_adapter_config())
            hostile = copy.deepcopy(receipt)
            hostile.pop("receipt_digest")
            if mutation == "rank":
                config["r"] = 8
            elif mutation == "step":
                hostile["global_step"] = 640
            elif mutation == "row-sequence":
                hostile["exploratory_full644"]["row_sequence_prefix"] = "0..642"
            elif mutation == "terminal-revalidation":
                hostile["exploratory_full644"][
                    "indexed_source_and_target_vae_shards_reverified_after_training"
                ] = False
            elif mutation == "optimizer":
                hostile["optimizer"]["learning_rate"] = 9.0
            elif mutation == "dpo-beta":
                hostile["training_contract"]["dpo_beta"] = 999.0
            elif mutation == "distributed":
                hostile["distributed"]["backend"] = "gloo"
            elif mutation == "trainable-count":
                hostile["trainable_parameter_count"] = 1
            elif mutation == "bool-preference-weight":
                hostile["training_contract"]["preference_weight"] = True
            elif mutation == "bool-optimizer-grad-clip":
                hostile["optimizer"]["max_gradient_norm"] = True
            elif mutation == "bool-formal-heldout":
                hostile["exploratory_full644"]["formal_heldout_contribution"] = False
            elif mutation == "bool-broadcast-rank":
                hostile["distributed"]["lora_parameters_broadcast_from_rank"] = False
            elif mutation == "peft-rank-pattern":
                config["rank_pattern"] = {"diff_dec.transformer.blocks.0": 8}
            elif mutation == "peft-bool-alias":
                config["use_qalora"] = 0
            elif mutation == "peft-exclude-modules":
                config["exclude_modules"] = ["diff_dec.transformer.blocks.0"]
            elif mutation == "peft-init-loftq":
                config["init_lora_weights"] = "loftq"
            elif mutation == "peft-init-pissa":
                config["init_lora_weights"] = "pissa"
            elif mutation == "peft-lora-ga":
                config["lora_ga_config"] = {"direction": "stable"}
            elif mutation == "peft-alora":
                config["alora_invocation_tokens"] = [1]
            elif mutation == "peft-runtime-offload":
                config["runtime_config"] = {"ephemeral_gpu_offload": True}
            elif mutation == "peft-lora-bias":
                config["lora_bias"] = True
            elif mutation == "peft-bdlora":
                config["use_bdlora"] = {"enabled": True}
            elif mutation == "peft-arrow":
                config["arrow_config"] = {"top_k": 1}
            elif mutation == "peft-weight-tying":
                config["ensure_weight_tying"] = True
            elif mutation == "peft-task-type":
                config["task_type"] = "CAUSAL_LM"
            else:
                config["unrecognized_behavior_switch"] = True
            hostile["receipt_digest"] = inference.object_sha256(hostile)
            with self.subTest(mutation=mutation), self.assertRaises(
                inference.InferenceContractError
            ):
                inference.validate_adapter_contract(config, hostile)

    def test_full644_inference_receipt_publishes_exact_adapter_profile(self) -> None:
        root = Path("/tmp/full644-adapter-receipt-test")
        adapter = inference.AdapterBundle(
            checkpoint_root=root / "checkpoint-00000644",
            adapter_dir=root / "checkpoint-00000644/adapter",
            adapter_config_path=(
                root / "checkpoint-00000644/adapter/adapter_config.json"
            ),
            adapter_model_path=(
                root / "checkpoint-00000644/adapter/adapter_model.safetensors"
            ),
            training_receipt_path=root / "checkpoint-00000644/receipt.json",
        )
        identity = inference.validate_adapter_contract(
            _full644_adapter_config(), _full644_training_receipt()
        )
        self.assertEqual(
            identity["peft_version"], inference.FULL644_PEFT_VERSION
        )
        args = argparse.Namespace(
            instruction="Make the actor crouch.",
            method_source_revision="1" * 40,
            method_source_archive_sha256="2" * 64,
            expected_checkpoint_tree_sha256=trainer.CHECKPOINT_TREE_SHA256,
            num_inference_steps=40,
            seed=42,
        )
        common = dict(
            args=args,
            source_path=root / "source.mp4",
            source_sha256="3" * 64,
            source_metadata={
                "frame_count": 81,
                "fps": 25.0,
                "source_derived_bucket_hw": [416, 576],
                "external_shared_i0": False,
            },
            output_path=root / "output.mp4",
            output_sha256="4" * 64,
            adapter=adapter,
            adapter_sha256="5" * 64,
            adapter_identity=identity,
            bernini_revision=trainer.BERNINI_OFFICIAL_COMMIT,
            veomni_revision=trainer.VEOMNI_TESTED_COMMIT,
            inference_file_hashes=inference.BERNINI_INFERENCE_FILE_HASHES,
            runtime_versions={
                "transformers": "5.5.4",
                "peft": inference.FULL644_PEFT_VERSION,
            },
            adapter_manifest_identity={
                "path": str(root / "checkpoint-00000644/checkpoint_manifest.json"),
                "sha256": "6" * 64,
                "manifest_digest": "7" * 64,
                "global_step": 644,
                "receipt_digest": identity["receipt_digest"],
                "file_count": 4,
                "adapter_config_sha256": "8" * 64,
                "adapter_model_sha256": "5" * 64,
                "training_receipt_sha256": "9" * 64,
                "optimizer_sha256": "a" * 64,
            },
        )
        receipt = inference.build_inference_receipt(
            **common, adapter_tensor_count=480
        )
        self.assertEqual(
            receipt["infer_lora_source_sha256"],
            inference.file_sha256(Path(inference.__file__).resolve()),
        )
        self.assertEqual(receipt["adapter"]["profile"], trainer.FULL644_EXPLORATORY_PROFILE)
        self.assertEqual(receipt["adapter"]["lora_rank"], 64)
        self.assertEqual(receipt["adapter"]["lora_alpha"], 64)
        self.assertEqual(receipt["adapter"]["target_module_count"], 240)
        self.assertEqual(
            receipt["runtime_versions"]["peft"],
            inference.FULL644_PEFT_VERSION,
        )
        with self.assertRaisesRegex(
            inference.InferenceContractError, "tensor count differs"
        ):
            inference.build_inference_receipt(
                **common, adapter_tensor_count=479
            )

    def test_full644_checkpoint_manifest_binds_adapter_bytes_and_complete_closure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}", encoding="ascii")
            model = adapter / "adapter_model.safetensors"
            model.write_bytes(b"exact-full644-adapter")
            (root / "optimizer.pt").write_bytes(b"exact-optimizer")
            receipt = {"receipt_digest": "a" * 64}
            (root / "receipt.json").write_bytes(
                inference.canonical_json_bytes(receipt) + b"\n"
            )
            manifest = trainer.build_checkpoint_content_manifest(
                root, global_step=644, receipt_digest="a" * 64
            )
            manifest_path = root / "checkpoint_manifest.json"
            manifest_path.write_bytes(
                inference.canonical_json_bytes(manifest) + b"\n"
            )
            bundle = inference.resolve_adapter_bundle(root)
            expected_sha = inference.file_sha256(manifest_path)
            identity = inference.validate_training_checkpoint_manifest(
                bundle, expected_sha256=expected_sha
            )
            self.assertEqual(
                identity["adapter_model_sha256"], inference.file_sha256(model)
            )
            self.assertEqual(identity["global_step"], 644)

            model.write_bytes(b"other-full644-adapter")
            with self.assertRaisesRegex(
                inference.InferenceContractError, "member bytes differ"
            ):
                inference.validate_training_checkpoint_manifest(
                    bundle, expected_sha256=expected_sha
                )

    def test_prompt_and_tokenizer_are_training_exact(self) -> None:
        seen = []

        def cleaner(value):
            seen.append(value)
            return "Change the actor's action."

        prompt = inference.build_training_prompt("  raw edit  ", prompt_cleaner=cleaner)
        self.assertEqual(seen, ["  raw edit  "])
        self.assertEqual(
            prompt,
            inference.MV2V_SYSTEM_PROMPT + "Change the actor's action.",
        )
        self.assertNotIn("\n", prompt)
        tokenizer = inference.tokenizer_load_kwargs()
        self.assertIs(tokenizer["fix_mistral_regex"], True)
        self.assertEqual(tokenizer["padding_side"], "right")
        self.assertIs(tokenizer["local_files_only"], True)
        positive = inference.training_prompt_tokenizer_kwargs()
        self.assertNotIn("truncation", positive)
        self.assertNotIn("max_length", positive)
        self.assertNotIn("padding", positive)
        negative = inference.renderer_negative_tokenizer_kwargs()
        self.assertIs(negative["truncation"], True)
        self.assertEqual(negative["max_length"], 512)
        self.assertEqual(negative["padding"], "max_length")
        with self.assertRaisesRegex(inference.InferenceContractError, "instruction"):
            inference.build_training_prompt("\x00", prompt_cleaner=lambda value: value)
        # The official renderer cleans only the positive prompt.  Its default
        # Chinese negative prompt must retain full-width punctuation before
        # tokenization.
        self.assertIn("，", inference.DEFAULT_NEGATIVE_PROMPT)

    def test_sampler_and_renderer_are_frozen_to_mv2v_81f_shift5(self) -> None:
        checkpoint = Path("/abs/Bernini-R-1.3B-Diffusers")
        overrides = inference.inference_renderer_config_overrides(checkpoint)
        self.assertEqual(overrides["wan22_base"], str(checkpoint))
        self.assertEqual(overrides["shift"], 5.0)
        self.assertIs(overrides["use_unipc"], True)
        self.assertIs(overrides["skip_transformer_2"], True)
        sampling = inference.sampler_contract(steps=40, seed=7)
        self.assertEqual(sampling["num_frames"], 81)
        self.assertEqual(sampling["guidance_mode"], "v2v_apg")
        self.assertEqual(sampling["flow_shift"], 5.0)
        self.assertEqual(sampling["num_inference_steps"], 40)
        self.assertEqual(sampling["seed"], 7)
        self.assertEqual(sampling["momentum"], 0.0)

    def test_source_onset_policy_is_an_explicit_latent_boundary_ablation(self) -> None:
        torch = self._import_torch_or_skip()
        generated = torch.full((1, 2, 21, 2, 2), 8.0)
        source = torch.full_like(generated, 2.0)
        self.assertIs(
            inference.apply_source_onset_policy(generated, source, "none"),
            generated,
        )
        hard = inference.apply_source_onset_policy(generated, source, "hard1")
        self.assertTrue(torch.equal(hard[:, :, 0], source[:, :, 0]))
        self.assertTrue(torch.equal(hard[:, :, 1:], generated[:, :, 1:]))
        ramp = inference.apply_source_onset_policy(generated, source, "ramp3")
        self.assertTrue(torch.equal(ramp[:, :, 0], source[:, :, 0]))
        self.assertTrue(torch.equal(ramp[:, :, 1], torch.full_like(ramp[:, :, 1], 5.0)))
        self.assertTrue(torch.equal(ramp[:, :, 2], torch.full_like(ramp[:, :, 2], 6.5)))
        self.assertTrue(torch.equal(ramp[:, :, 3:], generated[:, :, 3:]))
        self.assertTrue(torch.equal(generated, torch.full_like(generated, 8.0)))
        with self.assertRaisesRegex(inference.InferenceContractError, "unknown"):
            inference.apply_source_onset_policy(generated, source, "silent-fallback")

    @staticmethod
    def _import_torch_or_skip():
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("torch is unavailable")
        return torch

    def test_exact_video_and_four_rank_contract_fail_closed(self) -> None:
        inference.validate_exact_video_metadata(81, 25.0)
        inference.validate_exact_video_metadata(81, 25.0005)
        for frames, fps in ((41, 25.0), (81, 16.0), (81, math.nan)):
            with self.subTest(frames=frames, fps=fps), self.assertRaises(
                inference.InferenceContractError
            ):
                inference.validate_exact_video_metadata(frames, fps)
        contract = inference.inference_distributed_contract(
            {"WORLD_SIZE": "4", "RANK": "2", "LOCAL_RANK": "2"}
        )
        self.assertEqual(contract.ulysses_size, 4)
        with self.assertRaisesRegex(inference.InferenceContractError, "exactly 4"):
            inference.inference_distributed_contract(
                {"WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"}
            )

    def test_adapter_metadata_binds_all_240_attention_targets_and_receipt(self) -> None:
        targets = inference.expected_lora_target_modules()
        self.assertEqual(len(targets), 240)
        self.assertTrue(
            all(name.startswith("diff_dec.transformer.blocks.") for name in targets)
        )
        self.assertFalse(any("t5_text_encoder" in name for name in targets))
        identity = inference.validate_adapter_contract(
            _adapter_config(), _training_receipt()
        )
        self.assertEqual(identity["global_step"], 1)
        self.assertEqual(identity["transformers_version"], "5.5.4")

        bad_config = _adapter_config()
        bad_config["target_modules"] = bad_config["target_modules"][:-1]
        with self.assertRaisesRegex(inference.InferenceContractError, "Wan projection scope"):
            inference.validate_adapter_contract(bad_config, _training_receipt())

        bad_receipt = _training_receipt()
        bad_receipt["last_preclip_gradient_norm"] = 0.0
        bad_receipt["receipt_digest"] = inference.object_sha256(
            {key: value for key, value in bad_receipt.items() if key != "receipt_digest"}
        )
        with self.assertRaisesRegex(inference.InferenceContractError, "positive"):
            inference.validate_adapter_contract(_adapter_config(), bad_receipt)

    def test_training_receipt_tampering_and_tokenizer_version_are_rejected(self) -> None:
        tampered = _training_receipt()
        tampered["training_contract"]["tokenizer_fix_mistral_regex"] = False
        with self.assertRaisesRegex(inference.InferenceContractError, "digest mismatch"):
            inference.validate_adapter_contract(_adapter_config(), tampered)

        rebound = copy.deepcopy(tampered)
        rebound["receipt_digest"] = inference.object_sha256(
            {key: value for key, value in rebound.items() if key != "receipt_digest"}
        )
        with self.assertRaisesRegex(
            inference.InferenceContractError, "tokenizer_fix_mistral_regex"
        ):
            inference.validate_adapter_contract(_adapter_config(), rebound)

    def test_strict_adapter_state_requires_exact_keys_bytes_and_count(self) -> None:
        saved = {
            key: f"tensor-{index}"
            for index, key in enumerate(inference.expected_adapter_state_keys())
        }
        loaded = dict(saved)
        count = inference.validate_adapter_state_dicts(
            saved, loaded, tensor_equal=lambda left, right: left == right
        )
        self.assertEqual(count, 480)
        changed = inference.expected_adapter_state_keys()[7]
        loaded[changed] = "tampered"
        with self.assertRaisesRegex(inference.InferenceContractError, "tensor mismatch"):
            inference.validate_adapter_state_dicts(
                saved, loaded, tensor_equal=lambda left, right: left == right
            )
        missing = dict(saved)
        missing.pop(next(iter(missing)))
        with self.assertRaisesRegex(inference.InferenceContractError, "key mismatch"):
            inference.validate_adapter_state_dicts(
                saved, missing, tensor_equal=lambda left, right: left == right
            )

    def test_adapter_bundle_accepts_only_training_checkpoint_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "checkpoint-00000001"
            adapter = root / "adapter"
            adapter.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter / "adapter_model.safetensors").write_bytes(b"weights")
            (root / "receipt.json").write_text("{}", encoding="utf-8")
            bundle = inference.resolve_adapter_bundle(root)
            self.assertEqual(bundle.checkpoint_root, root)
            self.assertEqual(bundle.adapter_dir, adapter)
            self.assertEqual(inference.resolve_adapter_bundle(adapter), bundle)

    def test_inference_receipt_records_no_privileged_condition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            adapter = inference.AdapterBundle(
                checkpoint_root=root / "checkpoint-00000001",
                adapter_dir=root / "checkpoint-00000001/adapter",
                adapter_config_path=root / "checkpoint-00000001/adapter/adapter_config.json",
                adapter_model_path=root / "checkpoint-00000001/adapter/adapter_model.safetensors",
                training_receipt_path=root / "checkpoint-00000001/receipt.json",
            )
            args = argparse.Namespace(
                instruction="Make the actor crouch.",
                method_source_revision="1" * 40,
                method_source_archive_sha256="2" * 64,
                expected_checkpoint_tree_sha256=trainer.CHECKPOINT_TREE_SHA256,
                num_inference_steps=40,
                seed=42,
            )
            metadata = {
                "frame_count": 81,
                "fps": 25.0,
                "source_derived_bucket_hw": [416, 576],
                "external_shared_i0": False,
            }
            receipt = inference.build_inference_receipt(
                args=args,
                source_path=root / "source.mp4",
                source_sha256="3" * 64,
                source_metadata=metadata,
                output_path=root / "output.mp4",
                output_sha256="4" * 64,
                adapter=adapter,
                adapter_sha256="5" * 64,
                adapter_identity={
                    "receipt_digest": "6" * 64,
                    "global_step": 1,
                    "target_modules_sha256": "7" * 64,
                },
                adapter_tensor_count=480,
                bernini_revision=trainer.BERNINI_OFFICIAL_COMMIT,
                veomni_revision=trainer.VEOMNI_TESTED_COMMIT,
                inference_file_hashes=inference.BERNINI_INFERENCE_FILE_HASHES,
                runtime_versions={"transformers": "5.5.4"},
            )
            model_input = receipt["input"]
            self.assertEqual(
                model_input["accepted_model_conditions"],
                ["source_video", "edit_instruction"],
            )
            self.assertIs(model_input["target_video_argument"], False)
            self.assertIs(model_input["target_accessed_by_inference"], False)
            self.assertIs(model_input["external_mask_or_swept_tube"], False)
            self.assertIs(model_input["external_shared_i0"], False)
            self.assertEqual(receipt["output"]["frame_count"], 81)
            self.assertEqual(receipt["output"]["fps"], 25.0)
            digest = receipt.pop("receipt_digest")
            self.assertEqual(digest, inference.object_sha256(receipt))

    def test_base_only_receipt_has_no_fake_adapter_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            args = argparse.Namespace(
                instruction="Make the actor crouch.",
                method_source_revision="1" * 40,
                method_source_archive_sha256="2" * 64,
                expected_checkpoint_tree_sha256=trainer.CHECKPOINT_TREE_SHA256,
                num_inference_steps=40,
                seed=42,
            )
            receipt = inference.build_inference_receipt(
                args=args,
                source_path=root / "source.mp4",
                source_sha256="3" * 64,
                source_metadata={
                    "frame_count": 81,
                    "fps": 25.0,
                    "source_derived_bucket_hw": [416, 576],
                    "external_shared_i0": False,
                },
                output_path=root / "output.mp4",
                output_sha256="4" * 64,
                adapter=None,
                adapter_sha256=None,
                adapter_identity=None,
                adapter_tensor_count=0,
                bernini_revision=trainer.BERNINI_OFFICIAL_COMMIT,
                veomni_revision=trainer.VEOMNI_TESTED_COMMIT,
                inference_file_hashes=inference.BERNINI_INFERENCE_FILE_HASHES,
                runtime_versions={"transformers": "5.5.4"},
            )
            adaptation = receipt["adapter"]
            self.assertIs(adaptation["enabled"], False)
            self.assertEqual(adaptation["mode"], "frozen_base_no_adapter")
            self.assertEqual(adaptation["tensor_count"], 0)
            self.assertNotIn("checkpoint_root", adaptation)
            self.assertNotIn("adapter_model_path", adaptation)
            self.assertNotIn("adapter_model_sha256", adaptation)
            self.assertIs(receipt["scientific_claim_authorized"], False)

    def test_model_consumption_receipt_binds_retained_source_evidence(self) -> None:
        root = Path("/tmp/full644-retained-source-receipt-test")
        source_path = root / "source.mp4"
        source_sha256 = "3" * 64
        source_authority = {
            "path": str(source_path),
            "sha256": source_sha256,
            "size": 123,
            "mode": 0o100444,
            "device": 1,
            "inode": 2,
            "uid": 3,
            "gid": 4,
            "nlink": 1,
            "rdev": 0,
            "blocks": 8,
            "mtime_ns": 9,
            "ctime_ns": 10,
        }
        evidence = {
            "consumption_input_digest": "a" * 64,
            "task_input_digest": "b" * 64,
            "model_capture_digest": "c" * 64,
            "model_view_root": "/proc/self/fd/20",
            "adapter_capture_digest": None,
            "adapter_view_root": None,
            "fd_view_files_authorized": 52,
            "inherited_fd_binding_digest": "d" * 64,
            "inherited_fd_count": 55,
            "ptrace_authorization_used": False,
            "source_video_sha256": source_sha256,
            "source_video_physical_authority_digest": inference.object_sha256(
                source_authority
            ),
            "all_ranks_use_retained_source_fd": True,
            "four_rank_attestation": {
                "world_size": 4,
                "all_ranks_replayed_exact_fd_views": True,
                "rank_evidence_digest": "e" * 64,
                "ordered_rank_evidence_digests": ["e" * 64] * 4,
            },
        }
        args = argparse.Namespace(
            instruction="Make the actor crouch.",
            method_source_revision="1" * 40,
            method_source_archive_sha256="2" * 64,
            expected_checkpoint_tree_sha256=trainer.CHECKPOINT_TREE_SHA256,
            num_inference_steps=40,
            seed=42,
        )
        common = dict(
            args=args,
            source_path=source_path,
            source_sha256=source_sha256,
            source_metadata={
                "frame_count": 81,
                "fps": 25.0,
                "source_derived_bucket_hw": [416, 576],
                "external_shared_i0": False,
            },
            output_path=root / "output.mp4",
            output_sha256="4" * 64,
            adapter=None,
            adapter_sha256=None,
            adapter_identity=None,
            adapter_tensor_count=0,
            bernini_revision=trainer.BERNINI_OFFICIAL_COMMIT,
            veomni_revision=trainer.VEOMNI_TESTED_COMMIT,
            inference_file_hashes=inference.BERNINI_INFERENCE_FILE_HASHES,
            runtime_versions={"transformers": "5.5.4"},
            source_file_authority=source_authority,
        )
        receipt = inference.build_inference_receipt(
            **common, model_consumption_evidence=evidence
        )
        self.assertEqual(
            receipt["model_consumption"]["source_video_sha256"],
            source_sha256,
        )
        hostile = copy.deepcopy(evidence)
        hostile["source_video_sha256"] = "f" * 64
        with self.assertRaisesRegex(
            inference.InferenceContractError, "retained source evidence differs"
        ):
            inference.build_inference_receipt(
                **common, model_consumption_evidence=hostile
            )


if __name__ == "__main__":
    unittest.main()
