from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_lora as inference  # noqa: E402
import self_generated_action_preservation_v2 as preservation  # noqa: E402
import train_lora as legacy  # noqa: E402
import train_self_generated_action_quotient_v1 as v2_trainer  # noqa: E402


LOSS_COMPONENTS = {
    "action",
    "onset",
    "nuisance",
    "noop",
    "functional_code",
    "functional_temporal_dc",
    "functional_total",
}


def _resign(receipt: dict) -> dict:
    receipt["receipt_digest"] = inference.object_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_digest"}
    )
    return receipt


def _v2_config(scope: str) -> dict:
    return {
        "peft_type": "LORA",
        "r": 8,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": sorted(
            inference.V2_PEFT_COMPACT_TARGET_MODULES[scope]
        ),
        "modules_to_save": None,
        "use_dora": False,
        "use_rslora": False,
    }


def _v2_receipt(arm: str, *, step: int = 5) -> dict:
    spec = preservation.arm_spec(arm)
    targets = inference.expected_v2_lora_target_modules(spec.route_scope)
    args = argparse.Namespace(
        objective_family="preservation_v2",
        arm=arm,
        max_steps=20,
        method_source_revision="1" * 40,
        method_source_archive_sha256="2" * 64,
        seed=v2_trainer.V2_CANARY_SEED,
        source_manifest_sha256="3" * 64,
    )
    baseline = step == 0
    components = {
        key: 0.0 if baseline else 0.125 + index * 0.01
        for index, key in enumerate(sorted(LOSS_COMPONENTS))
    }
    return v2_trainer.checkpoint_receipt(
        args=args,
        manifest={"manifest_digest": "4" * 64},
        step=step,
        loss=0.0 if baseline else 0.75,
        grad_norm=0.0 if baseline else 0.25,
        target_modules=targets,
        trainable_count=123456,
        bernini_revision=legacy.BERNINI_OFFICIAL_COMMIT,
        veomni_revision=legacy.VEOMNI_TESTED_COMMIT,
        transformers_version="5.5.4",
        initial_digest="5" * 64,
        teacher_cache_seed=v2_trainer.V2_CANARY_SEED,
        teacher_cache_sha256="6" * 64,
        loss_components=components,
    )


def _legacy_config() -> dict:
    return {
        "peft_type": "LORA",
        "r": 8,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": sorted(inference.PEFT_COMPACT_TARGET_MODULES),
        "modules_to_save": None,
        "use_dora": False,
        "use_rslora": False,
    }


def _legacy_receipt() -> dict:
    targets = inference.expected_lora_target_modules()
    receipt = {
        "schema_version": legacy.RECEIPT_SCHEMA,
        "global_step": 1,
        "last_loss": 0.75,
        "last_preclip_gradient_norm": 0.25,
        "bernini_commit": legacy.BERNINI_OFFICIAL_COMMIT,
        "bernini_training_files_index_sha256": inference.object_sha256(
            legacy.BERNINI_PINNED_FILE_HASHES
        ),
        "veomni_commit": legacy.VEOMNI_TESTED_COMMIT,
        "method_source_revision": "1" * 40,
        "method_source_archive_sha256": "2" * 64,
        "checkpoint_tree_sha256": legacy.CHECKPOINT_TREE_SHA256,
        "training_contract": {
            "model": "Bernini-R-1.3B-Diffusers renderer-only",
            "single_expert": "transformer_1",
            "mv2v_flow_shift": 5.0,
            "num_frames": 81,
            "latent_frames": 21,
            "task_source_name": legacy.TASK_SOURCE_NAME,
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
    return _resign(receipt)


class PreservationV2AdapterContractTests(unittest.TestCase):
    def test_inference_mirrors_trainer_v2_constants_without_legacy_import_dependency(self) -> None:
        self.assertEqual(
            inference.V2_TRAINING_RECEIPT_SCHEMA,
            v2_trainer.RECEIPT_SCHEMA_V2,
        )
        self.assertEqual(inference.V2_OBJECTIVE, v2_trainer.METHOD_V2)
        self.assertEqual(inference.V2_CANARY_SEED, v2_trainer.V2_CANARY_SEED)
        self.assertEqual(inference.V2_SAVE_STEPS, v2_trainer.V2_SAVE_STEPS)
        self.assertEqual(inference.V2_SIGMA_BINS, v2_trainer.V2_SIGMA_BINS)

    def test_all_attention_and_cross_qo_derive_exact_signed_routes(self) -> None:
        cases = (
            ("v2_func025_all", "all_attention", 240, 480),
            ("v2_func025_cross_qo", "cross_attn2_qo", 60, 120),
        )
        for arm, scope, module_count, tensor_count in cases:
            with self.subTest(scope=scope):
                identity = inference.validate_adapter_contract(
                    _v2_config(scope), _v2_receipt(arm)
                )
                targets = identity["target_modules"]
                self.assertEqual(identity["lora_route_scope"], scope)
                self.assertEqual(len(targets), module_count)
                self.assertEqual(
                    len(inference.expected_adapter_state_keys(targets)),
                    tensor_count,
                )
                self.assertEqual(
                    identity["target_modules_sha256"],
                    inference.object_sha256(targets),
                )
                if scope == "cross_attn2_qo":
                    self.assertTrue(all(".attn2." in name for name in targets))
                    self.assertTrue(
                        all(
                            name.endswith(".to_q")
                            or name.endswith(".to_out.0")
                            for name in targets
                        )
                    )
                    self.assertFalse(any(".attn1." in name for name in targets))

    def test_hostile_scope_list_count_and_digest_are_each_rejected(self) -> None:
        original = _v2_receipt("v2_func025_cross_qo")

        unsigned_scope = copy.deepcopy(original)
        unsigned_scope["training_contract"]["lora_route_scope"] = "all_attention"
        with self.assertRaisesRegex(
            inference.InferenceContractError, "digest mismatch"
        ):
            inference.validate_adapter_contract(
                _v2_config("cross_attn2_qo"), unsigned_scope
            )

        wrong_scope = _resign(copy.deepcopy(unsigned_scope))
        with self.assertRaisesRegex(
            inference.InferenceContractError, "arm/route scope"
        ):
            inference.validate_adapter_contract(
                _v2_config("cross_attn2_qo"), wrong_scope
            )

        reversed_list = copy.deepcopy(original)
        reversed_list["target_modules"] = list(
            reversed(reversed_list["target_modules"])
        )
        _resign(reversed_list)
        with self.assertRaisesRegex(
            inference.InferenceContractError, "explicit target_modules"
        ):
            inference.validate_adapter_contract(
                _v2_config("cross_attn2_qo"), reversed_list
            )

        wrong_count = copy.deepcopy(original)
        wrong_count["target_module_count"] += 1
        _resign(wrong_count)
        with self.assertRaisesRegex(
            inference.InferenceContractError, "target_module_count"
        ):
            inference.validate_adapter_contract(
                _v2_config("cross_attn2_qo"), wrong_count
            )

        wrong_digest = copy.deepcopy(original)
        wrong_digest["target_modules_sha256"] = "f" * 64
        _resign(wrong_digest)
        with self.assertRaisesRegex(
            inference.InferenceContractError, "target module digest"
        ):
            inference.validate_adapter_contract(
                _v2_config("cross_attn2_qo"), wrong_digest
            )

    def test_scope_specific_peft_config_is_canonical_and_fail_closed(self) -> None:
        receipt = _v2_receipt("v2_func025_cross_qo")
        for target_modules in (
            sorted(inference.PEFT_COMPACT_TARGET_MODULES),
            ["to_out.0", "to_q"],
            receipt["target_modules"],
            ["to_q", "to_q"],
        ):
            with self.subTest(target_modules_len=len(target_modules)):
                config = _v2_config("cross_attn2_qo")
                config["target_modules"] = target_modules
                with self.assertRaisesRegex(
                    inference.InferenceContractError, "canonical PEFT"
                ):
                    inference.validate_adapter_contract(config, receipt)

    def test_inference_receipt_carries_dynamic_route_count_and_baseline_status(self) -> None:
        identity = inference.validate_adapter_contract(
            _v2_config("cross_attn2_qo"),
            _v2_receipt("v2_func025_cross_qo", step=0),
        )
        root = Path("/tmp/preservation-v2-contract-test")
        bundle = inference.AdapterBundle(
            checkpoint_root=root / "checkpoint-00000000",
            adapter_dir=root / "checkpoint-00000000/adapter",
            adapter_config_path=root / "checkpoint-00000000/adapter/adapter_config.json",
            adapter_model_path=(
                root / "checkpoint-00000000/adapter/adapter_model.safetensors"
            ),
            training_receipt_path=root / "checkpoint-00000000/receipt.json",
        )
        args = argparse.Namespace(
            instruction="Make the actor crouch.",
            method_source_revision="1" * 40,
            method_source_archive_sha256="2" * 64,
            expected_checkpoint_tree_sha256=legacy.CHECKPOINT_TREE_SHA256,
            num_inference_steps=40,
            seed=42,
        )
        result = inference.build_inference_receipt(
            args=args,
            source_path=root / "source.mp4",
            source_sha256="3" * 64,
            source_metadata={"source_derived_bucket_hw": [416, 576]},
            output_path=root / "output.mp4",
            output_sha256="4" * 64,
            adapter=bundle,
            adapter_sha256="5" * 64,
            adapter_identity=identity,
            adapter_tensor_count=120,
            bernini_revision=legacy.BERNINI_OFFICIAL_COMMIT,
            veomni_revision=legacy.VEOMNI_TESTED_COMMIT,
            inference_file_hashes=inference.BERNINI_INFERENCE_FILE_HASHES,
            runtime_versions={"transformers": "5.5.4"},
        )
        adapter = result["adapter"]
        self.assertEqual(adapter["lora_route_scope"], "cross_attn2_qo")
        self.assertEqual(adapter["target_module_count"], 60)
        self.assertEqual(adapter["tensor_count"], 120)
        self.assertTrue(adapter["frozen_baseline"])

    def test_narrow_state_keys_reject_legacy_or_attn1_leakage(self) -> None:
        targets = inference.expected_v2_lora_target_modules("cross_attn2_qo")
        keys = inference.expected_adapter_state_keys(targets)
        saved = {key: f"value-{index}" for index, key in enumerate(keys)}
        self.assertEqual(
            inference.validate_adapter_state_dicts(
                saved,
                dict(saved),
                tensor_equal=lambda left, right: left == right,
                expected_targets=targets,
            ),
            120,
        )

        hostile_loaded = dict(saved)
        hostile_loaded[
            "base_model.model.diff_dec.transformer.blocks.0.attn1.to_q.lora_A.weight"
        ] = "leak"
        with self.assertRaisesRegex(
            inference.InferenceContractError, "key mismatch"
        ):
            inference.validate_adapter_state_dicts(
                saved,
                hostile_loaded,
                tensor_equal=lambda left, right: left == right,
                expected_targets=targets,
            )

        legacy_loaded = {
            key: key for key in inference.expected_adapter_state_keys()
        }
        with self.assertRaisesRegex(
            inference.InferenceContractError, "key mismatch"
        ):
            inference.validate_adapter_state_dicts(
                saved,
                legacy_loaded,
                tensor_equal=lambda left, right: left == right,
                expected_targets=targets,
            )

    def test_checkpoint_zero_requires_receipt_and_tensor_zero_effect_baseline(self) -> None:
        receipt = _v2_receipt("v2_func025_cross_qo", step=0)
        identity = inference.validate_adapter_contract(
            _v2_config("cross_attn2_qo"), receipt
        )
        self.assertTrue(identity["frozen_baseline"])

        targets = identity["target_modules"]
        keys = inference.expected_adapter_state_keys(targets)
        saved = {
            key: 0 if ".lora_B.weight" in key else 1
            for key in keys
        }
        self.assertEqual(
            inference.validate_adapter_state_dicts(
                saved,
                dict(saved),
                tensor_equal=lambda left, right: left == right,
                expected_targets=targets,
                require_zero_effect=True,
                tensor_is_zero=lambda value: value == 0,
            ),
            120,
        )
        hostile = dict(saved)
        first_b = next(key for key in keys if ".lora_B.weight" in key)
        hostile[first_b] = 1
        with self.assertRaisesRegex(
            inference.InferenceContractError, "zero-effect baseline"
        ):
            inference.validate_adapter_state_dicts(
                hostile,
                dict(hostile),
                tensor_equal=lambda left, right: left == right,
                expected_targets=targets,
                require_zero_effect=True,
                tensor_is_zero=lambda value: value == 0,
            )

        false_baseline = copy.deepcopy(receipt)
        false_baseline["last_preclip_gradient_norm"] = 0.1
        _resign(false_baseline)
        with self.assertRaisesRegex(
            inference.InferenceContractError, "untouched zero-loss/zero-grad"
        ):
            inference.validate_adapter_contract(
                _v2_config("cross_attn2_qo"), false_baseline
            )

        false_trained = _v2_receipt("v2_func025_cross_qo", step=5)
        false_trained["last_preclip_gradient_norm"] = 0.0
        _resign(false_trained)
        with self.assertRaisesRegex(inference.InferenceContractError, "positive"):
            inference.validate_adapter_contract(
                _v2_config("cross_attn2_qo"), false_trained
            )

    def test_legacy_validation_result_and_default_480_keys_are_unchanged(self) -> None:
        with mock.patch.object(
            inference,
            "_load_preservation_v2_contract",
            side_effect=AssertionError("legacy path loaded v2 code"),
        ):
            identity = inference.validate_adapter_contract(
                _legacy_config(), _legacy_receipt()
            )
        self.assertEqual(
            set(identity),
            {
                "global_step",
                "receipt_digest",
                "target_modules_sha256",
                "transformers_version",
                "method_source_revision",
                "method_source_archive_sha256",
            },
        )
        self.assertNotIn("lora_route_scope", identity)
        self.assertEqual(len(inference.expected_adapter_state_keys()), 480)


if __name__ == "__main__":
    unittest.main()
