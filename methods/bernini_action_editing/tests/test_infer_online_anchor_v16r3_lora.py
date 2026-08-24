from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

import infer_online_anchor_v16r3_lora as method


CONFIG_SHA = "a" * 64
MODEL_SHA = "b" * 64
RECEIPT_SHA = "c" * 64


def valid_config():
    config = method._expected_peft_config_without_targets()
    config["target_modules"] = ["to_q", "to_k", "to_v", "to_out.0"]
    return config


def valid_receipt():
    return {
        "schema_version": method.TRAINING_RECEIPT_SCHEMA,
        "adapter_config_sha256": CONFIG_SHA,
        "adapter_model_sha256": MODEL_SHA,
        "bernini_commit": method.delegated.trainer.BERNINI_OFFICIAL_COMMIT,
        "veomni_commit": method.delegated.trainer.VEOMNI_TESTED_COMMIT,
        "global_step": method.GLOBAL_STEP,
        "max_steps": method.GLOBAL_STEP,
        "complete": True,
        "scientific_claim_authorized": False,
        "claim_scope": method.TRAINING_CLAIM_SCOPE,
        "last_reporting_scalar_is_not_a_joint_backpropagated_objective": True,
        "method_source_revision": "d" * 64,
        "method_source_archive_sha256": "e" * 64,
        "training_contract": {
            "method": method.TRAINING_METHOD,
            "training_objective": (
                "real_source_target_owned_routed_teacher_delta_v14r2"
            ),
            "route_operator": "self_target_owned_activity_kernel25_v14r2",
            "route_transport": (
                "self_target_owned_activity_kernel25_attn_output_v14r2"
            ),
            "dynaedit_sga_anc_reserved_for_decode_solver": True,
            "lora_rank": method.LORA_RANK,
            "lora_alpha": method.LORA_ALPHA,
            "lora_scope": method.LORA_SCOPE,
            "lora_target_module_count": method.TARGET_MODULE_COUNT,
            "lora_target_modules_sha256": method.TARGET_MODULES_SHA256,
            "trainable_parameter_count": method.TRAINABLE_PARAMETER_COUNT,
            "full_attention_lora_enabled": True,
            "full644_optimizer_schedule": "exact644_unique_rows_once",
            "all_full644_rows_targeted_exactly_once": True,
            "single_continuous_fresh_from_base_exact644_run": True,
            "single_continuous_fresh_from_base_exact644_parameter_trajectory": True,
            "starts_from_frozen_base_checkpoint_not_prior_adapter": True,
            "qk_only_zero_rms_backward_policy": method.ZERO_RMS_POLICY,
            "qk_only_zero_rms_backward_scope": method.ZERO_RMS_SCOPE,
            "qk_only_zero_rms_forward_values_changed": False,
            "qk_only_zero_rms_zero_subgradient": 0.0,
            "s279_endpoint_canary_covered": True,
            "sample_retry_or_skip_for_v16r3": False,
            "seed_or_timestep_changed_for_v16r3": False,
            "loss_scale_changed_for_v16r3": False,
            "component_preallreduce_finite_gate_relaxed": False,
            "nonfinite_gradient_committed": False,
            "manual_or_visual_review_controls_optimizer_admission": False,
            "scientific_claim_authorized": False,
        },
        "v16r3_zero_rms_backward_summary": {
            "policy": method.ZERO_RMS_POLICY,
            "scope": method.ZERO_RMS_SCOPE,
            "finite_nonnegative_forward_values_bit_exact": True,
            "zero_forward_value": 0.0,
            "zero_backward_subgradient": 0.0,
            "positive_backward_matches_standard_sqrt": True,
            "negative_or_nonfinite_values_masked": False,
            "loss_scale_changed": False,
            "seed_or_timestep_changed": False,
            "sample_retry_or_skip": False,
            "component_preallreduce_finite_gate_relaxed": False,
            "nonfinite_gradient_committed": False,
            "policy_fixed_from_step_one": True,
            "single_continuous_fresh_from_base_exact644": True,
            "scientific_claim_authorized": False,
            "s279_endpoint_canary": {
                "step": 279,
                "target_iid": "4aeb0557a94b4db3",
                "target_family": "fall",
                "expected_calls": copy.deepcopy(method.S279_EXPECTED_CALLS),
                "observed_calls": copy.deepcopy(method.S279_EXPECTED_CALLS),
                "covered_by_checkpoint": True,
            },
        },
        "memory_gate": {
            "passed": True,
            "dummy_or_padding_allocations": False,
            "true_training_tensors_only": True,
            "minimum_reserved_fraction": 0.75,
        },
        "gradient_coverage": {
            "tensor_count": method.ADAPTER_TENSOR_COUNT,
            "nonzero_tensor_count": method.ADAPTER_TENSOR_COUNT,
        },
    }


def validate(config=None, receipt=None):
    return method.validate_v16r3_adapter_contract(
        valid_config() if config is None else config,
        valid_receipt() if receipt is None else receipt,
        expected_adapter_config_sha256=CONFIG_SHA,
        expected_adapter_model_sha256=MODEL_SHA,
        expected_training_receipt_sha256=RECEIPT_SHA,
    )


class InferOnlineAnchorV16R3LoraTest(unittest.TestCase):
    def tearDown(self):
        method._ACTIVE_EXPECTED_IDENTITY = None

    def test_valid_contract_returns_exact_strict_loader_identity(self):
        identity = validate()
        self.assertTrue(identity["v16r3_online_anchor"])
        self.assertEqual(identity["global_step"], 644)
        self.assertEqual(identity["lora_rank"], 256)
        self.assertEqual(identity["lora_alpha"], 256)
        self.assertEqual(identity["target_module_count"], 240)
        self.assertEqual(len(identity["target_modules"]), 240)
        self.assertEqual(identity["adapter_config_sha256"], CONFIG_SHA)
        self.assertEqual(identity["adapter_model_sha256"], MODEL_SHA)
        self.assertEqual(identity["training_receipt_sha256"], RECEIPT_SHA)

    def test_contract_rejects_schema_step_complete_and_declared_hash_drift(self):
        mutations = (
            ("schema", lambda row: row.__setitem__("schema_version", "v16r2")),
            ("step", lambda row: row.__setitem__("global_step", 643)),
            ("complete", lambda row: row.__setitem__("complete", False)),
            (
                "config sha",
                lambda row: row.__setitem__("adapter_config_sha256", "f" * 64),
            ),
            (
                "model sha",
                lambda row: row.__setitem__("adapter_model_sha256", "f" * 64),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                receipt = valid_receipt()
                mutate(receipt)
                with self.assertRaises(method.V16R3InferenceContractError):
                    validate(receipt=receipt)

    def test_contract_rejects_rank_alpha_or_target_scope_drift(self):
        configs = []
        rank = valid_config()
        rank["r"] = 64
        configs.append(("rank", rank))
        alpha = valid_config()
        alpha["lora_alpha"] = 64
        configs.append(("alpha", alpha))
        target = valid_config()
        target["target_modules"] = ["to_q", "to_k", "to_v"]
        configs.append(("target", target))
        optional = valid_config()
        optional["use_dora"] = True
        configs.append(("optional PEFT semantic", optional))
        for label, config in configs:
            with self.subTest(label=label):
                with self.assertRaises(method.V16R3InferenceContractError):
                    validate(config=config)

    def test_contract_rejects_canary_and_no_manual_admission_drift(self):
        canary = valid_receipt()
        canary["v16r3_zero_rms_backward_summary"]["s279_endpoint_canary"][
            "observed_calls"
        ][0]["timestep"] = 999.0
        with self.assertRaises(method.V16R3InferenceContractError):
            validate(receipt=canary)

        manual = valid_receipt()
        manual["training_contract"][
            "manual_or_visual_review_controls_optimizer_admission"
        ] = True
        with self.assertRaises(method.V16R3InferenceContractError):
            validate(receipt=manual)

    def test_parser_requires_all_three_external_artifact_hashes(self):
        parser = method.build_parser()
        actions = {option: action for action in parser._actions for option in action.option_strings}
        self.assertTrue(actions["--expected-adapter-config-sha256"].required)
        self.assertTrue(actions["--expected-adapter-model-sha256"].required)
        self.assertTrue(actions["--expected-training-receipt-sha256"].required)
        self.assertIs(
            actions["--expected-training-receipt-sha256"],
            actions["--expected-adapter-receipt-sha256"],
        )

    def test_physical_binding_hashes_config_and_receipt_before_gpu_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint-00000644"
            adapter = root / "adapter"
            adapter.mkdir(parents=True)
            config_raw = b"{}\n"
            receipt_raw = b"{\"complete\":true}\n"
            model_raw = b"synthetic-safetensors-placeholder"
            (adapter / "adapter_config.json").write_bytes(config_raw)
            (adapter / "adapter_model.safetensors").write_bytes(model_raw)
            (root / "receipt.json").write_bytes(receipt_raw)
            args = SimpleNamespace(
                base_only=False,
                adapter_checkpoint=str(root),
                adapter_checkpoint_manifest=None,
                adapter_checkpoint_manifest_sha256=None,
                expected_adapter_config_sha256=hashlib.sha256(config_raw).hexdigest(),
                expected_adapter_model_sha256=hashlib.sha256(model_raw).hexdigest(),
                expected_training_receipt_sha256=hashlib.sha256(receipt_raw).hexdigest(),
            )
            identity = method._identity_from_args(args)
            self.assertEqual(identity.adapter_dir, adapter.resolve(strict=True))
            method._verify_artifact_hashes(identity, include_model=True)
            (root / "receipt.json").write_bytes(b"{}\n")
            with self.assertRaises(method.V16R3InferenceContractError):
                method._verify_artifact_hashes(identity, include_model=False)

    def test_inference_receipt_explicitly_labels_v16r3_file_identity(self):
        identity = validate()
        legacy = {
            "schema_version": method.delegated.INFERENCE_RECEIPT_SCHEMA,
            "infer_lora_source_sha256": "1" * 64,
            "adapter": {
                "enabled": True,
                "adapter_model_sha256": MODEL_SHA,
                "training_receipt_digest": RECEIPT_SHA,
                "strictly_reloaded": True,
                "safe_merged_for_inference": True,
                "tensor_count": method.ADAPTER_TENSOR_COUNT,
            },
            "experimental_inference": True,
            "scientific_claim_authorized": False,
            "receipt_digest": "2" * 64,
        }
        annotated = method._annotate_inference_receipt(
            legacy,
            adapter_identity=identity,
            wrapper_source_sha256="3" * 64,
        )
        self.assertEqual(annotated["schema_version"], method.INFERENCE_RECEIPT_SCHEMA)
        self.assertNotIn("training_receipt_digest", annotated["adapter"])
        self.assertEqual(
            annotated["adapter"]["training_receipt_sha256"], RECEIPT_SHA
        )
        training = annotated["adapter"]["training_identity"]
        self.assertEqual(training["schema_version"], method.TRAINING_RECEIPT_SCHEMA)
        self.assertEqual(training["adapter_config_sha256"], CONFIG_SHA)
        self.assertEqual(training["adapter_model_sha256"], MODEL_SHA)
        self.assertTrue(training["complete"])
        v16r3 = annotated["adapter"]["v16r3"]
        self.assertEqual(v16r3["decode_mode"], "adapter_only_direct_rv2v")
        self.assertFalse(v16r3["online_anchor_route_applied"])
        self.assertEqual(v16r3["lora_rank"], 256)
        self.assertEqual(v16r3["lora_alpha"], 256)
        self.assertEqual(v16r3["target_module_count"], 240)
        self.assertEqual(v16r3["training_receipt_sha256"], RECEIPT_SHA)
        self.assertEqual(v16r3["adapter_sha256"], MODEL_SHA)
        self.assertEqual(v16r3["config_sha256"], CONFIG_SHA)
        self.assertEqual(v16r3["receipt_sha256"], RECEIPT_SHA)
        unsigned = dict(annotated)
        digest = unsigned.pop("receipt_digest")
        self.assertEqual(digest, method.delegated.object_sha256(unsigned))

    def test_strict_loader_wrapper_checks_bytes_then_delegates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint-00000644"
            adapter_dir = root / "adapter"
            adapter_dir.mkdir(parents=True)
            config_raw = b"config"
            model_raw = b"model"
            receipt_raw = b"receipt"
            config_path = adapter_dir / "adapter_config.json"
            model_path = adapter_dir / "adapter_model.safetensors"
            receipt_path = root / "receipt.json"
            config_path.write_bytes(config_raw)
            model_path.write_bytes(model_raw)
            receipt_path.write_bytes(receipt_raw)
            identity = method.ExpectedArtifactIdentity(
                checkpoint_root=root,
                adapter_dir=adapter_dir,
                adapter_config_path=config_path,
                adapter_model_path=model_path,
                training_receipt_path=receipt_path,
                adapter_config_sha256=hashlib.sha256(config_raw).hexdigest(),
                adapter_model_sha256=hashlib.sha256(model_raw).hexdigest(),
                training_receipt_sha256=hashlib.sha256(receipt_raw).hexdigest(),
            )
            bundle = method.delegated.AdapterBundle(
                root, adapter_dir, config_path, model_path, receipt_path
            )
            method._ACTIVE_EXPECTED_IDENTITY = identity
            sentinel = (object(), method.ADAPTER_TENSOR_COUNT)
            with mock.patch.object(
                method,
                "_DELEGATED_STRICT_LOAD_AND_MERGE",
                return_value=sentinel,
            ) as loader:
                result = method._strict_load_and_merge_adapter(
                    object(), bundle, method.expected_target_modules()
                )
            self.assertIs(result, sentinel)
            loader.assert_called_once()


if __name__ == "__main__":
    unittest.main()
