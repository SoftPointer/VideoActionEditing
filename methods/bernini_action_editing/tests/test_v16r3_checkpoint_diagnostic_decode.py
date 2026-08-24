from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

import infer_online_anchor_v16r3_checkpoint_lora as route_off
import infer_online_anchor_v16r3_route_decode as route_decode
import v16r3_checkpoint_contract as contract


CONFIG_SHA = "a" * 64
MODEL_SHA = "b" * 64
RECEIPT_SHA = "c" * 64


def valid_config():
    config = contract._expected_peft_config_without_targets()
    config["target_modules"] = ["to_q", "to_k", "to_v", "to_out.0"]
    return config


def valid_receipt(step: int):
    terminal = step == contract.MAX_STEPS
    covered = step >= contract.S279_STEP
    target_iids = sorted(f"{index:016x}" for index in range(step))
    families = sorted(f"family{index:02d}" for index in range(min(step, 28)))
    fallback_summary = {
        "policy": contract.FALLBACK_POLICY,
        "reason": contract.FALLBACK_REASON,
        "fallback_count": 0,
        "fallback_steps": [],
        "fallback_target_iids": [],
        "fallback_geometry": [],
        "optimizer_state_reset_count": 0,
        "failed_candidates_committed": False,
        "parameter_values_exactly_restored_before_each_retry": True,
        "optimizer_state_restored": False,
        "optimizer_state_reset_before_each_retry": True,
        "committed_retry_gradient": "primary_action_only_clipped",
        "retry_limit_per_failed_candidate": 1,
        "committed_retries_reprobed_by_frozen_authority": True,
        "action_descent_gate_relaxed": False,
        "optimizer_history_matches_uninterrupted_adamw": True,
        "continuous_parameter_trajectory_from_frozen_base": True,
        "scientific_claim_authorized": False,
    }
    training_contract = {
        "method": contract.TRAINING_METHOD,
        "training_objective": contract.TRAINING_OBJECTIVE,
        "route_operator": contract.ROUTE_OPERATOR,
        "route_transport": contract.REQUIRED_DECODE_TRANSPORT,
        "target_owned_qk_route_v14r2": True,
        "anchor_donor_cached_fields": ["query", "key"],
        "anchor_donor_value_cached_or_used_by_route": False,
        "anchor_donor_hidden_or_attention_output_cached_or_used_by_route": False,
        "anchor_donor_rgb_latent_or_absolute_spatial_coordinate_used_by_route": False,
        "anchor_to_target_appearance_correspondence_used": False,
        "anchor_qk_time_constant_caption_offset_removed_before_support_and_kernel": True,
        "anchor_qk_phase0_only_difference_produces_zero_route": True,
        "dynaedit_sga_anc_reserved_for_decode_solver": True,
        "lora_rank": contract.LORA_RANK,
        "lora_alpha": contract.LORA_ALPHA,
        "lora_scope": contract.LORA_SCOPE,
        "lora_target_module_count": contract.TARGET_MODULE_COUNT,
        "lora_target_modules_sha256": contract.TARGET_MODULES_SHA256,
        "trainable_parameter_count": contract.TRAINABLE_PARAMETER_COUNT,
        "full_attention_lora_enabled": True,
        "full644_optimizer_schedule": "exact644_unique_rows_once",
        "all_full644_rows_targeted_exactly_once": terminal,
        "single_continuous_fresh_from_base_exact644_run": True,
        "single_continuous_fresh_from_base_exact644_parameter_trajectory": True,
        "starts_from_frozen_base_checkpoint_not_prior_adapter": True,
        "micro_semantics": "different_seed_same_iid_role1_action_anchor",
        "anchor_route_replay_uses_per_capture": 2,
        "teacher_delta_mode": "raw",
        "routed_teacher_mode": "same_action_route_only",
        "student_route_off_branch_stop_gradient": True,
        "action_objective_backpropagates_only_routed_student_query": True,
        "routed_teacher_cross_caption_source_branch": False,
        "source_reconstruction_weight": None,
        "source_reconstruction_weight_argument": 0.025,
        "base_replay_scale": 0.025,
        "replay_combine_mode": "action_priority_pcgrad_010",
        "qk_only_zero_rms_backward_policy": contract.ZERO_RMS_POLICY,
        "qk_only_zero_rms_backward_scope": list(contract.ZERO_RMS_SCOPE),
        "qk_only_zero_rms_forward_values_changed": False,
        "qk_only_zero_rms_zero_subgradient": 0.0,
        "s279_endpoint_canary_covered": covered,
        "s279_endpoint_canary_target_iid": contract.S279_TARGET_IID,
        "sample_retry_or_skip_for_v16r3": False,
        "seed_or_timestep_changed_for_v16r3": False,
        "loss_scale_changed_for_v16r3": False,
        "component_preallreduce_finite_gate_relaxed": False,
        "nonfinite_gradient_committed": False,
        "manual_or_visual_review_controls_optimizer_admission": False,
        "qwen_or_other_verifier_controls_optimizer_admission": False,
        "scientific_claim_authorized": False,
        "actual_distinct_target_iid_count": step,
        "actual_distinct_target_iids": target_iids,
        "actual_distinct_same_iid_role1_donor_count": step,
        "actual_distinct_same_iid_role1_donor_iids": target_iids,
        "actual_distinct_target_family_count": min(step, 28),
        "actual_distinct_target_families": families,
        "family_round_robin_first28_cover_all_families": (
            True if step >= 28 else None
        ),
        "actual_action_descent_fallback_policy": contract.FALLBACK_POLICY,
        "actual_action_descent_fallback_count": 0,
        "actual_action_descent_fallback_steps": [],
        "actual_action_descent_fallback_target_iids": [],
        "actual_action_descent_failed_candidates_committed": False,
        "actual_action_descent_fallback_parameter_values_exactly_restored": True,
        "actual_action_descent_fallback_optimizer_state_restored": False,
        "actual_action_descent_fallback_optimizer_state_reset_count": 0,
        "actual_action_descent_fallback_uses_primary_action_only": True,
        "actual_action_descent_fallback_retry_limit": 1,
        "actual_action_descent_fallback_reprobes_frozen_authority": True,
        "actual_action_descent_gate_relaxed": False,
        "optimizer_history_matches_uninterrupted_adamw": True,
    }
    return {
        "schema_version": contract.TRAINING_RECEIPT_SCHEMA,
        "adapter_config_sha256": CONFIG_SHA,
        "adapter_model_sha256": MODEL_SHA,
        "bernini_commit": contract.delegated.trainer.BERNINI_OFFICIAL_COMMIT,
        "veomni_commit": contract.delegated.trainer.VEOMNI_TESTED_COMMIT,
        "global_step": step,
        "max_steps": contract.MAX_STEPS,
        "complete": True,
        "scientific_claim_authorized": False,
        "claim_scope": contract.TRAINING_CLAIM_SCOPE,
        "last_reporting_scalar_is_not_a_joint_backpropagated_objective": True,
        "method_source_revision": "d" * 64,
        "method_source_archive_sha256": "e" * 64,
        "training_contract": training_contract,
        "v16_full644_summary": {
            "manifest_row_count": contract.MAX_STEPS,
            "manifest_family_count": 28,
            "manifest_sha256": "1" * 64,
            "manifest_digest": "2" * 64,
            "target_prefix_row_count": step,
            "target_prefix_iids_sha256": "3" * 64,
            "target_prefix_exact_once": True,
            "family_round_robin_first28_cover_all_families": (
                True if step >= 28 else None
            ),
            "actual_target_family_count": min(step, 28),
            "all_full644_rows_targeted_exactly_once": terminal,
            "donor_selection_count": 2 * step,
            "same_iid_role1_donor_count": 2 * step,
            "distinct_donor_iid_count": step,
            "anchor_cross_appearance": False,
            "pair_decode_count": step,
            "manual_or_visual_review_controls_optimizer_admission": False,
            "qwen_or_other_verifier_controls_optimizer_admission": False,
            "all_rows_admitted_from_sealed_manifest_without_per_sample_review": True,
            "scientific_claim_authorized": False,
        },
        "v16r2_actual_action_descent_fallback_summary": fallback_summary,
        "v16r3_zero_rms_backward_summary": {
            "policy": contract.ZERO_RMS_POLICY,
            "scope": list(contract.ZERO_RMS_SCOPE),
            "active_qk_route": "qk_only_target_gated_hard_temporal_kernel_contrast_output",
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
                "step": contract.S279_STEP,
                "target_iid": contract.S279_TARGET_IID,
                "target_family": contract.S279_TARGET_FAMILY,
                "expected_calls": [dict(row) for row in contract.S279_EXPECTED_CALLS],
                "observed_calls": (
                    [dict(row) for row in contract.S279_EXPECTED_CALLS]
                    if covered
                    else []
                ),
                "covered_by_checkpoint": covered,
            },
        },
        "memory_gate": {
            "capture_phase": "after_two_real_component_backwards_before_actual_update_audit_clones",
            "actual_update_audit_allocations_excluded": True,
            "passed": True,
            "dummy_or_padding_allocations": False,
            "true_training_tensors_only": True,
            "minimum_reserved_fraction": 0.75,
            "per_rank": [
                {"rank": rank, "reserved_fraction": 0.75} for rank in range(4)
            ],
        },
        "gradient_coverage": {
            "tensor_count": contract.ADAPTER_TENSOR_COUNT,
            "nonzero_tensor_count": (
                240 if step == 1 else contract.ADAPTER_TENSOR_COUNT
            ),
            "nonzero_names_sha256": (
                contract.S1_NONZERO_NAMES_SHA256
                if step == 1
                else contract.FULL_NONZERO_NAMES_SHA256
            ),
        },
    }


def validate(step: int, receipt=None):
    return contract.validate_v16r3_checkpoint_contract(
        valid_config(),
        valid_receipt(step) if receipt is None else receipt,
        expected_global_step=step,
        expected_adapter_config_sha256=CONFIG_SHA,
        expected_adapter_model_sha256=MODEL_SHA,
        expected_training_receipt_sha256=RECEIPT_SHA,
    )


def write_synthetic_adapter(path: Path, *, drop_last: bool = False) -> None:
    keys = contract.delegated.expected_adapter_state_keys(
        contract.expected_target_modules()
    )
    if drop_last:
        keys = keys[:-1]
    header = {"__metadata__": {"format": "pt"}}
    offset = 0
    for key in keys:
        shape = (
            [contract.LORA_RANK, 1536]
            if key.endswith(".lora_A.weight")
            else [1536, contract.LORA_RANK]
        )
        size = 4 * shape[0] * shape[1]
        header[key] = {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        offset += size
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        handle.truncate(8 + len(raw) + offset)


class V16R3CheckpointContractTest(unittest.TestCase):
    def test_accepts_registered_prefixes_on_both_sides_of_s279(self):
        for step in (1, 256, 359, 644):
            with self.subTest(step=step):
                identity = validate(step)
                self.assertEqual(identity["global_step"], step)
                self.assertEqual(identity["max_steps"], 644)
                self.assertIs(identity["terminal_full644_checkpoint"], step == 644)
                self.assertEqual(identity["target_module_count"], 240)
                self.assertEqual(identity["adapter_tensor_count"], 480)

    def test_rejects_step_max_prefix_and_s279_semantic_drift(self):
        mutations = []
        wrong_step = valid_receipt(256)
        wrong_step["global_step"] = 255
        mutations.append(("step", wrong_step, 256))
        wrong_max = valid_receipt(256)
        wrong_max["max_steps"] = 256
        mutations.append(("max", wrong_max, 256))
        wrong_count = valid_receipt(64)
        wrong_count["v16_full644_summary"]["donor_selection_count"] = 127
        mutations.append(("donors", wrong_count, 64))
        early_false_claim = valid_receipt(256)
        early_false_claim["v16r3_zero_rms_backward_summary"][
            "s279_endpoint_canary"
        ]["covered_by_checkpoint"] = True
        mutations.append(("early canary", early_false_claim, 256))
        late_missing_calls = valid_receipt(359)
        late_missing_calls["v16r3_zero_rms_backward_summary"][
            "s279_endpoint_canary"
        ]["observed_calls"] = []
        mutations.append(("late canary", late_missing_calls, 359))
        for label, receipt, step in mutations:
            with self.subTest(label=label):
                with self.assertRaises(contract.V16R3CheckpointContractError):
                    validate(step, receipt)

    def test_rejects_unregistered_step_and_wrong_checkpoint_directory(self):
        with self.assertRaises(contract.V16R3CheckpointContractError):
            contract.require_save_step(2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint-00000004"
            adapter = root / "adapter"
            adapter.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text("{}", encoding="ascii")
            (adapter / "adapter_model.safetensors").write_bytes(b"x")
            (root / "receipt.json").write_text("{}", encoding="ascii")
            with self.assertRaises(contract.V16R3CheckpointContractError):
                contract.resolve_checkpoint(root, expected_global_step=8)

    def test_safetensors_header_proves_exact_240_modules_480_tensors(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "adapter_model.safetensors"
            write_synthetic_adapter(path)
            inventory = contract.validate_adapter_safetensors_inventory(path)
            self.assertEqual(inventory["tensor_count"], 480)
            self.assertEqual(inventory["target_module_count"], 240)
            self.assertEqual(
                inventory["parameter_element_count"],
                contract.TRAINABLE_PARAMETER_COUNT,
            )
            broken = Path(temporary) / "broken.safetensors"
            write_synthetic_adapter(broken, drop_last=True)
            with self.assertRaises(contract.V16R3CheckpointContractError):
                contract.validate_adapter_safetensors_inventory(broken)


class V16R3RouteOffWrapperTest(unittest.TestCase):
    def tearDown(self):
        route_off._ACTIVE_IDENTITY = None

    def test_parser_requires_step_and_external_three_sha(self):
        parser = route_off.build_parser()
        actions = {
            option: action
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertTrue(actions["--expected-training-global-step"].required)
        self.assertEqual(
            tuple(actions["--expected-training-global-step"].choices),
            contract.SAVE_STEPS,
        )
        for option in (
            "--expected-adapter-config-sha256",
            "--expected-adapter-model-sha256",
            "--expected-training-receipt-sha256",
        ):
            self.assertTrue(actions[option].required)

    def test_receipt_annotation_preserves_dynamic_checkpoint_identity(self):
        identity = validate(64)
        delegated_receipt = {
            "schema_version": route_off.delegated.INFERENCE_RECEIPT_SCHEMA,
            "infer_lora_source_sha256": "1" * 64,
            "adapter": {
                "enabled": True,
                "adapter_model_sha256": MODEL_SHA,
                "training_receipt_digest": RECEIPT_SHA,
                "strictly_reloaded": True,
                "safe_merged_for_inference": True,
                "tensor_count": 480,
            },
            "receipt_digest": "2" * 64,
        }
        annotated = route_off._annotate_inference_receipt(
            delegated_receipt,
            adapter_identity=identity,
            wrapper_source_sha256="3" * 64,
        )
        self.assertEqual(annotated["schema_version"], route_off.INFERENCE_RECEIPT_SCHEMA)
        v16r3 = annotated["adapter"]["v16r3"]
        self.assertEqual(v16r3["global_step"], 64)
        self.assertEqual(v16r3["max_steps"], 644)
        self.assertFalse(v16r3["terminal_full644_checkpoint"])
        self.assertFalse(v16r3["online_anchor_route_applied"])

    def test_main_restores_every_delegated_patch_on_failure(self):
        originals = (
            route_off.delegated.build_parser,
            route_off.delegated.validate_cli,
            route_off.delegated.activate_model_consumption_authority,
            route_off.delegated.validate_adapter_contract,
            route_off.delegated._strict_load_and_merge_adapter,
            route_off.delegated.build_inference_receipt,
        )

        def fail(_argv):
            self.assertIs(route_off.delegated.build_parser, route_off.build_parser)
            self.assertIs(
                route_off.delegated.validate_adapter_contract,
                route_off.validate_adapter_contract,
            )
            raise RuntimeError("synthetic stop")

        with mock.patch.object(route_off.delegated, "main", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "synthetic stop"):
                route_off.main([])
        self.assertEqual(
            originals,
            (
                route_off.delegated.build_parser,
                route_off.delegated.validate_cli,
                route_off.delegated.activate_model_consumption_authority,
                route_off.delegated.validate_adapter_contract,
                route_off.delegated._strict_load_and_merge_adapter,
                route_off.delegated.build_inference_receipt,
            ),
        )


class FakeAnchorError(RuntimeError):
    pass


class V16R3RouteDecodeWrapperTest(unittest.TestCase):
    def tearDown(self):
        route_decode._ACTIVE_CHECKPOINT = None
        route_decode._ACTIVE_RUNNER = None

    @staticmethod
    def fake_runner():
        return SimpleNamespace(
            AnchorEventInferenceError=FakeAnchorError,
            _canonical_json=lambda value: json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
            _attention_lora_checkpoint=object(),
            source_audit=SimpleNamespace(model_freeze_certificate=object()),
            main=lambda _argv: 0,
        )

    def authenticated(self):
        bundle = contract.CheckpointBundle(
            checkpoint_root=Path("/tmp/checkpoint-00000064"),
            adapter_dir=Path("/tmp/checkpoint-00000064/adapter"),
            adapter_config_path=Path("/tmp/checkpoint-00000064/adapter/adapter_config.json"),
            adapter_model_path=Path("/tmp/checkpoint-00000064/adapter/adapter_model.safetensors"),
            training_receipt_path=Path("/tmp/checkpoint-00000064/receipt.json"),
        )
        identity = validate(64)
        return {
            **identity,
            "bundle": bundle,
            "training_receipt": valid_receipt(64),
        }

    def test_route_checkpoint_adapter_returns_actual_v16r3_schema(self):
        route_decode._ACTIVE_RUNNER = self.fake_runner()
        with mock.patch.object(contract, "validate_runtime_versions"), mock.patch.object(
            contract, "authenticate_checkpoint", return_value=self.authenticated()
        ) as authenticate:
            result = route_decode._attention_lora_checkpoint(
                "/tmp/checkpoint-00000064",
                expected_global_step=64,
                expected_training_objective=contract.TRAINING_OBJECTIVE,
                expected_route_operator=contract.ROUTE_OPERATOR,
                expected_adapter_model_sha256=MODEL_SHA,
                expected_adapter_config_sha256=CONFIG_SHA,
                expected_receipt_sha256=RECEIPT_SHA,
            )
        authenticate.assert_called_once()
        self.assertEqual(result["schema_version"], contract.TRAINING_RECEIPT_SCHEMA)
        self.assertEqual(result["global_step"], 64)
        self.assertEqual(
            result["required_decode_transport"],
            contract.REQUIRED_DECODE_TRANSPORT,
        )
        self.assertEqual(result["model_sha256"], MODEL_SHA)
        expected_digest = hashlib.sha256(
            route_decode._ACTIVE_RUNNER._canonical_json(result["binding"])
        ).hexdigest()
        self.assertEqual(result["binding_sha256"], expected_digest)

    def test_route_main_restores_checkpoint_and_freeze_monkeypatches(self):
        fake = self.fake_runner()
        original_checkpoint = fake._attention_lora_checkpoint
        original_certificate = fake.source_audit.model_freeze_certificate
        replacement = object()

        def fail(_argv):
            self.assertIs(
                fake._attention_lora_checkpoint,
                route_decode._attention_lora_checkpoint,
            )
            self.assertIs(fake.source_audit.model_freeze_certificate, replacement)
            raise RuntimeError("synthetic stop")

        fake.main = fail
        route_decode._ACTIVE_RUNNER = fake
        with mock.patch.object(
            route_decode.trained_editor,
            "build_model_freeze_certificate",
            return_value=replacement,
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic stop"):
                route_decode.main([])
        self.assertIs(fake._attention_lora_checkpoint, original_checkpoint)
        self.assertIs(fake.source_audit.model_freeze_certificate, original_certificate)
        self.assertIsNone(route_decode._ACTIVE_RUNNER)


if __name__ == "__main__":
    unittest.main()
