from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(METHOD_ROOT))

import motion_residual as motion  # noqa: E402
import train_delta_lora as delta_train  # noqa: E402
from spt_v2 import generator_native_sparse_router as sparse_router  # noqa: E402
from spt_v2 import infer_c2fr_lora as inference  # noqa: E402


SHA1 = "1" * 40
SHA256 = "2" * 64


def _runtime_schedule_audit() -> dict:
    return {
        "schedule_sha256": inference.sigma_strata.SCHEDULE_SHA256,
        "timesteps": list(inference.sigma_strata.PINNED_TIMESTEPS),
        "positive_sigmas": list(
            inference.sigma_strata.PINNED_POSITIVE_SIGMAS
        ),
        "positive_sigmas_float32_be_hex": list(
            inference.sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX
        ),
        "terminal_sigma": 0.0,
        "terminal_sigma_float32_be_hex": (
            inference.sigma_strata.TERMINAL_SIGMA_FLOAT32_HEX
        ),
    }


def _args(**overrides) -> argparse.Namespace:
    values = {
        "instruction": "Make the actor crouch.",
        "num_inference_steps": 40,
        "seed": 42,
        "alpha": 1.0,
        "max_generate_fraction": 0.12,
        "energy_coverage": 0.85,
        "expected_bernini_commit": inference.trainer.BERNINI_OFFICIAL_COMMIT,
        "expected_veomni_commit": inference.trainer.VEOMNI_TESTED_COMMIT,
        "expected_checkpoint_tree_sha256": (
            inference.trainer.CHECKPOINT_TREE_SHA256
        ),
        "method_source_revision": SHA1,
        "method_source_archive_sha256": SHA256,
        "checkpoint": "/checkpoint/Bernini-R-1.3B-Diffusers",
        "adapter_checkpoint": "/adapter/checkpoint-00000064",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _valid_metadata(scope: str = "cross_q") -> tuple[dict, dict]:
    targets = motion.select_lora_scope(
        inference._audited_attention_projection_names(), scope
    )
    noop_sha256 = hashlib.sha256(
        motion.DEFAULT_NOOP_INSTRUCTION.encode("utf-8")
    ).hexdigest()
    value = {
        "method": delta_train.METHOD_NAME,
        "method_source_revision": "3" * 40,
        "method_source_archive_sha256": "4" * 64,
        "bernini_commit": inference.trainer.BERNINI_OFFICIAL_COMMIT,
        "veomni_commit": inference.trainer.VEOMNI_TESTED_COMMIT,
        "checkpoint_tree_sha256": inference.trainer.CHECKPOINT_TREE_SHA256,
        "lora_scope": scope,
        "target_modules": targets,
        "paired_cells": [
            "source_query_action",
            "source_query_noop",
            "executable_target_query_action",
            "executable_target_query_noop",
        ],
        "posterior_statistic": "mode",
        "branch_state_mode": inference.REQUIRED_BRANCH_STATE_MODE,
        "minimum_training_sigma": 0.1,
        "inverse_sigma_weight_floor": (
            inference.REQUIRED_INVERSE_SIGMA_WEIGHT_FLOOR
        ),
        "shared_source_sigma_noise": True,
        "exact_same_noisy_query": True,
        "clean_reconstruction_formula": "x_clean = y - sigma * velocity",
        "clean_reconstruction_numeric_program": (
            "fp32_noisy_minus_cpu_fp32_0d_sigma_times_native_bf16_velocity"
        ),
        "training_sigma_representation": "cpu_fp32_0d",
        "branch_prediction_dtype_before_clean_reconstruction": "bfloat16",
        "predicted_clean_delta_formula": "-sigma * (v_action - v_noop)",
        "target_clean_delta_formula": "executable_target_clean - source_clean",
        "target_projection": inference.REQUIRED_TARGET_PROJECTION,
        "target_projection_idempotent": True,
        "motion_loss_multiplier": "1 / sigma",
        "copy_boundary_loss_multiplier": "not_enabled",
        "clean_field_loss_weight_range": [
            1.0,
            1.0 / inference.REQUIRED_INVERSE_SIGMA_WEIGHT_FLOOR,
        ],
        "motion_objective": inference.REQUIRED_MOTION_OBJECTIVE,
        "motion_representation": inference.REQUIRED_MOTION_REPRESENTATION,
        "copy_loss_weight": 0.0,
        "boundary_gauge_loss_weight": 0.0,
        "boundary_gauge": (
            "zero_first_latent_phase_of_raw_predicted_clean_delta"
        ),
        "bridge_fractions": [0.0, 1.0],
        "bridge_consistency_weight": (
            inference.REQUIRED_BRIDGE_CONSISTENCY_WEIGHT
        ),
        "causal_ema_decay": None,
        "charbonnier_scale": 0.1,
        "high_noise_floor": 1.0,
        "high_noise_power": 2.0,
        "inference_sigma_schedule_sha256": (
            inference.sigma_strata.SCHEDULE_SHA256
        ),
        "inference_sigma_selector": "absolute_global_step_mod_40",
        "routing_file_sha256": "d" * 64,
        "expected_routing_jsonl_sha256": "d" * 64,
        "eligible_route_stream_count": 359,
        "noop_instruction_sha256": noop_sha256,
    }
    immutable = {
        "value": value,
        "digest": inference.trainer.object_sha256(value),
    }
    supervision = {
        "inference_conditions": ["source_video", "edit_instruction"],
        "target_used_as_condition": False,
        "external_mask_track_pose_trajectory": False,
        "paired_action_noop_forward_every_optimizer_step": True,
        "action_noop_forwards_per_optimizer_step": 4,
        "counterfactual_noop_forward": True,
        "branch_state_mode": inference.REQUIRED_BRANCH_STATE_MODE,
        "exact_same_noisy_query": True,
        "minimum_training_sigma": 0.1,
        "inverse_sigma_weight_floor": (
            inference.REQUIRED_INVERSE_SIGMA_WEIGHT_FLOOR
        ),
        "only_text_condition_differs": True,
        "shared_source_posterior_mode": True,
        "shared_sigma": True,
        "shared_diffusion_noise": True,
        "unreviewed_full_target_weight": 0.0,
        "clean_reconstruction_formula": "x_clean = y - sigma * velocity",
        "clean_reconstruction_numeric_program": (
            "fp32_noisy_minus_cpu_fp32_0d_sigma_times_native_bf16_velocity"
        ),
        "training_sigma_representation": "cpu_fp32_0d",
        "branch_prediction_dtype_before_clean_reconstruction": "bfloat16",
        "predicted_clean_delta_formula": "-sigma * (v_action - v_noop)",
        "target_clean_delta_formula": "executable_target_clean - source_clean",
        "target_projection": inference.REQUIRED_TARGET_PROJECTION,
        "target_projection_idempotent": True,
        "motion_loss_multiplier": "1 / sigma",
        "copy_boundary_loss_multiplier": "not_enabled",
        "boundary_gauge_enabled": False,
        "boundary_gauge_loss_weight": 0.0,
        "boundary_gauge_field": (
            "raw_predicted_action_minus_noop_clean_field"
        ),
        "boundary_gauge_target": "zero_first_latent_phase",
        "boundary_gauge_uses_target_appearance": False,
        "motion_objective": inference.REQUIRED_MOTION_OBJECTIVE,
        "causal_boundary_quotient_enabled": True,
        "causal_boundary_projection_enabled": True,
        "temporal_quotient_enabled": False,
        "causal_boundary_gauge_loss_weight": 0.0,
        "copy_calibration_enabled": False,
        "copy_calibration_weight": 0.0,
        "bridge_endpoints": [0.0, 1.0],
        "bridge_consistency_enabled": True,
        "bridge_consistency_weight": (
            inference.REQUIRED_BRIDGE_CONSISTENCY_WEIGHT
        ),
        "bridge_query_formula": (
            "y_beta=(1-sigma)*((1-beta)*source+beta*executable_target)"
            "+sigma*epsilon"
        ),
        "causal_ema_enabled": False,
        "causal_ema_decay": None,
        "charbonnier_scale": 0.1,
        "inference_sigma_stratification": "exact_40_step_flow_shift_5_cycle",
        "inference_sigma_schedule_sha256": (
            inference.sigma_strata.SCHEDULE_SHA256
        ),
    }
    receipt = {
        "schema_version": delta_train.RECEIPT_SCHEMA,
        "method": delta_train.METHOD_NAME,
        "global_step": 40,
        "immutable_contract": immutable,
        "bernini_commit": inference.trainer.BERNINI_OFFICIAL_COMMIT,
        "veomni_commit": inference.trainer.VEOMNI_TESTED_COMMIT,
        "checkpoint": {
            "tree_sha256": inference.trainer.CHECKPOINT_TREE_SHA256,
        },
        "supervision": supervision,
        "inference_sigma_strata": (
            inference.sigma_strata.build_sigma_strata_receipt(
                completed_optimizer_steps=40
            )
        ),
        "dataset": {
            "rows": 644,
            "routing": {
                "default_tier": "reject",
                "explicit_route_counts": {
                    "full_pair": 0,
                    "motion_only": 359,
                    "reject": 285,
                },
                "file_sha256": "d" * 64,
            },
        },
        "adapter": {
            "rank": inference.trainer.LORA_RANK,
            "alpha": inference.trainer.LORA_ALPHA,
            "scope": scope,
            "target_module_count": len(targets),
            "target_modules": targets,
            "target_modules_sha256": inference.trainer.object_sha256(targets),
            "initialization_digest": "5" * 64,
            "checkpoint_parameter_digest": "c" * 64,
        },
        "distributed": {
            "world_size": 4,
            "ulysses_size": 4,
            "same_pair_all_ranks": True,
            "explicit_lora_gradient_all_reduce": True,
        },
        "transformers_version": "4.51.3",
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
    }
    receipt["receipt_digest"] = inference.trainer.object_sha256(receipt)
    config = {
        "peft_type": "LORA",
        "r": inference.trainer.LORA_RANK,
        "lora_alpha": inference.trainer.LORA_ALPHA,
        "lora_dropout": 0.0,
        "bias": "none",
        "modules_to_save": None,
        "use_dora": False,
        "use_rslora": False,
        "target_modules": targets,
    }
    return config, receipt


def _redigest(receipt: dict) -> None:
    receipt.pop("receipt_digest", None)
    receipt["receipt_digest"] = inference.trainer.object_sha256(receipt)


class C2FRLoRAContractTests(unittest.TestCase):
    def test_cli_adds_only_adapter_to_frozen_81f_contract(self) -> None:
        parser = inference.build_parser()
        destinations = {action.dest for action in parser._actions}
        self.assertIn("adapter_checkpoint", destinations)
        self.assertTrue(
            {"source_video", "instruction", "alpha", "energy_coverage"}
            <= destinations
        )
        self.assertTrue(
            destinations.isdisjoint(
                {
                    "target_video",
                    "mask",
                    "track",
                    "flow",
                    "pose",
                    "trajectory",
                    "planner_checkpoint",
                    "noop_instruction",
                    "first_frame_anchor",
                }
            )
        )
        args = parser.parse_args(
            [
                "--bernini-root",
                "/b",
                "--veomni-root",
                "/v",
                "--checkpoint",
                "/c",
                "--adapter-checkpoint",
                "/a",
                "--source-video",
                "/source.mp4",
                "--instruction",
                "move",
                "--output",
                "/out.mp4",
                "--method-source-revision",
                SHA1,
                "--method-source-archive-sha256",
                SHA256,
            ]
        )
        self.assertEqual(args.num_inference_steps, 40)
        self.assertEqual(inference.ADAPTER_SCALE, 1.0)
        inference.validate_cli(_args())
        for overrides in (
            {"alpha": 0.5},
            {"alpha": 1.5},
            {"max_generate_fraction": 0.08},
            {"energy_coverage": 0.7},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(
                inference.C2FRLoRAInferenceError
            ):
                inference.validate_cli(_args(**overrides))

    def test_v4_projected_bridge_robust_q0_cross_q_adapter_is_accepted(self) -> None:
        config, receipt = _valid_metadata()
        self.assertEqual(
            receipt["schema_version"],
            "bernini-r-1p3b-c2fr-lora-receipt-v4",
        )
        sigma_receipt = receipt["inference_sigma_strata"]
        self.assertEqual(sigma_receipt["complete_cycles"], 1)
        self.assertEqual(sigma_receipt["partial_cycle_steps"], 0)
        self.assertEqual(
            sigma_receipt["histogram_by_schedule_index"], [1] * 40
        )
        self.assertEqual(
            sigma_receipt["schedule"]["schedule_sha256"],
            inference.sigma_strata.SCHEDULE_SHA256,
        )
        identity = inference.validate_same_state_training_adapter_contract(
            config, receipt
        )
        self.assertEqual(
            identity["branch_state_mode"], "source_target_bridge_clean_field"
        )
        self.assertEqual(identity["scope"], "cross_q")
        self.assertEqual(len(identity["targets"]), 30)
        self.assertEqual(identity["global_step"], 40)
        self.assertEqual(identity["minimum_training_sigma"], 0.1)
        self.assertEqual(
            identity["inverse_sigma_weight_floor"],
            inference.REQUIRED_INVERSE_SIGMA_WEIGHT_FLOOR,
        )
        self.assertEqual(identity["boundary_gauge_loss_weight"], 0.0)
        self.assertEqual(
            identity["bridge_consistency_weight"],
            inference.REQUIRED_BRIDGE_CONSISTENCY_WEIGHT,
        )
        self.assertEqual(identity["charbonnier_scale"], 0.1)
        self.assertEqual(
            identity["inference_sigma_schedule_sha256"],
            inference.sigma_strata.SCHEDULE_SHA256,
        )
        self.assertEqual(
            identity["motion_representation"],
            inference.REQUIRED_MOTION_REPRESENTATION,
        )
        self.assertLessEqual(
            identity["minimum_training_sigma"],
            inference.OFFICIAL_LAST_POSITIVE_SIGMA,
        )

    def test_v1_v2_and_v3_receipts_are_rejected_even_with_valid_digest(self) -> None:
        for schema in (
            "bernini-r-1p3b-cdf-lora-receipt-v1",
            "bernini-r-1p3b-c2fr-lora-receipt-v2",
            "bernini-r-1p3b-c2fr-lora-receipt-v3",
        ):
            config, receipt = _valid_metadata()
            receipt["schema_version"] = schema
            _redigest(receipt)
            with self.subTest(schema=schema), self.assertRaisesRegex(
                inference.C2FRLoRAInferenceError, "v4"
            ):
                inference.validate_same_state_training_adapter_contract(
                    config, receipt
                )

    def test_pre_v4_motion_representations_are_rejected(self) -> None:

        for location, key, value in (
            (
                "immutable",
                "boundary_gauge",
                "zero_temporal_mean_of_predicted_action_minus_noop_clean_field",
            ),
            (
                "immutable",
                "motion_representation",
                "source-relative-causal-boundary-multilag-v1",
            ),
            ("supervision", "boundary_gauge_target", "zero_temporal_mean"),
            ("supervision", "temporal_quotient_enabled", True),
            ("supervision", "causal_ema_enabled", True),
            ("supervision", "causal_boundary_projection_enabled", False),
        ):
            config, receipt = _valid_metadata()
            if location == "immutable":
                contract = receipt["immutable_contract"]
                contract["value"][key] = value
                contract["digest"] = inference.trainer.object_sha256(
                    contract["value"]
                )
            else:
                receipt["supervision"][key] = value
            _redigest(receipt)
            with self.subTest(location=location, key=key), self.assertRaises(
                inference.C2FRLoRAInferenceError
            ):
                inference.validate_same_state_training_adapter_contract(
                    config, receipt
                )

    def test_two_clean_path_or_formula_drift_is_rejected(self) -> None:
        for location, key, value in (
            ("immutable", "branch_state_mode", "separate_clean_paths"),
            ("immutable", "bridge_fractions", [0.0]),
            ("immutable", "bridge_consistency_weight", 0.0),
            ("immutable", "causal_ema_decay", 0.25),
            ("immutable", "target_projection", "raw_target-source"),
            ("immutable", "target_projection_idempotent", False),
            ("immutable", "exact_same_noisy_query", False),
            ("immutable", "predicted_clean_delta_formula", "v_action-v_noop"),
            (
                "immutable",
                "clean_reconstruction_numeric_program",
                "fp32_noisy_minus_gpu_fp32_sigma_times_fp32_velocity",
            ),
            ("immutable", "training_sigma_representation", "gpu_fp32_1d"),
            (
                "immutable",
                "branch_prediction_dtype_before_clean_reconstruction",
                "float32",
            ),
            ("immutable", "inverse_sigma_weight_floor", 0.5),
            ("supervision", "only_text_condition_differs", False),
            ("supervision", "shared_sigma", False),
            ("supervision", "bridge_consistency_enabled", False),
            ("supervision", "causal_ema_decay", 0.25),
            ("supervision", "target_projection", "raw_target-source"),
        ):
            config, receipt = _valid_metadata()
            if location == "immutable":
                contract = receipt["immutable_contract"]
                contract["value"][key] = value
                contract["digest"] = inference.trainer.object_sha256(
                    contract["value"]
                )
            else:
                receipt["supervision"][key] = value
            _redigest(receipt)
            with self.subTest(location=location, key=key), self.assertRaises(
                inference.C2FRLoRAInferenceError
            ):
                inference.validate_same_state_training_adapter_contract(
                    config, receipt
                )

    def test_fewer_than_40_steps_and_inexact_sigma_strata_are_rejected(self) -> None:
        config, receipt = _valid_metadata()
        receipt["global_step"] = 39
        receipt["inference_sigma_strata"] = (
            inference.sigma_strata.build_sigma_strata_receipt(
                completed_optimizer_steps=39
            )
        )
        _redigest(receipt)
        with self.assertRaisesRegex(
            inference.C2FRLoRAInferenceError, "complete 40-sigma cycle"
        ):
            inference.validate_same_state_training_adapter_contract(config, receipt)

        config, receipt = _valid_metadata()
        receipt["inference_sigma_strata"]["selection"]["formula"] = (
            "random training sigma"
        )
        _redigest(receipt)
        with self.assertRaisesRegex(
            inference.C2FRLoRAInferenceError, "sigma strata"
        ):
            inference.validate_same_state_training_adapter_contract(config, receipt)

    def test_training_receipt_requires_hash_bound_strict_359_cohort(self) -> None:
        mutations = (
            ("rows", 643),
            ("default_tier", "motion_only"),
            (
                "explicit_route_counts",
                {"full_pair": 0, "motion_only": 358, "reject": 286},
            ),
            ("routing_file_sha256", "e" * 64),
            ("eligible_route_stream_count", 358),
        )
        for key, changed in mutations:
            config, receipt = _valid_metadata()
            if key == "rows":
                receipt["dataset"][key] = changed
            elif key in ("default_tier", "explicit_route_counts"):
                receipt["dataset"]["routing"][key] = changed
            else:
                receipt["immutable_contract"]["value"][key] = changed
                receipt["immutable_contract"]["digest"] = (
                    inference.trainer.object_sha256(
                        receipt["immutable_contract"]["value"]
                    )
                )
            _redigest(receipt)
            with self.subTest(key=key), self.assertRaisesRegex(
                inference.C2FRLoRAInferenceError, "strict-359"
            ):
                inference.validate_same_state_training_adapter_contract(
                    config, receipt
                )

    def test_training_must_cover_final_positive_unipc_sigma(self) -> None:
        config, receipt = _valid_metadata()
        receipt["immutable_contract"]["value"]["minimum_training_sigma"] = 0.25
        receipt["supervision"]["minimum_training_sigma"] = 0.25
        receipt["immutable_contract"]["digest"] = inference.trainer.object_sha256(
            receipt["immutable_contract"]["value"]
        )
        _redigest(receipt)
        with self.assertRaisesRegex(
            inference.C2FRLoRAInferenceError, "40-step shift-5 schedule"
        ):
            inference.validate_same_state_training_adapter_contract(config, receipt)

    def test_training_receipt_requires_four_rank_explicit_all_reduce(self) -> None:
        for key, value in (
            ("world_size", 1),
            ("ulysses_size", 1),
            ("same_pair_all_ranks", False),
            ("explicit_lora_gradient_all_reduce", False),
        ):
            config, receipt = _valid_metadata()
            receipt["distributed"][key] = value
            _redigest(receipt)
            with self.subTest(key=key), self.assertRaisesRegex(
                inference.C2FRLoRAInferenceError, "four-rank"
            ):
                inference.validate_same_state_training_adapter_contract(
                    config, receipt
                )

    def test_receipt_module_hash_and_runtime_scope_are_exact(self) -> None:
        config, receipt = _valid_metadata()
        receipt["adapter"]["target_modules"] = receipt["adapter"][
            "target_modules"
        ][:-1]
        receipt["adapter"]["target_module_count"] -= 1
        receipt["adapter"]["target_modules_sha256"] = (
            inference.trainer.object_sha256(receipt["adapter"]["target_modules"])
        )
        receipt["immutable_contract"]["value"]["target_modules"] = receipt[
            "adapter"
        ]["target_modules"]
        receipt["immutable_contract"]["digest"] = inference.trainer.object_sha256(
            receipt["immutable_contract"]["value"]
        )
        _redigest(receipt)
        with self.assertRaises(inference.C2FRLoRAInferenceError):
            inference.validate_same_state_training_adapter_contract(config, receipt)

        config, receipt = _valid_metadata(scope="cross_q_out")
        with self.assertRaisesRegex(
            inference.C2FRLoRAInferenceError, "cross_q-only"
        ):
            inference.validate_same_state_training_adapter_contract(config, receipt)

    def test_compact_peft_suffixes_must_cover_every_receipt_target(self) -> None:
        config, receipt = _valid_metadata()
        config["target_modules"] = ["attn2.to_q"]
        identity = inference.validate_same_state_training_adapter_contract(
            config, receipt
        )
        self.assertEqual(
            identity["serialized_target_modules"],
            ["attn2.to_q"],
        )
        config["target_modules"] = [
            "diff_dec.transformer.blocks.0.attn2.to_q"
        ]
        with self.assertRaisesRegex(
            inference.C2FRLoRAInferenceError, "do not cover"
        ):
            inference.validate_same_state_training_adapter_contract(config, receipt)

    def test_dora_and_rslora_are_rejected(self) -> None:
        for flag in ("use_dora", "use_rslora"):
            config, receipt = _valid_metadata()
            config[flag] = True
            with self.subTest(flag=flag), self.assertRaises(
                inference.C2FRLoRAInferenceError
            ):
                inference.validate_same_state_training_adapter_contract(
                    config, receipt
                )

    def test_training_receipt_requires_exact_checkpoint_parameter_digest(self) -> None:
        config, receipt = _valid_metadata()
        receipt["adapter"].pop("checkpoint_parameter_digest")
        _redigest(receipt)
        with self.assertRaisesRegex(
            inference.C2FRLoRAInferenceError, "checkpoint parameter digest"
        ):
            inference.validate_same_state_training_adapter_contract(config, receipt)

    def test_frozen_sparse_router_remains_an_isolated_v2_control(self) -> None:
        contract = sparse_router.runtime_contract()
        self.assertEqual(
            contract["same_state_input"],
            "raw_action_condition_clean_minus_raw_noop_condition_clean",
        )
        self.assertEqual(
            contract["training_alignment"],
            "exact_-sigma*(v_action-v_noop)_clean_delta",
        )
        self.assertEqual(
            contract["official_apg_role"],
            "parity_certificate_only_not_routed_delta",
        )
        self.assertEqual(
            sparse_router.GeneratorNativeSparseRouterConfig().static_delta_retention,
            0.0,
        )

    def test_receipt_binds_adapter_and_preserves_source_only_inputs(self) -> None:
        config, training_receipt = _valid_metadata()
        identity = inference.validate_same_state_training_adapter_contract(
            config, training_receipt
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            bundle = SimpleNamespace(
                checkpoint_root=root / "checkpoint-00000040",
                adapter_config_path=root / "adapter/adapter_config.json",
                adapter_model_path=root / "adapter/adapter_model.safetensors",
                training_receipt_path=root / "receipt.json",
            )
            receipt = inference.build_inference_receipt(
                args=_args(),
                source_path=root / "source.mp4",
                source_sha256="6" * 64,
                source_metadata={"source_derived_bucket_hw": [480, 496]},
                output_path=root / "out.mp4",
                output_sha256="7" * 64,
                noop_identity={"frozen_t5": True},
                execution_trace={
                    "trace_digest": "8" * 64,
                    "runtime_unipc_schedule_audit": _runtime_schedule_audit(),
                },
                bernini_revision=inference.trainer.BERNINI_OFFICIAL_COMMIT,
                veomni_revision=inference.trainer.VEOMNI_TESTED_COMMIT,
                inference_file_hashes={},
                wan_diffusion_path=root / "bernini/models/wan_diffusion.py",
                wan_diffusion_sha256=inference.tri.PINNED_WAN_DIFFUSION_SHA256,
                runtime_versions={"peft": "0.15.2"},
                adapter_bundle=bundle,
                adapter_identity=identity,
                adapter_config_sha256="9" * 64,
                adapter_model_sha256="a" * 64,
                training_receipt_file_sha256="b" * 64,
                adapter_tensor_count=60,
                active_lora_module_count=30,
            )
        self.assertEqual(
            receipt["schema_version"],
            (
                "bernini-c2fr-projected-bridge-robust-q0-dense-lora-"
                "inference-receipt-v4"
            ),
        )
        self.assertTrue(receipt["base_model"]["base_weights_frozen"])
        self.assertTrue(receipt["base_model"]["adapter_loaded"])
        self.assertFalse(receipt["adapter"]["merged"])
        self.assertEqual(receipt["adapter"]["scale"], 1.0)
        self.assertEqual(receipt["adapter"]["adapter_model_sha256"], "a" * 64)
        self.assertTrue(receipt["adapter"]["strict_tensor_reload_equal"])
        self.assertTrue(
            receipt["adapter"][
                "parameter_digest_verified_after_safetensors_reload"
            ]
        )
        self.assertEqual(receipt["adapter"]["checkpoint_parameter_digest"], "c" * 64)
        self.assertNotIn("router_config", receipt["sampling"])
        self.assertEqual(
            receipt["sampling"]["routing_contract"]["execution_field"],
            "dense_causal_boundary_action_minus_noop_clean_field",
        )
        self.assertFalse(
            receipt["sampling"]["routing_contract"][
                "temporal_mean_subtraction_at_execution"
            ]
        )
        self.assertTrue(
            receipt["sampling"]["routing_contract"][
                "callback_clean_first_phase_bit_exact"
            ]
        )
        self.assertFalse(
            receipt["sampling"]["routing_contract"][
                "final_generated_latent_first_phase_bit_exact_claimed"
            ]
        )
        self.assertEqual(
            receipt["input"]["accepted_external_conditions"],
            ["source_video", "edit_instruction"],
        )
        self.assertFalse(receipt["input"]["target_accessed_by_inference"])
        alignment = receipt["training_inference_alignment"]
        self.assertFalse(alignment["raw_condition_field_routed"])
        self.assertTrue(
            alignment["causal_boundary_projected_field_executed"]
        )
        self.assertTrue(
            alignment["closed_loop_query_gap_mitigated_by_endpoint_consistency"]
        )
        self.assertFalse(alignment["closed_loop_query_gap_proven_closed"])
        self.assertEqual(alignment["bridge_endpoints"], [0.0, 1.0])
        self.assertEqual(
            alignment["clean_reconstruction_numeric_program"],
            "fp32_noisy_minus_cpu_fp32_0d_sigma_times_native_bf16_velocity",
        )
        self.assertEqual(alignment["training_sigma_representation"], "cpu_fp32_0d")
        self.assertEqual(
            alignment["branch_prediction_dtype_before_clean_reconstruction"],
            "bfloat16",
        )
        self.assertEqual(
            alignment["training_motion_representation"],
            inference.REQUIRED_MOTION_REPRESENTATION,
        )
        self.assertEqual(
            alignment["target_projection"],
            inference.REQUIRED_TARGET_PROJECTION,
        )
        self.assertEqual(
            alignment["training_boundary_gauge"],
            "zero_first_latent_phase_of_raw_predicted_clean_delta",
        )
        self.assertEqual(alignment["training_boundary_gauge_loss_weight"], 0.0)
        self.assertEqual(
            alignment["inference_field_execution"],
            "dense_causal_boundary_action_minus_noop_clean_field",
        )
        self.assertEqual(
            alignment["training_sigma_schedule_sha256"],
            inference.sigma_strata.SCHEDULE_SHA256,
        )
        self.assertFalse(alignment["temporal_mean_subtraction_at_execution"])
        self.assertFalse(alignment["temporal_low_pass_at_execution"])
        self.assertTrue(alignment["first_phase_exact_zero_by_projection"])
        self.assertTrue(alignment["dense_training_and_inference_support_operator"])
        self.assertFalse(alignment["binary_support_operator"])
        self.assertFalse(alignment["same_noisy_state_gap"])
        self.assertFalse(alignment["motion_representation_gap"])
        self.assertFalse(alignment["support_execution_gap"])
        self.assertTrue(
            alignment["training_covers_every_positive_inference_sigma"]
        )
        self.assertEqual(
            alignment["official_last_positive_sigma"],
            inference.sigma_strata.PINNED_POSITIVE_SIGMAS[-1],
        )
        self.assertTrue(alignment["runtime_schedule_bit_exact"])
        self.assertEqual(
            receipt["sampling"]["runtime_unipc_schedule_audit"],
            _runtime_schedule_audit(),
        )
        self.assertIn(
            "inference_sigma_strata.py", receipt["method_files_sha256"]
        )
        candidate = copy.deepcopy(receipt)
        declared = candidate.pop("receipt_digest")
        self.assertEqual(inference.trainer.object_sha256(candidate), declared)

    def test_runtime_trace_requires_every_exact_timestep_and_sigma_bit(self) -> None:
        branch_records = []
        dense_records = []
        for index, (timestep, sigma) in enumerate(
            zip(
                inference.sigma_strata.PINNED_TIMESTEPS,
                inference.sigma_strata.PINNED_POSITIVE_SIGMAS,
            )
        ):
            branch_records.append(
                inference.tri.TriBranchStepRecord(
                    step_index=index,
                    timestep=float(timestep),
                    sigma=float(sigma),
                    model_id="transformer_1",
                    transformer_forwards=3,
                    shared_negative_forwards=1,
                    action_forwards=1,
                    noop_forwards=1,
                    original_scheduler_calls=1,
                    callback_correction_rms=0.0,
                    raw_action_noop_delta_rms=0.0,
                    guided_action_noop_delta_rms=0.0,
                    guided_action_noop_delta_l2=0.0,
                    action_noop_exact_parity=False,
                    effective_guidance_scale=inference.frozen.base.OMEGA_TEXT,
                    official_action_parity_rms_error=0.0,
                    official_action_parity_max_abs_error=0.0,
                    official_action_exact_parity=True,
                    sample_dtype="torch.float32",
                    branch_velocity_dtype="torch.bfloat16",
                    official_model_output_dtype="torch.bfloat16",
                )
            )
            dense_records.append(
                inference.DenseCausalBoundaryStepRecord(
                    step_index=index,
                    timestep=float(timestep),
                    sigma=float(sigma),
                    raw_field_rms=0.0,
                    causal_field_rms=0.0,
                    executed_change_rms=0.0,
                    first_phase_max_abs=0.0,
                )
            )
        tri_trace = inference.tri.TriBranchTrace(
            records=branch_records, sample_calls=1
        )
        dense_trace = inference.DenseCausalBoundaryExecutionTrace(
            alpha=1.0, records=dense_records
        )
        payload = inference.validate_dense_execution_trace(
            tri_trace,
            dense_trace,
            runtime_schedule_audit=_runtime_schedule_audit(),
        )
        self.assertEqual(
            payload["runtime_unipc_schedule_audit"],
            _runtime_schedule_audit(),
        )

        wrong_records = list(branch_records)
        wrong_records[7] = replace(
            wrong_records[7], sigma=wrong_records[7].sigma + 1.0e-6
        )
        with self.assertRaisesRegex(
            inference.C2FRLoRAInferenceError, "schedule index 7"
        ):
            inference.validate_dense_execution_trace(
                inference.tri.TriBranchTrace(
                    records=wrong_records, sample_calls=1
                ),
                dense_trace,
                runtime_schedule_audit=_runtime_schedule_audit(),
            )

        wrong_audit = _runtime_schedule_audit()
        wrong_audit["positive_sigmas_float32_be_hex"][3] = "00000000"
        with self.assertRaisesRegex(
            inference.C2FRLoRAInferenceError, "schedule audit differs"
        ):
            inference.validate_dense_execution_trace(
                tri_trace,
                dense_trace,
                runtime_schedule_audit=wrong_audit,
            )

    def test_main_loads_adapter_before_the_original_unipc_hook(self) -> None:
        source = inspect.getsource(inference.main)
        load_index = source.index("_strict_load_same_state_adapter")
        hook_index = source.index("tri.tri_branch_unipc_hook")
        sample_index = source.index("renderer.sample")
        self.assertLess(load_index, hook_index)
        self.assertLess(hook_index, sample_index)
        self.assertNotIn("merge_and_unload", source)
        self.assertNotIn("apply_adapter_strength", source)
        self.assertIn("TracedDenseCausalBoundaryCallback", source)
        self.assertIn("validate_dense_execution_trace", source)
        self.assertIn("audit_runtime_unipc_schedule", source)
        self.assertIn("initialize=True", source)
        self.assertIn("initialize=False", source)
        self.assertNotIn("TracedGeneratorNativeSparseCallback", source)

    def test_auh_launcher_is_four_gpu_hash_bound_and_uses_new_entry(self) -> None:
        launcher = (
            METHOD_ROOT / "spt_v2/scripts/auh_infer_c2fr_lora.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("--nproc_per_node=4", launcher)
        self.assertIn("--gres=gpu:mi210:4", launcher)
        self.assertIn("infer_c2fr_lora.py", launcher)
        self.assertIn("--adapter-checkpoint", launcher)
        self.assertIn("adapter_model.safetensors", launcher)
        self.assertIn("receipt.json", launcher)
        self.assertIn("BERNINI_C2FR_RANK_CACHE_ROOT", launcher)
        self.assertIn("dense_causal_boundary=true", launcher)
        self.assertIn("temporal_low_pass=false", launcher)
        self.assertIn("binary_support=false", launcher)
        self.assertNotIn("BERNINI_C2FR_LORA_GENERATE_CAP", launcher)
        self.assertNotIn("BERNINI_C2FR_LORA_ENERGY_COVERAGE", launcher)
        self.assertNotIn("--noop-instruction", launcher)
        self.assertNotIn("--planner-checkpoint", launcher)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is unavailable")
class DenseCausalBoundaryCallbackTests(unittest.TestCase):
    def test_dense_callback_executes_robust_q0_without_temporal_low_pass(self) -> None:
        layout = inference.tri.PackedLatentLayout.from_spatial_shape(
            (1, 16, 21, 10, 10)
        )
        phase_shape = (1, 21, 5, 5, 64)
        source = torch.randn(phase_shape)
        noop = torch.randn(phase_shape)
        raw = torch.full(phase_shape, 1.5)
        raw[:, 9:, 2, 3, :] = 4.0
        action = noop + raw
        action_spatial = sparse_router.phase_video_to_spatial(
            action, layout=layout
        )
        noop_spatial = sparse_router.phase_video_to_spatial(noop, layout=layout)
        zeros = torch.zeros_like(action_spatial)
        fields = inference.tri.CleanFieldStep(
            step_index=0,
            timestep=900.0,
            sigma=0.9,
            model_id="transformer_1",
            noisy=zeros,
            negative_velocity=zeros,
            action_velocity=zeros,
            noop_velocity=zeros,
            negative_clean=zeros,
            action_condition_clean=action_spatial,
            noop_condition_clean=noop_spatial,
            action_guided_clean=action_spatial,
            noop_guided_clean=noop_spatial,
            action_delta_clean=action_spatial - noop_spatial,
        )
        callback = inference.TracedDenseCausalBoundaryCallback(
            source_clean=source,
            layout=layout,
            alpha=1.0,
        )
        result = callback(fields)
        result_phase = sparse_router.spatial_to_phase_video(result, layout=layout)
        computed_raw = action - noop
        expected_field = sparse_router.causal_boundary_projection(computed_raw)
        expected = source + expected_field
        self.assertTrue(torch.equal(result_phase, expected))
        self.assertTrue(
            torch.equal(result_phase[:, :1], source[:, :1])
        )
        self.assertTrue(
            torch.equal(
                expected_field,
                computed_raw - computed_raw[:, :1],
            )
        )
        self.assertEqual(len(callback.trace.records), 1)
        self.assertEqual(callback.trace.records[0].first_phase_max_abs, 0.0)
        contract = inference.dense_causal_boundary_runtime_contract()
        self.assertFalse(contract["temporal_low_pass_at_execution"])
        self.assertEqual(
            contract["support_operator"],
            "dense_generator_native_field_no_binary_gate",
        )
        self.assertTrue(contract["first_phase_exact_zero"])
        self.assertTrue(contract["callback_clean_first_phase_bit_exact"])
        self.assertFalse(
            contract["final_generated_latent_first_phase_bit_exact_claimed"]
        )

        with self.assertRaisesRegex(
            inference.C2FRLoRAInferenceError, "alpha exactly 1.0"
        ):
            inference.TracedDenseCausalBoundaryCallback(
                source_clean=source,
                layout=layout,
                alpha=0.5,
            )


@unittest.skipIf(torch is None, "torch is unavailable")
class AdapterParameterDigestTests(unittest.TestCase):
    def test_reloaded_parameter_digest_is_byte_exact(self) -> None:
        class TinyAdapter(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer = torch.nn.Module()
                self.layer.lora_A = torch.nn.ModuleDict(
                    {"default": torch.nn.Linear(2, 1, bias=False)}
                )
                with torch.no_grad():
                    self.layer.lora_A["default"].weight.copy_(
                        torch.tensor([[1.0, 2.0]])
                    )

        model = TinyAdapter()
        expected = delta_train._checkpoint_parameter_digest(
            [
                (name, parameter)
                for name, parameter in model.named_parameters()
                if inference.trainer.is_lora_parameter_name(name)
            ]
        )
        self.assertEqual(
            inference.validate_loaded_adapter_parameter_digest(
                model, expected_digest=expected
            ),
            expected,
        )
        with torch.no_grad():
            model.layer.lora_A["default"].weight.add_(1.0)
        with self.assertRaisesRegex(
            inference.C2FRLoRAInferenceError, "differs"
        ):
            inference.validate_loaded_adapter_parameter_digest(
                model, expected_digest=expected
            )

    def test_strict_scope_scan_accepts_real_peft_moduledict_protocol(self) -> None:
        targets = motion.select_lora_scope(
            inference._audited_attention_projection_names(), "cross_q"
        )

        class FakeLoRALayer:
            def __init__(self) -> None:
                self.lora_A = torch.nn.ModuleDict(
                    {"default": torch.nn.Linear(1, 1, bias=False)}
                )
                self.lora_B = torch.nn.ModuleDict(
                    {"default": torch.nn.Linear(1, 1, bias=False)}
                )
                self.scaling = {"default": 1.0}

        class FakePeftModel:
            def __init__(self) -> None:
                self.layers = [(target, FakeLoRALayer()) for target in targets]
                self.named = []
                for target, layer in self.layers:
                    self.named.extend(
                        [
                            (
                                f"base_model.model.{target}.lora_A.default.weight",
                                layer.lora_A["default"].weight,
                            ),
                            (
                                f"base_model.model.{target}.lora_B.default.weight",
                                layer.lora_B["default"].weight,
                            ),
                        ]
                    )
                for _, parameter in self.named:
                    parameter.requires_grad_(False)

            def named_modules(self):
                return iter(self.layers)

            def named_parameters(self):
                return iter(self.named)

            def parameters(self):
                return (parameter for _, parameter in self.named)

            def eval(self):
                return self

        model = FakePeftModel()
        digest = delta_train._checkpoint_parameter_digest(model.named)
        identity = {
            "targets": targets,
            "checkpoint_parameter_digest": digest,
        }
        config = {"target_modules": targets}
        bundle = SimpleNamespace(
            adapter_dir=Path("/adapter"),
            adapter_model_path=Path("/adapter/model"),
        )
        with mock.patch.object(
            inference.adapter_loader,
            "_strict_load_adapter",
            return_value=(model, 2 * len(targets)),
        ):
            loaded, tensor_count, module_count = (
                inference._strict_load_same_state_adapter(
                    base_model=object(),
                    bundle=bundle,
                    adapter_config=config,
                    identity=identity,
                )
            )
        self.assertIs(loaded, model)
        self.assertEqual(tensor_count, 60)
        self.assertEqual(module_count, 30)


if __name__ == "__main__":
    unittest.main()
