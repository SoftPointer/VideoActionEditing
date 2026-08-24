#!/usr/bin/env python3
"""Fail-closed contracts for Bernini v5 prior-tangent inference."""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
import sys
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_prior_tangent_lora as inference  # noqa: E402


SHA1 = "1" * 40
SHA256 = "2" * 64


def _args(**overrides):
    values = {
        "bernini_root": "/bernini",
        "veomni_root": "/veomni",
        "checkpoint": "/checkpoint",
        "adapter_checkpoint": "/adapter",
        "source_video": "/source.mp4",
        "instruction": "Make the dog pick up the bone.",
        "output": "/output.mp4",
        "num_inference_steps": 40,
        "seed": 2027,
        "alpha": 1.0,
        "max_generate_fraction": inference.frozen.DEFAULT_GENERATE_CAP,
        "energy_coverage": inference.frozen.DEFAULT_ENERGY_COVERAGE,
        "expected_bernini_commit": inference.trainer.BERNINI_OFFICIAL_COMMIT,
        "expected_veomni_commit": inference.trainer.VEOMNI_TESTED_COMMIT,
        "expected_checkpoint_tree_sha256": inference.trainer.CHECKPOINT_TREE_SHA256,
        "method_source_revision": SHA1,
        "method_source_archive_sha256": SHA256,
        "execution_arm": "main",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _valid_adapter_and_receipt():
    targets = inference.motion.select_lora_scope(
        inference._audited_attention_projection_names(), "cross_q"
    )
    gamma = list(inference.tangent.correction_gamma_schedule())
    value = {
        "method": inference.METHOD_NAME,
        "schema_version": inference.v5_train.RECEIPT_SCHEMA,
        "method_source_revision": SHA1,
        "method_source_archive_sha256": SHA256,
        "bernini_commit": inference.trainer.BERNINI_OFFICIAL_COMMIT,
        "veomni_commit": inference.trainer.VEOMNI_TESTED_COMMIT,
        "checkpoint_tree_sha256": inference.trainer.CHECKPOINT_TREE_SHA256,
        "checkpoint_path": "/checkpoint",
        "dataset_signature": "dataset",
        "dataset_summary_sha256": "3" * 64,
        "dataset_index_sha256": "4" * 64,
        "routing_digest": "5" * 64,
        "routing_file_sha256": "6" * 64,
        "expected_routing_jsonl_sha256": "6" * 64,
        "eligible_route_count": 359,
        "eligible_route_stream_sha256": "7" * 64,
        "seed": 2027,
        "frames": 81,
        "latent_phases": 21,
        "learning_rate": inference.v5_train.LEARNING_RATE,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "lora_rank": 8,
        "lora_alpha": 8,
        "lora_scope": "cross_q",
        "target_modules": targets,
        "target_modules_sha256": inference.trainer.object_sha256(targets),
        "bridge_fractions": [0.0, 1.0],
        "bridge_query": "source_and_executable_target_same_epsilon_sigma_timestep",
        "branches_per_endpoint": [
            "base_negative_adapter_off_no_grad",
            "base_noop_adapter_off_no_grad",
            "base_action_adapter_off_no_grad",
            "adapted_action_adapter_on_grad",
        ],
        "forwards_per_endpoint": 4,
        "forwards_per_optimizer_step": 8,
        "base_apg": {
            "guidance_scale": 4.0,
            "eta": 0.5,
            "norm_threshold": 50.0,
            "momentum": 0.0,
            "negative_prompt_sha256": hashlib.sha256(
                inference.v5_train.DEFAULT_NEGATIVE_PROMPT.encode("utf-8")
            ).hexdigest(),
            "negative_tokenization": (
                "official_renderer_unconditional_verbatim"
            ),
            "clean_reconstruction": (
                "fp32_noisy_minus_cpu_fp32_0d_sigma_times_native_bf16_velocity"
            ),
        },
        "raw_prior": "raw_frozen_prior=base_guided_action-base_guided_noop",
        "prior": "causal_frozen_prior=Q0(raw_frozen_prior)",
        "adapter_correction": "Q0(adapted_guided_action-base_guided_action)",
        "teacher_correction": "Q0((target-source)-causal_frozen_prior)",
        "phase_zero_contract": (
            "executed_motion_exact_zero_source_exactly_preserved"
        ),
        "trust_region": {
            "kappa_parallel": 0.5,
            "kappa_perp": 0.15,
            "epsilon": 1.0e-6,
            "phase_dim": 1,
        },
        "gamma_schedule": gamma,
        "gamma_schedule_sha256": inference.trainer.object_sha256(gamma),
        "gamma_contract": (
            "0-23 full; 24-34 inclusive cosine taper "
            "(gamma24=1,gamma34=0); 35-39 exact causal frozen prior"
        ),
        "sigma_schedule": "exact_40_step_flow_shift_5_cycle",
        "sigma_schedule_sha256": inference.sigma_strata.SCHEDULE_SHA256,
        "sigma_selector": "absolute_global_step_mod_40",
        "minimum_training_sigma": 0.1,
        "inverse_sigma_weight_floor": inference.sigma_strata.PINNED_POSITIVE_SIGMAS[-1],
        "loss": {
            "field": 1.0,
            "bridge": 0.05,
            "late_replay": 0.10,
            "late_replay_gate": "1-gamma",
            "robust_distance": "charbonnier_scale_0.1",
            "outer_multiplier": "1/max(sigma,final_positive_unipc_sigma)",
        },
        "inference_conditions": ["source_video", "action_instruction"],
        "training_only_conditions": ["target_video"],
        "forbidden_inference_conditions": [
            "target_video",
            "mask",
            "track",
            "swept_tube",
            "pose",
            "trajectory",
            "optical_flow",
            "first_frame_anchor",
        ],
        "noop_instruction_sha256": hashlib.sha256(
            inference.motion.DEFAULT_NOOP_INSTRUCTION.encode("utf-8")
        ).hexdigest(),
    }
    immutable = {"value": value, "digest": inference.trainer.object_sha256(value)}
    supervision = {
        "method": inference.METHOD_NAME,
        "source_target_bridge": True,
        "four_branch_endpoint": True,
        "base_branches_adapter_disabled": True,
        "base_branches_no_grad": True,
        "adapted_action_only_trainable_forward": True,
        "causal_frozen_prior": "Q0(base_action-base_noop)",
        "executed_motion_phase_zero": "exact_zero",
        "official_apg_momentum": 0.0,
        "official_apg_guidance_scale": 4.0,
        "official_apg_eta": 0.5,
        "official_apg_norm_threshold": 50.0,
        "negative_prompt_sha256": inference.v5_train.NEGATIVE_PROMPT_SHA256,
        "negative_tokenization": "official_renderer_unconditional_verbatim",
        "field_loss_weight": 1.0,
        "bridge_loss_weight": 0.05,
        "late_replay_loss_weight": 0.10,
        "late_replay_gate": "1-gamma",
        "target_used_as_model_condition": False,
        "target_used_as_offline_teacher": True,
        "inference_conditions": ["source_video", "action_instruction"],
        "external_mask_track_flow_pose_trajectory": False,
        "post_video_acceptance": "pending",
        "production_claim_forbidden": True,
    }
    receipt = {
        "schema_version": inference.v5_train.RECEIPT_SCHEMA,
        "method": inference.METHOD_NAME,
        "global_step": 40,
        "immutable_contract": immutable,
        "bernini_commit": inference.trainer.BERNINI_OFFICIAL_COMMIT,
        "veomni_commit": inference.trainer.VEOMNI_TESTED_COMMIT,
        "checkpoint": {
            "path": "/checkpoint",
            "tree_sha256": inference.trainer.CHECKPOINT_TREE_SHA256,
        },
        "dataset": {
            "rows": 644,
            "routing": {
                "default_tier": "reject",
                "explicit_route_counts": {
                    "full_pair": 0,
                    "motion_only": 359,
                    "reject": 285,
                },
                "file_sha256": "6" * 64,
            },
        },
        "supervision": supervision,
        "inference_sigma_strata": inference.sigma_strata.build_sigma_strata_receipt(
            completed_optimizer_steps=40
        ),
        "adapter": {
            "rank": 8,
            "alpha": 8,
            "scope": "cross_q",
            "target_module_count": 30,
            "target_modules": targets,
            "target_modules_sha256": inference.trainer.object_sha256(targets),
            "initialization_digest": "8" * 64,
            "checkpoint_parameter_digest": "9" * 64,
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
        "r": 8,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "bias": "none",
        "modules_to_save": None,
        "use_dora": False,
        "use_rslora": False,
        "target_modules": targets,
    }
    return config, receipt


def _redigest(receipt):
    receipt.pop("receipt_digest", None)
    receipt["receipt_digest"] = inference.trainer.object_sha256(receipt)


class _FakeArray:
    def __init__(self, shape):
        self.shape = tuple(shape)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        product = 1
        for value in shape:
            product *= value
        original = 1
        for value in self.shape:
            original *= value
        if product != original:
            raise AssertionError("reshape changed element count")
        return _FakeArray(shape)


class PriorTangentInferencePureTests(unittest.TestCase):
    def test_auh_launcher_uses_cluster_compatible_private_scratch(self):
        launcher = (
            METHOD_ROOT / "scripts" / "auh_infer_prior_tangent_lora.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn('scratch_parent="${SLURM_TMPDIR:-/tmp}"', launcher)
        self.assertIn('task_scratch="$(mktemp -d ', launcher)
        self.assertNotIn("SLURM_TMPDIR must be set by Slurm", launcher)

    def test_cli_is_81f_40_step_and_arm_bound(self):
        parser = inference.build_parser()
        destinations = {action.dest for action in parser._actions}
        self.assertIn("execution_arm", destinations)
        self.assertNotIn("mask", destinations)
        self.assertNotIn("flow", destinations)
        args = _args()
        inference.validate_cli(args)
        self.assertEqual(args.execution_arm, "main")
        for arm in inference.EXECUTION_ARMS:
            inference.validate_cli(_args(execution_arm=arm))
        for arm in ("frozen_prior", "causal_frozen_prior"):
            inference.validate_cli(_args(execution_arm=arm, adapter_checkpoint=None))
        for arm in ("main", "parallel_only"):
            with self.assertRaises(inference.PriorTangentInferenceError):
                inference.validate_cli(_args(execution_arm=arm, adapter_checkpoint=None))
        with self.assertRaises(inference.PriorTangentInferenceError):
            inference.validate_cli(_args(execution_arm="unbound"))
        with self.assertRaises(inference.PriorTangentInferenceError):
            inference.validate_cli(_args(num_inference_steps=41))

    def test_packed_reshape_is_explicitly_b_21_s_d_and_reversible(self):
        layout = inference.tri.PackedLatentLayout.from_spatial_shape(
            (1, 16, 21, 8, 10)
        )
        packed = _FakeArray(layout.packed_shape)
        phase = inference.packed_to_phase_grid(packed, layout=layout)
        self.assertEqual(phase.shape, (1, 21, 20, 64))
        restored = inference.phase_grid_to_packed(phase, layout=layout)
        self.assertEqual(restored.shape, layout.packed_shape)
        wrong = inference.tri.PackedLatentLayout.from_spatial_shape(
            (1, 16, 20, 8, 10)
        )
        with self.assertRaises(inference.PriorTangentInferenceError):
            inference.packed_to_phase_grid(_FakeArray(wrong.packed_shape), layout=wrong)

    def test_training_receipt_accepts_only_exact_v5_cross_q_contract(self):
        config, receipt = _valid_adapter_and_receipt()
        identity = inference.validate_training_adapter_contract(config, receipt)
        self.assertEqual(identity["scope"], "cross_q")
        self.assertEqual(len(identity["targets"]), 30)
        self.assertEqual(
            receipt["immutable_contract"]["value"]["base_apg"]
            ["negative_tokenization"],
            "official_renderer_unconditional_verbatim",
        )

        mutations = []
        bad = copy.deepcopy(receipt)
        bad["schema_version"] = "legacy"
        mutations.append(bad)
        bad = copy.deepcopy(receipt)
        bad["immutable_contract"]["value"]["trust_region"]["kappa_perp"] = 1.0
        bad["immutable_contract"]["digest"] = inference.trainer.object_sha256(
            bad["immutable_contract"]["value"]
        )
        _redigest(bad)
        mutations.append(bad)
        bad = copy.deepcopy(receipt)
        bad["supervision"]["external_mask_track_flow_pose_trajectory"] = True
        _redigest(bad)
        mutations.append(bad)
        bad = copy.deepcopy(receipt)
        bad["adapter"]["scope"] = "all_qkv"
        _redigest(bad)
        mutations.append(bad)
        for candidate in mutations:
            with self.subTest(candidate=candidate.get("schema_version")):
                with self.assertRaises(inference.PriorTangentInferenceError):
                    inference.validate_training_adapter_contract(config, candidate)

        bad_config = dict(config)
        bad_config["target_modules"] = config["target_modules"][:-1]
        with self.assertRaises(inference.PriorTangentInferenceError):
            inference.validate_training_adapter_contract(bad_config, receipt)

    def test_gamma_schedule_and_ablation_trace_are_fail_closed(self):
        audit = {
            "schedule_sha256": inference.sigma_strata.SCHEDULE_SHA256,
            "timesteps": list(inference.sigma_strata.PINNED_TIMESTEPS),
            "positive_sigmas": list(inference.sigma_strata.PINNED_POSITIVE_SIGMAS),
            "positive_sigmas_float32_be_hex": list(
                inference.sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX
            ),
            "terminal_sigma": 0.0,
            "terminal_sigma_float32_be_hex": (
                inference.sigma_strata.TERMINAL_SIGMA_FLOAT32_HEX
            ),
        }
        for arm in inference.EXECUTION_ARMS:
            routed = arm in ("main", "parallel_only")
            records = []
            for step, (timestep, sigma) in enumerate(
                zip(
                    inference.sigma_strata.PINNED_TIMESTEPS,
                    inference.sigma_strata.PINNED_POSITIVE_SIGMAS,
                )
            ):
                gamma = inference.tangent.correction_gamma(step) if routed else 0.0
                records.append(
                    inference.PriorTangentStepRecord(
                        step_index=step,
                        timestep=timestep,
                        sigma=sigma,
                        gamma=gamma,
                        execution_arm=arm,
                        model_id="transformer_1",
                        transformer_forwards=4,
                        frozen_negative_forwards=1,
                        frozen_noop_forwards=1,
                        frozen_action_forwards=1,
                        adapted_action_forwards=1,
                        original_scheduler_calls=1,
                        official_frozen_action_apg_exact=True,
                        official_frozen_action_apg_rms_error=0.0,
                        official_frozen_action_apg_max_abs_error=0.0,
                        raw_prior_rms=1.0,
                        prior_phase0_rms=0.2,
                        q0_prior_rms=0.8,
                        raw_correction_rms=0.1 if routed else 0.0,
                        trusted_correction_rms=0.05 if routed else 0.0,
                        executed_correction_rms=gamma * 0.05,
                        trusted_first_phase_max_abs=0.0,
                        phase_cells=20,
                        gamma_zero_exact_frozen_prior=(
                            (routed and gamma == 0.0)
                            or arm == "causal_frozen_prior"
                        ),
                        adapter_correction_routed=routed,
                        adapter_loaded=True,
                    )
                )
            trace = inference.PriorTangentTrace(
                execution_arm=arm,
                adapter_loaded=True,
                records=records,
                sample_calls=1,
            )
            payload = inference.validate_execution_trace(
                trace,
                execution_arm=arm,
                adapter_loaded=True,
                runtime_schedule_audit=audit,
            )
            self.assertEqual(payload["certificate"]["execution_arm"], arm)
            self.assertEqual(
                payload["certificate"]["transformer_forwards"], 160
            )
        bad = copy.deepcopy(trace)
        # Dataclasses are frozen; rebuild one mutated record explicitly.
        record = asdict_for_test(bad.records[24])
        record["gamma"] = 0.25
        bad.records[24] = inference.PriorTangentStepRecord(**record)
        with self.assertRaises(inference.PriorTangentInferenceError):
            inference.validate_execution_trace(
                bad,
                execution_arm=bad.execution_arm,
                adapter_loaded=True,
                runtime_schedule_audit=audit,
            )

    def test_inference_receipt_binds_arm_adapter_seed_and_hashes(self):
        execution = {
            "runtime_unipc_schedule_audit": {
                "schedule_sha256": inference.sigma_strata.SCHEDULE_SHA256,
                "timesteps": list(inference.sigma_strata.PINNED_TIMESTEPS),
                "positive_sigmas": list(inference.sigma_strata.PINNED_POSITIVE_SIGMAS),
                "positive_sigmas_float32_be_hex": list(
                    inference.sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX
                ),
                "terminal_sigma": 0.0,
                "terminal_sigma_float32_be_hex": (
                    inference.sigma_strata.TERMINAL_SIGMA_FLOAT32_HEX
                ),
            }
        }
        base_receipt = {
            "base_model": {},
            "sampling": {},
            "input": {
                "source_video_path": "/source.mp4",
                "source_video_sha256": "a" * 64,
            },
            "output": {"path": "/output.mp4", "sha256": "b" * 64},
        }
        identity = {
            "receipt_digest": "c" * 64,
            "global_step": 40,
            "training_method_source_revision": SHA1,
            "training_method_source_archive_sha256": SHA256,
            "scope": "cross_q",
            "targets": [f"target.{i}" for i in range(30)],
            "target_modules_sha256": "d" * 64,
            "serialized_target_modules": [f"target.{i}" for i in range(30)],
            "initialization_digest": "e" * 64,
            "checkpoint_parameter_digest": "f" * 64,
        }

        class Bundle:
            checkpoint_root = Path("/adapter")
            adapter_config_path = Path("/adapter/adapter/adapter_config.json")
            adapter_model_path = Path("/adapter/adapter/adapter_model.safetensors")
            training_receipt_path = Path("/adapter/receipt.json")

        with mock.patch.object(
            inference.frozen, "build_inference_receipt", return_value=base_receipt
        ), mock.patch.object(inference, "_method_hashes", return_value={"x": "0" * 64}):
            receipt = inference.build_inference_receipt(
                args=_args(execution_arm="causal_frozen_prior"),
                source_path=Path("/source.mp4"),
                source_sha256="a" * 64,
                source_metadata={},
                output_path=Path("/output.mp4"),
                output_sha256="b" * 64,
                noop_identity={},
                execution_trace=execution,
                bernini_revision=inference.trainer.BERNINI_OFFICIAL_COMMIT,
                veomni_revision=inference.trainer.VEOMNI_TESTED_COMMIT,
                inference_file_hashes={},
                wan_diffusion_path=Path("/wan.py"),
                wan_diffusion_sha256="0" * 64,
                runtime_versions={},
                adapter_bundle=Bundle(),
                adapter_identity=identity,
                adapter_config_sha256="1" * 64,
                adapter_model_sha256="2" * 64,
                training_receipt_file_sha256="3" * 64,
                adapter_tensor_count=60,
                active_lora_module_count=30,
            )
        self.assertEqual(receipt["sampling"]["execution_arm"], "causal_frozen_prior")
        self.assertTrue(receipt["sampling"]["diagnostic_ablation"])
        self.assertFalse(receipt["sampling"]["main_method_claim"])
        self.assertEqual(receipt["adapter"]["adapter_model_sha256"], "2" * 64)
        self.assertEqual(receipt["input"]["source_video_sha256"], "a" * 64)
        self.assertEqual(receipt["output"]["sha256"], "b" * 64)
        candidate = dict(receipt)
        digest = candidate.pop("receipt_digest")
        self.assertEqual(digest, inference.trainer.object_sha256(candidate))


def asdict_for_test(value):
    return {field: getattr(value, field) for field in value.__dataclass_fields__}


try:
    import torch
except ImportError:  # pragma: no cover - local orchestration image
    torch = None


@unittest.skipIf(torch is None, "torch is unavailable")
class PriorTangentInferenceTensorTests(unittest.TestCase):
    def _raw(self, *, step, adapted_offset=0.03, arm="main"):
        torch.manual_seed(7)
        layout = inference.tri.PackedLatentLayout.from_spatial_shape(
            (1, 1, 21, 2, 2)
        )
        sample = torch.randn(layout.packed_shape, dtype=torch.float32)
        negative = torch.randn(layout.packed_shape, dtype=torch.float32).to(torch.bfloat16)
        noop = torch.randn(layout.packed_shape, dtype=torch.float32).to(torch.bfloat16)
        action = torch.randn(layout.packed_shape, dtype=torch.float32).to(torch.bfloat16)
        adapted = (action.float() + adapted_offset).to(torch.bfloat16)
        sigma = torch.tensor(0.5, dtype=torch.float32)
        spatial_sample = inference.tri._packed_to_spatial(sample, layout)
        spatial_negative = inference.tri._packed_to_spatial(negative, layout)
        spatial_action = inference.tri._packed_to_spatial(action, layout)
        negative_clean = inference.tri.pinned_raw_condition_clean(
            spatial_sample, spatial_negative, sigma
        )
        action_clean = inference.tri.pinned_raw_condition_clean(
            spatial_sample, spatial_action, sigma
        )
        apg = inference.tri.APGParameters(4.0, 0.8, False, 0.5, 50.0, 0.0)
        guided = inference.tri._normalized_guidance(
            action_clean,
            negative_clean,
            4.0,
            inference.tri._MomentumBuffer(0.0, branch="fixture"),
            0.5,
            50.0,
        )
        official = inference.tri._spatial_to_packed(
            (spatial_sample - guided) / sigma, layout
        ).to(torch.bfloat16)
        source = torch.randn((1, 1, 21, 2, 2), dtype=torch.float32)
        source_phase = inference.packed_to_phase_grid(
            inference.tri._spatial_to_packed(source, layout), layout=layout
        )
        return inference.RawFourBranchStep(
            step_index=step,
            timestep=torch.tensor(float(1000 - step)),
            timestep_float=float(1000 - step),
            sigma=sigma,
            sigma_float=0.5,
            model_id="transformer_1",
            sample_packed=sample,
            official_model_output=official,
            frozen_negative_velocity_packed=negative,
            frozen_noop_velocity_packed=noop,
            frozen_action_velocity_packed=action,
            adapted_action_velocity_packed=adapted,
            source_phase=source_phase,
            execution_arm=arm,
            adapter_loaded=True,
            apg=apg,
            layout=layout,
        )

    @staticmethod
    def _run(raw):
        return inference.project_prior_tangent_step(
            raw,
            base_action_momentum=inference.tri._MomentumBuffer(0.0, branch="a0"),
            base_noop_momentum=inference.tri._MomentumBuffer(0.0, branch="n0"),
            adapted_action_momentum=inference.tri._MomentumBuffer(0.0, branch="at"),
        )

    def test_main_gamma_zero_is_exact_frozen_prior_not_adapter_correction(self):
        projected, record = self._run(self._raw(step=35, arm="main"))
        self.assertEqual(tuple(projected.model_output.shape), (1, 21, 4))
        self.assertEqual(record.gamma, 0.0)
        self.assertTrue(record.gamma_zero_exact_frozen_prior)
        self.assertTrue(record.adapter_correction_routed)
        self.assertEqual(record.trusted_first_phase_max_abs, 0.0)

    def test_frozen_prior_ignores_adapted_action_and_causal_prior_removes_phase0(self):
        first, first_record = self._run(
            self._raw(step=0, adapted_offset=0.03, arm="frozen_prior")
        )
        second, _ = self._run(
            self._raw(step=0, adapted_offset=3.0, arm="frozen_prior")
        )
        self.assertTrue(torch.equal(first.model_output, second.model_output))
        self.assertFalse(first_record.adapter_correction_routed)
        self.assertFalse(first_record.gamma_zero_exact_frozen_prior)
        causal, causal_record = self._run(
            self._raw(step=0, adapted_offset=3.0, arm="causal_frozen_prior")
        )
        self.assertFalse(torch.equal(first.model_output, causal.model_output))
        self.assertTrue(causal_record.gamma_zero_exact_frozen_prior)
        self.assertGreaterEqual(causal_record.raw_prior_rms, 0.0)
        self.assertGreaterEqual(causal_record.prior_phase0_rms, 0.0)
        self.assertGreaterEqual(causal_record.q0_prior_rms, 0.0)

    def test_parallel_only_disables_perpendicular_cap_but_keeps_schedule(self):
        _, main = self._run(self._raw(step=20, arm="main"))
        _, parallel = self._run(self._raw(step=20, arm="parallel_only"))
        self.assertEqual(main.gamma, 1.0)
        self.assertEqual(parallel.gamma, 1.0)
        self.assertTrue(parallel.adapter_correction_routed)
        self.assertLessEqual(
            parallel.trusted_correction_rms, main.trusted_correction_rms + 1.0e-6
        )


if __name__ == "__main__":
    unittest.main()
