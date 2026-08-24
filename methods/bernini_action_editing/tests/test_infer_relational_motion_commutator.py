#!/usr/bin/env python3
"""Strict loader and five-branch runtime tests for RMC v7 inference."""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from dataclasses import replace
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_relational_motion_commutator as inference  # noqa: E402

try:
    import torch  # noqa: F401
except ImportError:
    HAS_TORCH = False
else:
    HAS_TORCH = True


SHA1 = "1" * 40
SHA256 = "2" * 64


def _resign(receipt):
    receipt.pop("receipt_digest", None)
    receipt["receipt_digest"] = inference.trainer.object_sha256(receipt)
    return receipt


def _valid_adapter_and_receipt():
    targets = inference.expected_lora_targets()

    class Dataset:
        root = Path("/data")
        signature = "strict-dataset-signature"

        def __len__(self):
            return 644

    class Router:
        digest = "5" * 64
        file_sha256 = inference.v7_train.v5.STRICT_ROUTING_SHA256

        def receipt(self):
            return {
                "path": "/route.jsonl",
                "default_tier": "reject",
                "file_sha256": self.file_sha256,
                "routing_digest": self.digest,
                "explicit_route_counts": {
                    "full_pair": 0,
                    "motion_only": 359,
                    "reject": 285,
                },
            }

    class Distributed:
        world_size = 4
        ulysses_size = 4

    args = argparse.Namespace(
        method_source_revision=SHA1,
        method_source_archive_sha256=SHA256,
        expected_bernini_commit=inference.trainer.BERNINI_OFFICIAL_COMMIT,
        expected_veomni_commit=inference.trainer.VEOMNI_TESTED_COMMIT,
        expected_checkpoint_tree_sha256=inference.trainer.CHECKPOINT_TREE_SHA256,
        seed=20260808,
        weight_decay=0.0,
        max_grad_norm=1.0,
        teacher_mode="target_only",
        max_steps=40,
    )
    dataset = Dataset()
    router = Router()
    summary = {"sha256": "3" * 64, "index_sha256": "4" * 64}
    eligible = [
        (
            index,
            SimpleNamespace(
                iid=f"iid-{index:03d}",
                tier="motion_only",
                full_target_weight=0.0,
            ),
        )
        for index in range(359)
    ]
    loss_config = inference.objective.RelationalCommutatorLossConfig(
        relational_auxiliary_weight=0.0,
        commutator_config=inference.MAIN_COMMUTATOR_CONFIG,
    )
    immutable = inference.v7_train._immutable_contract(
        args=args,
        dataset=dataset,
        dataset_summary=summary,
        router=router,
        eligible_routes=eligible,
        target_modules=targets,
        checkpoint=Path("/checkpoint"),
        loss_config=loss_config,
    )
    step_audit = [
        {
            "optimizer_step": index + 1,
            "row_index": index,
            "iid": f"iid-{index:03d}",
            "seed": index,
            "sigma_schedule_index": index,
            "sigma_timestep": inference.sigma_strata.PINNED_TIMESTEPS[index],
            "teacher_mode": "target_only",
            "metrics_timing": inference.v7_train.METRICS_TIMING,
            "rho": inference.commutator.release_rho(index),
        }
        for index in range(40)
    ]
    class Parameter:
        def numel(self):
            return 1

    parameter = Parameter()
    named_trainable = [("adapter.lora_A.default.weight", parameter)]
    parameter_names = [named_trainable[0][0]]
    optimizer_payload = {
        "schema_version": inference.v7_train.OPTIMIZER_SCHEMA,
        "global_step": 40,
        "state": "test",
    }
    with mock.patch.object(
        inference.v7_train.v4,
        "_checkpoint_parameter_digest",
        return_value="7" * 64,
    ), mock.patch.object(
        inference.v7_train.v4,
        "_stable_recursive_digest",
        return_value="8" * 64,
    ):
        receipt = inference.v7_train._build_receipt(
            args=args,
            global_step=40,
            metrics={"loss_total": 1.0},
            step_audit=step_audit,
            dataset=dataset,
            dataset_summary=summary,
            router=router,
            checkpoint=Path("/checkpoint"),
            bernini_revision=inference.trainer.BERNINI_OFFICIAL_COMMIT,
            veomni_revision=inference.trainer.VEOMNI_TESTED_COMMIT,
            distributed=Distributed(),
            backend="nccl",
            target_modules=targets,
            named_trainable=named_trainable,
            initialization_digest="6" * 64,
            transformers_version="test-transformers",
            immutable=immutable,
            optimizer_payload=optimizer_payload,
        )
    # The trainer can certify only source-level preflight.  A post-save
    # finalizer is the sole transition from pending to inference-ready.
    self_ready = receipt["inference_loader_parity"]
    if self_ready != immutable["value"]["inference_loader_parity"]:
        raise AssertionError("fixture lost inference-loader parity binding")
    if receipt["inference_loader_parity_pending"] is not True:
        raise AssertionError("trainer fixture must remain pending before save")
    pending_receipt_digest = receipt["receipt_digest"]
    artifact = {
        "schema_version": inference.v7_train.ARTIFACT_VALIDATION_SCHEMA,
        "verified": True,
        "status": "post_save_strict_reload_complete",
        "adapter_config_sha256": "9" * 64,
        "adapter_model_sha256": "a" * 64,
        "serialized_target_pattern_count": 17,
        "expanded_target_module_count": 46,
        "adapter_tensor_count": 92,
        "active_lora_module_count": 46,
        "strict_tensor_reload_equal": True,
        "parameter_digest_verified_after_safetensors_reload": True,
        "checkpoint_parameter_digest": receipt["adapter"][
            "checkpoint_parameter_digest"
        ],
        "pending_receipt_digest": pending_receipt_digest,
        "validator_method_source_revision": SHA1,
        "validator_method_source_archive_sha256": SHA256,
        "bernini_commit": inference.trainer.BERNINI_OFFICIAL_COMMIT,
        "veomni_commit": inference.trainer.VEOMNI_TESTED_COMMIT,
        "checkpoint_tree_sha256": inference.trainer.CHECKPOINT_TREE_SHA256,
    }
    artifact["digest"] = inference.trainer.object_sha256(artifact)
    receipt["artifact_validation"] = artifact
    receipt["inference_loader_parity_pending"] = False
    _resign(receipt)
    adapter_config = {
        "peft_type": "LORA",
        "r": 8,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "bias": "none",
        "modules_to_save": None,
        "use_dora": False,
        "use_rslora": False,
        "target_modules": list(
            reversed(inference.expected_serialized_target_patterns())
        ),
    }
    return adapter_config, receipt


def _runtime_schedule_audit():
    return {
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


class FakeAdapter:
    def __init__(self, diffusion):
        self.diffusion = diffusion

    @contextmanager
    def disable_adapter(self):
        previous = self.diffusion.adapter_enabled
        self.diffusion.adapter_enabled = False
        try:
            yield
        finally:
            self.diffusion.adapter_enabled = previous


class FakeScheduler:
    def __init__(self):
        import torch

        self.sigmas = torch.tensor(
            list(inference.sigma_strata.PINNED_POSITIVE_SIGMAS) + [0.0],
            dtype=torch.float32,
        )
        self.timesteps = torch.tensor(
            inference.sigma_strata.PINNED_TIMESTEPS, dtype=torch.int64
        )
        self.step_index = 0
        self.received = []
        self.fail_at = None

    def step(self, model_output, timestep, sample):
        if self.fail_at is not None and self.step_index == self.fail_at:
            raise RuntimeError("injected scheduler failure")
        self.received.append(model_output)
        self.step_index += 1
        return sample


class FakeDiffusion:
    use_unipc = True

    def __init__(self):
        import torch

        self.scheduler = FakeScheduler()
        self.adapter_enabled = True
        self.calls = []
        self.official_objects = []
        self.action = torch.zeros((1, 2, 3), dtype=torch.float32)
        self.negative = torch.ones((1, 2, 3), dtype=torch.float32)
        self.noop = torch.full((1, 2, 3), 2.0, dtype=torch.float32)
        self.noisy = torch.zeros((1, 21, 8), dtype=torch.float32)
        self.rotary = object()

    def _velocity(self, prompt):
        import torch

        phase = torch.arange(21, dtype=torch.float32).reshape(1, 21, 1)
        if prompt is self.negative:
            name, value = "negative", torch.zeros_like(self.noisy)
        elif prompt is self.noop:
            name = "noop"
            value = (0.01 if self.adapter_enabled else 0.0) * phase.expand_as(
                self.noisy
            )
        elif prompt is self.action:
            name = "action"
            value = (0.12 if self.adapter_enabled else 0.02) * phase.expand_as(
                self.noisy
            )
        else:
            raise AssertionError("unknown prompt object")
        state = "adapted" if self.adapter_enabled else "frozen"
        self.calls.append(f"{state}_{name}")
        return value.to(torch.bfloat16)

    def shared_step(
        self,
        noisy_latents,
        timesteps,
        rotary_embs,
        batch_vae_seqlen,
        model_id,
        cond_embeds,
        batch_text_seqlen,
    ):
        if noisy_latents is not self.noisy or rotary_embs is not self.rotary:
            raise AssertionError("same-state object identity changed")
        return self._velocity(cond_embeds)

    def sample(
        self,
        *,
        guidance_mode,
        num_inference_steps,
        flow_shift,
        prompt_embeds,
        uncond_prompt_embeds,
        prompt_embeds_t2=None,
        uncond_embeds_t2=None,
        omega_txt=1.0,
        omega_scale=1.0,
        eta=1.0,
        norm_threshold=0.0,
        momentum=0.0,
    ):
        import torch

        layout = inference.tri.PackedLatentLayout.from_spatial_shape(
            (1, 2, 21, 2, 2)
        )
        official_momentum = inference.tri._MomentumBuffer(
            0.0, branch="vendor_action"
        )
        for index, timestep_value in enumerate(
            inference.sigma_strata.PINNED_TIMESTEPS
        ):
            timestep = torch.tensor(timestep_value, dtype=torch.int64)
            negative_v = self.shared_step(
                self.noisy,
                timestep,
                self.rotary,
                [21],
                "transformer_1",
                uncond_prompt_embeds,
                [int(uncond_prompt_embeds.shape[1])],
            )
            action_v = self.shared_step(
                self.noisy,
                timestep,
                self.rotary,
                [21],
                "transformer_1",
                prompt_embeds,
                [int(prompt_embeds.shape[1])],
            )
            sigma = self.scheduler.sigmas[index]
            sample_spatial = inference.tri._packed_to_spatial(self.noisy, layout)
            negative_spatial = inference.tri._packed_to_spatial(negative_v, layout)
            action_spatial = inference.tri._packed_to_spatial(action_v, layout)
            negative_clean = inference.tri.pinned_raw_condition_clean(
                sample_spatial, negative_spatial, sigma
            )
            action_condition = inference.tri.pinned_raw_condition_clean(
                sample_spatial, action_spatial, sigma
            )
            action_clean = inference.tri._normalized_guidance(
                action_condition,
                negative_clean,
                float(omega_txt),
                official_momentum,
                float(eta),
                float(norm_threshold),
            )
            official = inference.tri._spatial_to_packed(
                (sample_spatial - action_clean) / sigma, layout
            ).to(torch.bfloat16)
            self.official_objects.append(official)
            self.scheduler.step(official, timestep, self.noisy)
        return self.noisy


class LoaderContractTests(unittest.TestCase):
    class _RuntimeAttentionModel:
        def __init__(self):
            self._modules = [
                (name, SimpleNamespace(weight=object()))
                for name in inference.v6_scope.canonical_attention_modules()
            ]

        def named_modules(self):
            yield "", self
            yield from self._modules

    def test_low_level_loader_accepts_receipt_exact46_without_legacy_scope(self):
        targets = inference.expected_lora_targets()
        validator = (
            inference.v5.adapter_loader.validate_runtime_exact_lora_targets
        )
        self.assertEqual(
            validator(self._RuntimeAttentionModel(), targets), targets
        )
        self.assertNotIn(
            inference.REQUIRED_LORA_SCOPE,
            inference.v5.adapter_loader.motion.MODULE_SCOPES,
        )

        with self.assertRaisesRegex(
            inference.v5.adapter_loader.DeltaInferenceError, "sorted unique"
        ):
            validator(self._RuntimeAttentionModel(), list(reversed(targets)))
        with self.assertRaisesRegex(
            inference.v5.adapter_loader.DeltaInferenceError, "lacks exact"
        ):
            validator(
                self._RuntimeAttentionModel(),
                sorted(targets[:-1] + ["diff_dec.transformer.blocks.30.attn2.to_q"]),
            )

    def test_valid_completed_target_only_exact46_contract(self):
        adapter_config, receipt = _valid_adapter_and_receipt()
        identity = inference.validate_training_adapter_contract(
            adapter_config, receipt
        )
        self.assertEqual(identity["global_step"], 40)
        self.assertEqual(len(identity["targets"]), 46)
        self.assertEqual(
            identity["serialized_target_modules"],
            inference.expected_serialized_target_patterns(),
        )
        self.assertEqual(
            inference._expand_serialized_target_patterns(
                identity["serialized_target_modules"]
            ),
            identity["targets"],
        )

    def test_rejects_canary_v6_pending_and_nonexact46(self):
        adapter_config, receipt = _valid_adapter_and_receipt()
        cases = []

        canary = copy.deepcopy(receipt)
        canary.update(
            {
                "global_step": 1,
                "max_steps": 1,
                "formal_40_sigma_cycle_complete": False,
            }
        )
        cases.append((adapter_config, _resign(canary)))

        v6 = copy.deepcopy(receipt)
        v6["schema_version"] = inference.v7_train.v6_runtime.RECEIPT_SCHEMA
        cases.append((adapter_config, _resign(v6)))

        pending = copy.deepcopy(receipt)
        pending["inference_loader_parity_pending"] = True
        cases.append((adapter_config, _resign(pending)))

        parity_drift = copy.deepcopy(receipt)
        parity_drift["inference_loader_parity"]["verified"] = False
        cases.append((adapter_config, _resign(parity_drift)))

        artifact_pending = copy.deepcopy(receipt)
        artifact_pending["artifact_validation"]["verified"] = False
        cases.append((adapter_config, _resign(artifact_pending)))

        artifact_hash_drift = copy.deepcopy(receipt)
        artifact_hash_drift["artifact_validation"]["adapter_model_sha256"] = (
            "b" * 64
        )
        cases.append((adapter_config, _resign(artifact_hash_drift)))

        transition_drift = copy.deepcopy(receipt)
        transition_drift["artifact_validation"]["pending_receipt_digest"] = (
            "c" * 64
        )
        transition_artifact = transition_drift["artifact_validation"]
        transition_artifact.pop("digest")
        transition_artifact["digest"] = inference.trainer.object_sha256(
            transition_artifact
        )
        cases.append((adapter_config, _resign(transition_drift)))

        missing = copy.deepcopy(adapter_config)
        missing["target_modules"] = missing["target_modules"][:-1]
        cases.append((missing, receipt))

        duplicate = copy.deepcopy(adapter_config)
        duplicate["target_modules"][-1] = duplicate["target_modules"][0]
        cases.append((duplicate, receipt))

        for config, candidate in cases:
            with self.subTest(case=len(config.get("target_modules", []))):
                with self.assertRaises(
                    inference.RelationalMotionCommutatorInferenceError
                ):
                    inference.validate_training_adapter_contract(config, candidate)

    def test_relational_auxiliary_checkpoint_remains_fail_closed(self):
        adapter_config, receipt = _valid_adapter_and_receipt()
        receipt = copy.deepcopy(receipt)
        immutable = receipt["immutable_contract"]
        immutable["value"]["teacher_mode"] = "relational_auxiliary"
        immutable["digest"] = inference.trainer.object_sha256(immutable["value"])
        _resign(receipt)
        with self.assertRaisesRegex(
            inference.RelationalMotionCommutatorInferenceError,
            "teacher_mode",
        ):
            inference.validate_training_adapter_contract(adapter_config, receipt)

    def test_v8_radius_configs_scale_both_priors_but_not_floor(self):
        base = inference.MAIN_FEASIBLE_QUOTIENT_CONFIG
        for scale in inference.V8_RADIUS_SCALE_CHOICES:
            with self.subTest(scale=scale):
                config = inference.feasible_quotient_config_for_radius_scale(
                    scale
                )
                self.assertEqual(
                    config.frozen_quotient_radius_ratio,
                    scale * base.frozen_quotient_radius_ratio,
                )
                self.assertEqual(
                    config.noop_dynamics_radius_ratio,
                    scale * base.noop_dynamics_radius_ratio,
                )
                self.assertEqual(config.radius_floor, base.radius_floor)
                self.assertEqual(config.epsilon, base.epsilon)
                self.assertEqual(
                    inference.validated_feasible_quotient_radius_scale(
                        config,
                        operator_mode=(
                            inference.V8_RECONSTRUCTION_SECTION_FQT
                        ),
                    ),
                    scale,
                )

    def test_runtime_contract_labels_scaled_radius_and_fails_closed(self):
        base = inference.MAIN_FEASIBLE_QUOTIENT_CONFIG
        config = inference.feasible_quotient_config_for_radius_scale(2.5)
        contract = inference.runtime_contract(
            operator_mode=inference.V8_RECONSTRUCTION_SECTION_FQT,
            feasible_quotient_config=config,
            v8_training_matched=True,
        )
        self.assertEqual(contract["v8_radius_scale"], 2.5)
        self.assertEqual(
            contract["audited_v8_radius_scales"], [1.0, 2.5, 4.0]
        )
        self.assertTrue(
            contract["v8_radius_scale_inference_only_ablation"]
        )
        self.assertFalse(contract["operator_training_matched"])
        self.assertFalse(contract["main_operator"])
        self.assertFalse(contract["training_matched"])
        self.assertTrue(contract["inference_only_ablation"])
        with self.assertRaisesRegex(
            inference.RelationalMotionCommutatorInferenceError,
            "trained V8",
        ):
            inference.runtime_contract(
                operator_mode=inference.V8_RECONSTRUCTION_SECTION_FQT,
                feasible_quotient_config=config,
                v8_training_matched=False,
            )
        with self.assertRaisesRegex(
            inference.RelationalMotionCommutatorInferenceError,
            "V7",
        ):
            inference.runtime_contract(
                operator_mode=inference.V7_RESIDUAL_ACTION_SECTION,
                feasible_quotient_config=config,
            )
        arbitrary = inference.gauge.FeasibleQuotientConfig(
            frozen_quotient_radius_ratio=3.0,
            noop_dynamics_radius_ratio=0.75,
            radius_floor=base.radius_floor,
            epsilon=base.epsilon,
        )
        with self.assertRaisesRegex(
            inference.RelationalMotionCommutatorInferenceError,
            "outside the audited",
        ):
            inference.runtime_contract(
                operator_mode=inference.V8_RECONSTRUCTION_SECTION_FQT,
                feasible_quotient_config=arbitrary,
                v8_training_matched=True,
            )


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for projector tests")
class ProjectorTests(unittest.TestCase):
    def test_v8_radius_uses_training_matched_local_fp32_action_clean(self):
        import torch

        local = torch.zeros((1, 2, 3), dtype=torch.float32)
        official_roundtrip = torch.full_like(local, 1.0e-3)
        self.assertIs(
            inference.select_frozen_action_clean_for_operator(
                local,
                official_roundtrip,
                operator_mode=inference.V8_RECONSTRUCTION_SECTION_FQT,
            ),
            local,
        )
        self.assertIs(
            inference.select_frozen_action_clean_for_operator(
                local,
                official_roundtrip,
                operator_mode=inference.V7_RESIDUAL_ACTION_SECTION,
            ),
            official_roundtrip,
        )

    @staticmethod
    def _raw(step_index):
        import torch

        layout = inference.tri.PackedLatentLayout.from_spatial_shape(
            (1, 2, 21, 2, 2)
        )
        sample = torch.zeros(layout.packed_shape, dtype=torch.float32)
        zero = torch.zeros(layout.packed_shape, dtype=torch.bfloat16)
        phase = torch.arange(21, dtype=torch.float32).reshape(1, 21, 1)
        adapted_action = (0.5 * phase.expand(layout.packed_shape)).to(
            torch.bfloat16
        )
        sigma = torch.tensor(
            inference.sigma_strata.PINNED_POSITIVE_SIGMAS[step_index],
            dtype=torch.float32,
        )
        return inference.RawRelationalMotionCommutatorStep(
            step_index=step_index,
            timestep=torch.tensor(
                inference.sigma_strata.PINNED_TIMESTEPS[step_index],
                dtype=torch.int64,
            ),
            timestep_float=float(
                inference.sigma_strata.PINNED_TIMESTEPS[step_index]
            ),
            sigma=sigma,
            sigma_float=float(sigma.item()),
            model_id="transformer_1",
            sample_packed=sample,
            official_model_output=zero.clone(),
            frozen_negative_velocity_packed=zero.clone(),
            frozen_noop_velocity_packed=zero.clone(),
            frozen_action_velocity_packed=zero.clone(),
            adapted_noop_velocity_packed=zero.clone(),
            adapted_action_velocity_packed=adapted_action,
            apg=inference.tri.APGParameters(
                guidance_scale=1.0,
                omega_scale=1.0,
                scale_transformer_2=False,
                eta=1.0,
                norm_threshold=0.0,
                momentum=0.0,
            ),
            layout=layout,
        )

    @staticmethod
    def _project(raw):
        buffers = [
            inference.tri._MomentumBuffer(0.0, branch=name)
            for name in ("a0", "n0", "nt", "at")
        ]
        return inference.project_relational_motion_commutator_step(
            raw,
            frozen_action_momentum=buffers[0],
            frozen_noop_momentum=buffers[1],
            adapted_noop_momentum=buffers[2],
            adapted_action_momentum=buffers[3],
        )

    def test_early_step_is_hard_bounded_and_late_step_aliases_official(self):
        early = self._raw(0)
        projected, record = self._project(early)
        self.assertEqual(record.rho, 1.0)
        self.assertIsNot(projected.model_output, early.official_model_output)
        self.assertLessEqual(
            record.bounded_increment_max_violation,
            max(inference.MAIN_COMMUTATOR_CONFIG.epsilon, 1.0e-6),
        )
        self.assertGreater(record.raw_commutator_correction_rms, 0.0)
        self.assertLess(
            record.bounded_commutator_correction_rms,
            record.raw_commutator_correction_rms,
        )

        late = self._raw(31)
        projected, record = self._project(late)
        self.assertEqual(record.rho, 0.0)
        self.assertIs(projected.model_output, late.official_model_output)
        self.assertTrue(record.exact_official_model_output_object)
        self.assertEqual(record.scheduler_boundary_correction_rms, 0.0)

    def test_official_action_apg_mismatch_fails_closed(self):
        raw = self._raw(0)
        raw = replace(
            raw,
            official_model_output=raw.official_model_output
            + raw.official_model_output.new_full((), 1.0),
        )
        with self.assertRaisesRegex(
            inference.RelationalMotionCommutatorInferenceError,
            "official model_output",
        ):
            self._project(raw)

    def test_only_audited_kappa_arms_are_allowed(self):
        main = inference.runtime_contract()
        self.assertTrue(main["main_operator"])
        ablation = inference.commutator.MotionCommutatorConfig(
            max_correction_increment_ratio=0.5,
            correction_increment_rms_floor=1.0e-3,
            temporal_smoothing=True,
        )
        self.assertTrue(
            inference.runtime_contract(ablation)["experimental_operator_ablation"]
        )
        invalid = inference.commutator.MotionCommutatorConfig(
            max_correction_increment_ratio=0.3,
            correction_increment_rms_floor=1.0e-3,
            temporal_smoothing=True,
        )
        with self.assertRaises(
            inference.RelationalMotionCommutatorInferenceError
        ):
            inference.runtime_contract(invalid)

    def test_v8_reconstruction_section_uses_noop_phase_and_full_quotient(self):
        early = replace(
            self._raw(0),
            operator_mode=inference.V8_RECONSTRUCTION_SECTION_FQT,
        )
        projected, record = self._project(early)
        self.assertEqual(
            record.operator_mode,
            inference.V8_RECONSTRUCTION_SECTION_FQT,
        )
        self.assertTrue(record.exact_noop_phase_zero)
        self.assertFalse(record.exact_official_model_output_object)
        self.assertGreater(record.full_quotient_raw_rms, 0.0)
        self.assertGreater(record.full_quotient_radius_mean, 0.0)
        self.assertGreater(record.gauge_phase_increment_tolerance, 0.0)
        self.assertLessEqual(
            record.gauge_phase_increment_rms_error,
            record.gauge_phase_increment_tolerance,
        )
        self.assertGreaterEqual(record.full_quotient_saturated_fraction, 0.0)
        self.assertLessEqual(record.full_quotient_saturated_fraction, 1.0)
        self.assertTrue(record.v8_local_fp32_frozen_action_for_radius)
        self.assertTrue(record.v8_scheduler_model_output_fp32)
        self.assertEqual(projected.model_output.dtype, torch.float32)
        self.assertLessEqual(
            record.scheduler_clean_roundtrip_max_abs_error,
            record.post_boundary_increment_tolerance,
        )
        self.assertLessEqual(
            record.post_boundary_increment_max_violation,
            record.post_boundary_increment_tolerance,
        )
        self.assertGreaterEqual(
            record.frozen_action_clean_roundtrip_rms_error, 0.0
        )
        self.assertGreaterEqual(
            record.frozen_action_clean_roundtrip_max_abs_error, 0.0
        )
        self.assertIsNot(projected.model_output, early.official_model_output)

        late = replace(
            self._raw(31),
            operator_mode=inference.V8_RECONSTRUCTION_SECTION_FQT,
        )
        projected, record = self._project(late)
        self.assertEqual(record.rho, 0.0)
        self.assertTrue(record.exact_noop_phase_zero)
        self.assertTrue(record.rho_zero_selected_noop_clean_object)
        self.assertTrue(record.rho_zero_noop_velocity_exact_parity)
        self.assertEqual(record.rho_zero_noop_velocity_rms_error, 0.0)
        self.assertFalse(record.exact_official_model_output_object)
        self.assertIsNot(projected.model_output, late.official_model_output)
        self.assertEqual(projected.model_output.dtype, torch.float32)


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for hook tests")
class FiveBranchHookTests(unittest.TestCase):
    @staticmethod
    def _run(diffusion):
        import torch

        adapter = FakeAdapter(diffusion)
        source = torch.zeros((1, 2, 21, 2, 2), dtype=torch.float32)
        original_sample = diffusion.sample
        original_shared = diffusion.shared_step
        original_scheduler = diffusion.scheduler.step
        with mock.patch.object(
            inference.tri,
            "validate_runtime_source_identity",
            return_value="0" * 64,
        ), mock.patch.object(
            inference.tri, "_validate_scheduler_contract", return_value=None
        ):
            with inference.relational_motion_commutator_unipc_hook(
                diffusion,
                adapter_model=adapter,
                source_clean=source,
                noop_prompt_embeds=diffusion.noop,
                latent_shape=source.shape,
                bernini_commit=inference.tri.PINNED_BERNINI_COMMIT,
                wan_diffusion_path=Path("/not/read/while/mocked.py"),
            ) as trace:
                result = diffusion.sample(
                    guidance_mode="v2v_apg",
                    num_inference_steps=40,
                    flow_shift=5.0,
                    prompt_embeds=diffusion.action,
                    uncond_prompt_embeds=diffusion.negative,
                    omega_txt=1.0,
                    omega_scale=1.0,
                    eta=1.0,
                    norm_threshold=0.0,
                    momentum=0.0,
                )
        return (
            result,
            trace,
            original_sample,
            original_shared,
            original_scheduler,
        )

    def test_fake_sampler_executes_200_forwards_and_40_original_steps(self):
        diffusion = FakeDiffusion()
        result, trace, original_sample, original_shared, original_scheduler = (
            self._run(diffusion)
        )
        self.assertIs(result, diffusion.noisy)
        self.assertEqual(len(diffusion.calls), 200)
        self.assertEqual(len(diffusion.scheduler.received), 40)
        per_step = [
            "frozen_negative",
            "frozen_noop",
            "frozen_action",
            "adapted_noop",
            "adapted_action",
        ]
        for index in range(40):
            self.assertEqual(diffusion.calls[index * 5 : (index + 1) * 5], per_step)
        validated = inference.validate_execution_trace(
            trace, runtime_schedule_audit=_runtime_schedule_audit()
        )
        self.assertEqual(validated["totals"]["transformer_forwards"], 200)
        self.assertEqual(validated["totals"]["original_scheduler_calls"], 40)
        self.assertEqual(
            validated["totals"]["rho_zero_exact_official_steps"],
            len(inference.LATE_EXACT_STEPS),
        )
        for index, (official, received) in enumerate(
            zip(diffusion.official_objects, diffusion.scheduler.received)
        ):
            self.assertEqual(
                received is official,
                inference.commutator.release_rho(index) == 0.0,
            )
        self.assertNotIn("sample", vars(diffusion))
        self.assertNotIn("shared_step", vars(diffusion))
        self.assertNotIn("step", vars(diffusion.scheduler))
        self.assertEqual(diffusion.sample, original_sample)
        self.assertEqual(diffusion.shared_step, original_shared)
        self.assertEqual(diffusion.scheduler.step, original_scheduler)

    def test_exception_restores_all_instance_wrappers(self):
        import torch

        diffusion = FakeDiffusion()
        diffusion.scheduler.fail_at = 3
        adapter = FakeAdapter(diffusion)
        source = torch.zeros((1, 2, 21, 2, 2), dtype=torch.float32)
        with mock.patch.object(
            inference.tri,
            "validate_runtime_source_identity",
            return_value="0" * 64,
        ), mock.patch.object(
            inference.tri, "_validate_scheduler_contract", return_value=None
        ):
            with self.assertRaisesRegex(RuntimeError, "injected scheduler failure"):
                with inference.relational_motion_commutator_unipc_hook(
                    diffusion,
                    adapter_model=adapter,
                    source_clean=source,
                    noop_prompt_embeds=diffusion.noop,
                    latent_shape=source.shape,
                    bernini_commit=inference.tri.PINNED_BERNINI_COMMIT,
                    wan_diffusion_path=Path("/not/read/while/mocked.py"),
                ):
                    diffusion.sample(
                        guidance_mode="v2v_apg",
                        num_inference_steps=40,
                        flow_shift=5.0,
                        prompt_embeds=diffusion.action,
                        uncond_prompt_embeds=diffusion.negative,
                        omega_txt=1.0,
                        omega_scale=1.0,
                        eta=1.0,
                        norm_threshold=0.0,
                        momentum=0.0,
                    )
        self.assertNotIn("sample", vars(diffusion))
        self.assertNotIn("shared_step", vars(diffusion))
        self.assertNotIn("step", vars(diffusion.scheduler))

    def test_v8_fake_sampler_replays_noop_section_at_rho_zero(self):
        diffusion = FakeDiffusion()
        import torch

        adapter = FakeAdapter(diffusion)
        source = torch.zeros((1, 2, 21, 2, 2), dtype=torch.float32)
        with mock.patch.object(
            inference.tri,
            "validate_runtime_source_identity",
            return_value="0" * 64,
        ), mock.patch.object(
            inference.tri, "_validate_scheduler_contract", return_value=None
        ):
            with inference.relational_motion_commutator_unipc_hook(
                diffusion,
                adapter_model=adapter,
                source_clean=source,
                noop_prompt_embeds=diffusion.noop,
                latent_shape=source.shape,
                bernini_commit=inference.tri.PINNED_BERNINI_COMMIT,
                wan_diffusion_path=Path("/not/read/while/mocked.py"),
                operator_mode=inference.V8_RECONSTRUCTION_SECTION_FQT,
            ) as trace:
                diffusion.sample(
                    guidance_mode="v2v_apg",
                    num_inference_steps=40,
                    flow_shift=5.0,
                    prompt_embeds=diffusion.action,
                    uncond_prompt_embeds=diffusion.negative,
                    omega_txt=1.0,
                    omega_scale=1.0,
                    eta=1.0,
                    norm_threshold=0.0,
                    momentum=0.0,
                )
        validated = inference.validate_execution_trace(
            trace, runtime_schedule_audit=_runtime_schedule_audit()
        )
        self.assertEqual(validated["totals"]["transformer_forwards"], 200)
        self.assertEqual(
            validated["totals"]["rho_zero_exact_official_steps"], 0
        )
        self.assertEqual(
            validated["totals"]["rho_zero_noop_clean_section_steps"],
            len(inference.LATE_EXACT_STEPS),
        )
        self.assertTrue(all(
            record.exact_noop_phase_zero for record in trace.records
        ))
        self.assertTrue(all(
            record.v8_scheduler_model_output_fp32 for record in trace.records
        ))
        self.assertTrue(all(
            value.dtype == torch.float32 for value in diffusion.scheduler.received
        ))
        self.assertNotIn("sample", vars(diffusion))
        self.assertNotIn("shared_step", vars(diffusion))
        self.assertNotIn("step", vars(diffusion.scheduler))

    def test_core_hook_accepts_only_trained_v8_radius_ablation(self):
        diffusion = FakeDiffusion()
        import torch

        adapter = FakeAdapter(diffusion)
        source = torch.zeros((1, 2, 21, 2, 2), dtype=torch.float32)
        config = inference.feasible_quotient_config_for_radius_scale(4.0)
        common = {
            "adapter_model": adapter,
            "source_clean": source,
            "noop_prompt_embeds": diffusion.noop,
            "latent_shape": source.shape,
            "bernini_commit": inference.tri.PINNED_BERNINI_COMMIT,
            "wan_diffusion_path": Path("/not/read/while/mocked.py"),
            "operator_mode": inference.V8_RECONSTRUCTION_SECTION_FQT,
            "feasible_quotient_config": config,
        }
        with mock.patch.object(
            inference.tri,
            "validate_runtime_source_identity",
            return_value="0" * 64,
        ), mock.patch.object(
            inference.tri, "_validate_scheduler_contract", return_value=None
        ):
            with self.assertRaisesRegex(
                inference.RelationalMotionCommutatorInferenceError,
                "trained V8",
            ):
                with inference.relational_motion_commutator_unipc_hook(
                    diffusion,
                    v8_training_matched=False,
                    **common,
                ):
                    pass
            with inference.relational_motion_commutator_unipc_hook(
                diffusion,
                v8_training_matched=True,
                **common,
            ) as trace:
                contract = trace.as_dict()["contract"]
                self.assertEqual(contract["v8_radius_scale"], 4.0)
                self.assertTrue(
                    contract["v8_radius_scale_inference_only_ablation"]
                )
                self.assertFalse(contract["operator_training_matched"])


if __name__ == "__main__":
    unittest.main()
