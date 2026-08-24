#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import self_guided_action_field_v1 as sgaf  # noqa: E402
from self_guided_action_field_v1 import (  # noqa: E402
    ActionFieldConfig,
    NativeRV2VActionFieldPatch,
    SelfGuidedActionFieldError,
    clip_action_delta_by_native_text_rms,
    module_state_hash_certificate,
    smooth_action_gate,
)


class GateTests(unittest.TestCase):
    def test_smooth_gate_endpoints_and_midpoint(self) -> None:
        self.assertEqual(
            smooth_action_gate(0.1, zero_below=0.2, full_above=0.6), 0.0
        )
        self.assertEqual(
            smooth_action_gate(0.8, zero_below=0.2, full_above=0.6), 1.0
        )
        self.assertAlmostEqual(
            smooth_action_gate(0.4, zero_below=0.2, full_above=0.6), 0.5
        )

    def test_invalid_gate_rejected(self) -> None:
        with self.assertRaises(SelfGuidedActionFieldError):
            smooth_action_gate(0.5, zero_below=0.7, full_above=0.6)


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None,
    "PyTorch is required",
)
class TensorAndPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        cls.torch = torch

    def setUp(self) -> None:
        torch = self.torch
        self.action = torch.full((1, 3, 4), 0.11)
        self.negative = torch.full((1, 2, 4), -0.17)
        self.target = torch.full((1, 4, 4), 0.31)
        self.source = torch.full((1, 5, 4), -0.29)

    def _config(self, *, scale: float, steps: int = 2, maximum_ratio: float = 100.0):
        return ActionFieldConfig(
            target_patch_tokens=1,
            effective_scale=scale,
            target_latent_shape=(1, 16, 1, 2, 2),
            expected_condition_prefix_tokens=5,
            expected_steps=steps,
            native_text_guidance_scale=4.0,
            sigma_zero_below=0.0,
            sigma_full_above=0.1,
            maximum_delta_to_native_text_rms=maximum_ratio,
        )

    def _diffusion(self, **flags):
        return _FakeDiffusion(
            torch=self.torch,
            prompts={
                "action": self.action,
                "negative": self.negative,
                "target": self.target,
                "source": self.source,
            },
            **flags,
        )

    def _run(self, *, scale: float = 1.0, diffusion=None, steps: int = 2):
        diffusion = diffusion or self._diffusion()
        patch = NativeRV2VActionFieldPatch(
            diffusion,
            target_t2v_embeds=self.target,
            source_t2v_embeds=self.source,
            config=self._config(scale=scale, steps=steps),
        )
        before = (
            diffusion.sample.__func__,
            diffusion.shared_step.__func__,
            diffusion.scheduler.step.__func__,
        )
        patch.install()
        try:
            result = diffusion.sample(
                prompt_embeds=self.action,
                uncond_prompt_embeds=self.negative,
                num_inference_steps=steps,
                guidance_mode="v2v_apg",
                omega_txt=4.0,
                omega_scale=0.75,
                flow_shift=5.0,
                eta=0.6,
                norm_threshold=(50.0, 50.0),
                momentum=0.25,
            )
        finally:
            patch.restore()
        receipt = patch.finalize()
        after = (
            diffusion.sample.__func__,
            diffusion.shared_step.__func__,
            diffusion.scheduler.step.__func__,
        )
        self.assertEqual(before, after)
        self.assertNotIn("sample", vars(diffusion))
        self.assertNotIn("shared_step", vars(diffusion))
        self.assertNotIn("step", vars(diffusion.scheduler))
        return diffusion, patch, receipt, result

    def test_rms_clip_preserves_small_and_clips_large(self) -> None:
        torch = self.torch
        native = torch.ones(1, 2, 3)
        small, raw, native_rms, multiplier = clip_action_delta_by_native_text_rms(
            torch.full_like(native, 0.5), native, maximum_ratio=1.5
        )
        self.assertTrue(torch.equal(small, torch.full_like(native, 0.5)))
        self.assertAlmostEqual(raw, 0.5)
        self.assertAlmostEqual(native_rms, 1.0)
        self.assertEqual(multiplier, 1.0)
        clipped, raw, _, multiplier = clip_action_delta_by_native_text_rms(
            torch.full_like(native, 4.0), native, maximum_ratio=1.5
        )
        self.assertAlmostEqual(raw, 4.0)
        self.assertAlmostEqual(multiplier, 0.375)
        self.assertTrue(torch.allclose(clipped, torch.full_like(native, 1.5)))

    def test_two_official_plus_three_target_only_queries_per_step(self) -> None:
        diffusion, _, receipt, _ = self._run(scale=1.0)
        self.assertEqual(receipt["native_rv2v_forwards"], 4)
        self.assertEqual(receipt["frozen_t2v_teacher_forwards"], 6)
        self.assertEqual(receipt["original_scheduler_calls"], 2)
        self.assertEqual(
            receipt["per_step_call_graph"],
            [
                "native-negative",
                "native-action",
                "t2v-uncond",
                "t2v-target",
                "t2v-source",
            ],
        )
        self.assertEqual(
            [row["prompt"] for row in diffusion.forward_calls],
            [
                "negative",
                "action",
                "negative",
                "target",
                "source",
                "negative",
                "action",
                "negative",
                "target",
                "source",
            ],
        )
        for offset in (0, 5):
            native_negative, native_action, t2v_u, t2v_t, t2v_s = (
                diffusion.forward_calls[offset : offset + 5]
            )
            self.assertIs(native_negative["noisy"], native_action["noisy"])
            self.assertIs(native_negative["timestep"], native_action["timestep"])
            self.assertIs(native_negative["rotary"], native_action["rotary"])
            self.assertEqual(native_negative["vae_len"], [6])
            for teacher in (t2v_u, t2v_t, t2v_s):
                self.assertIs(teacher["noisy"], t2v_u["noisy"])
                self.assertIs(teacher["timestep"], native_action["timestep"])
                self.assertIs(teacher["rotary"], t2v_u["rotary"])
                self.assertEqual(tuple(teacher["noisy"].shape), (1, 1, 1536))
                self.assertEqual(tuple(teacher["rotary"].shape), (1, 1, 1, 8))
                self.assertEqual(teacher["vae_len"], [1])

    def test_scheduler_algebra_uses_official_apg_base_without_omega_division(self) -> None:
        torch = self.torch
        diffusion, patch, receipt, _ = self._run(scale=0.75)
        self.assertEqual(len(diffusion.scheduler.calls), 2)
        self.assertTrue(all(row["native_official_apg_exact_parity"] for row in receipt["trace"]))
        target_buffer = sgaf._MomentumBuffer(0.25, branch="test-target")
        source_buffer = sgaf._MomentumBuffer(0.25, branch="test-source")
        apg = sgaf._APGParameters(
            guidance_scale=4.0,
            eta=0.6,
            norm_threshold=50.0,
            momentum=0.25,
        )
        for index, scheduler_call in enumerate(diffusion.scheduler.calls):
            record = diffusion.pre_scheduler[index]
            sigma = diffusion.scheduler.sigmas[index]
            target_guided = sgaf._guided_velocity(
                scheduler_call["sample"],
                record["teacher_uncond"],
                record["teacher_target"],
                sigma,
                shape=(1, 16, 1, 2, 2),
                parameters=apg,
                momentum_buffer=target_buffer,
                output_like=record["official"],
            )
            source_guided = sgaf._guided_velocity(
                scheduler_call["sample"],
                record["teacher_uncond"],
                record["teacher_source"],
                sigma,
                shape=(1, 16, 1, 2, 2),
                parameters=apg,
                momentum_buffer=source_buffer,
                output_like=record["official"],
            )
            expected = record["official"] + 0.75 * (target_guided - source_guided)
            self.assertTrue(torch.equal(scheduler_call["model_output"], expected))
            divided = record["official"] + (0.75 / 4.0) * (
                target_guided - source_guided
            )
            self.assertFalse(torch.equal(scheduler_call["model_output"], divided))
            raw_cfg_delta = 4.0 * (
                record["teacher_target"].float()
                - record["teacher_source"].float()
            )
            self.assertFalse(torch.allclose(target_guided - source_guided, raw_cfg_delta))
        self.assertIs(receipt["injection_divided_by_omega_txt"], False)
        self.assertEqual(patch.teacher_call_count, 6)

    def test_scale_zero_is_bit_and_object_exact_noop_with_no_teacher_queries(self) -> None:
        torch = self.torch
        baseline = self._diffusion()
        baseline_result = baseline.sample(
            prompt_embeds=self.action,
            uncond_prompt_embeds=self.negative,
            num_inference_steps=2,
            guidance_mode="v2v_apg",
            omega_txt=4.0,
            omega_scale=0.75,
            flow_shift=5.0,
            eta=0.6,
            norm_threshold=(50.0, 50.0),
            momentum=0.25,
        )
        patched = self._diffusion()
        patched, _, receipt, patched_result = self._run(
            scale=0.0, diffusion=patched
        )
        self.assertTrue(torch.equal(baseline_result, patched_result))
        self.assertEqual(receipt["frozen_t2v_teacher_forwards"], 0)
        self.assertIs(receipt["scale_zero_exact_noop"], True)
        self.assertEqual(len(patched.forward_calls), 4)
        self.assertTrue(
            all(
                call["model_output"] is original
                for call, original in zip(
                    patched.scheduler.calls, patched.official_outputs
                )
            )
        )
        self.assertTrue(
            all(
                row["scale_zero_exact_model_output_object"]
                for row in receipt["trace"]
            )
        )

    def test_wrong_native_state_and_metadata_fail_before_scheduler(self) -> None:
        for flag, message in (
            ("mismatch_action_state", "exact same object"),
            ("wrong_action_prompt", "exact authenticated object"),
            ("wrong_model_id", "model_id"),
            ("bad_batch_vae", "batch_vae_seqlen"),
            ("wrong_hidden_dim", "noisy geometry"),
            ("copied_shared_timestep", "zero-stride expand"),
        ):
            with self.subTest(flag=flag):
                diffusion = self._diffusion(**{flag: True})
                patch = NativeRV2VActionFieldPatch(
                    diffusion,
                    target_t2v_embeds=self.target,
                    source_t2v_embeds=self.source,
                    config=self._config(scale=1.0),
                )
                patch.install()
                try:
                    with self.assertRaisesRegex(SelfGuidedActionFieldError, message):
                        diffusion.sample(
                            prompt_embeds=self.action,
                            uncond_prompt_embeds=self.negative,
                            num_inference_steps=2,
                            guidance_mode="v2v_apg",
                            omega_txt=4.0,
                            eta=0.6,
                            norm_threshold=(50.0, 50.0),
                            momentum=0.25,
                        )
                finally:
                    patch.restore()
                self.assertEqual(diffusion.scheduler.calls, [])
                self.assertNotIn("sample", vars(diffusion))
                self.assertNotIn("shared_step", vars(diffusion))
                self.assertNotIn("step", vars(diffusion.scheduler))

    def test_native_apg_mismatch_fails_before_original_scheduler(self) -> None:
        diffusion = self._diffusion(corrupt_official_apg=True)
        patch = NativeRV2VActionFieldPatch(
            diffusion,
            target_t2v_embeds=self.target,
            source_t2v_embeds=self.source,
            config=self._config(scale=1.0),
        )
        patch.install()
        try:
            with self.assertRaisesRegex(
                SelfGuidedActionFieldError, "native APG differs"
            ):
                diffusion.sample(
                    prompt_embeds=self.action,
                    uncond_prompt_embeds=self.negative,
                    num_inference_steps=2,
                    guidance_mode="v2v_apg",
                    omega_txt=4.0,
                    eta=0.6,
                    norm_threshold=(50.0, 50.0),
                    momentum=0.25,
                )
        finally:
            patch.restore()
        self.assertEqual(diffusion.scheduler.calls, [])

    def test_scheduler_failure_still_allows_full_rollback(self) -> None:
        diffusion = self._diffusion(scheduler_failure=True)
        patch = NativeRV2VActionFieldPatch(
            diffusion,
            target_t2v_embeds=self.target,
            source_t2v_embeds=self.source,
            config=self._config(scale=1.0),
        )
        patch.install()
        try:
            with self.assertRaisesRegex(RuntimeError, "scheduler failure"):
                diffusion.sample(
                    prompt_embeds=self.action,
                    uncond_prompt_embeds=self.negative,
                    num_inference_steps=2,
                    guidance_mode="v2v_apg",
                    omega_txt=4.0,
                    eta=0.6,
                    norm_threshold=(50.0, 50.0),
                    momentum=0.25,
                )
        finally:
            patch.restore()
        self.assertNotIn("sample", vars(diffusion))
        self.assertNotIn("shared_step", vars(diffusion))
        self.assertNotIn("step", vars(diffusion.scheduler))
        with self.assertRaises(SelfGuidedActionFieldError):
            patch.finalize()

    def test_parameter_and_buffer_freeze_certificate(self) -> None:
        torch = self.torch

        class Frozen(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.arange(4, dtype=torch.float32), requires_grad=False
                )
                self.register_buffer(
                    "running", torch.arange(3, dtype=torch.bfloat16)
                )
                self.register_buffer("scalar", torch.tensor(0.5))

        model = Frozen()
        before = module_state_hash_certificate(model)
        self.assertEqual(before, module_state_hash_certificate(model))
        self.assertEqual(before["parameter_tensors"], 1)
        self.assertEqual(before["buffer_tensors"], 2)
        model.running.add_(1)
        after = module_state_hash_certificate(model)
        self.assertNotEqual(
            before["parameters_and_buffers_sha256"],
            after["parameters_and_buffers_sha256"],
        )
        model.weight.requires_grad_(True)
        with self.assertRaisesRegex(SelfGuidedActionFieldError, "trainable"):
            module_state_hash_certificate(model)


class _FakeScheduler:
    def __init__(self, *, torch, fail: bool = False) -> None:
        self.torch = torch
        self.fail = fail
        self.config = SimpleNamespace(
            _class_name="UniPCMultistepScheduler",
            flow_shift=5.0,
            prediction_type="flow_prediction",
            predict_x0=True,
            use_flow_sigmas=True,
            thresholding=False,
            solver_order=2,
            solver_type="bh2",
        )
        self.calls = []
        self.step_index = None

    def set_timesteps(self, steps: int) -> None:
        if steps != 2:
            raise RuntimeError("fake is pinned to two steps")
        self.timesteps = self.torch.tensor([900.0, 700.0], dtype=self.torch.float32)
        self.sigmas = self.torch.tensor([0.9, 0.7, 0.0], dtype=self.torch.float32)
        self.step_index = None

    def index_for_timestep(self, timestep):
        matches = (self.timesteps == timestep).nonzero().reshape(-1)
        return int(matches[0].item())

    def step(self, model_output, timestep, sample, return_dict=False):
        if self.fail:
            raise RuntimeError("scheduler failure")
        self.calls.append(
            {
                "model_output": model_output,
                "timestep": timestep,
                "sample": sample,
                "return_dict": return_dict,
            }
        )
        if self.step_index is None:
            self.step_index = 0
        self.step_index += 1
        return (sample - 0.01 * model_output,)


class _FakeDiffusion:
    use_unipc = True
    transformer_2 = None

    def __init__(
        self,
        *,
        torch,
        prompts,
        mismatch_action_state=False,
        wrong_action_prompt=False,
        wrong_model_id=False,
        bad_batch_vae=False,
        wrong_hidden_dim=False,
        copied_shared_timestep=False,
        corrupt_official_apg=False,
        scheduler_failure=False,
    ) -> None:
        self.torch = torch
        self.prompts = prompts
        self.scheduler = _FakeScheduler(torch=torch, fail=scheduler_failure)
        self.mismatch_action_state = mismatch_action_state
        self.wrong_action_prompt = wrong_action_prompt
        self.wrong_model_id = wrong_model_id
        self.bad_batch_vae = bad_batch_vae
        self.wrong_hidden_dim = wrong_hidden_dim
        self.copied_shared_timestep = copied_shared_timestep
        self.corrupt_official_apg = corrupt_official_apg
        self.transformer = SimpleNamespace(
            config=SimpleNamespace(
                num_attention_heads=12,
                attention_head_dim=128,
            )
        )
        self.forward_calls = []
        self.official_outputs = []
        self.pre_scheduler = []
        channel = torch.arange(64, dtype=torch.float32)
        vectors = {
            "negative": torch.sin(channel * 0.17) * 0.3,
            "action": torch.cos(channel * 0.11) * 0.45 + 0.05,
            "target": torch.sin(channel * 0.07 + 0.4) * 0.65,
            "source": torch.cos(channel * 0.13 - 0.2) * 0.55,
        }
        self.prompt_names = {id(value): name for name, value in prompts.items()}
        self.prompt_vectors = {
            id(prompts[name]): vector.to(torch.bfloat16)
            for name, vector in vectors.items()
        }
        self.wrong_prompt = torch.full((1, 6, 4), 0.99)
        self.prompt_names[id(self.wrong_prompt)] = "wrong"
        self.prompt_vectors[id(self.wrong_prompt)] = torch.ones(64, dtype=torch.bfloat16)

    def shared_step(
        self,
        model_id,
        noisy_latents,
        timesteps,
        cond_embeds,
        rotary_embs,
        batch_vae_seqlen=None,
        batch_text_seqlen=None,
        **kwargs,
    ):
        del kwargs
        name = self.prompt_names[id(cond_embeds)]
        self.forward_calls.append(
            {
                "model_id": model_id,
                "noisy": noisy_latents,
                "timestep": timesteps,
                "prompt": name,
                "prompt_object": cond_embeds,
                "rotary": rotary_embs,
                "vae_len": batch_vae_seqlen,
                "text_len": batch_text_seqlen,
            }
        )
        vector = self.prompt_vectors[id(cond_embeds)].to(noisy_latents.device)
        return vector.reshape(1, 1, 64).expand(
            noisy_latents.shape[0], noisy_latents.shape[1], 64
        ).clone()

    def sample(
        self,
        prompt_embeds=None,
        prompt_embeds_t2=None,
        uncond_prompt_embeds=None,
        uncond_embeds_t2=None,
        num_inference_steps=2,
        guidance_mode="v2v_apg",
        omega_txt=4.0,
        omega_scale=0.75,
        flow_shift=5.0,
        eta=0.6,
        norm_threshold=(50.0, 50.0),
        momentum=0.25,
    ):
        del prompt_embeds_t2, uncond_embeds_t2, omega_scale, flow_shift
        torch = self.torch
        self.scheduler.set_timesteps(num_inference_steps)
        sample = torch.linspace(-0.4, 0.6, 64, dtype=torch.float32).reshape(1, 1, 64)
        vendor_momentum = sgaf._MomentumBuffer(momentum, branch="fake-vendor")
        parameters = sgaf._APGParameters(
            guidance_scale=float(omega_txt),
            eta=float(eta),
            norm_threshold=float(
                norm_threshold[0]
                if isinstance(norm_threshold, (tuple, list))
                else norm_threshold
            ),
            momentum=float(momentum),
        )
        for timestep in self.scheduler.timesteps:
            hidden_dim = 64 if self.wrong_hidden_dim else 1536
            source_prefix = torch.zeros(1, 5, hidden_dim, dtype=torch.float32)
            target_hidden = torch.zeros(1, 1, hidden_dim, dtype=torch.float32)
            vi_inp = torch.cat((source_prefix, target_hidden), dim=1)
            rotary = torch.zeros(1, 1, 6, 8, dtype=torch.float32)
            vae_len = [5] if self.bad_batch_vae else [6]
            shared_timestep = timestep.expand(1)
            if self.copied_shared_timestep:
                shared_timestep = shared_timestep.clone()
            negative = self.shared_step(
                model_id="wrong" if self.wrong_model_id else "transformer_1",
                noisy_latents=vi_inp,
                timesteps=shared_timestep,
                cond_embeds=uncond_prompt_embeds,
                rotary_embs=rotary,
                batch_vae_seqlen=vae_len,
                batch_text_seqlen=[uncond_prompt_embeds.shape[1]],
            )
            action_noisy = vi_inp.clone() if self.mismatch_action_state else vi_inp
            action_prompt = self.wrong_prompt if self.wrong_action_prompt else prompt_embeds
            action = self.shared_step(
                model_id="transformer_1",
                noisy_latents=action_noisy,
                timesteps=shared_timestep,
                cond_embeds=action_prompt,
                rotary_embs=rotary,
                batch_vae_seqlen=vae_len,
                batch_text_seqlen=[action_prompt.shape[1]],
            )
            official = sgaf._guided_velocity(
                sample,
                negative[:, -1:, :],
                action[:, -1:, :],
                self.scheduler.sigmas[
                    0 if self.scheduler.step_index is None else self.scheduler.step_index
                ],
                shape=(1, 16, 1, 2, 2),
                parameters=parameters,
                momentum_buffer=vendor_momentum,
                output_like=sample,
            )
            if self.corrupt_official_apg:
                official = official + 0.125
            self.official_outputs.append(official)
            if len(self.forward_calls) >= 5:
                current = self.forward_calls[-5:]
                if [row["prompt"] for row in current] == [
                    "negative",
                    "action",
                    "negative",
                    "target",
                    "source",
                ]:
                    self.pre_scheduler.append(
                        {
                            "official": official,
                            "teacher_uncond": self.prompt_vectors[
                                id(self.prompts["negative"])
                            ].reshape(1, 1, 64),
                            "teacher_target": self.prompt_vectors[
                                id(self.prompts["target"])
                            ].reshape(1, 1, 64),
                            "teacher_source": self.prompt_vectors[
                                id(self.prompts["source"])
                            ].reshape(1, 1, 64),
                        }
                    )
            sample = self.scheduler.step(
                official, timestep, sample, return_dict=False
            )[0]
        return sample


if __name__ == "__main__":
    unittest.main()
