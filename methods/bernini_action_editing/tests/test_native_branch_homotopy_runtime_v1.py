#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import native_branch_homotopy_runtime_v1 as runtime  # noqa: E402
import self_guided_action_field_v1 as sgaf  # noqa: E402


class RuntimeStaticContractTests(unittest.TestCase):
    def test_generic_dog_and_human_p_r_geometry(self) -> None:
        dog = runtime.NativeBranchHomotopyRuntimeConfig(
            target_latent_shape=(1, 16, 21, 60, 62)
        )
        human = runtime.NativeBranchHomotopyRuntimeConfig(
            target_latent_shape=(1, 16, 21, 64, 58)
        )
        dog.validate()
        human.validate()
        self.assertEqual((dog.target_patch_tokens, dog.reference_patch_tokens), (19_530, 930))
        self.assertEqual((human.target_patch_tokens, human.reference_patch_tokens), (19_488, 928))
        self.assertEqual(dog.low_vi_tokens, 2 * 19_530 + 4 * 930)
        self.assertEqual(human.high_i_tokens, 19_488 + 4 * 928)

    def test_defaults_and_native_orders_are_fully_pinned(self) -> None:
        config = runtime.NativeBranchHomotopyRuntimeConfig(
            target_latent_shape=(1, 16, 21, 2, 2)
        )
        config.validate()
        self.assertEqual(config.expected_steps, 40)
        self.assertEqual(config.expected_num_frames, 81)
        self.assertEqual(config.expected_flow_shift, 5.0)
        self.assertEqual(config.omega_image, 4.5)
        self.assertEqual(config.omega_text, 4.0)
        self.assertEqual(config.eta, 0.5)
        self.assertEqual(config.image_norm_threshold, 50.0)
        self.assertEqual(config.text_norm_threshold, 50.0)
        self.assertEqual(config.momentum, 0.0)
        self.assertEqual(
            runtime.EXPECTED_PATCH_SOURCE_IDS,
            (1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 0.0),
        )
        self.assertEqual(
            runtime.PER_STEP_FORWARD_ORDER,
            (
                "low-vi-negative",
                "low-vi-action",
                "high-none-negative",
                "high-i-negative",
                "high-i-action",
            ),
        )

    def test_nonzero_momentum_and_changed_apg_constants_fail(self) -> None:
        for kwargs, message in (
            ({"momentum": 0.1}, "momentum"),
            ({"eta": 1.0}, "eta"),
            ({"omega_image": 3.0}, "omega_image"),
            ({"text_norm_threshold": 40.0}, "text_norm_threshold"),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(
                    runtime.NativeBranchHomotopyRuntimeError, message
                ):
                    runtime.NativeBranchHomotopyRuntimeConfig(
                        target_latent_shape=(1, 16, 21, 2, 2), **kwargs
                    ).validate()


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None,
    "PyTorch is required",
)
class RuntimePatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        cls.torch = torch

    def setUp(self) -> None:
        torch = self.torch
        self._saved_vendor = sys.modules.get(runtime.VENDOR_APG_MODULE)
        vendor = ModuleType(runtime.VENDOR_APG_MODULE)

        class MomentumBuffer:
            def __init__(self, momentum):
                self.momentum = momentum
                self.running_average = 0

            def update(self, update_value):
                self.running_average = update_value + self.momentum * self.running_average

        MomentumBuffer.__module__ = runtime.VENDOR_APG_MODULE

        def normalize_diff(diff, base_pred, momentum_buffer, eta, norm_threshold):
            import torch.nn.functional as torch_f

            if momentum_buffer is not None:
                momentum_buffer.update(diff)
                diff = momentum_buffer.running_average
            if norm_threshold > 0:
                ones = torch.ones_like(diff)
                diff_norm = diff.norm(p=2, dim=[-1, -2, -4], keepdim=True)
                scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
                diff = diff * scale_factor
            v0, v1 = diff.double(), base_pred.double()
            v1 = torch_f.normalize(v1, dim=[-1, -2, -4])
            parallel = (v0 * v1).sum(dim=[-1, -2, -4], keepdim=True) * v1
            orthogonal = v0 - parallel
            return orthogonal.to(diff.dtype) + eta * parallel.to(diff.dtype)

        def normalized_guidance_chain(
            pred_uncond,
            preds,
            scales,
            momentum_buffers,
            eta,
            norm_thresholds,
        ):
            bases = [pred_uncond] + list(preds)
            result = pred_uncond
            for index, cond in enumerate(preds):
                normalized = normalize_diff(
                    cond - bases[index],
                    cond,
                    momentum_buffers[index],
                    eta,
                    norm_thresholds[index],
                )
                result = result + scales[index] * normalized
            return result

        normalized_guidance_chain.__module__ = runtime.VENDOR_APG_MODULE
        vendor.MomentumBuffer = MomentumBuffer
        vendor.normalized_guidance_chain = normalized_guidance_chain
        sys.modules[runtime.VENDOR_APG_MODULE] = vendor
        self.vendor = vendor
        self.low_action = torch.full((1, 3, 4), 0.21)
        self.low_negative = torch.full((1, 2, 4), -0.17)
        self.high_action = torch.full((1, 5, 4), 0.39)

    def tearDown(self) -> None:
        if self._saved_vendor is None:
            sys.modules.pop(runtime.VENDOR_APG_MODULE, None)
        else:
            sys.modules[runtime.VENDOR_APG_MODULE] = self._saved_vendor

    def _config(self) -> runtime.NativeBranchHomotopyRuntimeConfig:
        return runtime.NativeBranchHomotopyRuntimeConfig(
            target_latent_shape=(1, 16, 21, 2, 2)
        )

    def _diffusion(self, **flags):
        return _FakeDiffusion(torch=self.torch, **flags)

    def _sample_kwargs(self, diffusion):
        torch = self.torch
        return {
            "prompt_embeds": self.low_action,
            "uncond_prompt_embeds": self.low_negative,
            "image_vae_latents": None,
            "multi_video_vae_latents": [torch.full((1, 16, 21, 2, 2), 0.2)],
            "multi_image_vae_latents": [
                torch.full((1, 16, 1, 2, 2), float(index + 1) / 10.0)
                for index in range(4)
            ],
            "width": 16,
            "height": 16,
            "num_frames": 81,
            "num_inference_steps": 40,
            "guidance_mode": "v2v_apg",
            "omega_vid": 1.25,
            "omega_img": 4.5,
            "omega_txt": 4.0,
            "omega_scale": 0.8,
            "flow_shift": 5.0,
            "seed": 7,
            "eta": 0.5,
            "norm_threshold": (50.0, 50.0),
            "momentum": 0.0,
            "device": "cpu",
        }

    def _run(self, diffusion=None):
        diffusion = diffusion or self._diffusion()
        patch = runtime.NativeBranchHomotopyRuntimePatch(
            diffusion,
            r2v_action_prompt_embeds=self.high_action,
            config=self._config(),
        )
        before = (
            diffusion.sample.__func__,
            diffusion.shared_step.__func__,
            diffusion.transformer.patch_vae_latent.__func__,
            diffusion.scheduler.step.__func__,
        )
        patch.install()
        try:
            result = diffusion.sample(**self._sample_kwargs(diffusion))
        finally:
            patch.restore()
        receipt = patch.finalize()
        after = (
            diffusion.sample.__func__,
            diffusion.shared_step.__func__,
            diffusion.transformer.patch_vae_latent.__func__,
            diffusion.scheduler.step.__func__,
        )
        self.assertEqual(before, after)
        self.assertNotIn("sample", vars(diffusion))
        self.assertNotIn("shared_step", vars(diffusion))
        self.assertNotIn("patch_vae_latent", vars(diffusion.transformer))
        self.assertNotIn("step", vars(diffusion.scheduler))
        return diffusion, patch, receipt, result

    def test_exact40_has_five_forwards_ten_patches_and_one_scheduler(self) -> None:
        diffusion, _, receipt, result = self._run()
        self.assertEqual(tuple(result.shape), (1, 21, 64))
        self.assertEqual(diffusion.shared_call_count, 200)
        self.assertEqual(receipt["transformer_forwards"], 200)
        self.assertEqual(receipt["low_vi_forwards"], 80)
        self.assertEqual(receipt["high_r2v4_forwards"], 120)
        self.assertEqual(receipt["patch_vae_latent_calls"], 400)
        self.assertEqual(receipt["original_scheduler_calls"], 40)
        self.assertEqual(
            receipt["schedule_preflight"]["live_schedule_digest"],
            native_schedule.PINNED_NATIVE_UNIPC40_SCHEDULE_DIGEST,
        )
        self.assertEqual(receipt["schedule_preflight"]["timestep_count"], 40)
        self.assertEqual(
            receipt["schedule_preflight"]["sigma_count_including_terminal"], 41
        )
        self.assertTrue(
            receipt["schedule_preflight"][
                "validated_before_first_transformer_forward"
            ]
        )
        self.assertEqual(receipt["per_step_forward_order"], list(runtime.PER_STEP_FORWARD_ORDER))
        self.assertTrue(receipt["low_official_apg_exact_parity_all_steps"])
        self.assertEqual(len(receipt["trace"]), 40)
        self.assertTrue(all(row["transformer_forwards"] == 5 for row in receipt["trace"]))
        self.assertTrue(all(row["original_scheduler_calls"] == 1 for row in receipt["trace"]))
        self.assertTrue(all(row["freeze_safe_no_grad_outputs"] for row in receipt["trace"]))
        self.assertEqual(
            [call["prompt"] for call in diffusion.forward_calls[:5]],
            ["low-negative", "low-action", "low-negative", "low-negative", "high-action"],
        )

    def test_fp32_conditions_are_authenticated_after_native_bfloat16_conversion(self) -> None:
        torch = self.torch
        diffusion, _, receipt, _ = self._run(
            self._diffusion(transformer_dtype=torch.bfloat16)
        )
        self.assertEqual(receipt["patch_vae_latent_calls"], 400)
        self.assertEqual(diffusion.transformer.patch_log[0][0].dtype, torch.bfloat16)
        self.assertEqual(diffusion.transformer.patch_log[1][0].dtype, torch.bfloat16)
        self.assertEqual(diffusion.transformer.patch_log[2][0].dtype, torch.bfloat16)
        self.assertTrue(
            self._sample_kwargs(diffusion)["multi_video_vae_latents"][0].dtype
            == torch.float32
        )

    def test_smoothstep_endpoints_are_direct_scheduler_objects(self) -> None:
        diffusion, _, receipt, _ = self._run()
        first = receipt["trace"][0]
        last = receipt["trace"][-1]
        self.assertEqual(first["endpoint"], "high_r2v4_apg")
        self.assertEqual(first["high_r2v4_weight"], 1.0)
        self.assertFalse(first["scheduler_received_original_model_output_object"])
        self.assertTrue(first["endpoint_direct_return_verified"])
        self.assertEqual(last["endpoint"], "low_official_v2v_apg")
        self.assertEqual(last["high_r2v4_weight"], 0.0)
        self.assertTrue(last["scheduler_received_original_model_output_object"])
        self.assertIs(diffusion.scheduler.received_objects[-1], diffusion.official_outputs[-1])
        transitions = [row for row in receipt["trace"] if row["endpoint"] == "transition"]
        self.assertTrue(transitions)
        self.assertTrue(all(0.0 < row["high_r2v4_weight"] < 1.0 for row in transitions))

    def test_wrong_patch_order_fails_before_integration_and_restores(self) -> None:
        diffusion = self._diffusion(wrong_patch_order=True)
        patch = runtime.NativeBranchHomotopyRuntimePatch(
            diffusion,
            r2v_action_prompt_embeds=self.high_action,
            config=self._config(),
        )
        patch.install()
        try:
            with self.assertRaisesRegex(
                runtime.NativeBranchHomotopyRuntimeError,
                "source-id order",
            ):
                diffusion.sample(**self._sample_kwargs(diffusion))
        finally:
            patch.restore()
        self.assertEqual(diffusion.scheduler.call_count, 0)
        self.assertNotIn("sample", vars(diffusion))

    def test_altered_middle_sigma_fails_before_any_forward_or_integration(self) -> None:
        diffusion = self._diffusion(altered_middle_sigma=True)
        patch = runtime.NativeBranchHomotopyRuntimePatch(
            diffusion,
            r2v_action_prompt_embeds=self.high_action,
            config=self._config(),
        )
        patch.install()
        try:
            with self.assertRaisesRegex(
                runtime.NativeBranchHomotopyRuntimeError,
                "live exact40 shift-5 timestep/sigma schedule differs",
            ):
                diffusion.sample(**self._sample_kwargs(diffusion))
        finally:
            patch.restore()
        self.assertEqual(diffusion.shared_call_count, 0)
        self.assertEqual(diffusion.scheduler.call_count, 0)
        self.assertEqual(diffusion.transformer.patch_log, [])

    def test_low_prompt_state_timestep_and_rotary_identity_are_closed(self) -> None:
        for flag, message in (
            ("wrong_action_prompt", "prompt object"),
            ("copied_action_state", "exact same object"),
            ("copied_action_timestep", "exact same object"),
            ("copied_action_rotary", "exact same object"),
        ):
            with self.subTest(flag=flag):
                diffusion = self._diffusion(**{flag: True})
                patch = runtime.NativeBranchHomotopyRuntimePatch(
                    diffusion,
                    r2v_action_prompt_embeds=self.high_action,
                    config=self._config(),
                )
                patch.install()
                try:
                    with self.assertRaisesRegex(
                        runtime.NativeBranchHomotopyRuntimeError,
                        message,
                    ):
                        diffusion.sample(**self._sample_kwargs(diffusion))
                finally:
                    patch.restore()
                self.assertEqual(diffusion.scheduler.call_count, 0)

    def test_corrupt_official_low_apg_fails_before_original_scheduler(self) -> None:
        diffusion = self._diffusion(corrupt_official_apg=True)
        patch = runtime.NativeBranchHomotopyRuntimePatch(
            diffusion,
            r2v_action_prompt_embeds=self.high_action,
            config=self._config(),
        )
        patch.install()
        try:
            with self.assertRaisesRegex(
                runtime.NativeBranchHomotopyRuntimeError,
                "locally rebuilt low V2V APG differs",
            ):
                diffusion.sample(**self._sample_kwargs(diffusion))
        finally:
            patch.restore()
        self.assertEqual(diffusion.scheduler.call_count, 0)

    def test_vendor_function_alias_is_rejected_by_identity(self) -> None:
        original = self.vendor.normalized_guidance_chain

        def alias(*args, **kwargs):
            return original(*args, **kwargs)

        alias.__module__ = "not_the_pinned_vendor"
        self.vendor.normalized_guidance_chain = alias
        with self.assertRaisesRegex(
            runtime.NativeBranchHomotopyRuntimeError,
            "module/function identity",
        ):
            runtime.NativeBranchHomotopyRuntimePatch(
                self._diffusion(),
                r2v_action_prompt_embeds=self.high_action,
                config=self._config(),
            )

    def test_nonzero_sample_momentum_and_trainable_transformer_fail_closed(self) -> None:
        diffusion = self._diffusion(trainable=True)
        with self.assertRaisesRegex(
            runtime.NativeBranchHomotopyRuntimeError,
            "not freeze-safe",
        ):
            runtime.NativeBranchHomotopyRuntimePatch(
                diffusion,
                r2v_action_prompt_embeds=self.high_action,
                config=self._config(),
            )
        diffusion = self._diffusion()
        patch = runtime.NativeBranchHomotopyRuntimePatch(
            diffusion,
            r2v_action_prompt_embeds=self.high_action,
            config=self._config(),
        )
        kwargs = self._sample_kwargs(diffusion)
        kwargs["momentum"] = 0.1
        patch.install()
        try:
            with self.assertRaisesRegex(
                runtime.NativeBranchHomotopyRuntimeError,
                "sample/condition contract",
            ):
                diffusion.sample(**kwargs)
        finally:
            patch.restore()
        self.assertEqual(diffusion.scheduler.call_count, 0)


if importlib.util.find_spec("torch") is not None:
    import torch
    import source_self_native_ref_contrastive_v3 as native_schedule

    class _FakeScheduler:
        def __init__(self) -> None:
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
            self.sigmas = torch.tensor(
                (*native_schedule.NATIVE_UNIPC40_SIGMAS, 0.0),
                dtype=torch.float32,
            )
            self.timesteps = torch.tensor(
                native_schedule.NATIVE_UNIPC40_TIMESTEPS,
                dtype=torch.int64,
            )
            self.step_index = 0
            self.call_count = 0
            self.received_objects = []

        def step(self, model_output, timestep, sample, return_dict=False):
            del timestep, return_dict
            self.received_objects.append(model_output)
            self.call_count += 1
            self.step_index += 1
            return (sample - 0.001 * model_output,)


    class _FakeTransformer(torch.nn.Module):
        def __init__(self, *, trainable=False, dtype=torch.float32) -> None:
            super().__init__()
            self.dtype = dtype
            self.config = SimpleNamespace(
                num_attention_heads=12,
                attention_head_dim=128,
                in_channels=16,
            )
            self.anchor = torch.nn.Parameter(
                torch.zeros(()), requires_grad=bool(trainable)
            )
            self.eval()
            self.patch_log = []

        def patch_vae_latent(self, value, source_id=None):
            sid = float(source_id)
            self.patch_log.append((value, sid))
            if value.ndim == 3:
                tokens = int(value.shape[1])
                mean = value.float().mean(dim=-1, keepdim=True)
            else:
                tokens = int(value.shape[2] * (value.shape[3] // 2) * (value.shape[4] // 2))
                mean = value.float().mean().reshape(1, 1, 1).expand(1, tokens, 1)
            latent = (mean + sid).to(dtype=self.dtype).expand(1, tokens, 1536).contiguous()
            rotary = torch.full((1, 1, tokens, 4), sid, dtype=torch.float32)
            return latent, rotary


    class _FakeDiffusion:
        use_unipc = True
        transformer_2 = None

        def __init__(
            self,
            *,
            torch,
            wrong_patch_order=False,
            wrong_action_prompt=False,
            copied_action_state=False,
            copied_action_timestep=False,
            copied_action_rotary=False,
            corrupt_official_apg=False,
            trainable=False,
            transformer_dtype=torch.float32,
            altered_middle_sigma=False,
        ) -> None:
            self.torch = torch
            self.transformer = _FakeTransformer(
                trainable=trainable,
                dtype=transformer_dtype,
            )
            self.scheduler = _FakeScheduler()
            if altered_middle_sigma:
                self.scheduler.sigmas[20] += torch.tensor(0.001, dtype=torch.float32)
            self.wrong_patch_order = wrong_patch_order
            self.wrong_action_prompt = wrong_action_prompt
            self.copied_action_state = copied_action_state
            self.copied_action_timestep = copied_action_timestep
            self.copied_action_rotary = copied_action_rotary
            self.corrupt_official_apg = corrupt_official_apg
            self.shared_call_count = 0
            self.forward_calls = []
            self.official_outputs = []

        def shared_step(
            self,
            model_id,
            noisy_latents,
            timesteps,
            cond_embeds,
            rotary_embs,
            batch_vae_seqlen,
            batch_text_seqlen,
        ):
            del model_id, batch_vae_seqlen, batch_text_seqlen
            self.shared_call_count += 1
            if cond_embeds.shape[1] == 2:
                prompt = "low-negative"
            elif cond_embeds.shape[1] == 3:
                prompt = "low-action"
            elif cond_embeds.shape[1] == 5:
                prompt = "high-action"
            else:
                prompt = "unknown"
            self.forward_calls.append(
                {
                    "prompt": prompt,
                    "noisy": noisy_latents,
                    "timestep": timesteps,
                    "rotary": rotary_embs,
                }
            )
            context = noisy_latents.float().mean(dim=1, keepdim=True)
            text = cond_embeds.float().mean()
            rotary = rotary_embs.float().mean(dim=(1, 2, 3), keepdim=False).reshape(1, 1, 1)
            return noisy_latents[:, :, :64] + context[:, :, :1] + text + 0.01 * rotary

        def sample(
            self,
            prompt_embeds,
            uncond_prompt_embeds,
            image_vae_latents,
            multi_video_vae_latents,
            multi_image_vae_latents,
            width,
            height,
            num_frames,
            num_inference_steps,
            guidance_mode,
            omega_vid,
            omega_img,
            omega_txt,
            omega_scale,
            flow_shift,
            seed,
            eta,
            norm_threshold,
            momentum,
            device,
        ):
            del (
                image_vae_latents,
                width,
                height,
                num_frames,
                guidance_mode,
                omega_vid,
                omega_img,
                omega_scale,
                flow_shift,
                seed,
                device,
            )
            sample = torch.linspace(-0.2, 0.3, 21 * 64, dtype=torch.float32).reshape(1, 21, 64)
            low_momentum = sgaf._MomentumBuffer(momentum, branch="fake-official-low")
            for index in range(num_inference_steps):
                timestep = self.scheduler.timesteps[index]
                video = self.transformer.patch_vae_latent(
                    multi_video_vae_latents[0].to(dtype=self.transformer.dtype),
                    source_id=1.0,
                )
                vi_refs = []
                for ref_index, ref in enumerate(multi_image_vae_latents):
                    vi_sid = float(ref_index + 2)
                    i_sid = float(ref_index + 1)
                    if self.wrong_patch_order and index == 0 and ref_index == 0:
                        vi_sid = 9.0
                    vi_refs.append(
                        self.transformer.patch_vae_latent(
                            ref.to(dtype=self.transformer.dtype), source_id=vi_sid
                        )
                    )
                    self.transformer.patch_vae_latent(
                        ref.to(dtype=self.transformer.dtype), source_id=i_sid
                    )
                target = self.transformer.patch_vae_latent(
                    sgaf._packed_to_spatial(
                        sample,
                        (1, 16, 21, 2, 2),
                    ).to(dtype=self.transformer.dtype),
                    source_id=0.0,
                )
                vi_latent = torch.cat(
                    [video[0], *(part[0] for part in vi_refs), target[0]], dim=1
                )
                vi_rotary = torch.cat(
                    [video[1], *(part[1] for part in vi_refs), target[1]], dim=2
                )
                expanded = timestep.expand(1)

                def forward(text, *, action=False):
                    latent = vi_latent.clone() if action and self.copied_action_state else vi_latent
                    rotary = vi_rotary.clone() if action and self.copied_action_rotary else vi_rotary
                    step = expanded.clone() if action and self.copied_action_timestep else expanded
                    prompt = text.clone() if action and self.wrong_action_prompt else text
                    return self.shared_step(
                        model_id="transformer_1",
                        noisy_latents=latent,
                        timesteps=step,
                        cond_embeds=prompt,
                        rotary_embs=rotary,
                        batch_vae_seqlen=[int(latent.shape[1])],
                        batch_text_seqlen=[int(prompt.shape[1])],
                    )[:, -sample.shape[1] :, :]

                negative = forward(uncond_prompt_embeds)
                action = forward(prompt_embeds, action=True)
                sigma = self.scheduler.sigmas[index]
                official = sgaf._guided_velocity(
                    sample,
                    negative,
                    action,
                    sigma,
                    shape=(1, 16, 21, 2, 2),
                    parameters=sgaf._APGParameters(
                        guidance_scale=omega_txt,
                        eta=eta,
                        norm_threshold=norm_threshold[0],
                        momentum=momentum,
                    ),
                    momentum_buffer=low_momentum,
                    output_like=sample,
                )
                if self.corrupt_official_apg and index == 0:
                    official = official + 0.001
                self.official_outputs.append(official)
                sample = self.scheduler.step(
                    official,
                    timestep,
                    sample,
                    return_dict=False,
                )[0]
            return sample.contiguous()


if __name__ == "__main__":
    unittest.main()
