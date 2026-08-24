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

import self_guided_action_field_v1 as sgaf  # noqa: E402
import source_self_native_ref_contrastive_v3 as schedule_contract  # noqa: E402
import t2v_v2v_branch_homotopy_runtime_v1 as runtime  # noqa: E402


class RuntimeStaticContractTests(unittest.TestCase):
    def test_defaults_orders_and_exact40_regions_are_pinned(self) -> None:
        config = runtime.T2VV2VBranchHomotopyRuntimeConfig(
            target_latent_shape=(1, 16, 21, 2, 2)
        )
        config.validate()
        self.assertEqual(config.expected_steps, 40)
        self.assertEqual(config.expected_num_frames, 81)
        self.assertEqual(config.expected_flow_shift, 5.0)
        self.assertEqual(config.omega_text, 4.0)
        self.assertEqual(config.eta, 0.5)
        self.assertEqual(config.norm_threshold, 50.0)
        self.assertEqual(config.momentum, 0.0)
        self.assertEqual(config.expected_text_dim, 4096)
        self.assertEqual(config.target_patch_tokens, 21)
        self.assertEqual(config.low_source_v2v_tokens, 42)
        self.assertEqual(config.high_pure_t2v_tokens, 21)
        self.assertEqual(runtime.EXPECTED_PATCH_SOURCE_IDS, (1.0, 0.0))
        self.assertEqual(
            runtime.PER_STEP_FORWARD_ORDER,
            (
                "low-source-v2v-negative",
                "low-source-v2v-action",
                "high-pure-t2v-negative",
                "high-pure-t2v-action",
            ),
        )
        self.assertEqual(runtime.HIGH_ENDPOINT_STEP_INDICES, tuple(range(0, 9)))
        self.assertEqual(runtime.TRANSITION_STEP_INDICES, tuple(range(9, 26)))
        self.assertEqual(runtime.LOW_ENDPOINT_STEP_INDICES, tuple(range(26, 40)))
        self.assertEqual(
            runtime.PINNED_SCHEDULE_DIGEST,
            "46f3dcb6e2d65cb7921e5217e2a20dfe008b366cfacafe455fee4d3c45f63ae2",
        )

    def test_changed_apg_or_schedule_constants_fail(self) -> None:
        for kwargs, message in (
            ({"momentum": 0.1}, "momentum"),
            ({"eta": 1.0}, "eta"),
            ({"omega_text": 3.0}, "omega_text"),
            ({"norm_threshold": 40.0}, "norm_threshold"),
            ({"expected_flow_shift": 4.0}, "expected_flow_shift"),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(
                    runtime.T2VV2VBranchHomotopyRuntimeError,
                    message,
                ):
                    runtime.T2VV2VBranchHomotopyRuntimeConfig(
                        target_latent_shape=(1, 16, 21, 2, 2),
                        **kwargs,
                    ).validate()


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None,
    "PyTorch is required",
)
class RuntimePatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        torch.set_num_threads(1)
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
                self.running_average = (
                    update_value + self.momentum * self.running_average
                )

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
            parallel = (v0 * v1).sum(
                dim=[-1, -2, -4], keepdim=True
            ) * v1
            orthogonal = v0 - parallel
            return orthogonal.to(diff.dtype) + eta * parallel.to(diff.dtype)

        def normalized_guidance(
            pred_cond,
            pred_uncond,
            guidance_scale,
            momentum_buffer=None,
            eta=1.0,
            norm_threshold=0.0,
        ):
            vendor.single_call_count += 1
            normalized = normalize_diff(
                pred_cond - pred_uncond,
                pred_cond,
                momentum_buffer,
                eta,
                norm_threshold,
            )
            return pred_uncond + guidance_scale * normalized

        def normalized_guidance_chain(
            pred_uncond,
            preds,
            scales,
            momentum_buffers,
            eta,
            norm_thresholds,
        ):
            vendor.chain_call_count += 1
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

        normalized_guidance.__module__ = runtime.VENDOR_APG_MODULE
        normalized_guidance_chain.__module__ = runtime.VENDOR_APG_MODULE
        vendor.MomentumBuffer = MomentumBuffer
        vendor.single_call_count = 0
        vendor.chain_call_count = 0
        vendor.normalized_guidance = normalized_guidance
        vendor.normalized_guidance_chain = normalized_guidance_chain
        sys.modules[runtime.VENDOR_APG_MODULE] = vendor
        self.vendor = vendor

        embed_shape = (1, 512, 4096)
        self.low_action = torch.full(embed_shape, 0.21, dtype=torch.float32)
        self.negative = torch.full(embed_shape, -0.17, dtype=torch.float32)
        self.high_action = torch.full(embed_shape, 0.39, dtype=torch.float32)

    def tearDown(self) -> None:
        if self._saved_vendor is None:
            sys.modules.pop(runtime.VENDOR_APG_MODULE, None)
        else:
            sys.modules[runtime.VENDOR_APG_MODULE] = self._saved_vendor

    def _config(self) -> runtime.T2VV2VBranchHomotopyRuntimeConfig:
        return runtime.T2VV2VBranchHomotopyRuntimeConfig(
            target_latent_shape=(1, 16, 21, 2, 2)
        )

    def _diffusion(self, **flags):
        return _FakeDiffusion(torch=self.torch, vendor=self.vendor, **flags)

    def _sample_kwargs(self, diffusion):
        torch = self.torch
        return {
            "prompt_embeds": self.low_action,
            "prompt_embeds_t2": None,
            "uncond_prompt_embeds": self.negative,
            "uncond_embeds_t2": None,
            "num_frames": 81,
            "width": 16,
            "height": 16,
            "image_vae_latents": None,
            "multi_video_vae_latents": [
                torch.full((1, 16, 21, 2, 2), 0.2, dtype=torch.float32)
            ],
            "multi_image_vae_latents": None,
            "num_inference_steps": 40,
            "guidance_mode": "v2v_apg",
            "omega_vid": 3.0,
            "omega_img": 3.0,
            "omega_txt": 4.0,
            "omega_scale": 0.75,
            "flow_shift": 5.0,
            "seed": 7,
            "device": "cpu",
            "eta": 0.5,
            "norm_threshold": (50.0, 50.0),
            "momentum": 0.0,
        }

    def _run(self, diffusion=None):
        diffusion = diffusion or self._diffusion()
        patch = runtime.T2VV2VBranchHomotopyRuntimePatch(
            diffusion,
            t2v_action_prompt_embeds=self.high_action,
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

    def test_exact40_four_forwards_two_patches_one_scheduler(self) -> None:
        diffusion, _, receipt, result = self._run()
        self.assertEqual(tuple(result.shape), (1, 16, 21, 2, 2))
        self.assertEqual(result.dtype, self.torch.float32)
        self.assertEqual(diffusion.shared_call_count, 160)
        self.assertEqual(receipt["transformer_forwards"], 160)
        self.assertEqual(receipt["low_source_v2v_forwards"], 80)
        self.assertEqual(receipt["high_pure_t2v_forwards"], 80)
        self.assertEqual(receipt["patch_vae_latent_calls"], 80)
        self.assertEqual(receipt["original_scheduler_calls"], 40)
        self.assertEqual(receipt["terminal_sigma"], 0.0)
        self.assertEqual(
            receipt["exact40_shift5_schedule_digest"],
            schedule_contract.PINNED_NATIVE_UNIPC40_SCHEDULE_DIGEST,
        )
        self.assertTrue(receipt["low_stock_apg_exact_parity_all_steps"])
        # 40 stock V2V calls plus 40 independently rebuilt low calls and 40
        # pure-T2V high calls all use the exact helper called by official
        # ``v2v_apg``/``t2v_apg``.  The chain helper must remain unused.
        self.assertEqual(self.vendor.single_call_count, 120)
        self.assertEqual(self.vendor.chain_call_count, 0)
        self.assertEqual(
            receipt["branch_apg"]["function"],
            f"{runtime.VENDOR_APG_MODULE}.normalized_guidance",
        )
        self.assertEqual(
            receipt["runtime_source_identity_enforcement"],
            "external_canary_required",
        )
        self.assertEqual(
            [call["prompt"] for call in diffusion.forward_calls[:4]],
            ["negative", "low-action", "negative", "high-action"],
        )
        self.assertEqual(
            [call["tokens"] for call in diffusion.forward_calls[:4]],
            [42, 42, 21, 21],
        )
        first = diffusion.forward_calls[:4]
        self.assertIs(first[0]["noisy"], first[1]["noisy"])
        self.assertIs(first[0]["timestep"], first[1]["timestep"])
        self.assertIs(first[0]["rotary"], first[1]["rotary"])
        self.assertIs(first[2]["noisy"], first[3]["noisy"])
        self.assertIs(first[2]["timestep"], first[3]["timestep"])
        self.assertIs(first[2]["rotary"], first[3]["rotary"])
        self.assertIs(first[1]["timestep"], first[2]["timestep"])
        self.assertIs(first[0]["prompt_object"], self.negative)
        self.assertIs(first[2]["prompt_object"], self.negative)
        self.assertIs(first[1]["prompt_object"], self.low_action)
        self.assertIs(first[3]["prompt_object"], self.high_action)
        self.assertTrue(
            self.torch.equal(first[0]["noisy"][:, -21:, :], first[2]["noisy"])
        )
        self.assertTrue(
            all(row["transformer_forwards"] == 4 for row in receipt["trace"])
        )
        self.assertTrue(
            all(row["freeze_safe_no_grad_outputs"] for row in receipt["trace"])
        )

    def test_endpoint_partition_and_direct_scheduler_objects(self) -> None:
        diffusion, _, receipt, _ = self._run()
        self.assertEqual(
            [row["endpoint"] for row in receipt["trace"]],
            ["high_pure_t2v_apg"] * 9
            + ["transition"] * 17
            + ["low_source_v2v_apg"] * 14,
        )
        self.assertTrue(
            all(row["endpoint_direct_return_verified"] for row in receipt["trace"][:9])
        )
        self.assertTrue(
            all(
                not row["endpoint_direct_return_verified"]
                for row in receipt["trace"][9:26]
            )
        )
        self.assertTrue(
            all(row["endpoint_direct_return_verified"] for row in receipt["trace"][26:])
        )
        self.assertIs(
            diffusion.scheduler.received_objects[-1],
            diffusion.official_outputs[-1],
        )
        self.assertIsNot(
            diffusion.scheduler.received_objects[0],
            diffusion.official_outputs[0],
        )

    def test_full_live_schedule_must_match_before_first_scheduler_step(self) -> None:
        for flag in ("altered_middle_sigma", "wrong_terminal_sigma", "altered_timestep"):
            with self.subTest(flag=flag):
                diffusion = self._diffusion(**{flag: True})
                patch = runtime.T2VV2VBranchHomotopyRuntimePatch(
                    diffusion,
                    t2v_action_prompt_embeds=self.high_action,
                    config=self._config(),
                )
                patch.install()
                try:
                    with self.assertRaisesRegex(
                        runtime.T2VV2VBranchHomotopyRuntimeError,
                        "live exact40 shift-5",
                    ):
                        diffusion.sample(**self._sample_kwargs(diffusion))
                finally:
                    patch.restore()
                self.assertEqual(diffusion.scheduler.call_count, 0)
                self.assertNotIn("sample", vars(diffusion))

    def test_corrupt_stock_low_apg_fails_before_original_scheduler(self) -> None:
        diffusion = self._diffusion(corrupt_official_apg=True)
        patch = runtime.T2VV2VBranchHomotopyRuntimePatch(
            diffusion,
            t2v_action_prompt_embeds=self.high_action,
            config=self._config(),
        )
        patch.install()
        try:
            with self.assertRaisesRegex(
                runtime.T2VV2VBranchHomotopyRuntimeError,
                "rebuilt low source-V2V APG differs",
            ):
                diffusion.sample(**self._sample_kwargs(diffusion))
        finally:
            patch.restore()
        self.assertEqual(diffusion.scheduler.call_count, 0)

    def test_wrong_patch_order_fails_before_integration(self) -> None:
        diffusion = self._diffusion(wrong_patch_order=True)
        patch = runtime.T2VV2VBranchHomotopyRuntimePatch(
            diffusion,
            t2v_action_prompt_embeds=self.high_action,
            config=self._config(),
        )
        patch.install()
        try:
            with self.assertRaisesRegex(
                runtime.T2VV2VBranchHomotopyRuntimeError,
                "source-id order",
            ):
                diffusion.sample(**self._sample_kwargs(diffusion))
        finally:
            patch.restore()
        self.assertEqual(diffusion.scheduler.call_count, 0)

    def test_source_only_condition_contract_rejects_image_references(self) -> None:
        diffusion = self._diffusion()
        patch = runtime.T2VV2VBranchHomotopyRuntimePatch(
            diffusion,
            t2v_action_prompt_embeds=self.high_action,
            config=self._config(),
        )
        kwargs = self._sample_kwargs(diffusion)
        kwargs["multi_image_vae_latents"] = [
            self.torch.zeros((1, 16, 1, 2, 2), dtype=self.torch.float32)
        ]
        patch.install()
        try:
            with self.assertRaisesRegex(
                runtime.T2VV2VBranchHomotopyRuntimeError,
                "source-video-only",
            ):
                diffusion.sample(**kwargs)
        finally:
            patch.restore()
        self.assertEqual(diffusion.scheduler.call_count, 0)

    def test_bfloat16_native_condition_conversion_is_authenticated(self) -> None:
        diffusion, _, receipt, _ = self._run(
            self._diffusion(transformer_dtype=self.torch.bfloat16)
        )
        self.assertEqual(receipt["patch_vae_latent_calls"], 80)
        self.assertEqual(diffusion.transformer.patch_log[0][0].dtype, self.torch.bfloat16)
        self.assertEqual(diffusion.transformer.patch_log[1][0].dtype, self.torch.bfloat16)

    def test_vendor_single_alias_and_trainable_transformer_are_rejected(self) -> None:
        original = self.vendor.normalized_guidance

        def alias(*args, **kwargs):
            return original(*args, **kwargs)

        alias.__module__ = runtime.VENDOR_APG_MODULE
        alias.__name__ = "normalized_guidance"
        self.vendor.normalized_guidance = alias
        with self.assertRaisesRegex(
            runtime.T2VV2VBranchHomotopyRuntimeError,
            "module/function identity|signature differs",
        ):
            runtime.T2VV2VBranchHomotopyRuntimePatch(
                self._diffusion(),
                t2v_action_prompt_embeds=self.high_action,
                config=self._config(),
            )
        self.vendor.normalized_guidance = original
        with self.assertRaisesRegex(
            runtime.T2VV2VBranchHomotopyRuntimeError,
            "not freeze-safe",
        ):
            runtime.T2VV2VBranchHomotopyRuntimePatch(
                self._diffusion(trainable=True),
                t2v_action_prompt_embeds=self.high_action,
                config=self._config(),
            )


if importlib.util.find_spec("torch") is not None:
    import torch

    class _FakeScheduler:
        def __init__(
            self,
            *,
            altered_middle_sigma=False,
            wrong_terminal_sigma=False,
            altered_timestep=False,
        ) -> None:
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
            self.timesteps = torch.tensor(
                schedule_contract.NATIVE_UNIPC40_TIMESTEPS,
                # Pinned Diffusers UniPC materializes the live timeline as
                # integer timesteps; runtime validation must not assume FP32.
                dtype=torch.int64,
            )
            self.sigmas = torch.tensor(
                (*schedule_contract.NATIVE_UNIPC40_SIGMAS, 0.0),
                dtype=torch.float32,
            )
            if altered_middle_sigma:
                self.sigmas[17] += 0.001
            if wrong_terminal_sigma:
                self.sigmas[40] = 0.001
            if altered_timestep:
                self.timesteps[18] += 1
            self.step_index = 0
            self.call_count = 0
            self.received_objects = []

        def set_timesteps(self, num_inference_steps):
            if num_inference_steps != 40:
                raise ValueError("fake is exact40 only")

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
                text_dim=4096,
            )
            self.anchor = torch.nn.Parameter(
                torch.zeros(()),
                requires_grad=bool(trainable),
            )
            self.eval()
            self.patch_log = []

        def patch_vae_latent(self, hidden_states, source_id=None):
            sid = float(source_id)
            self.patch_log.append((hidden_states, sid))
            tokens = int(
                hidden_states.shape[2]
                * (hidden_states.shape[3] // 2)
                * (hidden_states.shape[4] // 2)
            )
            mean = (
                hidden_states.float().mean().reshape(1, 1, 1).expand(1, tokens, 1)
            )
            latent = (
                (mean + sid)
                .to(dtype=self.dtype)
                .expand(1, tokens, 1536)
                .contiguous()
            )
            rotary = torch.full((1, 1, tokens, 4), sid, dtype=torch.float32)
            return latent, rotary


    class _FakeDiffusion:
        use_unipc = True
        transformer_2 = None

        def __init__(
            self,
            *,
            torch,
            vendor,
            altered_middle_sigma=False,
            wrong_terminal_sigma=False,
            altered_timestep=False,
            wrong_patch_order=False,
            corrupt_official_apg=False,
            trainable=False,
            transformer_dtype=torch.float32,
        ) -> None:
            self.torch = torch
            self.vendor = vendor
            self.transformer = _FakeTransformer(
                trainable=trainable,
                dtype=transformer_dtype,
            )
            self.scheduler = _FakeScheduler(
                altered_middle_sigma=altered_middle_sigma,
                wrong_terminal_sigma=wrong_terminal_sigma,
                altered_timestep=altered_timestep,
            )
            self.wrong_patch_order = wrong_patch_order
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
            text_mean = float(cond_embeds[0, 0, 0].item())
            if text_mean < 0:
                prompt = "negative"
            elif text_mean < 0.3:
                prompt = "low-action"
            else:
                prompt = "high-action"
            self.forward_calls.append(
                {
                    "prompt": prompt,
                    "prompt_object": cond_embeds,
                    "tokens": int(noisy_latents.shape[1]),
                    "noisy": noisy_latents,
                    "timestep": timesteps,
                    "rotary": rotary_embs,
                }
            )
            context = noisy_latents.float().mean(dim=1, keepdim=True)
            text = cond_embeds.float().mean()
            rotary = rotary_embs.float().mean(
                dim=(1, 2, 3), keepdim=False
            ).reshape(1, 1, 1)
            return (
                noisy_latents[:, :, :64].float()
                + context[:, :, :1].float()
                + text
                + 0.01 * rotary
            ).to(dtype=self.transformer.dtype)

        def sample(
            self,
            prompt_embeds=None,
            prompt_embeds_t2=None,
            uncond_prompt_embeds=None,
            uncond_embeds_t2=None,
            num_frames=1,
            width=832,
            height=480,
            image_vae_latents=None,
            multi_video_vae_latents=None,
            multi_image_vae_latents=None,
            num_inference_steps=50,
            guidance_mode="rv2v",
            omega_vid=3.0,
            omega_img=3.0,
            omega_txt=4.0,
            omega_scale=0.75,
            flow_shift=5.0,
            seed=42,
            device="cuda",
            eta=1.0,
            norm_threshold=(50.0, 50.0),
            momentum=0.0,
        ):
            del (
                prompt_embeds_t2,
                uncond_embeds_t2,
                num_frames,
                width,
                height,
                image_vae_latents,
                multi_image_vae_latents,
                guidance_mode,
                omega_vid,
                omega_img,
                omega_scale,
                flow_shift,
                seed,
                device,
            )
            self.scheduler.set_timesteps(num_inference_steps)
            sample = torch.linspace(
                -0.2,
                0.3,
                21 * 64,
                dtype=torch.float32,
            ).reshape(1, 21, 64)
            official_momentum = self.vendor.MomentumBuffer(momentum)
            nt0 = norm_threshold[0] if isinstance(norm_threshold, (list, tuple)) else norm_threshold
            for index in range(num_inference_steps):
                timestep = self.scheduler.timesteps[index]
                source_id = 9.0 if self.wrong_patch_order and index == 0 else 1.0
                source = self.transformer.patch_vae_latent(
                    multi_video_vae_latents[0].to(dtype=self.transformer.dtype),
                    source_id=source_id,
                )
                target = self.transformer.patch_vae_latent(
                    sgaf._packed_to_spatial(
                        sample,
                        (1, 16, 21, 2, 2),
                    ).to(dtype=self.transformer.dtype),
                    source_id=0.0,
                )
                low_latent = torch.cat([source[0], target[0]], dim=1).to(
                    dtype=self.transformer.dtype
                )
                low_rotary = torch.cat([source[1], target[1]], dim=2)
                expanded = timestep.expand(1)

                def forward(text):
                    return self.shared_step(
                        model_id="transformer_1",
                        noisy_latents=low_latent,
                        timesteps=expanded,
                        cond_embeds=text,
                        rotary_embs=low_rotary,
                        batch_vae_seqlen=[int(low_latent.shape[1])],
                        batch_text_seqlen=[int(text.shape[1])],
                    )[:, -sample.shape[1] :, :]

                negative = forward(uncond_prompt_embeds)
                action = forward(prompt_embeds)
                sigma = self.scheduler.sigmas[index]
                sample_spatial = sgaf._packed_to_spatial(
                    sample,
                    (1, 16, 21, 2, 2),
                )
                negative_clean = sample_spatial - sigma * sgaf._packed_to_spatial(
                    negative,
                    (1, 16, 21, 2, 2),
                )
                action_clean = sample_spatial - sigma * sgaf._packed_to_spatial(
                    action,
                    (1, 16, 21, 2, 2),
                )
                guided = self.vendor.normalized_guidance(
                    pred_cond=action_clean,
                    pred_uncond=negative_clean,
                    guidance_scale=omega_txt,
                    momentum_buffer=official_momentum,
                    eta=eta,
                    norm_threshold=nt0,
                )
                official = sgaf._spatial_to_packed(
                    (sample_spatial - guided) / sigma,
                    (1, 16, 21, 2, 2),
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
            return sgaf._packed_to_spatial(
                sample,
                (1, 16, 21, 2, 2),
            ).contiguous()


if __name__ == "__main__":
    unittest.main()
