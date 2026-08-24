from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if _TORCH_AVAILABLE:
    import braid_dual_native_apg_runtime_v1 as runtime  # noqa: E402
    import self_guided_action_field_v1 as sgaf  # noqa: E402
else:
    runtime = None  # type: ignore[assignment]
    sgaf = None  # type: ignore[assignment]


@unittest.skipUnless(_TORCH_AVAILABLE, "torch required")
class StaticContractTests(unittest.TestCase):
    def test_reference_and_diagnostic_modes_are_explicit(self) -> None:
        reference = runtime.BraidDualNativeAPGConfig(
            target_latent_shape=(1, 16, 21, 2, 2),
            sp_rank=1,
            reset_source_costate=False,
        )
        reference.validate()
        self.assertEqual(reference.forward_mode, runtime.REFERENCE_4F)
        self.assertEqual(reference.forward_order, runtime.REFERENCE_4F_ORDER)
        self.assertEqual(reference.forwards_per_step, 4)
        self.assertEqual(reference.block_index, 15)
        self.assertEqual(runtime.BLOCK15_AUTHORITY, "infrastructure_canary_only_not_an_authorized_braid_reset_boundary")

        with self.assertRaisesRegex(
            runtime.BraidDualNativeAPGRuntimeError, "explicit opt-in"
        ):
            runtime.BraidDualNativeAPGConfig(
                target_latent_shape=(1, 16, 21, 2, 2),
                sp_rank=1,
                reset_source_costate=False,
                forward_mode=runtime.SHARED_NEGATIVE_3F_DIAGNOSTIC,
            ).validate()
        diagnostic = runtime.BraidDualNativeAPGConfig(
            target_latent_shape=(1, 16, 21, 2, 2),
            sp_rank=1,
            reset_source_costate=False,
            forward_mode=runtime.SHARED_NEGATIVE_3F_DIAGNOSTIC,
            allow_shared_negative_diagnostic=True,
        )
        diagnostic.validate()
        self.assertEqual(diagnostic.forward_order, runtime.SHARED_NEGATIVE_3F_ORDER)
        self.assertEqual(diagnostic.forwards_per_step, 3)

    def test_layout_partitions_global_source_target_and_append_padding(self) -> None:
        layouts = [
            runtime.BraidSP4RoleLayout.build(
                total_tokens=42,
                condition_tokens=21,
                sp_rank=rank,
            )
            for rank in range(4)
        ]
        self.assertTrue(all(row.local_length == 11 for row in layouts))
        self.assertEqual(
            [(row.source_local_indices.numel(), row.target_local_indices.numel(), row.padding_local_indices.numel()) for row in layouts],
            [(11, 0, 0), (10, 1, 0), (0, 11, 0), (0, 9, 2)],
        )
        source_global = []
        target_global = []
        padding_global = []
        for row in layouts:
            start = row.shard_global_start
            source_global.extend((start + row.source_local_indices).tolist())
            target_global.extend((start + row.target_local_indices).tolist())
            padding_global.extend((start + row.padding_local_indices).tolist())
        self.assertEqual(source_global, list(range(21)))
        self.assertEqual(target_global, list(range(21, 42)))
        self.assertEqual(padding_global, [42, 43])
        with self.assertRaisesRegex(
            runtime.BraidDualNativeAPGRuntimeError, "observed rank-local"
        ):
            runtime.BraidSP4RoleLayout.build(
                total_tokens=42,
                condition_tokens=21,
                sp_rank=0,
                observed_local_length=42,
            )


@unittest.skipUnless(_TORCH_AVAILABLE, "torch required")
class Block15HookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        torch.set_num_threads(1)
        cls.torch = torch

    def _transformer(self):
        torch = self.torch

        class Transformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList(
                    [torch.nn.Identity() for _ in range(30)]
                )

        return Transformer().eval()

    def _run_pair(self, *, rank: int, reset: bool):
        torch = self.torch
        transformer = self._transformer()
        hook = runtime.BraidBlock15SourceCoStateHook(
            transformer, reset_enabled=reset
        )
        layout = runtime.BraidSP4RoleLayout.build(
            total_tokens=7,
            condition_tokens=3,
            sp_rank=rank,
        )
        hook.install()
        try:
            hook.begin_step(
                step_index=0,
                layout=layout,
                forward_order=runtime.REFERENCE_4F_ORDER,
            )
            base_negative = torch.zeros(1, layout.local_length, 4)
            with hook.leg("base_negative"):
                transformer.blocks[15](base_negative)
            base_positive = torch.arange(
                layout.local_length * 4, dtype=torch.float32
            ).reshape(1, layout.local_length, 4)
            # Preserve a signed-zero bit so equality checks are byte-level.
            base_positive[0, 0, 0] = -0.0
            with hook.leg("base_positive"):
                transformer.blocks[15](base_positive)
            action_negative = torch.ones_like(base_positive)
            with hook.leg("action_negative"):
                transformer.blocks[15](action_negative)
            action_positive = base_positive.clone() + 50.0
            action_positive[0, -1, -1] = -0.0
            action_before = action_positive.detach().clone()
            with hook.leg("action_positive"):
                observed = transformer.blocks[15](action_positive)
            record = hook.finish_step()
            return (
                hook,
                layout,
                base_positive,
                action_before,
                action_positive,
                observed,
                record,
            )
        finally:
            hook.remove()

    def test_reset_off_preserves_exact_output_object_identity(self) -> None:
        _, _, _, _, action_object, observed, record = self._run_pair(
            rank=1, reset=False
        )
        self.assertIs(observed, action_object)
        self.assertTrue(record.reset_off_returned_original_object)
        self.assertFalse(record.reset_returned_new_object)
        self.assertGreater(record.source_post_reset_mismatch_bytes, 0)
        self.assertEqual(record.target_post_reset_mismatch_bytes, 0)
        self.assertEqual(record.padding_post_reset_mismatch_bytes, 0)

    def test_reset_on_clamps_only_source_and_preserves_target_and_padding_bits(self) -> None:
        for rank in range(4):
            with self.subTest(rank=rank):
                (
                    hook,
                    layout,
                    base,
                    action,
                    _,
                    observed,
                    record,
                ) = self._run_pair(rank=rank, reset=True)
                source = layout.source_local_indices
                target = layout.target_local_indices
                padding = layout.padding_local_indices
                self.assertTrue(
                    runtime._raw_bytes_equal(
                        observed.index_select(1, source),
                        base.index_select(1, source),
                    )
                )
                self.assertTrue(
                    runtime._raw_bytes_equal(
                        observed.index_select(1, target),
                        action.index_select(1, target),
                    )
                )
                self.assertTrue(
                    runtime._raw_bytes_equal(
                        observed.index_select(1, padding),
                        action.index_select(1, padding),
                    )
                )
                self.assertEqual(record.source_post_reset_mismatch_bytes, 0)
                self.assertEqual(record.target_post_reset_mismatch_bytes, 0)
                self.assertEqual(record.padding_post_reset_mismatch_bytes, 0)
                receipt = hook.receipt()
                self.assertEqual(receipt["selection_authority"], runtime.BLOCK15_AUTHORITY)
                self.assertFalse(receipt["hidden_collective_or_reinjection"])
                self.assertFalse(receipt["training_authorized"])


if _TORCH_AVAILABLE:
    import torch

    class _FakeScheduler:
        def __init__(self):
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
            self.timesteps = torch.arange(40, 0, -1, dtype=torch.int64)
            self.sigmas = torch.cat(
                (torch.linspace(1.0, 0.1, 40, dtype=torch.float32), torch.zeros(1))
            )
            self.step_index = 0
            self.call_count = 0
            self.received_objects = []

        def set_timesteps(self, count):
            if count != 40:
                raise ValueError("fake is exact40")
            self.step_index = 0

        def step(self, model_output, timestep, sample, return_dict=False):
            del timestep, return_dict
            self.received_objects.append(model_output)
            self.call_count += 1
            self.step_index += 1
            return (sample - 0.001 * model_output,)


    class _ToyBlock(torch.nn.Module):
        def __init__(self, index):
            super().__init__()
            self.index = index

        def forward(self, hidden):
            return hidden + float(self.index + 1) / 1000.0


    class _FakeTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dtype = torch.float32
            self.config = SimpleNamespace(
                num_attention_heads=12,
                attention_head_dim=128,
                in_channels=16,
                text_dim=4096,
            )
            self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
            self.blocks = torch.nn.ModuleList([_ToyBlock(i) for i in range(30)])
            self.eval()


    class _FakeDiffusion:
        use_unipc = True
        transformer_2 = None

        def __init__(self, *, vendor, sp_rank=1, corrupt_official=False, double_scheduler=False):
            self.vendor = vendor
            self.sp_rank = sp_rank
            self.corrupt_official = corrupt_official
            self.double_scheduler = double_scheduler
            self.transformer = _FakeTransformer()
            self.scheduler = _FakeScheduler()
            self.shared_call_count = 0
            self.forward_log = []
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
            del model_id, timesteps, rotary_embs, batch_vae_seqlen, batch_text_seqlen
            self.shared_call_count += 1
            text = float(cond_embeds[0, 0, 0].item())
            label = "negative" if text < 0 else "base" if text < 0.3 else "action"
            layout = runtime.BraidSP4RoleLayout.build(
                total_tokens=42,
                condition_tokens=21,
                sp_rank=self.sp_rank,
            )
            start = layout.shard_global_start
            stop = min(layout.shard_global_stop_padded, 42)
            local = noisy_latents[:, start:stop, :].float() + text
            if local.shape[1] < layout.local_length:
                local = torch.cat(
                    (
                        local,
                        torch.zeros(
                            1,
                            layout.local_length - local.shape[1],
                            1536,
                            dtype=local.dtype,
                        ),
                    ),
                    dim=1,
                )
            block15 = self.transformer.blocks[15](local)
            self.forward_log.append(
                {
                    "label": label,
                    "noisy": noisy_latents,
                    "prompt": cond_embeds,
                    "block15": block15.detach().clone(),
                }
            )
            offset = block15.float().mean()
            return (
                noisy_latents[:, :, :64].float() + 0.01 * offset
            ).to(dtype=self.transformer.dtype)

        @torch.no_grad()
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
            source_mean = multi_video_vae_latents[0].float().mean()
            sample = torch.linspace(-0.2, 0.3, 21 * 64).reshape(1, 21, 64)
            official_buffer = self.vendor.MomentumBuffer(momentum)
            threshold = norm_threshold[0] if isinstance(norm_threshold, (list, tuple)) else norm_threshold
            for index in range(num_inference_steps):
                timestep = self.scheduler.timesteps[index]
                source_hidden = torch.full((1, 21, 1536), float(source_mean))
                target_hidden = sample.float().mean(dim=2, keepdim=True).expand(1, 21, 1536)
                noisy = torch.cat((source_hidden, target_hidden), dim=1).contiguous()
                rotary = torch.zeros(1, 1, 42, 4)
                expanded = timestep.expand(1)

                def forward(prompt):
                    return self.shared_step(
                        model_id="transformer_1",
                        noisy_latents=noisy,
                        timesteps=expanded,
                        cond_embeds=prompt,
                        rotary_embs=rotary,
                        batch_vae_seqlen=[42],
                        batch_text_seqlen=[512],
                    )[:, -21:, :]

                negative = forward(uncond_prompt_embeds)
                positive = forward(prompt_embeds)
                sigma = self.scheduler.sigmas[index]
                spatial = sgaf._packed_to_spatial(sample, (1, 16, 21, 2, 2))
                negative_clean = spatial - sigma * sgaf._packed_to_spatial(
                    negative, (1, 16, 21, 2, 2)
                )
                positive_clean = spatial - sigma * sgaf._packed_to_spatial(
                    positive, (1, 16, 21, 2, 2)
                )
                guided = self.vendor.normalized_guidance(
                    pred_cond=positive_clean,
                    pred_uncond=negative_clean,
                    guidance_scale=omega_txt,
                    momentum_buffer=official_buffer,
                    eta=eta,
                    norm_threshold=threshold,
                )
                official = sgaf._spatial_to_packed(
                    (spatial - guided) / sigma, (1, 16, 21, 2, 2)
                )
                if self.corrupt_official and index == 0:
                    official = official + 0.001
                self.official_outputs.append(official)
                sample = self.scheduler.step(
                    official, timestep, sample, return_dict=False
                )[0]
                if self.double_scheduler and index == 0:
                    self.scheduler.step(official, timestep, sample, return_dict=False)
            return sgaf._packed_to_spatial(sample, (1, 16, 21, 2, 2)).contiguous()


@unittest.skipUnless(_TORCH_AVAILABLE, "torch required")
class RuntimePatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def setUp(self) -> None:
        self._saved_vendor = sys.modules.get(runtime.VENDOR_APG_MODULE)
        vendor = ModuleType(runtime.VENDOR_APG_MODULE)

        class MomentumBuffer:
            def __init__(self, momentum):
                self.momentum = momentum
                self.running_average = 0

            def update(self, update_value):
                self.running_average = update_value + self.momentum * self.running_average

        MomentumBuffer.__module__ = runtime.VENDOR_APG_MODULE

        def normalized_guidance(
            pred_cond,
            pred_uncond,
            guidance_scale,
            momentum_buffer=None,
            eta=1.0,
            norm_threshold=0.0,
        ):
            vendor.call_count += 1
            diff = pred_cond - pred_uncond
            if momentum_buffer is not None:
                momentum_buffer.update(diff)
                diff = momentum_buffer.running_average
            if norm_threshold > 0:
                norm = diff.norm(p=2, dim=[-1, -2, -4], keepdim=True)
                diff = diff * torch.minimum(torch.ones_like(diff), norm_threshold / norm)
            base = torch.nn.functional.normalize(pred_cond.double(), dim=[-1, -2, -4])
            parallel = (diff.double() * base).sum(
                dim=[-1, -2, -4], keepdim=True
            ) * base
            orthogonal = diff.double() - parallel
            normalized = orthogonal.to(diff.dtype) + eta * parallel.to(diff.dtype)
            return pred_uncond + guidance_scale * normalized

        normalized_guidance.__module__ = runtime.VENDOR_APG_MODULE
        vendor.MomentumBuffer = MomentumBuffer
        vendor.normalized_guidance = normalized_guidance
        vendor.call_count = 0
        sys.modules[runtime.VENDOR_APG_MODULE] = vendor
        self.vendor = vendor
        shape = (1, 512, 4096)
        self.base = torch.full(shape, 0.2)
        self.action = torch.full(shape, 0.4)
        self.negative = torch.full(shape, -0.1)
        self.source = torch.full((1, 16, 21, 2, 2), 0.15)

    def tearDown(self) -> None:
        if self._saved_vendor is None:
            sys.modules.pop(runtime.VENDOR_APG_MODULE, None)
        else:
            sys.modules[runtime.VENDOR_APG_MODULE] = self._saved_vendor

    def _kwargs(self):
        return {
            "prompt_embeds": self.base,
            "prompt_embeds_t2": None,
            "uncond_prompt_embeds": self.negative,
            "uncond_embeds_t2": None,
            "num_frames": 81,
            "width": 16,
            "height": 16,
            "image_vae_latents": None,
            "multi_video_vae_latents": [self.source],
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

    def _run(self, *, mode=None, reset=True):
        if mode is None:
            mode = runtime.REFERENCE_4F
        diffusion = _FakeDiffusion(vendor=self.vendor, sp_rank=1)
        config = runtime.BraidDualNativeAPGConfig(
            target_latent_shape=(1, 16, 21, 2, 2),
            sp_rank=1,
            reset_source_costate=reset,
            forward_mode=mode,
            allow_shared_negative_diagnostic=(
                mode == runtime.SHARED_NEGATIVE_3F_DIAGNOSTIC
            ),
        )
        patch = runtime.BraidDualNativeAPGRuntimePatch(
            diffusion,
            action_prompt_embeds=self.action,
            config=config,
        )
        before = (
            diffusion.sample.__func__,
            diffusion.shared_step.__func__,
            diffusion.scheduler.step.__func__,
            self.vendor.normalized_guidance,
        )
        patch.install()
        try:
            result = diffusion.sample(**self._kwargs())
        finally:
            patch.restore()
        receipt = patch.finalize()
        after = (
            diffusion.sample.__func__,
            diffusion.shared_step.__func__,
            diffusion.scheduler.step.__func__,
            self.vendor.normalized_guidance,
        )
        self.assertEqual(before, after)
        self.assertFalse(diffusion.transformer.blocks[15]._forward_hooks)
        return diffusion, receipt, result

    def test_four_forward_reference_binds_two_vendor_states_and_steps_unipc_once(self) -> None:
        diffusion, receipt, result = self._run()
        self.assertEqual(tuple(result.shape), (1, 16, 21, 2, 2))
        self.assertEqual(diffusion.shared_call_count, 160)
        self.assertEqual(receipt["transformer_forwards"], 160)
        self.assertEqual(receipt["vendor_base_apg_calls"], 40)
        self.assertEqual(receipt["vendor_action_apg_calls"], 40)
        self.assertEqual(receipt["original_scheduler_calls"], 40)
        self.assertEqual(receipt["scheduler_execution"], "stock_base_V0_exact_object_only")
        self.assertEqual(receipt["forward_mode_authority"], "four_forward_reference")
        self.assertEqual(receipt["per_step_forward_order"], list(runtime.REFERENCE_4F_ORDER))
        self.assertNotEqual(
            receipt["base_apg_binding"]["buffer_object_id"],
            receipt["action_apg_binding"]["buffer_object_id"],
        )
        self.assertEqual(receipt["base_apg_binding"]["normalized_guidance_calls"], 40)
        self.assertEqual(receipt["action_apg_binding"]["normalized_guidance_calls"], 40)
        self.assertTrue(all(row["original_scheduler_calls"] == 1 for row in receipt["trace"]))
        self.assertTrue(all(row["base_stock_apg_exact_parity"] for row in receipt["trace"]))
        self.assertTrue(all(row["scheduler_received_stock_base_object"] for row in receipt["trace"]))
        self.assertTrue(all(row["block15"]["source_post_reset_mismatch_bytes"] == 0 for row in receipt["trace"]))
        self.assertEqual(
            diffusion.scheduler.received_objects,
            diffusion.official_outputs,
        )
        unsigned = dict(receipt)
        digest = unsigned.pop("runtime_digest")
        payload = json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        self.assertEqual(digest, hashlib.sha256(payload).hexdigest())
        self.assertFalse(receipt["optimizer_created"])
        self.assertFalse(receipt["training_authorized"])

    def test_three_forward_shared_negative_is_labeled_diagnostic(self) -> None:
        diffusion, receipt, _ = self._run(
            mode=runtime.SHARED_NEGATIVE_3F_DIAGNOSTIC,
            reset=False,
        )
        self.assertEqual(diffusion.shared_call_count, 120)
        self.assertEqual(receipt["transformer_forwards"], 120)
        self.assertEqual(receipt["forward_mode_authority"], "shared_negative_diagnostic_only")
        self.assertTrue(all(row["shared_negative"] for row in receipt["trace"]))
        self.assertTrue(
            all(not row["independent_complete_native_apg_pairs"] for row in receipt["trace"])
        )
        self.assertTrue(
            all(row["block15"]["reset_off_returned_original_object"] for row in receipt["trace"])
        )

    def test_corrupt_stock_apg_fails_before_original_scheduler(self) -> None:
        diffusion = _FakeDiffusion(vendor=self.vendor, corrupt_official=True)
        patch = runtime.BraidDualNativeAPGRuntimePatch(
            diffusion,
            action_prompt_embeds=self.action,
            config=runtime.BraidDualNativeAPGConfig(
                target_latent_shape=(1, 16, 21, 2, 2),
                sp_rank=1,
                reset_source_costate=True,
            ),
        )
        patch.install()
        try:
            with self.assertRaisesRegex(
                runtime.BraidDualNativeAPGRuntimeError,
                "captured vendor base APG differs",
            ):
                diffusion.sample(**self._kwargs())
        finally:
            patch.restore()
        self.assertEqual(diffusion.scheduler.call_count, 0)

    def test_second_scheduler_call_in_one_step_fails_before_reintegration(self) -> None:
        diffusion = _FakeDiffusion(vendor=self.vendor, double_scheduler=True)
        patch = runtime.BraidDualNativeAPGRuntimePatch(
            diffusion,
            action_prompt_embeds=self.action,
            config=runtime.BraidDualNativeAPGConfig(
                target_latent_shape=(1, 16, 21, 2, 2),
                sp_rank=1,
                reset_source_costate=False,
            ),
        )
        patch.install()
        try:
            with self.assertRaisesRegex(
                runtime.BraidDualNativeAPGRuntimeError,
                "before dual-native APG closure",
            ):
                diffusion.sample(**self._kwargs())
        finally:
            patch.restore()
        self.assertEqual(diffusion.scheduler.call_count, 1)


if __name__ == "__main__":
    unittest.main()
