from __future__ import annotations

import functools
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock
import torch


ROOT = Path(__file__).resolve().parents[1]
METHOD_ROOT = ROOT / "methods" / "bernini_action_editing"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


level_b = load(
    "infer_action_edit_level_b_renderer_0817_v1_test",
    METHOD_ROOT / "infer_action_edit_level_b_renderer_0817_v1.py",
)
product = load(
    "infer_action_edit_product_abi_0817_v1_level_b_test",
    METHOD_ROOT / "infer_action_edit_product_abi_0817_v1.py",
)


def authenticated_callable_identity_fixture(value):
    return value


class AuthenticatedOwnerFixture:
    @classmethod
    def load(cls):
        return cls


class FakeOwnership:
    def __init__(self, start: int):
        self.digest = hashlib.sha256(str(start).encode()).hexdigest()
        self.target_only = True
        self.target_suffix_start = start


class FakePredictorModule:
    @staticmethod
    def certify_closed_target_suffix_route(
        value, *, source_prefix_tokens, packed_total_tokens, audit_finite
    ):
        if (
            value.shape[1] != source_prefix_tokens
            or packed_total_tokens != 2 * source_prefix_tokens
            or audit_finite is not True
        ):
            raise RuntimeError("bad fake ownership request")
        return FakeOwnership(source_prefix_tokens)


class FakePlan:
    def __init__(self, ownership):
        self.ownership = ownership


class FakeInjection:
    def __init__(self, torch_module, hidden: int):
        self.torch = torch_module
        self.hidden = hidden
        self.calls = []

    def residual(self, _plan, *, block_index: int):
        self.calls.append(block_index)
        return self.torch.full(
            (1, level_b.PHASES, self.hidden),
            (block_index + 1) / 1000.0,
            dtype=self.torch.float32,
        )

    @staticmethod
    def validate_block_traversal(indices):
        if tuple(indices) != tuple(range(30)):
            raise RuntimeError("fake block traversal differs")


class FakeConditioner(torch.nn.Module):
    renderer_hidden_width = 4

    def __init__(self, torch_module):
        super().__init__()
        self.config = SimpleNamespace(
            source_token_width=4,
            instruction_token_width=6,
        )
        self.injection = FakeInjection(torch_module, self.renderer_hidden_width)

    def prepare_route(self, _source, _instruction, ownership):
        return FakePlan(ownership)


class FakeBlock:
    """A tiny torch Module allocated lazily after torch is imported."""

    @staticmethod
    def create(torch_module):
        class Block(torch_module.nn.Module):
            def forward(self, value):
                return value

        return Block()


class FakeSigmaContract:
    SCHEDULE_SHA256 = level_b.PINNED_UNIPC_SCHEDULE_SHA256
    SCHEDULER_CLASS = "UniPCMultistepScheduler"
    NUM_INFERENCE_STEPS = level_b.NUM_INFERENCE_STEPS
    FLOW_SHIFT = level_b.FLOW_SHIFT

    @staticmethod
    def audit_runtime_unipc_schedule(_scheduler, initialize=True):
        return {
            "schedule_sha256": level_b.PINNED_UNIPC_SCHEDULE_SHA256,
            "timesteps": list(level_b.PINNED_UNIPC_TIMESTEPS),
            "positive_sigmas_float32_be_hex": ["3f000000"]
            * level_b.NUM_INFERENCE_STEPS,
            "terminal_sigma_float32_be_hex": "00000000",
        }


class FakeScheduler:
    def __init__(self, torch_module, *, bad_timestep=False):
        self.torch = torch_module
        self.timesteps = list(level_b.PINNED_UNIPC_TIMESTEPS)
        if bad_timestep:
            self.timesteps[3] += 1
        self.config = SimpleNamespace(
            _class_name="UniPCMultistepScheduler",
            flow_shift=level_b.FLOW_SHIFT,
        )
        self.calls = 0

    def step(self, model_output, timestep, sample, return_dict=False):
        if return_dict is not False:
            raise RuntimeError("fake scheduler requires tuple output")
        self.calls += 1
        return (sample + 0.01 + model_output.float() * 0.0,)


class FakeDiffusion:
    use_unipc = True
    transformer_2 = None

    def __init__(
        self,
        torch_module,
        transformer,
        *,
        mutate_source=False,
        swap_action_prompt=False,
        third_forward=False,
        skip_last_block=False,
        bad_timestep=False,
        tamper_sample_wrapper=False,
        tamper_scheduler_wrapper=False,
        shared_step_extra=False,
        batch_vae_metadata=None,
    ):
        self.torch = torch_module
        self.transformer = transformer
        self.scheduler = FakeScheduler(torch_module, bad_timestep=bad_timestep)
        self.mutate_source = mutate_source
        self.swap_action_prompt = swap_action_prompt
        self.third_forward = third_forward
        self.skip_last_block = skip_last_block
        self.tamper_sample_wrapper = tamper_sample_wrapper
        self.tamper_scheduler_wrapper = tamper_scheduler_wrapper
        self.shared_step_extra = shared_step_extra
        self.batch_vae_metadata = batch_vae_metadata
        self.initial_packed = torch_module.linspace(
            -0.75,
            0.75,
            steps=level_b.PHASES * level_b.PATCH_VALUES,
            dtype=torch_module.float32,
        ).reshape(1, level_b.PHASES, level_b.PATCH_VALUES)

    def shared_step(
        self,
        model_id,
        noisy_latents,
        timesteps,
        cond_embeds,
        rotary_embs,
        batch_vae_seqlen,
        batch_text_seqlen,
        **kwargs,
    ):
        del kwargs
        value = noisy_latents
        blocks = self.transformer.blocks
        if self.skip_last_block:
            blocks = blocks[:-1]
        for block in blocks:
            value = block(value)
        one = value[..., :1]
        return one.repeat(1, 1, level_b.PATCH_VALUES)

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
        torch = self.torch
        del prompt_embeds_t2, uncond_embeds_t2, num_frames, width, height
        del image_vae_latents, multi_video_vae_latents, multi_image_vae_latents
        del omega_vid, omega_img, omega_txt, omega_scale, device, eta
        del norm_threshold, momentum, guidance_mode, flow_shift, seed
        source = torch.linspace(
            -1.0,
            1.0,
            steps=level_b.PHASES * 4,
            dtype=torch.float32,
        ).reshape(1, level_b.PHASES, 4)
        target = self.initial_packed.clone()
        rotary = object()
        for index, timestep_value in enumerate(self.scheduler.timesteps):
            current_source = source.clone()
            if self.mutate_source and index == 1:
                current_source[0, 0, 0] += 1.0
            target_hidden = target[..., :4]
            embedded = torch.cat((current_source, target_hidden), dim=1)
            timestep = torch.tensor([timestep_value], dtype=torch.int64)
            common = {
                "model_id": "transformer_1",
                "noisy_latents": embedded,
                "timesteps": timestep,
                "rotary_embs": rotary,
                "batch_vae_seqlen": (
                    [2 * level_b.PHASES]
                    if self.batch_vae_metadata is None
                    else self.batch_vae_metadata
                ),
                "batch_text_seqlen": [512],
            }
            hostile = (
                {"hostile_extra": object()}
                if self.shared_step_extra and index == 0
                else {}
            )
            self.shared_step(
                cond_embeds=uncond_prompt_embeds, **common, **hostile
            )
            action_prompt = (
                uncond_prompt_embeds if self.swap_action_prompt else prompt_embeds
            )
            action = self.shared_step(cond_embeds=action_prompt, **common)
            if self.third_forward and index == 0:
                self.shared_step(cond_embeds=prompt_embeds, **common)
            target_velocity = action[:, -level_b.PHASES :, :]
            target = self.scheduler.step(
                target_velocity,
                timestep,
                target,
                return_dict=False,
            )[0]
        if self.tamper_sample_wrapper:
            self.sample = lambda *_args, **_kwargs: None
        if self.tamper_scheduler_wrapper:
            self.scheduler.step = lambda *_args, **_kwargs: None
        # Inverse of a 1x2x2 Wan pack when the spatial patch grid is 1x1.
        return (
            target.reshape(1, level_b.PHASES, 1, 1, 2, 2, 16)
            .permute(0, 6, 1, 2, 4, 3, 5)
            .reshape(1, 16, level_b.PHASES, 2, 2)
        )


class FakeNoiseObserver:
    def __init__(self, initial):
        self.initial = initial

    def packed_for(self, device):
        return self.initial.to(device=device)


def fake_bundle(torch_module, **diffusion_kwargs):
    class Transformer(torch_module.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch_module.nn.ModuleList(
                [FakeBlock.create(torch_module) for _ in range(30)]
            )

    transformer = Transformer()
    conditioner = FakeConditioner(torch_module)
    offline = product.install_offline_action_plan_hooks(
        transformer=transformer,
        conditioner=conditioner,
        torch_module=torch_module,
    )
    original_hook_objects = tuple(
        block._forward_hooks[handle.id]
        for block, handle in zip(transformer.blocks, offline.handles)
    )
    diffusion = FakeDiffusion(
        torch_module, transformer, **diffusion_kwargs
    )
    distributed = SimpleNamespace(
        sp_rank=0,
        topology=SimpleNamespace(sp_size=1),
    )
    checkpoint = SimpleNamespace(predictor_module=FakePredictorModule)
    return SimpleNamespace(
        transformer=transformer,
        conditioner=conditioner,
        offline_hooks=offline,
        renderer=SimpleNamespace(diff_dec=diffusion),
        distributed=distributed,
        checkpoint=checkpoint,
        sigma_contract_module=FakeSigmaContract,
        original_hook_objects=original_hook_objects,
    )


def fake_internal_sampling(seed=20260817):
    return {
        "num_frames": 81,
        "num_inference_steps": 40,
        "guidance_mode": "v2v_apg",
        "omega_vid": 1.25,
        "omega_img": 0.0,
        "omega_txt": 4.0,
        "omega_scale": 0.8,
        "flow_shift": 5.0,
        "seed": seed,
        "eta": 0.5,
        "norm_threshold": (50.0, 50.0),
        "momentum": 0.0,
    }


class LevelBBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch

    def run_bridge(self, **diffusion_kwargs):
        torch = self.torch
        sample_changes = diffusion_kwargs.pop("sample_changes", {})
        nonzero_prompt_padding = diffusion_kwargs.pop(
            "nonzero_prompt_padding", False
        )
        bundle = fake_bundle(torch, **diffusion_kwargs)
        diffusion = bundle.renderer.diff_dec
        observer = FakeNoiseObserver(diffusion.initial_packed)
        policy = product.OfflineInferencePolicyV1(seed=20260817)
        action_prompt = torch.zeros(1, 512, 6, dtype=torch.float32)
        negative_prompt = torch.zeros(1, 512, 6, dtype=torch.float32)
        action_prompt[:, :3, :] = torch.linspace(
            -0.5, 0.5, steps=18, dtype=torch.float32
        ).reshape(1, 3, 6)
        negative_prompt[:, :3, :] = torch.linspace(
            0.5, -0.5, steps=18, dtype=torch.float32
        ).reshape(1, 3, 6)
        if nonzero_prompt_padding:
            action_prompt[0, 3, 0] = 1.0
        source_condition = torch.zeros(1, 16, 21, 2, 2)
        source_conditions = [source_condition]
        sampling = fake_internal_sampling()
        call_sampling = {**sampling, **sample_changes}
        with mock.patch.object(
            level_b,
            "_audit_native_unipc_scheduler_callable",
            return_value={"test_only_nonformal_fake_scheduler": True},
        ):
            with level_b.native_action_renderer_bridge(
                fresh_bundle=bundle,
                product_module=product,
                inference_policy=policy,
                patch_grid=(21, 1, 1),
                row_identity="fake-row",
                torch_module=torch,
                noise_observer=observer,
                source_condition_list=source_conditions,
                expected_width=16,
                expected_height=16,
                expected_device="cpu",
                expected_internal_sampling=sampling,
                expected_instruction_token_count=3,
            ) as bridge:
                result = diffusion.sample(
                    prompt_embeds=action_prompt,
                    uncond_prompt_embeds=negative_prompt,
                    prompt_embeds_t2=None,
                    uncond_embeds_t2=None,
                    width=16,
                    height=16,
                    image_vae_latents=None,
                    multi_video_vae_latents=source_conditions,
                    multi_image_vae_latents=None,
                    device="cpu",
                    **call_sampling,
                )
        return bundle, diffusion, bridge, result

    def test_fake_scheduler_is_rejected_without_nonformal_test_patch(self):
        torch = self.torch
        bundle = fake_bundle(torch)
        observer = FakeNoiseObserver(bundle.renderer.diff_dec.initial_packed)
        policy = product.OfflineInferencePolicyV1(seed=20260817)
        with self.assertRaisesRegex(
            level_b.LevelBRendererError, "exact pinned UniPC vendor class"
        ):
            with level_b.native_action_renderer_bridge(
                fresh_bundle=bundle,
                product_module=product,
                inference_policy=policy,
                patch_grid=(21, 1, 1),
                row_identity="fake-row",
                torch_module=torch,
                noise_observer=observer,
                source_condition_list=[torch.zeros(1)],
                expected_width=16,
                expected_height=16,
                expected_device="cpu",
                expected_internal_sampling=fake_internal_sampling(),
                expected_instruction_token_count=3,
            ):
                pass

    def test_scheduler_spoof_with_exact_public_name_is_rejected_by_source_sha(self):
        module_name = "diffusers.schedulers.scheduling_unipc_multistep"

        class SubstituteScheduler:
            def __init__(self):
                self.config = SimpleNamespace(
                    _class_name="UniPCMultistepScheduler"
                )

            def step(self, model_output, timestep, sample, return_dict=True):
                return sample

        SubstituteScheduler.__name__ = "UniPCMultistepScheduler"
        SubstituteScheduler.__qualname__ = "UniPCMultistepScheduler"
        SubstituteScheduler.__module__ = module_name
        SubstituteScheduler.step.__qualname__ = "UniPCMultistepScheduler.step"
        SubstituteScheduler.step.__module__ = module_name
        substitute_module = ModuleType(module_name)
        substitute_module.__file__ = str(Path(__file__).resolve())
        substitute_module.UniPCMultistepScheduler = SubstituteScheduler
        previous = sys.modules.get(module_name)
        sys.modules[module_name] = substitute_module
        try:
            with self.assertRaisesRegex(
                level_b.LevelBRendererError, "owned by its pinned source bytes"
            ):
                level_b._audit_native_unipc_scheduler_callable(
                    SubstituteScheduler(),
                    expected_path=str(Path(__file__).resolve()),
                    expected_sha256=level_b.PINNED_SITE_PACKAGE_SOURCE_HASHES[
                        "diffusers/schedulers/scheduling_unipc_multistep.py"
                    ],
                )
        finally:
            if previous is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous

    def test_exact40_native_bridge_invokes_only_action_heads(self):
        bundle, diffusion, bridge, result = self.run_bridge()
        trace = bridge.receipt()
        self.assertEqual(tuple(result.shape), (1, 16, 21, 2, 2))
        self.assertEqual(diffusion.scheduler.calls, 40)
        self.assertEqual(trace["negative_residual_head_calls"], 0)
        self.assertEqual(trace["action_residual_head_calls"], 1200)
        self.assertEqual(trace["native_shared_step_calls"], 80)
        self.assertEqual(trace["actual_contextual_instruction_length"], 3)
        self.assertEqual(trace["native_contextual_instruction_padded_length"], 512)
        self.assertTrue(
            all(
                row["native_batch_vae_seqlen"] == [42]
                for row in trace["route_receipts"]
            )
        )
        self.assertTrue(trace["forty_distinct_evolving_target_states"])
        self.assertEqual(
            bundle.conditioner.injection.calls,
            list(range(30)) * 40,
        )
        self.assertFalse(bundle.offline_hooks.restored)
        self.assertTrue(trace["same_level_a_exact30_hook_objects_reused_all40"])
        self.assertEqual(
            tuple(
                block._forward_hooks[handle.id]
                for block, handle in zip(
                    bundle.transformer.blocks, bundle.offline_hooks.handles
                )
            ),
            bundle.original_hook_objects,
        )

    def test_mutated_source_prefix_is_rejected(self):
        with self.assertRaisesRegex(
            level_b.LevelBRendererError, "prefix changed"
        ):
            self.run_bridge(mutate_source=True)

    def test_swapped_action_prompt_is_rejected(self):
        with self.assertRaisesRegex(
            level_b.LevelBRendererError, "second native shared_step"
        ):
            self.run_bridge(swap_action_prompt=True)

    def test_third_forward_is_rejected_before_unipc(self):
        with self.assertRaisesRegex(
            level_b.LevelBRendererError, "more than negative/action"
        ):
            self.run_bridge(third_forward=True)

    def test_missing_block_is_rejected(self):
        with self.assertRaisesRegex(
            level_b.LevelBRendererError, "exact30"
        ):
            self.run_bridge(skip_last_block=True)

    def test_wrong_live_timestep_is_rejected(self):
        with self.assertRaisesRegex(
            level_b.LevelBRendererError, "timestep differs"
        ):
            self.run_bridge(bad_timestep=True)

    def test_complete_internal_sampler_contract_rejects_one_changed_control(self):
        with self.assertRaisesRegex(
            level_b.LevelBRendererError, "internal sampler omega_scale differs"
        ):
            self.run_bridge(sample_changes={"omega_scale": 0.81})

    def test_internal_diffusion_sample_parameter_order_matches_vendor(self):
        self.assertEqual(
            level_b.INTERNAL_DIFFUSION_SAMPLE_PARAMETERS[:4],
            (
                "prompt_embeds",
                "prompt_embeds_t2",
                "uncond_prompt_embeds",
                "uncond_embeds_t2",
            ),
        )

    def test_nonempty_shared_step_variadic_kwargs_are_rejected(self):
        with self.assertRaisesRegex(
            level_b.LevelBRendererError, "variadic kwargs are not exactly empty"
        ):
            self.run_bridge(shared_step_extra=True)

    def test_nonzero_native_prompt_padding_is_rejected(self):
        with self.assertRaisesRegex(
            level_b.LevelBRendererError, "padding is not bit-exact zero"
        ):
            self.run_bridge(nonzero_prompt_padding=True)

    def test_native_batch_vae_metadata_requires_exact_builtin_list_int(self):
        for hostile in ((2 * level_b.PHASES,), [True], [2 * level_b.PHASES + 1]):
            with self.subTest(hostile=hostile), self.assertRaisesRegex(
                level_b.LevelBRendererError, "VAE sequence-length metadata"
            ):
                self.run_bridge(batch_vae_metadata=hostile)

    def test_sample_wrapper_tamper_is_rejected_on_exit(self):
        with self.assertRaisesRegex(
            level_b.LevelBRendererError, "sample wrapper identity changed"
        ):
            self.run_bridge(tamper_sample_wrapper=True)

    def test_scheduler_wrapper_tamper_is_rejected_on_exit(self):
        with self.assertRaisesRegex(
            level_b.LevelBRendererError, "step wrapper identity changed"
        ):
            self.run_bridge(tamper_scheduler_wrapper=True)

    def test_direct_block_call_outside_sample_is_rejected(self):
        torch = self.torch
        bundle = fake_bundle(torch)
        observer = FakeNoiseObserver(bundle.renderer.diff_dec.initial_packed)
        policy = product.OfflineInferencePolicyV1(seed=20260817)
        with mock.patch.object(
            level_b,
            "_audit_native_unipc_scheduler_callable",
            return_value={"test_only_nonformal_fake_scheduler": True},
        ):
            with self.assertRaisesRegex(Exception, "without an authenticated route"):
                with level_b.native_action_renderer_bridge(
                fresh_bundle=bundle,
                product_module=product,
                inference_policy=policy,
                patch_grid=(21, 1, 1),
                row_identity="fake-row",
                torch_module=torch,
                noise_observer=observer,
                source_condition_list=[torch.zeros(1)],
                expected_width=16,
                expected_height=16,
                    expected_device="cpu",
                    expected_internal_sampling=fake_internal_sampling(),
                    expected_instruction_token_count=3,
                ):
                    bundle.transformer.blocks[0](torch.zeros(1, 42, 4))

    def test_replaced_level_a_callback_is_rejected_before_sample(self):
        torch = self.torch
        bundle = fake_bundle(torch)
        handle = bundle.offline_hooks.handles[7]
        bundle.transformer.blocks[7]._forward_hooks[handle.id] = (
            lambda _module, _args, output: output
        )
        observer = FakeNoiseObserver(bundle.renderer.diff_dec.initial_packed)
        policy = product.OfflineInferencePolicyV1(seed=20260817)
        with mock.patch.object(
            level_b,
            "_audit_native_unipc_scheduler_callable",
            return_value={"test_only_nonformal_fake_scheduler": True},
        ):
            with self.assertRaisesRegex(
                level_b.LevelBRendererError, "callback provenance"
            ):
                with level_b.native_action_renderer_bridge(
                fresh_bundle=bundle,
                product_module=product,
                inference_policy=policy,
                patch_grid=(21, 1, 1),
                row_identity="fake-row",
                torch_module=torch,
                noise_observer=observer,
                source_condition_list=[torch.zeros(1)],
                expected_width=16,
                expected_height=16,
                    expected_device="cpu",
                    expected_internal_sampling=fake_internal_sampling(),
                    expected_instruction_token_count=3,
                ):
                    pass

    def test_extra_level_a_hook_is_rejected_before_sample(self):
        torch = self.torch
        bundle = fake_bundle(torch)
        bundle.transformer.blocks[3].register_forward_hook(
            lambda _module, _args, _output: None
        )
        observer = FakeNoiseObserver(bundle.renderer.diff_dec.initial_packed)
        policy = product.OfflineInferencePolicyV1(seed=20260817)
        with mock.patch.object(
            level_b,
            "_audit_native_unipc_scheduler_callable",
            return_value={"test_only_nonformal_fake_scheduler": True},
        ):
            with self.assertRaisesRegex(
                level_b.LevelBRendererError, "hook ownership"
            ):
                with level_b.native_action_renderer_bridge(
                fresh_bundle=bundle,
                product_module=product,
                inference_policy=policy,
                patch_grid=(21, 1, 1),
                row_identity="fake-row",
                torch_module=torch,
                noise_observer=observer,
                source_condition_list=[torch.zeros(1)],
                expected_width=16,
                expected_height=16,
                    expected_device="cpu",
                    expected_internal_sampling=fake_internal_sampling(),
                    expected_instruction_token_count=3,
                ):
                    pass


class NativeNoiseObserverTests(unittest.TestCase):
    def test_observer_forwards_exact_native_object_and_packs(self):
        import torch

        def canonical(shape, *, generator, device, dtype):
            return torch.randn(
                shape, generator=generator, device=device, dtype=dtype
            )

        module = SimpleNamespace(randn_tensor=canonical)
        observer = level_b.NativeInitialNoiseObserver(
            wan_diffusion_module=module,
            canonical_randn_tensor=canonical,
            expected_shape=(1, 16, 21, 2, 2),
            expected_device=torch.device("cpu"),
            expected_seed=7,
            torch_module=torch,
        )
        generator = torch.Generator(device="cpu").manual_seed(7)
        with level_b.observe_native_initial_noise(observer):
            returned = module.randn_tensor(
                (1, 16, 21, 2, 2),
                generator=generator,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            self.assertIsInstance(returned, torch.Tensor)
        receipt = observer.receipt()
        self.assertEqual(tuple(observer.packed_for("cpu").shape), (1, 21, 64))
        self.assertTrue(receipt["returned_object_forwarded_by_identity"])
        self.assertFalse(receipt["external_initial_noise_injection"])
        self.assertIs(module.randn_tensor, canonical)

    def test_observer_rejects_second_draw(self):
        import torch

        def canonical(shape, *, generator, device, dtype):
            return torch.randn(
                shape, generator=generator, device=device, dtype=dtype
            )

        module = SimpleNamespace(randn_tensor=canonical)
        observer = level_b.NativeInitialNoiseObserver(
            wan_diffusion_module=module,
            canonical_randn_tensor=canonical,
            expected_shape=(1, 16, 21, 2, 2),
            expected_device="cpu",
            expected_seed=9,
            torch_module=torch,
        )
        generator = torch.Generator(device="cpu").manual_seed(9)
        with self.assertRaisesRegex(
            level_b.LevelBRendererError, "more than one"
        ):
            with level_b.observe_native_initial_noise(observer):
                for _ in range(2):
                    module.randn_tensor(
                        (1, 16, 21, 2, 2),
                        generator=generator,
                        device="cpu",
                        dtype=torch.float32,
                    )


class TensorDigestMemoryTests(unittest.TestCase):
    def test_phase_owned_tensor_reference_is_popped_by_exact_identity(self):
        value = object()
        owner = {"tensor": value}
        self.assertIs(
            level_b._pop_exact_phase_owned_value(
                owner, "tensor", value, label="fixture tensor"
            ),
            value,
        )
        self.assertEqual(owner, {})
        with self.assertRaisesRegex(
            level_b.LevelBRendererError, "object identity differs"
        ):
            level_b._pop_exact_phase_owned_value(
                {"tensor": object()}, "tensor", value, label="fixture tensor"
            )

    def test_formal_lifecycle_never_restores_vae_weights_to_host(self):
        source = inspect.getsource(
            level_b._run_level_b_pre_d0_offline_inference_authenticated_core
        )
        self.assertNotIn('vae_value.to("cpu")', source)
        self.assertNotIn('vae.to("cpu")', source)
        self.assertIn("del tokenizer", source)
        self.assertIn("del vae", source)

    @staticmethod
    def historical_digest(value):
        item = value.detach().contiguous().cpu()
        metadata = level_b.canonical_json_bytes(
            {"shape": [int(x) for x in item.shape], "dtype": str(item.dtype)}
        )
        digest = hashlib.sha256()
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(bytes(item.reshape(-1).view(torch.uint8).tolist()))
        return digest.hexdigest()

    def test_streaming_tensor_digest_matches_historical_contiguous_and_views(self):
        values = (
            torch.arange(60, dtype=torch.float32).reshape(3, 4, 5),
            torch.arange(60, dtype=torch.float32).reshape(3, 4, 5).transpose(0, 2),
            torch.arange(60, dtype=torch.bfloat16).reshape(3, 4, 5).transpose(1, 2),
        )
        for value in values:
            with self.subTest(dtype=str(value.dtype), shape=tuple(value.shape)):
                self.assertEqual(
                    level_b.tensor_sha256(value, torch_module=torch),
                    self.historical_digest(value),
                )

    def test_streaming_tensor_digest_never_transfers_more_than_64k_raw_bytes(self):
        tracker = {"maximum": 0, "calls": 0}

        class FakeRaw:
            def __init__(self, payload):
                self.payload = payload

            def view(self, _dtype):
                return self

            def numel(self):
                return len(self.payload)

            def __getitem__(self, item):
                return FakeRaw(self.payload[item])

            def cpu(self):
                tracker["calls"] += 1
                tracker["maximum"] = max(tracker["maximum"], len(self.payload))
                return self

            def tolist(self):
                return list(self.payload)

        class FakeTensor:
            def __init__(self, payload):
                self.payload = payload
                self.device = SimpleNamespace(type="cuda")
                self.shape = (len(payload),)
                self.dtype = "fixture.uint8"

            def detach(self):
                return self

            def contiguous(self):
                return self

            def reshape(self, *_shape):
                return FakeRaw(self.payload)

            def cpu(self):
                raise AssertionError("whole tensor CPU transfer is forbidden")

        fake_torch = SimpleNamespace(Tensor=FakeTensor, uint8=object())
        payload = bytes(index % 256 for index in range(3 * 64 * 1024 + 17))
        observed = level_b.tensor_sha256(
            FakeTensor(payload), torch_module=fake_torch
        )
        metadata = level_b.canonical_json_bytes(
            {"shape": [len(payload)], "dtype": "fixture.uint8"}
        )
        reference = hashlib.sha256()
        reference.update(len(metadata).to_bytes(8, "big"))
        reference.update(metadata)
        reference.update(payload)
        self.assertEqual(observed, reference.hexdigest())
        self.assertEqual(tracker["maximum"], 64 * 1024)
        self.assertEqual(tracker["calls"], 4)


class World8CanonicalizationTests(unittest.TestCase):
    def test_rank_local_cuda_ordinals_normalize_to_identical_bytes(self):
        encoded = []
        for rank in range(8):
            value = {
                "device": f"cuda:{rank}",
                "nested": ["cpu", f"cuda:{rank}"],
                "rank_invariant": "cuda-kernel",
            }
            normalized = level_b._normalize_rank_local_cuda_indices(
                value, local_cuda_index=rank
            )
            encoded.append(level_b.canonical_json_bytes(normalized))
        self.assertEqual(len(set(encoded)), 1)
        self.assertIn(b"cuda:<rank-local>", encoded[0])

    def test_nonlocal_cuda_ordinal_is_not_erased(self):
        with self.assertRaisesRegex(
            level_b.LevelBRendererError, "non-local CUDA ordinal"
        ):
            level_b._normalize_rank_local_cuda_indices(
                {"device": "cuda:7"}, local_cuda_index=0
            )

    def test_every_rank0_only_phase_broadcasts_failure_and_all_peers_exit(self):
        phases = (
            "base_checkpoint_binding",
            "source_preprocessing",
            "vae_load_and_source_encode",
            "vae_output_decode",
            "ffmpeg_validate_and_output_staging",
            "create_only_output_precommit",
            "final_marker_readiness_status",
        )

        class SharedBroadcast:
            payload = None

        class FakeDistributed:
            def __init__(self, rank, shared):
                self.rank = rank
                self.shared = shared

            def broadcast_object_list(self, box, *, src, group):
                self.assertions = (src, group)
                if self.rank == src:
                    self.shared.payload = json.loads(json.dumps(box[0]))
                else:
                    box[0] = json.loads(json.dumps(self.shared.payload))

        for phase in phases:
            with self.subTest(phase=phase):
                shared = SharedBroadcast()

                def explode():
                    raise RuntimeError(f"injected {phase} failure")

                with self.assertRaises(level_b.LevelBRendererError) as root_error:
                    level_b._run_world8_rank0_collective_phase(
                        phase=phase,
                        rank=0,
                        operation=explode,
                        distributed_module=FakeDistributed(0, shared),
                        group="world8",
                    )
                peer_calls = []
                for rank in range(1, level_b.WORLD_SIZE):
                    with self.assertRaises(level_b.LevelBRendererError) as peer_error:
                        level_b._run_world8_rank0_collective_phase(
                            phase=phase,
                            rank=rank,
                            operation=lambda: peer_calls.append(rank),
                            distributed_module=FakeDistributed(rank, shared),
                            group="world8",
                        )
                    self.assertEqual(str(peer_error.exception), str(root_error.exception))
                self.assertEqual(peer_calls, [])


class ReleaseAuthorityAndTransactionTests(unittest.TestCase):
    @staticmethod
    def _write_release(root: Path, *, corrupt_official: bool = False):
        (root / "tools").mkdir()
        sources = {
            "action_preservation_decoded_eval_model_authority_v2.py": (
                METHOD_ROOT / "action_preservation_decoded_eval_model_authority_v2.py"
            ),
            "infer_action_edit_level_b_renderer_0817_v1.py": (
                METHOD_ROOT / "infer_action_edit_level_b_renderer_0817_v1.py"
            ),
            "infer_lora.py": METHOD_ROOT / "infer_lora.py",
            "tools/build_renderer_dataset.py": (
                METHOD_ROOT / "tools" / "build_renderer_dataset.py"
            ),
            "tools/materialize_vae.py": METHOD_ROOT / "tools" / "materialize_vae.py",
        }
        rows = []
        for relative in level_b.LEVEL_B_RELEASE_MEMBER_PATHS:
            target = root.joinpath(*relative.split("/"))
            payload = sources[relative].read_bytes()
            if corrupt_official and relative == "infer_lora.py":
                payload += b"\n# hostile alternate official source\n"
            target.write_bytes(payload)
            os.chmod(target, 0o444)
            rows.append(
                {
                    "path": relative,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size": len(payload),
                    "mode": 0o444,
                }
            )
        payload = {
            "schema_version": level_b.RUNTIME_RELEASE_SCHEMA,
            "authority": level_b.AUTHORITY,
            "member_count": len(rows),
            "members": rows,
            "release_digest": level_b.object_sha256(rows),
        }
        manifest = root / "RELEASE_MANIFEST.json"
        manifest.write_bytes(level_b.canonical_json_bytes(payload) + b"\n")
        os.chmod(manifest, 0o444)
        os.chmod(root / "tools", 0o555)
        os.chmod(root, 0o555)
        return manifest, hashlib.sha256(manifest.read_bytes()).hexdigest()

    def test_recomputed_self_authority_manifest_cannot_replace_official_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "release"
            root.mkdir()
            manifest, manifest_sha = self._write_release(
                root, corrupt_official=True
            )
            with mock.patch.object(
                level_b,
                "__file__",
                str(root / "infer_action_edit_level_b_renderer_0817_v1.py"),
            ):
                with self.assertRaisesRegex(
                    level_b.LevelBRendererError, "compile-time pin"
                ):
                    level_b._validate_level_b_runtime_release_manifest(
                        manifest.resolve(),
                        sealed_launcher_expected_manifest_sha256=manifest_sha,
                    )

    def test_exact_sealed_release_source_closure_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "release"
            root.mkdir()
            manifest, manifest_sha = self._write_release(root)
            with mock.patch.object(
                level_b,
                "__file__",
                str(root / "infer_action_edit_level_b_renderer_0817_v1.py"),
            ):
                receipt = level_b._validate_level_b_runtime_release_manifest(
                    manifest.resolve(),
                    sealed_launcher_expected_manifest_sha256=manifest_sha,
                )
            self.assertTrue(receipt["executed_level_b_member_verified"])
            self.assertTrue(receipt["compile_time_transitive_source_pins_verified"])
            self.assertEqual(receipt["exact_member_count"], 5)

    def test_verified_runtime_cannot_be_caller_constructed(self):
        with self.assertRaisesRegex(
            level_b.LevelBRendererError, "only be issued"
        ):
            level_b.VerifiedLevelBRuntime()
        self.assertFalse(hasattr(level_b.VerifiedLevelBRuntime, "_issue"))
        self.assertFalse(hasattr(level_b, "_VERIFIED_RUNTIME_SEAL"))
        self.assertFalse(
            hasattr(level_b, "_SEALED_LAUNCHER_MANIFEST_SHA_AT_IMPORT")
        )
        self.assertFalse(hasattr(level_b, "_make_level_b_runtime_authenticator"))
        self.assertFalse(
            hasattr(
                level_b,
                "_run_level_b_pre_d0_offline_inference_authenticated",
            )
        )
        forged = object.__new__(level_b.VerifiedLevelBRuntime)
        with self.assertRaisesRegex(
            level_b.LevelBRendererError, "forged, stale, or already consumed"
        ):
            forged.validate_at_use()
        with self.assertRaisesRegex(
            level_b.LevelBRendererError, "final and cannot be subclassed"
        ):
            class ForgedRuntime(level_b.VerifiedLevelBRuntime):
                pass

    def test_cpu_static_preflight_public_api_is_fixed_and_capability_only(self):
        signature = inspect.signature(
            level_b.run_level_b_cpu_static_runtime_preflight
        )
        parameters = tuple(signature.parameters.values())
        self.assertEqual(tuple(parameter.name for parameter in parameters), ("verified_runtime",))
        self.assertIs(parameters[0].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIs(parameters[0].default, inspect.Parameter.empty)
        self.assertIn(
            "run_level_b_cpu_static_runtime_preflight", level_b.__all__
        )
        forged = object.__new__(level_b.VerifiedLevelBRuntime)
        with self.assertRaisesRegex(
            level_b.LevelBRendererError, "forged, stale, or already consumed"
        ):
            level_b.run_level_b_cpu_static_runtime_preflight(
                verified_runtime=forged
            )

    def test_cpu_static_preflight_core_has_no_weight_constructor_or_output_call(self):
        nonlocals = inspect.getclosurevars(
            level_b.run_level_b_cpu_static_runtime_preflight
        ).nonlocals
        implementation = nonlocals["implementation"]
        sealed_closure = nonlocals["sealed_scoped_module_closure"]
        self.assertEqual(len(sealed_closure), 52)
        self.assertEqual(sum(row[1] == "bernini" for row in sealed_closure), 13)
        self.assertEqual(sum(row[1] == "veomni" for row in sealed_closure), 39)
        self.assertEqual(
            tuple(row[0] for row in sealed_closure),
            tuple(sorted(row[0] for row in sealed_closure)),
        )
        self.assertEqual(len(level_b.PINNED_BERNINI_RUNTIME_FILE_HASHES), 16)
        self.assertEqual(len(level_b.PINNED_VEOMNI_RUNTIME_FILE_HASHES), 39)
        self.assertEqual(
            level_b.PINNED_SITE_PACKAGE_SOURCE_HASHES[
                "botocore/vendored/six.py"
            ],
            "4ce39f422ee71467ccac8bed76beb05f8c321c7f0ceda9279ae2dfa3670106b3",
        )
        self.assertEqual(
            level_b.PINNED_SITE_PACKAGE_SOURCE_HASHES["six.py"],
            "c51c91f703d3d4b3696c923cb5fec213e05e75d9215393befac7f2fa6a3904df",
        )
        source = inspect.getsource(implementation)
        self.assertIn("object.__new__(AutoencoderKLWan)", source)
        self.assertIn("object.__new__(T5Tokenizer)", source)
        self.assertIn("object.__new__(BerniniRendererModel)", source)
        self.assertIn("object.__new__(GEN_Wanx22)", source)
        self.assertNotIn("torch.cuda.init", source)
        self.assertNotIn("subprocess.", source)
        self.assertNotIn("open(", source)
        self.assertNotIn("mkdir", source)
        self.assertNotIn("save_output", source)
        self.assertNotIn("vae_from_pretrained(", source)
        self.assertNotIn("tokenizer_from_pretrained(", source)
        self.assertIn("sys.dont_write_bytecode = True", source)
        self.assertIn('("OPENBLAS_MAIN_FREE", "GOTOBLAS_MAIN_FREE")', source)
        self.assertIn(
            'fail("CPU static preflight BLAS import environment differs")',
            source,
        )
        self.assertIn('os.environ.get("VEOMNI_VERBOSITY")', source)
        self.assertIn(
            'fail("CPU static preflight VeOmni logging environment differs")',
            source,
        )
        self.assertIn("require_instantiator_caller(21", source)
        self.assertIn("require_instantiator_caller(23", source)
        self.assertIn("args != ()", source)
        self.assertIn(
            "callback.__func__ is not NoWriteTemporaryDirectory.cleanup",
            source,
        )
        self.assertIn("sys.path[-1] is not jit_tempdir_name", source)
        self.assertIn("sys.path.pop()", source)
        self.assertIn("PathFinder.find_spec", source)
        self.assertIn("self.original.exec_module(module)", source)
        self.assertIn("module.__loader__ = self.original", source)
        self.assertIn("self.spec.loader = self.original", source)
        self.assertIn("meta_path_identity_is", source)
        self.assertIn("remote_template_factory_calls != 0", source)
        self.assertIn("caller.f_lineno != 30", source)
        self.assertIn("remote_template_factory_scope_restored", source)
        self.assertIn("sys.path_importer_cache.clear()", source)
        self.assertIn("audit_scoped_module_source_closure()", source)
        self.assertIn("audit_six_meta_path_importer(", source)
        self.assertIn("module_export_identity_verified", source)
        self.assertIn("sys.meta_path.pop() is not six_importer", source)
        self.assertIn("six_importer_scope_restored", source)
        self.assertNotIn("repr(", source)
        self.assertNotIn("atexit_register_previous(callback", source)
        self.assertNotIn("temporary_directory_previous(*args", source)

    def test_cpu_static_preflight_scoped_closure_is_captured_and_address_free(self):
        nonlocals = inspect.getclosurevars(
            level_b.run_level_b_cpu_static_runtime_preflight
        ).nonlocals
        sealed = nonlocals["sealed_scoped_module_closure"]
        self.assertEqual(
            level_b.object_sha256(sealed),
            "749b9063870c55ffc8a31137baef7abb9911d2193e2bca342f0a1ff66d04b0df",
        )
        with mock.patch.object(
            level_b, "PINNED_CPU_STATIC_SCOPED_MODULE_CLOSURE", ()
        ):
            self.assertIs(
                inspect.getclosurevars(
                    level_b.run_level_b_cpu_static_runtime_preflight
                ).nonlocals["sealed_scoped_module_closure"],
                sealed,
            )
        canonical = level_b.canonical_json_bytes(sealed)
        self.assertNotIn(b"repr", canonical)
        self.assertNotIn(b"0x", canonical)

    def test_cpu_static_preflight_audit_guard_is_real_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            readable = root / "readable.txt"
            readable.write_text("read-only fixture", encoding="utf-8")
            weight = root / "hostile.pt"
            weight.write_bytes(b"fixture")
            package = root / "guard_happy_package"
            package.mkdir()
            (package / "__init__.py").write_text(
                "VALUE = 'read-only-import-ok'\n", encoding="utf-8"
            )
            script = r'''
import importlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from methods.bernini_action_editing import infer_action_edit_level_b_renderer_0817_v1 as module

root = Path(sys.argv[1]).resolve()
sys.dont_write_bytecode = True
guard = module._install_cpu_static_preflight_audit_guard(
    base_checkpoint=root / "checkpoint"
)
import tempfile
assert tempfile.tempdir is None
try:
    os.lstat("/nonexistent")
except FileNotFoundError:
    pass
else:
    raise AssertionError("sealed tempfile sentinel unexpectedly exists")
previous_tempdir = tempfile.tempdir
sealed_tempdir = "/nonexistent"
tempfile.tempdir = sealed_tempdir
assert tempfile.tempdir is sealed_tempdir
assert tempfile.gettempdir() == sealed_tempdir
sys.path.insert(0, str(root))
loaded = importlib.import_module("guard_happy_package")
assert loaded.VALUE == "read-only-import-ok"
assert (root / "readable.txt").read_text(encoding="utf-8") == "read-only fixture"
with open(os.devnull, "r+"):
    pass
with open(os.devnull, "w"):
    pass
descriptor = os.open(os.devnull, os.O_RDWR | getattr(os, "O_CLOEXEC", 0))
os.close(descriptor)
descriptor = os.open(
    os.devnull,
    os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_CLOEXEC", 0),
)
os.close(descriptor)
tempfile.tempdir = previous_tempdir
assert tempfile.tempdir is None
receipt = guard.receipt()
blocked = []
operations = (
    lambda: open(os.devnull, "r+"),
    lambda: (root / "persistent.txt").write_text("forbidden", encoding="utf-8"),
    lambda: (root / "hostile.pt").read_bytes(),
    lambda: os.mkdir(root / "forbidden-directory"),
    lambda: subprocess.run(["/usr/bin/true"], check=False),
    lambda: socket.socket(),
    lambda: os.putenv("BERNINI_LEVEL_B_FORBIDDEN", "1"),
    lambda: os.unsetenv("BERNINI_LEVEL_B_FORBIDDEN"),
)
for operation in operations:
    try:
        operation()
    except module.LevelBRendererError:
        blocked.append(True)
    else:
        blocked.append(False)
print(json.dumps({
    "blocked": blocked,
    "devnull_counts": guard.devnull_open_counts,
    "receipt": receipt,
    "violation_count": guard.forbidden_violation_count,
}, sort_keys=True, separators=(",", ":")))
'''
            completed = subprocess.run(
                [sys.executable, "-B", "-c", script, str(root)],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(completed.stdout)
            self.assertEqual(result["blocked"], [True] * 8)
            self.assertEqual(
                result["devnull_counts"],
                {
                    "python-open-r+": 1,
                    "python-open-w": 1,
                    "os-open-O_RDWR": 1,
                    "os-open-O_WRONLY|O_CREAT|O_TRUNC": 1,
                },
            )
            self.assertEqual(
                result["receipt"]["devnull_open_counts"],
                result["devnull_counts"],
            )
            self.assertTrue(
                result["receipt"]["write_capable_open_exceptions_sealed"]
            )
            self.assertEqual(result["violation_count"], 8)
            self.assertFalse((root / "persistent.txt").exists())
            self.assertFalse((root / "forbidden-directory").exists())

    def test_cpu_static_preflight_rejects_preseeded_tempfile_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            script = r'''
import inspect
import sys
import tempfile
from pathlib import Path
from methods.bernini_action_editing import infer_action_edit_level_b_renderer_0817_v1 as module

root = Path(sys.argv[1]).resolve()
implementation = inspect.getclosurevars(
    module.run_level_b_cpu_static_runtime_preflight
).nonlocals["implementation"]
tempfile.tempdir = "/nonexistent"
try:
    implementation({
        "bernini_root": str(root),
        "veomni_root": str(root),
        "base_checkpoint": str(root / "checkpoint"),
    })
except module.LevelBRendererError as error:
    print(str(error))
else:
    raise AssertionError("preseeded tempfile cache was accepted")
'''
            environment = dict(os.environ)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "HIP_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
            ):
                environment[name] = ""
            environment["OPENBLAS_MAIN_FREE"] = "1"
            environment["GOTOBLAS_MAIN_FREE"] = "1"
            environment["VEOMNI_VERBOSITY"] = "ERROR"
            completed = subprocess.run(
                [sys.executable, "-B", "-c", script, str(root)],
                cwd=ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                completed.stdout.strip(),
                "CPU static preflight tempfile cache is not fresh",
            )

    def test_cpu_static_preflight_requires_exact_blas_import_environment(self):
        implementation = inspect.getclosurevars(
            level_b.run_level_b_cpu_static_runtime_preflight
        ).nonlocals["implementation"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            runtime = {
                "bernini_root": str(root),
                "veomni_root": str(root),
                "base_checkpoint": str(checkpoint),
            }
            for environment in (
                {
                    "CUDA_VISIBLE_DEVICES": "",
                    "HIP_VISIBLE_DEVICES": "",
                    "ROCR_VISIBLE_DEVICES": "",
                    "OPENBLAS_MAIN_FREE": "1",
                },
                {
                    "CUDA_VISIBLE_DEVICES": "",
                    "HIP_VISIBLE_DEVICES": "",
                    "ROCR_VISIBLE_DEVICES": "",
                    "OPENBLAS_MAIN_FREE": "1",
                    "GOTOBLAS_MAIN_FREE": "0",
                },
            ):
                with self.subTest(environment=environment), mock.patch.dict(
                    os.environ, environment, clear=True
                ):
                    with self.assertRaisesRegex(
                        level_b.LevelBRendererError,
                        "BLAS import environment differs",
                    ):
                        implementation(runtime)

    def test_cpu_static_preflight_requires_exact_veomni_logging_environment(self):
        implementation = inspect.getclosurevars(
            level_b.run_level_b_cpu_static_runtime_preflight
        ).nonlocals["implementation"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            runtime = {
                "bernini_root": str(root),
                "veomni_root": str(root),
                "base_checkpoint": str(checkpoint),
            }
            base_environment = {
                "CUDA_VISIBLE_DEVICES": "",
                "HIP_VISIBLE_DEVICES": "",
                "ROCR_VISIBLE_DEVICES": "",
                "OPENBLAS_MAIN_FREE": "1",
                "GOTOBLAS_MAIN_FREE": "1",
            }
            for value in (None, "INFO", "error"):
                environment = dict(base_environment)
                if value is not None:
                    environment["VEOMNI_VERBOSITY"] = value
                with self.subTest(value=value), mock.patch.dict(
                    os.environ, environment, clear=True
                ):
                    with self.assertRaisesRegex(
                        level_b.LevelBRendererError,
                        "VeOmni logging environment differs",
                    ):
                        implementation(runtime)

    def test_replaced_visible_validator_cannot_authorize_forged_runtime(self):
        forged = object.__new__(level_b.VerifiedLevelBRuntime)
        with mock.patch.object(
            level_b.VerifiedLevelBRuntime,
            "validate_at_use",
            return_value={"caller_signed": True},
        ):
            with self.assertRaisesRegex(
                level_b.LevelBRendererError,
                "forged, stale, or already consumed",
            ):
                level_b.run_level_b_pre_d0_offline_inference(
                    fresh_bundle=None,
                    verified_runtime=forged,
                    source_video_path="/caller/source.mp4",
                    expected_source_video_sha256="0" * 64,
                    edit_instruction="forged",
                    inference_seed=1,
                    output_mp4_path="/caller/output.mp4",
                )

    def test_introspected_registry_cannot_self_authorize_empty_release(self):
        nonlocals = inspect.getclosurevars(
            level_b.VerifiedLevelBRuntime.validate_at_use
        ).nonlocals
        registry = nonlocals.get("registry")
        self.assertIsInstance(registry, dict)
        forged = object.__new__(level_b.VerifiedLevelBRuntime)
        release = {
            "manifest_path": "/caller/forged/RELEASE_MANIFEST.json",
            "manifest_sha256": "0" * 64,
            "members": {},
        }
        unsigned = {
            "schema_version": level_b.RUNTIME_RELEASE_SCHEMA,
            "authority": level_b.AUTHORITY,
            "release": release,
            "sealed_launcher_manifest_sha256": "0" * 64,
            "sealed_launcher_pin_captured_before_level_b_source_exec": True,
            "bernini_root": "/caller/bernini",
            "veomni_root": "/caller/veomni",
            "base_checkpoint": "/caller/base",
            "base_checkpoint_tree_sha256": level_b.PINNED_BASE_CHECKPOINT_TREE_SHA256,
            "checkpoint_content_manifest": {},
            "ffmpeg": {},
            "ffprobe": {},
            "python": {},
            "stdlib_socket_source": {},
            "vendor_source_files": {},
            "diffusers_version": level_b.PINNED_DIFFUSERS_VERSION,
            "transformers_version": level_b.PINNED_TRANSFORMERS_VERSION,
            "torch_version": level_b.PINNED_TORCH_VERSION,
            "fixed_paths_and_hashes_are_not_product_inputs": True,
            "single_use_opaque_capability": True,
        }
        forged_receipt = {
            **unsigned,
            "runtime_digest": level_b.object_sha256(unsigned),
        }
        registry[id(forged)] = (forged, forged_receipt)
        with self.assertRaises(level_b.LevelBRendererError):
            forged.validate_at_use()
        self.assertNotIn(id(forged), registry)

    def test_authenticated_source_audit_rejects_replaced_module_symbol(self):
        module = sys.modules[__name__]
        original = authenticated_callable_identity_fixture
        path = Path(__file__).resolve()
        expected_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        setattr(
            module,
            "authenticated_callable_identity_fixture",
            lambda value: value,
        )
        try:
            with self.assertRaisesRegex(
                level_b.LevelBRendererError, "owned by its pinned source bytes"
            ):
                level_b._audit_callable_against_authenticated_source(
                    original,
                    label="hostile substituted vendor callable",
                    expected_module=__name__,
                    expected_qualname="authenticated_callable_identity_fixture",
                    expected_path=path,
                    expected_sha256=expected_sha,
                )
        finally:
            setattr(module, "authenticated_callable_identity_fixture", original)

    def test_inherited_classmethod_from_substitute_owner_is_rejected(self):
        class SubstituteOwner(AuthenticatedOwnerFixture):
            pass

        path = Path(__file__).resolve()
        with self.assertRaisesRegex(
            level_b.LevelBRendererError, "owned by its pinned source bytes"
        ):
            level_b._audit_callable_against_authenticated_source(
                SubstituteOwner.load,
                label="substitute inherited loader",
                expected_module=__name__,
                expected_qualname="AuthenticatedOwnerFixture.load",
                expected_path=path,
                expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                expected_bound_owner=AuthenticatedOwnerFixture,
            )

    @staticmethod
    def _install_wrapped_inherited_classmethod_fixture(directory, suffix):
        root = Path(directory).resolve()
        decorator_name = f"level_b_wrapped_decorator_{suffix}"
        definition_name = f"level_b_wrapped_definition_{suffix}"
        decorator_path = root / f"{decorator_name}.py"
        definition_path = root / f"{definition_name}.py"
        decorator_path.write_text(
            "from functools import wraps\n"
            "import inspect\n"
            "def validate_hf_hub_args(fn):\n"
            "    signature = inspect.signature(fn)\n"
            "    @wraps(fn)\n"
            "    def _inner_fn(*args, **kwargs):\n"
            "        signature.bind_partial(*args, **kwargs)\n"
            "        return fn(*args, **kwargs)\n"
            "    return _inner_fn\n",
            encoding="utf-8",
        )
        definition_path.write_text(
            f"from {decorator_name} import validate_hf_hub_args\n"
            "class ConfigMixin:\n"
            "    @classmethod\n"
            "    @validate_hf_hub_args\n"
            "    def load_config(cls, value=None):\n"
            "        return cls, value\n"
            "class AutoencoderKLWan(ConfigMixin):\n"
            "    pass\n",
            encoding="utf-8",
        )
        prior = {
            decorator_name: sys.modules.get(decorator_name),
            definition_name: sys.modules.get(definition_name),
        }
        decorator_module = load(decorator_name, decorator_path)
        definition_module = load(definition_name, definition_path)
        kwargs = {
            "label": "fixture VAE load_config",
            "expected_bound_owner": definition_module.AutoencoderKLWan,
            "method_name": "load_config",
            "expected_definition_module": definition_name,
            "expected_definition_owner_qualname": "ConfigMixin",
            "expected_definition_qualname": "ConfigMixin.load_config",
            "expected_definition_path": definition_path,
            "expected_definition_sha256": hashlib.sha256(
                definition_path.read_bytes()
            ).hexdigest(),
            "expected_wrapper_module": decorator_name,
            "expected_wrapper_factory_qualname": "validate_hf_hub_args",
            "expected_wrapper_path": decorator_path,
            "expected_wrapper_sha256": hashlib.sha256(
                decorator_path.read_bytes()
            ).hexdigest(),
        }
        return decorator_module, definition_module, kwargs, prior

    @staticmethod
    def _restore_wrapped_inherited_classmethod_fixture(prior):
        for name, module in prior.items():
            sys.modules.pop(name, None)
            if module is not None:
                sys.modules[name] = module

    def test_real_inherited_decorated_classmethod_two_layer_owner_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            _, definition, kwargs, prior = (
                self._install_wrapped_inherited_classmethod_fixture(
                    directory, "positive"
                )
            )
            try:
                receipt = level_b._audit_inherited_wrapped_classmethod_against_authenticated_sources(
                    definition.AutoencoderKLWan.load_config,
                    **kwargs,
                )
            finally:
                self._restore_wrapped_inherited_classmethod_fixture(prior)
        self.assertTrue(receipt["exact_mro_definition_owner_identity_verified"])
        self.assertTrue(receipt["exact_bound_owner_identity_verified"])
        self.assertTrue(
            receipt["wrapper"]["exact_decorator_factory_code_identity_verified"]
        )
        self.assertEqual(
            receipt["wrapper"]["decorator_factory_nested_code_identity_count"],
            1,
        )
        self.assertEqual(receipt["wrapper"]["closure_freevars"], ["fn", "signature"])
        self.assertTrue(
            receipt["unwrapped_definition"][
                "one_hop_wrapped_target_identity_verified"
            ]
        )

    def test_inherited_decorated_classmethod_moved_to_substitute_owner_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, definition, kwargs, prior = (
                self._install_wrapped_inherited_classmethod_fixture(
                    directory, "substitute_owner"
                )
            )
            try:
                descriptor = vars(definition.ConfigMixin)["load_config"]

                class SubstituteOwner(definition.ConfigMixin):
                    load_config = descriptor

                class Product(SubstituteOwner):
                    pass

                kwargs["expected_bound_owner"] = Product
                with self.assertRaisesRegex(
                    level_b.LevelBRendererError,
                    "inherited classmethod owner differs",
                ):
                    level_b._audit_inherited_wrapped_classmethod_against_authenticated_sources(
                        Product.load_config,
                        **kwargs,
                    )
            finally:
                self._restore_wrapped_inherited_classmethod_fixture(prior)

    def test_inherited_decorated_classmethod_hostile_wrapped_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            decorator, definition, kwargs, prior = (
                self._install_wrapped_inherited_classmethod_fixture(
                    directory, "hostile_target"
                )
            )
            original_descriptor = vars(definition.ConfigMixin)["load_config"]
            try:
                def hostile_original(cls, value=None):
                    return "hostile", cls, value

                hostile_original.__module__ = definition.__name__
                hostile_original.__qualname__ = "ConfigMixin.load_config"
                definition.ConfigMixin.load_config = classmethod(
                    decorator.validate_hf_hub_args(hostile_original)
                )
                with self.assertRaisesRegex(
                    level_b.LevelBRendererError,
                    "wrapper/definition ownership differs",
                ):
                    level_b._audit_inherited_wrapped_classmethod_against_authenticated_sources(
                        definition.AutoencoderKLWan.load_config,
                        **kwargs,
                    )
            finally:
                definition.ConfigMixin.load_config = original_descriptor
                self._restore_wrapped_inherited_classmethod_fixture(prior)

    def test_runtime_authenticator_accepts_manifest_path_only(self):
        self.assertEqual(
            tuple(inspect.signature(level_b.authenticate_level_b_runtime_release).parameters),
            ("manifest_path",),
        )
        with self.assertRaisesRegex(
            level_b.LevelBRendererError, "import-time sealed-launcher"
        ):
            level_b.authenticate_level_b_runtime_release("/caller/manifest.json")

    def test_materialize_and_builder_are_both_loaded_from_authenticated_bytes(self):
        names = ("tools.materialize_vae", "tools.build_renderer_dataset", "tools")
        prior = {name: sys.modules.get(name) for name in names}
        for name in names:
            sys.modules.pop(name, None)
        try:
            builder_path = METHOD_ROOT / "tools" / "build_renderer_dataset.py"
            materialize_path = METHOD_ROOT / "tools" / "materialize_vae.py"
            module = level_b._install_authenticated_materialize_module(
                path=materialize_path.resolve(),
                expected_sha256=hashlib.sha256(
                    materialize_path.read_bytes()
                ).hexdigest(),
                builder_path=builder_path.resolve(),
                expected_builder_sha256=hashlib.sha256(
                    builder_path.read_bytes()
                ).hexdigest(),
            )
            self.assertIs(
                module.raw_builder, sys.modules["tools.build_renderer_dataset"]
            )
            self.assertIs(
                sys.modules["tools"].materialize_vae,
                sys.modules["tools.materialize_vae"],
            )
        finally:
            for name in names:
                sys.modules.pop(name, None)
            for name, value in prior.items():
                if value is not None:
                    sys.modules[name] = value

    def test_late_commit_validation_failure_leaves_no_output_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            transaction, receipt, output, receipt_output, marker_output = (
                self._make_staged_transaction(root)
            )
            with mock.patch.object(
                level_b,
                "_nfs_reopen_precommit_product_pair",
                side_effect=level_b.LevelBRendererError("hostile late reopen failure"),
            ):
                with self.assertRaisesRegex(
                    level_b.LevelBRendererError, "late reopen failure"
                ):
                    level_b._publish_precommit_product_pair(
                        transaction=transaction, receipt=receipt
                    )
            self.assertFalse(output.exists())
            self.assertFalse(receipt_output.exists())
            self.assertFalse(marker_output.exists())
            self.assertFalse(transaction.directory.exists())

    def test_receipt_staging_failure_cleans_private_mp4_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            transaction, receipt, output, receipt_output, marker_output = (
                self._make_staged_transaction(root)
            )
            with mock.patch.object(
                level_b,
                "_write_new_staged_json",
                side_effect=level_b.LevelBRendererError("hostile receipt-stage failure"),
            ):
                with self.assertRaisesRegex(
                    level_b.LevelBRendererError, "receipt-stage failure"
                ):
                    level_b._publish_precommit_product_pair(
                        transaction=transaction, receipt=receipt
                    )
            self.assertFalse(output.exists())
            self.assertFalse(receipt_output.exists())
            self.assertFalse(marker_output.exists())
            self.assertFalse(transaction.directory.exists())

    def _make_staged_transaction(self, root):
        staging = root / ".stage"
        staging.mkdir()
        mp4_stage = staging / "payload.mp4"
        mp4_stage.write_bytes(b"real-validated-mp4-placeholder")
        os.chmod(mp4_stage, 0o444)
        mp4_sha, mp4_identity = level_b.stable_file_sha256(
            mp4_stage, label="test staged MP4"
        )
        output = root / "result.mp4"
        receipt_output = root / "result.mp4.receipt.json"
        marker_output = root / "result.mp4.COMMITTED.json"
        transaction = level_b._StagedProductPair(
            directory=staging,
            mp4_path=mp4_stage,
            receipt_path=staging / "receipt.json",
            commit_marker_path=staging / "COMMITTED.json",
            final_mp4_path=output,
            final_receipt_path=receipt_output,
            final_commit_marker_path=marker_output,
            mp4_sha256=mp4_sha,
            mp4_identity=mp4_identity,
        )
        marker_envelope = level_b._commit_marker_envelope(
            output_path=output,
            receipt_path=receipt_output,
            commit_marker_path=marker_output,
            mp4_sha256=mp4_sha,
        )
        unsigned = {
            "schema_version": level_b.RECEIPT_SCHEMA,
            "authority": level_b.AUTHORITY,
            "complete": True,
            "commit_marker_envelope": marker_envelope,
            "output_transaction": {
                "atomic_commit_marker_schema": level_b.OUTPUT_COMMIT_MARKER_SCHEMA,
                "commit_marker_path": str(marker_output),
                "commit_marker_is_only_consumer_completion_authority": True,
                "bare_mp4_or_receipt_pair_is_never_complete": True,
                "world8_canonical_gate_required_before_commit": True,
                "all8_precommit_reopen_required_before_marker": True,
                "marker_link_is_final_business_action": True,
                "marker_is_receipt_inode_alias": True,
            },
        }
        receipt = {
            **unsigned,
            "receipt_digest": level_b.object_sha256(unsigned),
        }
        return transaction, receipt, output, receipt_output, marker_output

    def _precommit_and_prepare_marker(self, transaction, receipt):
        precommit = level_b._publish_precommit_product_pair(
            transaction=transaction, receipt=receipt
        )
        gate_contract = receipt["commit_marker_envelope"][
            "world8_precommit_gate_contract"
        ]

        class SuccessfulDistributed:
            @staticmethod
            def all_gather_object(rows, local, *, group):
                self.assertEqual(group, "world8")
                for rank in range(level_b.WORLD_SIZE):
                    rows[rank] = json.loads(json.dumps(local))
                    rows[rank]["rank"] = rank

        gate = level_b._gather_world8_precommit_reopen_consensus(
            rank=0,
            output_path=transaction.final_mp4_path,
            receipt_path=transaction.final_receipt_path,
            commit_marker_path=transaction.final_commit_marker_path,
            expected_mp4_sha256=transaction.mp4_sha256,
            expected_receipt_sha256=precommit["receipt_sha256"],
            expected_gate_contract=gate_contract,
            distributed_module=SuccessfulDistributed(),
            group="world8",
        )
        publisher, readiness = (
            level_b._prepare_final_receipt_alias_marker_publisher(
                transaction=transaction,
                precommit_receipt=precommit,
                world8_gate_evidence=gate,
            )
        )
        return precommit, gate, publisher, readiness

    def _bind_guard(self, transaction, output, receipt_output, marker_output):
        guard = level_b._LevelBOutputRollbackGuard()
        guard.bind_output_paths(
            output_path=output,
            receipt_path=receipt_output,
            commit_marker_path=marker_output,
        )
        guard.register_stage(transaction)
        return guard

    def _assert_no_stage_or_public_product(
        self, transaction, output, receipt_output, marker_output
    ):
        self.assertFalse(transaction.directory.exists())
        self.assertFalse(output.exists())
        self.assertFalse(receipt_output.exists())
        self.assertFalse(marker_output.exists())
        with self.assertRaises(level_b.LevelBRendererError):
            level_b.validate_committed_level_b_product(
                output_mp4_path=output
            )

    def test_side_effect_phase_broadcast_failures_rollback_stage_and_commit(self):
        class FailingBroadcast:
            @staticmethod
            def broadcast_object_list(_box, *, src, group):
                self.assertEqual((src, group), (0, "world8"))
                raise RuntimeError("injected success-status broadcast failure")

        for phase in (
            "ffmpeg_validate_and_output_staging",
            "create_only_output_precommit",
        ):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                transaction, receipt, output, receipt_output, marker_output = (
                    self._make_staged_transaction(root)
                )
                guard = self._bind_guard(
                    transaction, output, receipt_output, marker_output
                )

                def operation():
                    if phase == "ffmpeg_validate_and_output_staging":
                        return {"staged": True}
                    precommit = level_b._publish_precommit_product_pair(
                        transaction=transaction, receipt=receipt
                    )
                    self.assertFalse(marker_output.exists())
                    return precommit

                with mock.patch.object(
                    level_b,
                    "_fsync_directory",
                    wraps=level_b._fsync_directory,
                ) as fsync_directory:
                    with self.assertRaisesRegex(
                        level_b.LevelBRendererError,
                        "collective broadcast failed",
                    ):
                        level_b._run_world8_rank0_collective_phase(
                            phase=phase,
                            rank=0,
                            operation=operation,
                            distributed_module=FailingBroadcast(),
                            group="world8",
                            rollback_on_failure=guard.rollback,
                        )
                labels = [
                    call.kwargs.get("label")
                    for call in fsync_directory.call_args_list
                ]
                self.assertIn("Level-B staging rollback parent", labels)
                if phase == "ffmpeg_validate_and_output_staging":
                    self.assertIn(
                        "Level-B staging directory after cleanup", labels
                    )
                # The helper rollback and outer retry are both idempotent.
                guard.rollback()
                self._assert_no_stage_or_public_product(
                    transaction, output, receipt_output, marker_output
                )

    def test_precommit_handoff_failure_uses_prelink_identity_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            transaction, receipt, output, receipt_output, marker_output = (
                self._make_staged_transaction(root)
            )
            guard = self._bind_guard(
                transaction, output, receipt_output, marker_output
            )

            class SharedBroadcast:
                payload = None

            class FakeDistributed:
                @staticmethod
                def broadcast_object_list(box, *, src, group):
                    SharedBroadcast.payload = box[0]

            def precommit_then_fail_handoff():
                level_b._publish_precommit_product_pair(
                    transaction=transaction, receipt=receipt
                )
                self.assertTrue(output.exists())
                self.assertTrue(receipt_output.exists())
                self.assertFalse(marker_output.exists())
                raise RuntimeError("injected precommit handoff failure")

            with self.assertRaisesRegex(
                level_b.LevelBRendererError, "precommit handoff failure"
            ):
                level_b._run_world8_rank0_collective_phase(
                    phase="create_only_output_precommit",
                    rank=0,
                    operation=precommit_then_fail_handoff,
                    distributed_module=FakeDistributed(),
                    group="world8",
                    rollback_on_failure=guard.rollback,
                )
            self._assert_no_stage_or_public_product(
                transaction, output, receipt_output, marker_output
            )

    def test_stage_handoff_failure_is_cleaned_inside_stage_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = root / "result.mp4"
            receipt_output = root / "result.mp4.receipt.json"
            marker_output = root / "result.mp4.COMMITTED.json"
            guard = level_b._LevelBOutputRollbackGuard()
            guard.bind_output_paths(
                output_path=output,
                receipt_path=receipt_output,
                commit_marker_path=marker_output,
            )

            def encoder(_frames, path, *, fps):
                self.assertEqual(fps, int(level_b.FPS))
                Path(path).write_bytes(b"validated-mp4")
                return {"encoder": "authenticated-test-double"}

            def validation(*, mp4_path, **_kwargs):
                digest, identity = level_b.stable_file_sha256(
                    mp4_path, label="hostile staged handoff MP4"
                )
                return {
                    "mp4_sha256": digest,
                    "mp4_file_identity": identity,
                }

            with mock.patch.object(
                level_b,
                "_validate_full_mp4_bytes_with_authenticated_tools",
                side_effect=validation,
            ), mock.patch.object(
                guard,
                "register_stage",
                side_effect=RuntimeError("injected stage handoff failure"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "stage handoff failure"
                ):
                    level_b._stage_real_mp4(
                        decoded_frames=object(),
                        output_path=output,
                        receipt_path=receipt_output,
                        commit_marker_path=marker_output,
                        authenticated_encoder=encoder,
                        encoder_authority={"authenticated": True},
                        expected_height=16,
                        expected_width=16,
                        runtime_receipt={
                            "ffmpeg": {"path": "/fixed/ffmpeg", "sha256": "a" * 64},
                            "ffprobe": {"path": "/fixed/ffprobe", "sha256": "b" * 64},
                        },
                        output_rollback_guard=guard,
                    )
            self.assertEqual(list(root.glob(".result.mp4.stage-*")), [])
            self.assertFalse(output.exists())
            self.assertFalse(receipt_output.exists())
            self.assertFalse(marker_output.exists())

    def test_stage_chmod_failure_after_mkdir_leaves_no_private_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = root / "result.mp4"
            receipt_output = root / "result.mp4.receipt.json"
            marker_output = root / "result.mp4.COMMITTED.json"
            guard = level_b._LevelBOutputRollbackGuard()
            guard.bind_output_paths(
                output_path=output,
                receipt_path=receipt_output,
                commit_marker_path=marker_output,
            )
            real_chmod = os.chmod
            injected = False

            def failing_stage_chmod(path, mode, *args, **kwargs):
                nonlocal injected
                if Path(path).parent == root and ".result.mp4.stage-" in Path(path).name:
                    injected = True
                    raise OSError("injected post-mkdir chmod failure")
                return real_chmod(path, mode, *args, **kwargs)

            with mock.patch.object(
                level_b.os, "chmod", side_effect=failing_stage_chmod
            ), mock.patch.object(
                level_b,
                "_fsync_directory",
                wraps=level_b._fsync_directory,
            ) as fsync_directory:
                with self.assertRaisesRegex(OSError, "post-mkdir chmod"):
                    level_b._stage_real_mp4(
                        decoded_frames=object(),
                        output_path=output,
                        receipt_path=receipt_output,
                        commit_marker_path=marker_output,
                        authenticated_encoder=lambda *_args, **_kwargs: self.fail(
                            "encoder must not run after stage chmod failure"
                        ),
                        encoder_authority={"authenticated": True},
                        expected_height=16,
                        expected_width=16,
                        runtime_receipt={
                            "ffmpeg": {"path": "/fixed/ffmpeg", "sha256": "a" * 64},
                            "ffprobe": {"path": "/fixed/ffprobe", "sha256": "b" * 64},
                        },
                        output_rollback_guard=guard,
                    )
            self.assertTrue(injected)
            self.assertEqual(list(root.glob(".result.mp4.stage-*")), [])
            self.assertFalse(output.exists())
            self.assertFalse(receipt_output.exists())
            self.assertFalse(marker_output.exists())
            self.assertIn(
                "failed Level-B staging parent",
                [call.kwargs.get("label") for call in fsync_directory.call_args_list],
            )

    def test_real_stage_mkdir_then_baseexception_leaves_no_private_directory(self):
        class FatalMkdirExit(BaseException):
            pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = root / "result.mp4"
            receipt_output = root / "result.mp4.receipt.json"
            marker_output = root / "result.mp4.COMMITTED.json"
            guard = level_b._LevelBOutputRollbackGuard()
            guard.bind_output_paths(
                output_path=output,
                receipt_path=receipt_output,
                commit_marker_path=marker_output,
            )
            real_mkdir = os.mkdir
            injected = False

            def mkdir_then_fatal(path, mode=0o777, *args, **kwargs):
                nonlocal injected
                real_mkdir(path, mode, *args, **kwargs)
                injected = True
                raise FatalMkdirExit("injected post-mkdir BaseException")

            with mock.patch.object(
                level_b.os, "mkdir", side_effect=mkdir_then_fatal
            ):
                with self.assertRaisesRegex(
                    FatalMkdirExit, "post-mkdir BaseException"
                ):
                    level_b._stage_real_mp4(
                        decoded_frames=object(),
                        output_path=output,
                        receipt_path=receipt_output,
                        commit_marker_path=marker_output,
                        authenticated_encoder=lambda *_args, **_kwargs: self.fail(
                            "encoder must not run after stage mkdir failure"
                        ),
                        encoder_authority={"authenticated": True},
                        expected_height=16,
                        expected_width=16,
                        runtime_receipt={
                            "ffmpeg": {"path": "/fixed/ffmpeg", "sha256": "a" * 64},
                            "ffprobe": {"path": "/fixed/ffprobe", "sha256": "b" * 64},
                        },
                        output_rollback_guard=guard,
                    )
            self.assertTrue(injected)
            self.assertEqual(list(root.glob(".result.mp4.stage-*")), [])
            self.assertFalse(output.exists())
            self.assertFalse(receipt_output.exists())
            self.assertFalse(marker_output.exists())

    def test_real_link_then_baseexception_is_closed_by_prelink_ledger(self):
        class FatalLinkExit(BaseException):
            pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            transaction, receipt, output, receipt_output, marker_output = (
                self._make_staged_transaction(root)
            )
            real_link = os.link
            injected = False

            def link_then_fatal(source, destination, **kwargs):
                nonlocal injected
                real_link(source, destination, **kwargs)
                if not injected:
                    injected = True
                    raise FatalLinkExit("injected post-link BaseException")

            with mock.patch.object(
                level_b.os, "link", side_effect=link_then_fatal
            ):
                with self.assertRaisesRegex(
                    FatalLinkExit, "post-link BaseException"
                ):
                    level_b._publish_precommit_product_pair(
                        transaction=transaction, receipt=receipt
                    )
            self.assertTrue(injected)
            self._assert_no_stage_or_public_product(
                transaction, output, receipt_output, marker_output
            )

    def test_precommit_all8_failures_leave_no_acceptable_product(self):
        class HostileDistributed:
            def __init__(self, mode):
                self.mode = mode

            def all_gather_object(self, rows, local, *, group):
                if self.mode == "all_gather_failure":
                    raise RuntimeError("injected precommit all_gather failure")
                for rank in range(level_b.WORLD_SIZE):
                    rows[rank] = json.loads(json.dumps(local))
                    rows[rank]["rank"] = rank
                if self.mode == "consensus_mismatch":
                    rows[-1]["receipt"]["mp4_sha256"] = "f" * 64
                elif self.mode == "duplicate_rank":
                    rows[-1]["rank"] = 0

            def broadcast_object_list(self, _box, *, src, group):
                if self.mode == "status_broadcast_failure":
                    raise RuntimeError(
                        "injected precommit status broadcast failure"
                    )

            def barrier(self, *, group):
                if self.mode == "barrier_failure":
                    raise RuntimeError("injected precommit barrier failure")

        modes = (
            "all_gather_failure",
            "consensus_mismatch",
            "duplicate_rank",
            "status_broadcast_failure",
            "barrier_failure",
        )
        for mode in modes:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                transaction, receipt, output, receipt_output, marker_output = (
                    self._make_staged_transaction(root)
                )

                def guarded_precommit(guard):
                    guard.bind_output_paths(
                        output_path=output,
                        receipt_path=receipt_output,
                        commit_marker_path=marker_output,
                    )
                    guard.register_stage(transaction)
                    precommit = level_b._publish_precommit_product_pair(
                        transaction=transaction, receipt=receipt
                    )
                    self.assertFalse(marker_output.exists())
                    with self.assertRaises(level_b.LevelBRendererError):
                        level_b.validate_committed_level_b_product(
                            output_mp4_path=output
                        )
                    gate_contract = receipt["commit_marker_envelope"][
                        "world8_precommit_gate_contract"
                    ]
                    gate = level_b._gather_world8_precommit_reopen_consensus(
                        rank=0,
                        output_path=output,
                        receipt_path=receipt_output,
                        commit_marker_path=marker_output,
                        expected_mp4_sha256=transaction.mp4_sha256,
                        expected_receipt_sha256=precommit["receipt_sha256"],
                        expected_gate_contract=gate_contract,
                        distributed_module=HostileDistributed(mode),
                        group="world8",
                    )
                    publisher, readiness = (
                        level_b._prepare_final_receipt_alias_marker_publisher(
                            transaction=transaction,
                            precommit_receipt=precommit,
                            world8_gate_evidence=gate,
                        )
                    )
                    level_b._run_world8_rank0_collective_phase(
                        phase="final_marker_readiness_status",
                        rank=0,
                        operation=lambda: readiness,
                        distributed_module=HostileDistributed(mode),
                        group="world8",
                        rollback_on_failure=guard.rollback,
                    )
                    HostileDistributed(mode).barrier(group="world8")
                    guard.disarm()
                    publisher.publish()

                with self.assertRaises(BaseException):
                    level_b._run_with_level_b_output_rollback(
                        guarded_precommit
                    )
                self._assert_no_stage_or_public_product(
                    transaction, output, receipt_output, marker_output
                )

    def test_successful_outer_guard_disarms_only_after_precommit_barrier(self):
        class SuccessfulDistributed:
            barrier_completed = False

            @staticmethod
            def all_gather_object(rows, local, *, group):
                for rank in range(level_b.WORLD_SIZE):
                    rows[rank] = json.loads(json.dumps(local))
                    rows[rank]["rank"] = rank

            @staticmethod
            def broadcast_object_list(_box, *, src, group):
                return None

            @classmethod
            def barrier(cls, *, group):
                cls.barrier_completed = True

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            transaction, receipt, output, receipt_output, marker_output = (
                self._make_staged_transaction(root)
            )

            def guarded_success(guard):
                guard.bind_output_paths(
                    output_path=output,
                    receipt_path=receipt_output,
                    commit_marker_path=marker_output,
                )
                guard.register_stage(transaction)
                precommit, gate, publisher, readiness = (
                    self._precommit_and_prepare_marker(transaction, receipt)
                )
                self.assertEqual(
                    readiness["receipt_sha256"], precommit["receipt_sha256"]
                )
                level_b._run_world8_rank0_collective_phase(
                    phase="final_marker_readiness_status",
                    rank=0,
                    operation=lambda: readiness,
                    distributed_module=SuccessfulDistributed(),
                    group="world8",
                )
                SuccessfulDistributed.barrier(group="world8")
                self.assertTrue(SuccessfulDistributed.barrier_completed)
                guard.disarm()
                publisher.publish()
                return gate

            gate = level_b._run_with_level_b_output_rollback(guarded_success)
            self.assertTrue(gate["all8_precommit_nfs_reopen_exact_consensus"])
            self.assertTrue(marker_output.exists())
            self.assertTrue(
                level_b.validate_committed_level_b_product(
                    output_mp4_path=output
                )["atomic_commit_marker_is_only_completion_authority"]
            )

    def test_bare_mp4_and_pair_without_marker_are_never_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = root / "result.mp4"
            output.write_bytes(b"not-authoritative-by-itself")
            os.chmod(output, 0o444)
            with self.assertRaisesRegex(
                level_b.LevelBRendererError, "receipt.*unavailable"
            ):
                level_b.validate_committed_level_b_product(
                    output_mp4_path=output
                )
            unsigned = {
                "schema_version": level_b.RECEIPT_SCHEMA,
                "authority": level_b.AUTHORITY,
                "complete": True,
            }
            receipt = output.with_name(output.name + ".receipt.json")
            receipt.write_bytes(
                level_b.canonical_json_bytes(
                    {**unsigned, "receipt_digest": level_b.object_sha256(unsigned)}
                )
                + b"\n"
            )
            os.chmod(receipt, 0o444)
            with self.assertRaisesRegex(
                level_b.LevelBRendererError, "atomic marker.*unavailable"
            ):
                level_b.validate_committed_level_b_product(
                    output_mp4_path=output
                )

    def test_commit_marker_is_linked_last_and_is_only_completion_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            transaction, receipt, output, receipt_output, marker_output = (
                self._make_staged_transaction(root)
            )
            real_link = os.link
            publication_order = []

            def observing_link(source, destination, **kwargs):
                publication_order.append(Path(destination).name)
                if Path(destination).name != marker_output.name:
                    self.assertFalse(marker_output.exists())
                return real_link(source, destination, **kwargs)

            with mock.patch.object(level_b.os, "link", side_effect=observing_link):
                precommit, gate, publisher, readiness = (
                    self._precommit_and_prepare_marker(transaction, receipt)
                )
                self.assertTrue(gate["commit_marker_absent_on_all8_reopens"])
                self.assertTrue(
                    readiness["marker_link_is_terminal_filesystem_action"]
                )
                publisher.publish()
            self.assertEqual(
                publication_order,
                [output.name, receipt_output.name, marker_output.name],
            )
            self.assertEqual(
                receipt_output.lstat().st_ino, marker_output.lstat().st_ino
            )
            self.assertEqual(receipt_output.lstat().st_nlink, 2)
            opened = level_b.validate_committed_level_b_product(
                output_mp4_path=output
            )
            self.assertTrue(
                opened["atomic_commit_marker_is_only_completion_authority"]
            )

    def test_crash_after_first_publication_rolls_back_all_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            transaction, receipt, output, receipt_output, marker_output = (
                self._make_staged_transaction(root)
            )
            real_link = os.link
            calls = 0

            def crashing_link(source, destination, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("simulated observer-visible commit crash")
                return real_link(source, destination, **kwargs)

            with mock.patch.object(level_b.os, "link", side_effect=crashing_link):
                with self.assertRaisesRegex(OSError, "simulated.*crash"):
                    level_b._publish_precommit_product_pair(
                        transaction=transaction, receipt=receipt
                    )
            self.assertFalse(output.exists())
            self.assertFalse(receipt_output.exists())
            self.assertFalse(marker_output.exists())
            self.assertFalse(transaction.directory.exists())

    def test_terminal_marker_link_has_no_post_link_patchable_hooks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            transaction, receipt, output, receipt_output, marker_output = (
                self._make_staged_transaction(root)
            )
            _precommit, _gate, publisher, _readiness = (
                self._precommit_and_prepare_marker(transaction, receipt)
            )
            with mock.patch.object(
                level_b, "_fsync_directory", side_effect=AssertionError("late fsync")
            ) as fsync_hook, mock.patch.object(
                level_b,
                "_nfs_reopen_precommit_product_pair",
                side_effect=AssertionError("late reopen"),
            ) as reopen_hook, mock.patch.object(
                level_b,
                "stable_file_sha256",
                side_effect=AssertionError("late hash"),
            ) as hash_hook, mock.patch.object(
                level_b.os, "fsync", side_effect=AssertionError("late os.fsync")
            ) as os_fsync_hook, mock.patch.object(
                level_b.os, "open", side_effect=AssertionError("late os.open")
            ) as os_open_hook, mock.patch.object(
                level_b.os, "link", side_effect=AssertionError("late live link lookup")
            ) as live_link_hook, mock.patch.object(
                Path, "lstat", side_effect=AssertionError("late lstat")
            ) as lstat_hook, mock.patch.object(
                Path, "unlink", side_effect=AssertionError("late unlink")
            ) as unlink_hook:
                publisher.publish()
            for hook in (
                fsync_hook,
                reopen_hook,
                hash_hook,
                os_fsync_hook,
                os_open_hook,
                live_link_hook,
                lstat_hook,
                unlink_hook,
            ):
                hook.assert_not_called()
            self.assertTrue(marker_output.exists())
            self.assertTrue(
                level_b.validate_committed_level_b_product(
                    output_mp4_path=output
                )["receipt_inode_alias_marker_verified"]
            )

    def test_persistent_precommit_unlink_failure_can_only_leave_rejected_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            transaction, receipt, output, receipt_output, marker_output = (
                self._make_staged_transaction(root)
            )
            guard = self._bind_guard(
                transaction, output, receipt_output, marker_output
            )
            level_b._publish_precommit_product_pair(
                transaction=transaction, receipt=receipt
            )
            real_unlink = Path.unlink

            def persistent_unlink(path, *args, **kwargs):
                if path in (output, receipt_output):
                    raise OSError("persistent precommit unlink failure")
                return real_unlink(path, *args, **kwargs)

            with mock.patch.object(Path, "unlink", persistent_unlink):
                with self.assertRaises(level_b.LevelBRendererError):
                    guard.rollback()
            self.assertTrue(output.exists())
            self.assertTrue(receipt_output.exists())
            self.assertFalse(marker_output.exists())
            with self.assertRaises(level_b.LevelBRendererError):
                level_b.validate_committed_level_b_product(
                    output_mp4_path=output
                )

    def test_persistent_precommit_lstat_failure_can_only_leave_rejected_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            transaction, receipt, output, receipt_output, marker_output = (
                self._make_staged_transaction(root)
            )
            guard = self._bind_guard(
                transaction, output, receipt_output, marker_output
            )
            level_b._publish_precommit_product_pair(
                transaction=transaction, receipt=receipt
            )
            real_lstat = Path.lstat

            def persistent_lstat(path, *args, **kwargs):
                if path in (output, receipt_output):
                    raise OSError("persistent precommit lstat failure")
                return real_lstat(path, *args, **kwargs)

            with mock.patch.object(Path, "lstat", persistent_lstat):
                with self.assertRaises(level_b.LevelBRendererError):
                    guard.rollback()
            self.assertTrue(output.exists())
            self.assertTrue(receipt_output.exists())
            self.assertFalse(marker_output.exists())
            with self.assertRaises(level_b.LevelBRendererError):
                level_b.validate_committed_level_b_product(
                    output_mp4_path=output
                )

    def test_final_marker_publish_failure_leaves_rejected_precommit_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            transaction, receipt, output, receipt_output, marker_output = (
                self._make_staged_transaction(root)
            )
            _precommit, _gate, publisher, _readiness = (
                self._precommit_and_prepare_marker(transaction, receipt)
            )
            publisher.link_function = mock.Mock(
                side_effect=OSError("injected terminal marker link failure")
            )
            with self.assertRaisesRegex(OSError, "terminal marker link failure"):
                publisher.publish()
            self.assertTrue(output.exists())
            self.assertTrue(receipt_output.exists())
            self.assertFalse(marker_output.exists())
            with self.assertRaises(level_b.LevelBRendererError):
                level_b.validate_committed_level_b_product(
                    output_mp4_path=output
                )

    def test_real_terminal_link_then_baseexception_is_irreversible_commit(self):
        class FatalAfterLink(BaseException):
            pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            transaction, receipt, output, receipt_output, marker_output = (
                self._make_staged_transaction(root)
            )
            _precommit, _gate, publisher, _readiness = (
                self._precommit_and_prepare_marker(transaction, receipt)
            )
            real_link = publisher.link_function

            def link_then_fatal(*args, **kwargs):
                real_link(*args, **kwargs)
                raise FatalAfterLink("injected exception after commit point")

            publisher.link_function = link_then_fatal
            with self.assertRaisesRegex(FatalAfterLink, "after commit point"):
                publisher.publish()
            self.assertTrue(marker_output.exists())
            self.assertTrue(
                level_b.validate_committed_level_b_product(
                    output_mp4_path=output
                )["receipt_inode_alias_marker_verified"]
            )

    def test_single_transient_stage_unlink_cannot_preserve_private_artifact(self):

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            transaction, _receipt, output, receipt_output, marker_output = (
                self._make_staged_transaction(root)
            )
            real_unlink = Path.unlink
            failed_once = False

            def transient_stage_unlink(path, *args, **kwargs):
                nonlocal failed_once
                if path == transaction.mp4_path and not failed_once:
                    failed_once = True
                    raise OSError("injected one-shot stage unlink failure")
                return real_unlink(path, *args, **kwargs)

            with mock.patch.object(Path, "unlink", transient_stage_unlink):
                transaction.cleanup_stage()
            self.assertTrue(failed_once)
            self._assert_no_stage_or_public_product(
                transaction, output, receipt_output, marker_output
            )


class ProductBoundaryAndVideoTests(unittest.TestCase):
    @staticmethod
    def file_sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @staticmethod
    def official_sample_kwargs():
        return {
            name: None for name in level_b.OFFICIAL_RENDERER_SAMPLE_KEYWORDS
        }

    @staticmethod
    def official_renderer():
        class Renderer:
            def sample(
                self,
                input_ids,
                attention_mask,
                uncond_input_ids,
                uncond_attention_mask,
                image_vae_latents,
                multi_video_vae_latents,
                multi_image_vae_latents,
                width,
                height,
                device,
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
            ):
                raise AssertionError("signature audit must not call the renderer")

        return Renderer()

    def test_decoded_array_requires_exact_float32_finite_unit_interval(self):
        import numpy as np

        valid = np.zeros((level_b.FRAME_COUNT, 2, 3, 3), dtype=np.float32)
        valid[0, 0, 0, 0] = np.float32(1.0)
        receipt = level_b._audit_decoded_video_array(
            valid,
            expected_height=2,
            expected_width=3,
            numpy_module=np,
        )
        self.assertTrue(receipt["finite"])
        self.assertTrue(receipt["closed_unit_interval"])
        hostile_values = (
            valid.astype(np.float64),
            np.full(valid.shape, np.nan, dtype=np.float32),
            np.full(valid.shape, np.float32(1.01), dtype=np.float32),
            valid.view(type("ArraySubclass", (np.ndarray,), {})),
        )
        for hostile in hostile_values:
            with self.subTest(dtype=str(hostile.dtype), type=type(hostile)), self.assertRaises(
                level_b.LevelBRendererError
            ):
                level_b._audit_decoded_video_array(
                    hostile,
                    expected_height=2,
                    expected_width=3,
                    numpy_module=np,
                )

    def test_decoded_frame_duration_fields_allow_only_closed_two_modes(self):
        absent = [{} for _ in range(level_b.FRAME_COUNT)]
        present = [
            {"pkt_duration": "512", "pkt_duration_time": "0.04"}
            for _ in range(level_b.FRAME_COUNT)
        ]
        self.assertEqual(
            level_b._decoded_frame_duration_field_mode(absent),
            "all-absent-packet-authoritative",
        )
        self.assertEqual(
            level_b._decoded_frame_duration_field_mode(present),
            "all-present-and-exact",
        )
        for hostile in (
            [*absent[:-1], {"pkt_duration": "512"}],
            [*absent[:-1], present[-1]],
        ):
            with self.assertRaisesRegex(
                level_b.LevelBRendererError, "mixed or partial"
            ):
                level_b._decoded_frame_duration_field_mode(hostile)

    def test_authenticated_encoder_forces_exact_ffmpeg_and_restores_ambient(self):
        ffmpeg = shutil.which("ffmpeg")
        alternate = shutil.which("false")
        if ffmpeg is None or alternate is None:
            self.skipTest("local ffmpeg/false executable unavailable")
        ffmpeg_path = Path(ffmpeg).resolve()
        alternate_path = Path(alternate).resolve()
        runtime = {
            "ffmpeg": {
                "path": str(ffmpeg_path),
                "sha256": self.file_sha(ffmpeg_path),
            }
        }
        selected = ModuleType("imageio_ffmpeg")
        selected.get_ffmpeg_exe = lambda: os.environ["IMAGEIO_FFMPEG_EXE"]
        observations = []

        def save_output(frames, path, *, fps):
            observations.append(
                (frames, path, fps, os.environ.get("IMAGEIO_FFMPEG_EXE"))
            )

        with mock.patch.dict(sys.modules, {"imageio_ffmpeg": selected}), mock.patch.dict(
            os.environ,
            {"IMAGEIO_FFMPEG_EXE": str(alternate_path)},
            clear=False,
        ):
            encoder, authority = level_b._make_authenticated_save_output_encoder(
                save_output=save_output, runtime_receipt=runtime
            )
            execution = encoder("frames", "/tmp/not-written.mp4", fps=25)
            self.assertEqual(
                os.environ.get("IMAGEIO_FFMPEG_EXE"), str(alternate_path)
            )
        self.assertEqual(
            observations,
            [("frames", "/tmp/not-written.mp4", 25, str(ffmpeg_path))],
        )
        self.assertEqual(authority["authenticated_ffmpeg_path"], str(ffmpeg_path))
        self.assertTrue(execution["imageio_ffmpeg_exe_explicitly_bound"])

    def test_authenticated_encoder_rejects_resolver_path_substitution(self):
        ffmpeg = shutil.which("ffmpeg")
        alternate = shutil.which("false")
        if ffmpeg is None or alternate is None:
            self.skipTest("local ffmpeg/false executable unavailable")
        ffmpeg_path = Path(ffmpeg).resolve()
        alternate_path = Path(alternate).resolve()
        runtime = {
            "ffmpeg": {
                "path": str(ffmpeg_path),
                "sha256": self.file_sha(ffmpeg_path),
            }
        }
        substituted = ModuleType("imageio_ffmpeg")
        substituted.get_ffmpeg_exe = lambda: str(alternate_path)
        calls = []

        def save_output(*args, **kwargs):
            calls.append((args, kwargs))

        with mock.patch.dict(sys.modules, {"imageio_ffmpeg": substituted}):
            encoder, _ = level_b._make_authenticated_save_output_encoder(
                save_output=save_output, runtime_receipt=runtime
            )
            with self.assertRaisesRegex(
                level_b.LevelBRendererError, "did not select"
            ):
                encoder("frames", "/tmp/not-written.mp4", fps=25)
        self.assertEqual(calls, [])

    def test_authenticated_encoder_timeout_restores_signal_and_environment(self):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            self.skipTest("local ffmpeg executable unavailable")
        ffmpeg_path = Path(ffmpeg).resolve()
        runtime = {
            "ffmpeg": {
                "path": str(ffmpeg_path),
                "sha256": self.file_sha(ffmpeg_path),
            }
        }
        selected = ModuleType("imageio_ffmpeg")
        selected.get_ffmpeg_exe = lambda: os.environ["IMAGEIO_FFMPEG_EXE"]
        handler_before = level_b.signal.getsignal(level_b.signal.SIGALRM)

        def save_output(*_args, **_kwargs):
            level_b.signal.raise_signal(level_b.signal.SIGALRM)

        with mock.patch.dict(sys.modules, {"imageio_ffmpeg": selected}):
            encoder, _ = level_b._make_authenticated_save_output_encoder(
                save_output=save_output, runtime_receipt=runtime
            )
            with self.assertRaisesRegex(
                level_b.LevelBRendererError, "encode exceeded wall timeout"
            ):
                encoder("frames", "/tmp/not-written.mp4", fps=25)
        self.assertEqual(
            level_b.signal.getsignal(level_b.signal.SIGALRM), handler_before
        )
        self.assertEqual(
            level_b.signal.getitimer(level_b.signal.ITIMER_REAL), (0.0, 0.0)
        )

    def test_encoder_signal_inspection_failures_restore_ambient_environment(self):
        ffmpeg = shutil.which("ffmpeg")
        alternate = shutil.which("false")
        if ffmpeg is None or alternate is None:
            self.skipTest("local ffmpeg/false executable unavailable")
        ffmpeg_path = Path(ffmpeg).resolve()
        alternate_path = Path(alternate).resolve()
        runtime = {
            "ffmpeg": {
                "path": str(ffmpeg_path),
                "sha256": self.file_sha(ffmpeg_path),
            }
        }
        selected = ModuleType("imageio_ffmpeg")
        selected.get_ffmpeg_exe = lambda: os.environ["IMAGEIO_FFMPEG_EXE"]

        for api in ("getsignal", "getitimer"):
            with self.subTest(api=api), mock.patch.dict(
                sys.modules, {"imageio_ffmpeg": selected}
            ), mock.patch.dict(
                os.environ,
                {"IMAGEIO_FFMPEG_EXE": str(alternate_path)},
                clear=False,
            ):
                encoder, _ = level_b._make_authenticated_save_output_encoder(
                    save_output=lambda *_args, **_kwargs: self.fail(
                        "save_output must not run after signal inspection failure"
                    ),
                    runtime_receipt=runtime,
                )
                with mock.patch.object(
                    level_b.signal,
                    api,
                    side_effect=RuntimeError(f"injected {api} failure"),
                ):
                    with self.assertRaisesRegex(RuntimeError, api):
                        encoder("frames", "/tmp/not-written.mp4", fps=25)
                self.assertEqual(
                    os.environ.get("IMAGEIO_FFMPEG_EXE"), str(alternate_path)
                )

    def test_nonmain_encoder_path_never_calls_signal_api_and_restores_env(self):
        ffmpeg = shutil.which("ffmpeg")
        alternate = shutil.which("false")
        if ffmpeg is None or alternate is None:
            self.skipTest("local ffmpeg/false executable unavailable")
        ffmpeg_path = Path(ffmpeg).resolve()
        alternate_path = Path(alternate).resolve()
        runtime = {
            "ffmpeg": {
                "path": str(ffmpeg_path),
                "sha256": self.file_sha(ffmpeg_path),
            }
        }
        selected = ModuleType("imageio_ffmpeg")
        selected.get_ffmpeg_exe = lambda: os.environ["IMAGEIO_FFMPEG_EXE"]
        observations = []
        results = []
        errors = []

        def save_output(frames, path, *, fps):
            observations.append(
                (frames, path, fps, os.environ.get("IMAGEIO_FFMPEG_EXE"))
            )

        with mock.patch.dict(
            sys.modules, {"imageio_ffmpeg": selected}
        ), mock.patch.dict(
            os.environ,
            {"IMAGEIO_FFMPEG_EXE": str(alternate_path)},
            clear=False,
        ), mock.patch.object(
            level_b.signal,
            "getsignal",
            side_effect=AssertionError("nonmain getsignal forbidden"),
        ) as getsignal, mock.patch.object(
            level_b.signal,
            "getitimer",
            side_effect=AssertionError("nonmain getitimer forbidden"),
        ) as getitimer, mock.patch.object(
            level_b.signal,
            "signal",
            side_effect=AssertionError("nonmain signal forbidden"),
        ) as install_signal, mock.patch.object(
            level_b.signal,
            "setitimer",
            side_effect=AssertionError("nonmain setitimer forbidden"),
        ) as setitimer:
            encoder, _ = level_b._make_authenticated_save_output_encoder(
                save_output=save_output, runtime_receipt=runtime
            )

            def run_encoder():
                try:
                    results.append(
                        encoder("frames", "/tmp/not-written.mp4", fps=25)
                    )
                except BaseException as error:
                    errors.append(error)

            worker = threading.Thread(target=run_encoder)
            worker.start()
            worker.join(timeout=5.0)
            self.assertFalse(worker.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(
                os.environ.get("IMAGEIO_FFMPEG_EXE"), str(alternate_path)
            )
            getsignal.assert_not_called()
            getitimer.assert_not_called()
            install_signal.assert_not_called()
            setitimer.assert_not_called()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["encoder_timeout_mode"], "no-signal-nonmain-thread")
        self.assertTrue(results[0]["nonmain_thread_signal_api_avoided"])
        self.assertEqual(
            observations,
            [("frames", "/tmp/not-written.mp4", 25, str(ffmpeg_path))],
        )

    @staticmethod
    def _install_torch_no_grad_method_fixture(directory, suffix):
        try:
            import torch.autograd.grad_mode as grad_mode
            import torch.utils._contextlib as torch_context
        except ImportError as error:
            raise unittest.SkipTest("local Torch contextlib fixture unavailable") from error
        expected_lines = (
            torch_context.context_decorator.__code__.co_firstlineno,
            torch_context._DecoratorContextManager.clone.__code__.co_firstlineno,
            grad_mode.no_grad.__init__.__code__.co_firstlineno,
            grad_mode.no_grad.__enter__.__code__.co_firstlineno,
            grad_mode.no_grad.__exit__.__code__.co_firstlineno,
        )
        if expected_lines != (70, 146, 75, 80, 84):
            raise unittest.SkipTest(
                f"local Torch context source does not match pinned line ABI: {expected_lines}"
            )
        module_name = f"level_b_no_grad_definition_{suffix}"
        definition_path = Path(directory).resolve() / f"{module_name}.py"
        definition_path.write_text(
            "import torch\n"
            "class BerniniRendererModel:\n"
            "    @torch.no_grad()\n"
            "    def sample(self, value=None):\n"
            "        return value\n",
            encoding="utf-8",
        )
        prior = sys.modules.get(module_name)
        definition = load(module_name, definition_path)
        descriptor = vars(definition.BerniniRendererModel)["sample"]
        self_original = descriptor.__wrapped__
        if self_original.__code__.co_firstlineno != 3:
            raise AssertionError("fixture original line drifted")
        kwargs = {
            "label": "fixture renderer sample",
            "expected_instance": definition.BerniniRendererModel(),
            "expected_class": definition.BerniniRendererModel,
            "method_name": "sample",
            "expected_definition_module": module_name,
            "expected_definition_class_qualname": "BerniniRendererModel",
            "expected_definition_qualname": "BerniniRendererModel.sample",
            "expected_definition_path": definition_path,
            "expected_definition_sha256": hashlib.sha256(
                definition_path.read_bytes()
            ).hexdigest(),
            "expected_original_firstlineno": 3,
            "expected_original_parameter_names": ("self", "value"),
            "expected_original_defaults": (None,),
            "expected_original_annotations": {},
            "expected_contextlib_path": Path(torch_context.__file__).resolve(),
            "expected_contextlib_sha256": hashlib.sha256(
                Path(torch_context.__file__).read_bytes()
            ).hexdigest(),
            "expected_grad_mode_path": Path(grad_mode.__file__).resolve(),
            "expected_grad_mode_sha256": hashlib.sha256(
                Path(grad_mode.__file__).read_bytes()
            ).hexdigest(),
            "torch_module": torch,
            "expected_torch_version": str(torch.__version__),
        }
        return definition, kwargs, module_name, prior, torch_context, grad_mode

    @staticmethod
    def _restore_torch_no_grad_method_fixture(module_name, prior):
        sys.modules.pop(module_name, None)
        if prior is not None:
            sys.modules[module_name] = prior

    def test_torch_no_grad_real_two_layer_method_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            definition, kwargs, name, prior, _, _ = (
                self._install_torch_no_grad_method_fixture(directory, "positive")
            )
            try:
                instance = kwargs["expected_instance"]
                receipt = level_b._audit_torch_no_grad_wrapped_instance_method(
                    instance.sample, **kwargs
                )
            finally:
                self._restore_torch_no_grad_method_fixture(name, prior)
        self.assertTrue(receipt["exact_bound_instance_identity_verified"])
        self.assertEqual(receipt["wrapper"]["code_firstlineno"], 113)
        self.assertEqual(
            receipt["wrapper"]["decorator_factory_nested_code_identity_count"], 1
        )
        self.assertTrue(
            receipt["no_grad_context"]["exact_context_owner_type_verified"]
        )

    def test_torch_no_grad_substitute_bound_owner_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            definition, kwargs, name, prior, _, _ = (
                self._install_torch_no_grad_method_fixture(directory, "owner")
            )
            try:
                class Substitute(definition.BerniniRendererModel):
                    pass

                kwargs["expected_instance"] = Substitute()
                with self.assertRaisesRegex(
                    level_b.LevelBRendererError, "torch no-grad authority differs"
                ):
                    level_b._audit_torch_no_grad_wrapped_instance_method(
                        kwargs["expected_instance"].sample, **kwargs
                    )
            finally:
                self._restore_torch_no_grad_method_fixture(name, prior)

    def test_torch_no_grad_hostile_wrapper_code_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            definition, kwargs, name, prior, _, _ = (
                self._install_torch_no_grad_method_fixture(directory, "wrapper")
            )
            original_descriptor = vars(definition.BerniniRendererModel)["sample"]
            original = original_descriptor.__wrapped__
            try:
                @functools.wraps(original)
                def hostile_wrapper(*args, **kwargs):
                    return original(*args, **kwargs)

                definition.BerniniRendererModel.sample = hostile_wrapper
                kwargs["expected_instance"] = definition.BerniniRendererModel()
                with self.assertRaisesRegex(
                    level_b.LevelBRendererError,
                    "wrapper/original executable ownership differs",
                ):
                    level_b._audit_torch_no_grad_wrapped_instance_method(
                        kwargs["expected_instance"].sample, **kwargs
                    )
            finally:
                definition.BerniniRendererModel.sample = original_descriptor
                self._restore_torch_no_grad_method_fixture(name, prior)

    def test_torch_no_grad_hostile_closure_original_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            definition, kwargs, name, prior, _, _ = (
                self._install_torch_no_grad_method_fixture(directory, "closure")
            )
            original_descriptor = vars(definition.BerniniRendererModel)["sample"]
            try:
                def hostile_original(self, value=None):
                    return "hostile", value

                hostile_original.__module__ = definition.__name__
                hostile_original.__qualname__ = "BerniniRendererModel.sample"
                definition.BerniniRendererModel.sample = torch.no_grad()(
                    hostile_original
                )
                kwargs["expected_instance"] = definition.BerniniRendererModel()
                with self.assertRaisesRegex(
                    level_b.LevelBRendererError,
                    "wrapper/original executable ownership differs",
                ):
                    level_b._audit_torch_no_grad_wrapped_instance_method(
                        kwargs["expected_instance"].sample, **kwargs
                    )
            finally:
                definition.BerniniRendererModel.sample = original_descriptor
                self._restore_torch_no_grad_method_fixture(name, prior)

    def test_torch_no_grad_hostile_context_factory_closure_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            definition, kwargs, name, prior, _, _ = (
                self._install_torch_no_grad_method_fixture(directory, "context")
            )
            wrapper = vars(definition.BerniniRendererModel)["sample"]
            freevars = tuple(wrapper.__code__.co_freevars)
            cell = wrapper.__closure__[freevars.index("ctx_factory")]
            original_factory = cell.cell_contents
            try:
                cell.cell_contents = lambda: torch.no_grad()
                with self.assertRaisesRegex(
                    level_b.LevelBRendererError, "context-factory owner differs"
                ):
                    level_b._audit_torch_no_grad_wrapped_instance_method(
                        kwargs["expected_instance"].sample, **kwargs
                    )
            finally:
                cell.cell_contents = original_factory
                self._restore_torch_no_grad_method_fixture(name, prior)

    def test_torch_no_grad_hostile_lifecycle_method_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            definition, kwargs, name, prior, _, grad_mode = (
                self._install_torch_no_grad_method_fixture(directory, "lifecycle")
            )
            original_enter = grad_mode.no_grad.__enter__
            try:
                def hostile_enter(self):
                    return None

                grad_mode.no_grad.__enter__ = hostile_enter
                with self.assertRaises(level_b.LevelBRendererError):
                    level_b._audit_torch_no_grad_wrapped_instance_method(
                        kwargs["expected_instance"].sample, **kwargs
                    )
            finally:
                grad_mode.no_grad.__enter__ = original_enter
                self._restore_torch_no_grad_method_fixture(name, prior)

    @staticmethod
    def _install_loaded_wrapped_vae_fixture(directory):
        definition_name = "diffusers.models.autoencoders.autoencoder_kl_wan"
        wrapper_name = "diffusers.utils.accelerate_utils"
        output_name = "diffusers.models.modeling_outputs"
        vae_name = "diffusers.models.autoencoders.vae"
        root = Path(directory).resolve()
        definition_path = root / "autoencoder_kl_wan.py"
        wrapper_path = root / "accelerate_utils.py"
        output_path = root / "modeling_outputs.py"
        vae_path = root / "vae.py"

        wrapper_lines = ["# exact fixture padding"] * 26
        wrapper_lines.extend(
            [
                "def apply_forward_hook(method):",  # line 27
                *["    # exact factory padding"] * 15,
                "    def wrapper(self, *args, **kwargs):",  # line 43
                "        return method(self, *args, **kwargs)",
                "    return wrapper",
            ]
        )
        wrapper_path.write_text("\n".join(wrapper_lines) + "\n", encoding="utf-8")

        output_path.write_text(
            "class AutoencoderKLOutput:\n"
            "    pass\n",
            encoding="utf-8",
        )
        vae_path.write_text(
            "class DecoderOutput:\n"
            "    pass\n"
            "class DiagonalGaussianDistribution:\n"
            "    pass\n",
            encoding="utf-8",
        )

        definition_lines = [
            f"from {wrapper_name} import apply_forward_hook",
            "import torch",
            f"from {output_name} import AutoencoderKLOutput",
            f"from {vae_name} import DecoderOutput, DiagonalGaussianDistribution",
            "class AutoencoderKLWan:",
        ]
        definition_lines.extend([""] * (1159 - len(definition_lines)))
        definition_lines.extend(
            [
                "    @apply_forward_hook",  # line 1160
                "    def encode(self, x: torch.Tensor, return_dict: bool=True) -> AutoencoderKLOutput | tuple[DiagonalGaussianDistribution]:",
                "        return x, return_dict",
            ]
        )
        definition_lines.extend([""] * (1217 - len(definition_lines)))
        definition_lines.extend(
            [
                "    @apply_forward_hook",  # line 1218
                "    def decode(self, z: torch.Tensor, return_dict: bool=True) -> DecoderOutput | torch.Tensor:",
                "        return z, return_dict",
            ]
        )
        definition_path.write_text(
            "\n".join(definition_lines) + "\n", encoding="utf-8"
        )

        prior = {
            wrapper_name: sys.modules.get(wrapper_name),
            definition_name: sys.modules.get(definition_name),
            output_name: sys.modules.get(output_name),
            vae_name: sys.modules.get(vae_name),
        }
        wrapper = load(wrapper_name, wrapper_path)
        load(output_name, output_path)
        load(vae_name, vae_path)
        definition = load(definition_name, definition_path)
        definition_sha, definition_identity = level_b.stable_file_sha256(
            definition_path, label="test wrapped VAE definition"
        )
        wrapper_sha, wrapper_identity = level_b.stable_file_sha256(
            wrapper_path, label="test wrapped VAE decorator"
        )
        output_sha, output_identity = level_b.stable_file_sha256(
            output_path, label="test wrapped VAE output annotations"
        )
        vae_sha, vae_identity = level_b.stable_file_sha256(
            vae_path, label="test wrapped VAE annotation classes"
        )
        runtime = {
            "vendor_source_files": {
                "site-packages:diffusers/models/autoencoders/autoencoder_kl_wan.py": {
                    "path": str(definition_path),
                    "sha256": definition_sha,
                    "file_identity": definition_identity,
                },
                "site-packages:diffusers/utils/accelerate_utils.py": {
                    "path": str(wrapper_path),
                    "sha256": wrapper_sha,
                    "file_identity": wrapper_identity,
                },
                "site-packages:diffusers/models/modeling_outputs.py": {
                    "path": str(output_path),
                    "sha256": output_sha,
                    "file_identity": output_identity,
                },
                "site-packages:diffusers/models/autoencoders/vae.py": {
                    "path": str(vae_path),
                    "sha256": vae_sha,
                    "file_identity": vae_identity,
                },
            }
        }
        return wrapper, definition, runtime, prior

    @staticmethod
    def _restore_loaded_wrapped_vae_fixture(prior):
        for name, module in prior.items():
            sys.modules.pop(name, None)
            if module is not None:
                sys.modules[name] = module

    @unittest.skipUnless(
        sys.version_info >= (3, 10), "PEP604 runtime annotation fixture requires 3.10+"
    )
    def test_loaded_vae_real_apply_forward_hook_layers_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            _, module, runtime, prior = self._install_loaded_wrapped_vae_fixture(
                directory
            )
            try:
                receipt = level_b._audit_loaded_vae_callables(
                    module.AutoencoderKLWan(),
                    expected_vae_class=module.AutoencoderKLWan,
                    runtime_receipt=runtime,
                )
            finally:
                self._restore_loaded_wrapped_vae_fixture(prior)
        self.assertTrue(receipt["exact_loaded_class_identity_verified"])
        for name in ("encode", "decode"):
            row = receipt["bound_callables"][name]
            self.assertTrue(row["exact_bound_instance_identity_verified"])
            self.assertTrue(row["exact_class_descriptor_identity_verified"])
            self.assertEqual(row["wrapper"]["closure_freevars"], ["method"])
            self.assertEqual(
                row["wrapper"]["decorator_factory_nested_code_identity_count"],
                1,
            )
            self.assertTrue(row["closure_original"]["exact_closure_identity_verified"])

    @unittest.skipUnless(
        sys.version_info >= (3, 10), "PEP604 runtime annotation fixture requires 3.10+"
    )
    def test_loaded_vae_substitute_class_and_instance_method_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, module, runtime, prior = self._install_loaded_wrapped_vae_fixture(
                directory
            )
            try:
                class Substitute(module.AutoencoderKLWan):
                    pass

                with self.assertRaisesRegex(
                    level_b.LevelBRendererError, "exact class"
                ):
                    level_b._audit_loaded_vae_callables(
                        Substitute(),
                        expected_vae_class=module.AutoencoderKLWan,
                        runtime_receipt=runtime,
                    )
                vae = module.AutoencoderKLWan()
                vae.encode = lambda value: value
                with self.assertRaisesRegex(
                    level_b.LevelBRendererError, "exact bound method"
                ):
                    level_b._audit_loaded_vae_callables(
                        vae,
                        expected_vae_class=module.AutoencoderKLWan,
                        runtime_receipt=runtime,
                    )
            finally:
                self._restore_loaded_wrapped_vae_fixture(prior)

    @unittest.skipUnless(
        sys.version_info >= (3, 10), "PEP604 runtime annotation fixture requires 3.10+"
    )
    def test_loaded_vae_hostile_class_wrapper_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, module, runtime, prior = self._install_loaded_wrapped_vae_fixture(
                directory
            )
            original_descriptor = vars(module.AutoencoderKLWan)["encode"]
            try:
                def hostile_wrapper(self, *args, **kwargs):
                    return args, kwargs

                hostile_wrapper.__module__ = "diffusers.utils.accelerate_utils"
                hostile_wrapper.__qualname__ = "apply_forward_hook.<locals>.wrapper"
                module.AutoencoderKLWan.encode = hostile_wrapper
                with self.assertRaisesRegex(
                    level_b.LevelBRendererError, "wrapper executable ownership differs"
                ):
                    level_b._audit_loaded_vae_callables(
                        module.AutoencoderKLWan(),
                        expected_vae_class=module.AutoencoderKLWan,
                        runtime_receipt=runtime,
                    )
            finally:
                module.AutoencoderKLWan.encode = original_descriptor
                self._restore_loaded_wrapped_vae_fixture(prior)

    @unittest.skipUnless(
        sys.version_info >= (3, 10), "PEP604 runtime annotation fixture requires 3.10+"
    )
    def test_loaded_vae_hostile_wrapper_closure_original_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            wrapper, module, runtime, prior = (
                self._install_loaded_wrapped_vae_fixture(directory)
            )
            original_descriptor = vars(module.AutoencoderKLWan)["encode"]
            try:
                def hostile_original(self, x, return_dict=True):
                    return "hostile", x, return_dict

                hostile_original.__module__ = module.__name__
                hostile_original.__qualname__ = "AutoencoderKLWan.encode"
                module.AutoencoderKLWan.encode = wrapper.apply_forward_hook(
                    hostile_original
                )
                with self.assertRaisesRegex(
                    level_b.LevelBRendererError, "closure-original ownership differs"
                ):
                    level_b._audit_loaded_vae_callables(
                        module.AutoencoderKLWan(),
                        expected_vae_class=module.AutoencoderKLWan,
                        runtime_receipt=runtime,
                    )
            finally:
                module.AutoencoderKLWan.encode = original_descriptor
                self._restore_loaded_wrapped_vae_fixture(prior)

    @unittest.skipUnless(
        sys.version_info >= (3, 10), "PEP604 runtime annotation fixture requires 3.10+"
    )
    def test_loaded_vae_empty_or_substituted_annotations_are_rejected(self):
        for mutation in ("empty", "substituted-return"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                _, module, runtime, prior = self._install_loaded_wrapped_vae_fixture(
                    directory
                )
                wrapper = vars(module.AutoencoderKLWan)["encode"]
                original = wrapper.__closure__[0].cell_contents
                annotations = dict(original.__annotations__)
                try:
                    if mutation == "empty":
                        original.__annotations__ = {}
                    else:
                        original.__annotations__["return"] = int
                    with self.assertRaisesRegex(
                        level_b.LevelBRendererError,
                        "signature/annotations differ",
                    ):
                        level_b._audit_loaded_vae_callables(
                            module.AutoencoderKLWan(),
                            expected_vae_class=module.AutoencoderKLWan,
                            runtime_receipt=runtime,
                        )
                finally:
                    original.__annotations__ = annotations
                    self._restore_loaded_wrapped_vae_fixture(prior)

    @unittest.skipUnless(
        sys.version_info >= (3, 10), "PEP604 runtime annotation fixture requires 3.10+"
    )
    def test_loaded_vae_union_order_container_and_class_symbol_tamper_are_rejected(self):
        for mutation in ("order", "container", "class-symbol"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                _, module, runtime, prior = self._install_loaded_wrapped_vae_fixture(
                    directory
                )
                wrapper = vars(module.AutoencoderKLWan)["encode"]
                original = wrapper.__closure__[0].cell_contents
                annotations = dict(original.__annotations__)
                output_module = sys.modules["diffusers.models.modeling_outputs"]
                vae_module = sys.modules["diffusers.models.autoencoders.vae"]
                exported = output_module.AutoencoderKLOutput
                try:
                    if mutation == "order":
                        original.__annotations__["return"] = (
                            tuple[vae_module.DiagonalGaussianDistribution]
                            | output_module.AutoencoderKLOutput
                        )
                    elif mutation == "container":
                        original.__annotations__["return"] = (
                            output_module.AutoencoderKLOutput
                            | list[vae_module.DiagonalGaussianDistribution]
                        )
                    else:
                        class SubstituteOutput:
                            pass

                        output_module.AutoencoderKLOutput = SubstituteOutput
                    with self.assertRaises(level_b.LevelBRendererError):
                        level_b._audit_loaded_vae_callables(
                            module.AutoencoderKLWan(),
                            expected_vae_class=module.AutoencoderKLWan,
                            runtime_receipt=runtime,
                        )
                finally:
                    original.__annotations__ = annotations
                    output_module.AutoencoderKLOutput = exported
                    self._restore_loaded_wrapped_vae_fixture(prior)

    def test_loaded_tokenizer_requires_exact_t5_class(self):
        base_name = "transformers.tokenization_utils_base"
        t5_name = "transformers.models.t5.tokenization_t5"
        prior = {name: sys.modules.get(name) for name in (base_name, t5_name)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            base_path = root / "tokenization_utils_base.py"
            t5_path = root / "tokenization_t5.py"
            base_path.write_text(
                "class PreTrainedTokenizerBase:\n"
                "    def __call__(self, value):\n"
                "        return value\n",
                encoding="utf-8",
            )
            t5_path.write_text(
                "class T5Tokenizer(PreTrainedTokenizerBase):\n"
                "    pass\n",
                encoding="utf-8",
            )
            base_module = load(base_name, base_path)
            t5_module = ModuleType(t5_name)
            t5_module.__file__ = str(t5_path)
            t5_module.__package__ = t5_name.rpartition(".")[0]
            t5_module.PreTrainedTokenizerBase = base_module.PreTrainedTokenizerBase
            sys.modules[t5_name] = t5_module
            exec(
                compile(t5_path.read_bytes(), str(t5_path), "exec"),
                t5_module.__dict__,
            )
            base_sha, base_identity = level_b.stable_file_sha256(
                base_path, label="test tokenizer base source"
            )
            t5_sha, t5_identity = level_b.stable_file_sha256(
                t5_path, label="test tokenizer T5 source"
            )
            runtime = {
                "vendor_source_files": {
                    "site-packages:transformers/tokenization_utils_base.py": {
                        "path": str(base_path),
                        "sha256": base_sha,
                        "file_identity": base_identity,
                    },
                    "site-packages:transformers/models/t5/tokenization_t5.py": {
                        "path": str(t5_path),
                        "sha256": t5_sha,
                        "file_identity": t5_identity,
                    },
                }
            }
            try:
                tokenizer = t5_module.T5Tokenizer()
                receipt = level_b._audit_loaded_tokenizer_callable(
                    tokenizer,
                    expected_tokenizer_class=t5_module.T5Tokenizer,
                    runtime_receipt=runtime,
                )
                self.assertTrue(receipt["exact_loaded_class_identity_verified"])

                class Substitute(t5_module.T5Tokenizer):
                    pass

                with self.assertRaisesRegex(
                    level_b.LevelBRendererError, "authority is absent"
                ):
                    level_b._audit_loaded_tokenizer_callable(
                        Substitute(),
                        expected_tokenizer_class=t5_module.T5Tokenizer,
                        runtime_receipt=runtime,
                    )
            finally:
                for name in (base_name, t5_name):
                    sys.modules.pop(name, None)
                    if prior[name] is not None:
                        sys.modules[name] = prior[name]

    def test_live_renderer_signature_binds_exact_source_only_kwargs(self):
        receipt = level_b.audit_official_renderer_sample_call(
            renderer=self.official_renderer(),
            sample_kwargs=self.official_sample_kwargs(),
        )
        self.assertTrue(receipt["live_bound_method"])
        self.assertEqual(
            receipt["exact_keyword_names_in_call_order"],
            list(level_b.OFFICIAL_RENDERER_SAMPLE_KEYWORDS),
        )
        self.assertEqual(
            receipt["bound_runtime_parameter_names"],
            list(level_b.OFFICIAL_RENDERER_SAMPLE_KEYWORDS),
        )
        self.assertFalse(receipt["caller_callback_or_custom_denoiser_present"])

    def test_live_renderer_signature_rejects_hidden_callable(self):
        kwargs = self.official_sample_kwargs()
        kwargs["multi_video_vae_latents"] = [lambda: None]
        with self.assertRaisesRegex(level_b.LevelBRendererError, "hidden callable"):
            level_b.audit_official_renderer_sample_call(
                renderer=self.official_renderer(), sample_kwargs=kwargs
            )

    def test_live_renderer_signature_rejects_target_kwarg(self):
        kwargs = self.official_sample_kwargs()
        kwargs["target_video"] = object()
        with self.assertRaisesRegex(
            level_b.LevelBRendererError, "keyword set/order differs"
        ):
            level_b.audit_official_renderer_sample_call(
                renderer=self.official_renderer(), sample_kwargs=kwargs
            )

    def test_live_renderer_signature_rejects_varkw_substitute(self):
        class Renderer:
            def sample(self, **kwargs):
                raise AssertionError("signature audit must not call the renderer")

        with self.assertRaisesRegex(
            level_b.LevelBRendererError, "does not explicitly declare"
        ):
            level_b.audit_official_renderer_sample_call(
                renderer=Renderer(), sample_kwargs=self.official_sample_kwargs()
            )

    def test_public_signature_has_no_teacher_or_callback_escape(self):
        receipt = level_b.validate_public_product_signature()
        self.assertFalse(receipt["target_anchor_teacher_callback_accepted"])
        names = receipt["parameters"]
        for fragment in level_b.FORBIDDEN_PUBLIC_ARGUMENT_FRAGMENTS:
            self.assertFalse(any(fragment in name.lower() for name in names))
        self.assertEqual(
            names,
            [
                "fresh_bundle",
                "verified_runtime",
                "source_video_path",
                "expected_source_video_sha256",
                "edit_instruction",
                "inference_seed",
                "output_mp4_path",
            ],
        )
        self.assertNotIn("runtime_assets", names)
        self.assertNotIn("validate_full_mp4_bytes", level_b.__dict__)
        self.assertNotIn(
            "_validate_full_mp4_bytes_with_authenticated_tools", level_b.__all__
        )
        self.assertNotIn("native_action_renderer_bridge", level_b.__all__)
        self.assertNotIn("InstalledNativeActionRendererBridge", level_b.__all__)

    def test_ascii_pseudo_mp4_is_rejected(self):
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        if ffmpeg is None or ffprobe is None:
            self.skipTest("ffmpeg/ffprobe unavailable")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fake.mp4"
            path.write_bytes(b"not a video" * 64)
            with self.assertRaises(level_b.LevelBRendererError):
                level_b._validate_full_mp4_bytes_with_authenticated_tools(
                    mp4_path=path.resolve(),
                    expected_height=16,
                    expected_width=16,
                    ffmpeg_path=Path(ffmpeg).resolve(),
                    ffprobe_path=Path(ffprobe).resolve(),
                    expected_ffmpeg_sha256=self.file_sha(Path(ffmpeg).resolve()),
                    expected_ffprobe_sha256=self.file_sha(Path(ffprobe).resolve()),
                )

    def test_real_exact81_mp4_is_ffprobed_and_fully_decoded(self):
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        if ffmpeg is None or ffprobe is None:
            self.skipTest("ffmpeg/ffprobe unavailable")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "real.mp4"
            command = [
                ffmpeg,
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=16x16:rate=25",
                "-frames:v",
                "81",
                "-pix_fmt",
                "yuv420p",
                "-y",
                str(path),
            ]
            completed = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            if completed.returncode != 0:
                self.skipTest(f"local ffmpeg encoder unavailable: {completed.stderr!r}")
            receipt = level_b._validate_full_mp4_bytes_with_authenticated_tools(
                mp4_path=path.resolve(),
                expected_height=16,
                expected_width=16,
                ffmpeg_path=Path(ffmpeg).resolve(),
                ffprobe_path=Path(ffprobe).resolve(),
                expected_ffmpeg_sha256=self.file_sha(Path(ffmpeg).resolve()),
                expected_ffprobe_sha256=self.file_sha(Path(ffprobe).resolve()),
            )
            self.assertTrue(receipt["complete_decode_verified"])
            self.assertEqual(receipt["full_decode_frame_count"], 81)
            self.assertEqual(receipt["full_decode_rgb_byte_count"], 81 * 16 * 16 * 3)
            self.assertTrue(receipt["show_frames_exact_pts_n_over_25"])
            self.assertTrue(receipt["show_packets_exact_pts_n_over_25"])
            self.assertEqual(receipt["ffprobe_codec_name"], "h264")
            self.assertEqual(receipt["ffprobe_pixel_format"], "yuv420p")
            self.assertIn(
                receipt["show_frames_packet_duration_field_mode"],
                ("all-present-and-exact", "all-absent-packet-authoritative"),
            )

    def test_real_exact81_vfr_mp4_is_rejected_by_pts_contract(self):
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        if ffmpeg is None or ffprobe is None:
            self.skipTest("ffmpeg/ffprobe unavailable")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vfr.mp4"
            command = [
                ffmpeg,
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=16x16:rate=25",
                "-vf",
                r"setpts=if(lt(N\,40)\,N\,40+(N-40)*2)",
                "-frames:v",
                "81",
                "-vsync",
                "vfr",
                "-pix_fmt",
                "yuv420p",
                "-y",
                str(path),
            ]
            completed = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            if completed.returncode != 0:
                self.skipTest(f"local VFR encoder unavailable: {completed.stderr!r}")
            with self.assertRaisesRegex(
                level_b.LevelBRendererError, "CFR|duration|PTS"
            ):
                level_b._validate_full_mp4_bytes_with_authenticated_tools(
                    mp4_path=path.resolve(),
                    expected_height=16,
                    expected_width=16,
                    ffmpeg_path=Path(ffmpeg).resolve(),
                    ffprobe_path=Path(ffprobe).resolve(),
                    expected_ffmpeg_sha256=self.file_sha(Path(ffmpeg).resolve()),
                    expected_ffprobe_sha256=self.file_sha(Path(ffprobe).resolve()),
                )


if __name__ == "__main__":
    unittest.main()
