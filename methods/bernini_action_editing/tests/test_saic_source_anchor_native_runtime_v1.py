#!/usr/bin/env python3

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch
    from torch import nn

    import inference_sigma_strata as strata
    import saic_source_anchor_adapter_v1 as anchor
    import saic_source_anchor_native_runtime_v1 as runtime
    import source_self_native_ref_contrastive_v3 as native

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    strata = None  # type: ignore[assignment]
    anchor = None  # type: ignore[assignment]
    runtime = None  # type: ignore[assignment]
    native = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:
    class UniPCMultistepScheduler:
        def __init__(self) -> None:
            self.config = {
                "_class_name": "UniPCMultistepScheduler",
                "num_train_timesteps": 1000,
                "flow_shift": 5.0,
                "prediction_type": "flow_prediction",
                "predict_x0": True,
                "use_flow_sigmas": True,
                "thresholding": False,
                "solver_order": 2,
                "solver_type": "bh2",
                "final_sigmas_type": "zero",
            }
            self.timesteps = torch.tensor(strata.PINNED_TIMESTEPS, dtype=torch.int64)
            self.sigmas = torch.tensor(
                (*strata.PINNED_POSITIVE_SIGMAS, 0.0), dtype=torch.float32
            )
            self.step_index = 0
            self.original_calls = 0

        def set_timesteps(self, steps: int) -> None:
            if steps != 40:
                raise ValueError("fake scheduler is exact40")
            self.step_index = 0

        def step(self, model_output, timestep, sample, return_dict=False):
            del model_output, timestep, return_dict
            self.original_calls += 1
            self.step_index += 1
            return (sample,)


    def _handle_receipt() -> dict[str, object]:
        return {
            "schema_version": anchor.SCHEMA_VERSION,
            "blocks": list(anchor.SOURCE_ANCHOR_BLOCK_INDICES),
            "projections": ["attn1.to_q", "attn1.to_out.0"],
            "rank": anchor.SOURCE_ANCHOR_RANK,
            "full_source_native_branches": list(anchor.FULL_SOURCE_BRANCHES),
            "active_sigma_indices": list(anchor.ACTIVE_SIGMA_INDICES),
            "exact40_schedule_sha256": strata.SCHEDULE_SHA256,
            "source_reference_padding_rows_exact_base": True,
            "prompt_role_agnostic_action_and_noop": True,
            "route_accepts_caller_rank_size_index_or_mask": False,
            "route_binds_live_parallel_native_mask_and_actual_scheduler_sigma": True,
            "accepted_timestep_representations": [
                "official_device_local_int64",
                "manual_device_local_float32",
            ],
            "only_registered_self_attention_qo_replaced": True,
            "base_parameters_frozen": True,
            "digest": "a" * 64,
        }


    def _fake_handle(transformer, route_log):
        handle = anchor.SAICSourceAnchorHandle.__new__(anchor.SAICSourceAnchorHandle)
        handle.transformer = transformer
        handle.restored = False
        handle.receipt = lambda: _handle_receipt()

        @contextmanager
        def route(*, branch, scheduler, timestep):
            route_object = SimpleNamespace(
                branch=branch,
                scheduler=scheduler,
                timestep=timestep,
                adapter_active=float(timestep.item())
                in {float(strata.PINNED_TIMESTEPS[i]) for i in anchor.ACTIVE_SIGMA_INDICES},
            )
            token = anchor._ACTIVE_ROUTE.set(route_object)  # noqa: SLF001
            route_log.append(("enter", branch.name, int(timestep.item())))
            try:
                yield route_object
            finally:
                route_log.append(("exit", branch.name, int(timestep.item())))
                anchor._ACTIVE_ROUTE.reset(token)  # noqa: SLF001

        handle.route = route
        return handle


    class _FakeTransformer:
        def __init__(self) -> None:
            self.patch_embedding = nn.Conv3d(
                16, 1536, kernel_size=(1, 2, 2), bias=True
            )
            self.config = SimpleNamespace(
                num_attention_heads=12,
                attention_head_dim=128,
                in_channels=16,
                out_channels=16,
                patch_size=(1, 2, 2),
                text_dim=4096,
            )


    class _FakeDiffusion:
        use_unipc = True
        transformer_2 = None
        switch_dit_boundary = 0.0

        def __init__(
            self,
            *,
            branch_name: str,
            teacher_queries: bool,
            copied_teacher_suffix: bool = False,
        ) -> None:
            self.branch_name = branch_name
            self.teacher_queries = teacher_queries
            self.copied_teacher_suffix = copied_teacher_suffix
            self.transformer = _FakeTransformer()
            self.scheduler = UniPCMultistepScheduler()
            self.shared_route_log: list[object] = []
            self.original_sample_calls = 0
            self.original_shared_calls = 0

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
            del model_id, timesteps, cond_embeds, rotary_embs
            if batch_vae_seqlen != [noisy_latents.shape[1]] or batch_text_seqlen != [512]:
                raise RuntimeError("fake metadata differs")
            self.original_shared_calls += 1
            self.shared_route_log.append(anchor.active_route())
            return torch.zeros(
                1, noisy_latents.shape[1], 64, dtype=torch.float32
            )

        def sample(
            self,
            prompt_embeds=None,
            prompt_embeds_t2=None,
            uncond_prompt_embeds=None,
            uncond_embeds_t2=None,
            num_frames=81,
            width=16,
            height=16,
            image_vae_latents=None,
            multi_video_vae_latents=None,
            multi_image_vae_latents=None,
            num_inference_steps=40,
            guidance_mode="v2v_apg",
            omega_vid=1.25,
            omega_img=0.0,
            omega_txt=4.0,
            omega_scale=0.8,
            flow_shift=5.0,
            seed=7,
            device="cpu",
            eta=0.5,
            norm_threshold=(50.0, 50.0),
            momentum=0.0,
            mask=None,
            target_video=None,
        ):
            del (
                prompt_embeds_t2,
                uncond_embeds_t2,
                num_frames,
                width,
                height,
                image_vae_latents,
                multi_video_vae_latents,
                multi_image_vae_latents,
                guidance_mode,
                omega_vid,
                omega_img,
                omega_txt,
                omega_scale,
                flow_shift,
                seed,
                device,
                eta,
                norm_threshold,
                momentum,
                mask,
                target_video,
            )
            self.original_sample_calls += 1
            self.scheduler.set_timesteps(num_inference_steps)
            target_tokens = 21
            total_tokens = 42 if self.branch_name == "V" else 46
            scheduler_sample = torch.zeros(1, target_tokens, 64)
            auxiliary = torch.full((1, 512, 4096), 0.5, dtype=torch.bfloat16)
            for index in range(num_inference_steps):
                scalar_timestep = self.scheduler.timesteps[index]
                model_timestep = scalar_timestep.expand(1)
                full = torch.zeros(1, total_tokens, 1536)
                rotary = torch.zeros(1, 1, total_tokens, 4)

                def forward(noisy, rope, prompt):
                    return self.shared_step(
                        model_id="transformer_1",
                        noisy_latents=noisy,
                        timesteps=model_timestep,
                        cond_embeds=prompt,
                        rotary_embs=rope,
                        batch_vae_seqlen=[noisy.shape[1]],
                        batch_text_seqlen=[512],
                    )

                negative = forward(full, rotary, uncond_prompt_embeds)
                if self.teacher_queries:
                    teacher_hidden = full[:, -target_tokens:, :]
                    teacher_rotary = rotary[:, :, -target_tokens:, :]
                    if self.copied_teacher_suffix and index == 0:
                        teacher_hidden = teacher_hidden.clone()
                    forward(teacher_hidden, teacher_rotary, prompt_embeds)
                    forward(teacher_hidden, teacher_rotary, auxiliary)
                action = forward(full, rotary, prompt_embeds)
                if self.teacher_queries:
                    forward(
                        full[:, -target_tokens:, :],
                        rotary[:, :, -target_tokens:, :],
                        uncond_prompt_embeds,
                    )
                self.scheduler.step(
                    action[:, -target_tokens:, :],
                    scalar_timestep,
                    scheduler_sample,
                    return_dict=False,
                )
                del negative
            return torch.zeros(1, 16, 21, 2, 2, dtype=torch.float32)


    # Runtime class authentication mirrors the pinned vendor objects without
    # importing either large package in this CPU-only test.
    UniPCMultistepScheduler.__module__ = (
        "diffusers.schedulers.scheduling_unipc_multistep"
    )
    _FakeDiffusion.__module__ = "bernini.models.wan_diffusion"
    _FakeDiffusion.__name__ = "GEN_Wanx22"


    class _Attention(nn.Module):
        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.to_q = nn.Linear(hidden, hidden, bias=False)
            self.to_k = nn.Linear(hidden, hidden, bias=False)
            self.to_v = nn.Linear(hidden, hidden, bias=False)
            self.to_out = nn.ModuleList(
                [nn.Linear(hidden, hidden, bias=False), nn.Identity()]
            )


    class _Block(nn.Module):
        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.attn1 = _Attention(hidden)
            self.attn2 = _Attention(hidden)
            self.gradient_checkpointing = False


    class _SmallTransformer(nn.Module):
        def __init__(self, hidden: int = 4) -> None:
            super().__init__()
            self.patch_embedding = nn.Conv3d(16, hidden, kernel_size=(1, 2, 2))
            self.blocks = nn.ModuleList(
                [_Block(hidden) for _ in range(anchor.TOTAL_BLOCKS_1P3B)]
            )
            self.gradient_checkpointing = False
            self.is_gradient_checkpointing = False

        def patch_vae_latent(self, value, source_id):
            del source_id
            return value, value


    class _ParallelState:
        ulysses_rank = 0
        ulysses_size = 1


def _sample_kwargs(branch_name: str, action, negative, source, references):
    return {
        "prompt_embeds": action,
        "prompt_embeds_t2": None,
        "uncond_prompt_embeds": negative,
        "uncond_embeds_t2": None,
        "num_frames": 81,
        "width": 16,
        "height": 16,
        "image_vae_latents": None,
        "multi_video_vae_latents": [source],
        "multi_image_vae_latents": references if branch_name == "VI" else None,
        "num_inference_steps": 40,
        "guidance_mode": "v2v_apg",
        "omega_vid": 1.25,
        "omega_img": 4.5 if branch_name == "VI" else 0.0,
        "omega_txt": 4.0,
        "omega_scale": 0.8,
        "flow_shift": 5.0,
        "seed": 7,
        "device": "cpu",
        "eta": 0.5,
        "norm_threshold": (50.0, 50.0),
        "momentum": 0.0,
    }


@unittest.skipUnless(_TORCH_AVAILABLE, "torch runtime is required")
class SourceAnchorNativeRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.action = torch.full((1, 512, 4096), 0.25, dtype=torch.bfloat16)
        self.negative = torch.full((1, 512, 4096), -0.25, dtype=torch.bfloat16)
        self.source = torch.zeros(1, 16, 21, 2, 2)
        self.references = [torch.zeros(1, 16, 1, 2, 2) for _ in range(4)]

    def _run(self, branch_name: str, *, teacher_queries: bool):
        diffusion = _FakeDiffusion(
            branch_name=branch_name, teacher_queries=teacher_queries
        )
        route_log: list[object] = []
        handle = _fake_handle(diffusion.transformer, route_log)
        before = (
            diffusion.sample.__func__,
            diffusion.shared_step.__func__,
            diffusion.scheduler.step.__func__,
        )
        with runtime.saic_source_anchor_native_runtime(
            diffusion,
            handle=handle,
            config=runtime.SourceAnchorNativeRuntimeConfig(
                target_latent_shape=(1, 16, 21, 2, 2),
                branch_name=branch_name,
            ),
        ) as patch:
            result = diffusion.sample(
                **_sample_kwargs(
                    branch_name,
                    self.action,
                    self.negative,
                    self.source,
                    self.references,
                )
            )
        receipt = patch.finalize()
        after = (
            diffusion.sample.__func__,
            diffusion.shared_step.__func__,
            diffusion.scheduler.step.__func__,
        )
        self.assertEqual(before, after)
        return diffusion, receipt, result, route_log

    def test_normal_noop_v_exact40_routes_both_official_forwards(self) -> None:
        diffusion, receipt, result, route_log = self._run(
            "V", teacher_queries=False
        )
        self.assertEqual(tuple(result.shape), (1, 16, 21, 2, 2))
        self.assertEqual(diffusion.original_sample_calls, 1)
        self.assertEqual(diffusion.original_shared_calls, 80)
        self.assertEqual(diffusion.scheduler.original_calls, 40)
        self.assertEqual(len(route_log), 160)
        self.assertTrue(all(route is not None for route in diffusion.shared_route_log))
        self.assertEqual(receipt["branch_name"], "V")
        self.assertEqual(receipt["active_schedule_indices"], list(range(35, 40)))
        self.assertTrue(receipt["action_and_noop_share_anchor_route"])

    def test_vi_routes_negative_action_but_never_target_only_teachers(self) -> None:
        diffusion, receipt, _, _ = self._run("VI", teacher_queries=True)
        self.assertEqual(diffusion.original_shared_calls, 200)
        for index in range(40):
            cell = diffusion.shared_route_log[index * 5 : (index + 1) * 5]
            self.assertIsNotNone(cell[0])
            self.assertIsNone(cell[1])
            self.assertIsNone(cell[2])
            self.assertIsNotNone(cell[3])
            self.assertIsNone(cell[4])
        self.assertEqual(receipt["target_only_teacher_forwards_unrouted"], 120)
        self.assertTrue(receipt["target_only_teacher_has_anchor_route"] is False)
        self.assertEqual(receipt["condition_tokens"], 25)
        self.assertEqual(receipt["total_tokens"], 46)

    def test_copied_teacher_suffix_fails_closed(self) -> None:
        diffusion = _FakeDiffusion(
            branch_name="VI", teacher_queries=True, copied_teacher_suffix=True
        )
        handle = _fake_handle(diffusion.transformer, [])
        with runtime.saic_source_anchor_native_runtime(
            diffusion,
            handle=handle,
            config=runtime.SourceAnchorNativeRuntimeConfig(
                target_latent_shape=(1, 16, 21, 2, 2), branch_name="VI"
            ),
        ):
            with self.assertRaisesRegex(
                runtime.SAICSourceAnchorNativeRuntimeError,
                "exact full-pack target suffix view",
            ):
                diffusion.sample(
                    **_sample_kwargs(
                        "VI",
                        self.action,
                        self.negative,
                        self.source,
                        self.references,
                    )
                )

    def test_branch_input_contract_rejects_missing_vi_references(self) -> None:
        diffusion = _FakeDiffusion(branch_name="VI", teacher_queries=False)
        handle = _fake_handle(diffusion.transformer, [])
        kwargs = _sample_kwargs(
            "VI", self.action, self.negative, self.source, self.references
        )
        kwargs["multi_image_vae_latents"] = None
        with runtime.saic_source_anchor_native_runtime(
            diffusion,
            handle=handle,
            config=runtime.SourceAnchorNativeRuntimeConfig(
                target_latent_shape=(1, 16, 21, 2, 2), branch_name="VI"
            ),
        ):
            with self.assertRaisesRegex(
                runtime.SAICSourceAnchorNativeRuntimeError,
                "reference count differs",
            ):
                diffusion.sample(**kwargs)


@unittest.skipUnless(_TORCH_AVAILABLE, "torch runtime is required")
class AdapterTimestepCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.transformer = _SmallTransformer()
        self.transformer.requires_grad_(False)
        self.handle = anchor.install_saic_source_anchor_adapter(self.transformer)
        self.scheduler = UniPCMultistepScheduler()
        target_tokens = 21
        mask = torch.zeros(42, dtype=torch.bool)
        mask[target_tokens:] = True
        self.branch = native.NativeRV2VBranch(
            name="V",
            latents=torch.zeros(1, 42, 4),
            rotary=torch.zeros(1, 1, 42, 2),
            target_mask=mask,
            total_tokens=42,
            condition_tokens=21,
            source_ids=(1.0, 0.0),
            concat_order=native.BRANCH_CONCAT_ORDER["V"],
        )
        self.parallel = mock.patch.object(
            anchor, "_get_live_parallel_state", return_value=_ParallelState()
        )
        self.parallel.start()
        self.addCleanup(self.parallel.stop)

    def tearDown(self) -> None:
        if not self.handle.restored and anchor.active_route() is None:
            self.handle.restore()

    def test_device_local_int64_official_and_float32_manual_both_route(self) -> None:
        cases = (
            torch.tensor([117], dtype=torch.int64),
            torch.tensor([117.0], dtype=torch.float32),
        )
        for timestep in cases:
            with self.subTest(dtype=str(timestep.dtype)):
                with self.handle.route(
                    branch=self.branch,
                    scheduler=self.scheduler,
                    timestep=timestep,
                ) as route:
                    receipt = route.receipt()
                    self.assertEqual(receipt["schedule_index"], 39)
                    self.assertEqual(receipt["timestep_dtype"], str(timestep.dtype))
                    self.assertEqual(
                        receipt["timestep_device"], str(self.branch.latents.device)
                    )
                    self.assertEqual(
                        receipt["sigma_float32_be_hex"],
                        strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[39],
                    )
                    self.assertEqual(
                        receipt["exact40_schedule_sha256"],
                        strata.SCHEDULE_SHA256,
                    )

    def test_int32_float64_ambiguous_and_schedule_mismatch_are_rejected(self) -> None:
        cases = (
            (torch.tensor([117], dtype=torch.int32), "INT64/FP32"),
            (torch.tensor([117.0], dtype=torch.float64), "INT64/FP32"),
            (torch.tensor([117.0, 117.0], dtype=torch.float32), "one detached"),
            (torch.tensor([118.0], dtype=torch.float32), "not one unique"),
        )
        for timestep, message in cases:
            with self.subTest(dtype=str(timestep.dtype), shape=tuple(timestep.shape)):
                with self.assertRaisesRegex(anchor.SAICSourceAnchorError, message):
                    with self.handle.route(
                        branch=self.branch,
                        scheduler=self.scheduler,
                        timestep=timestep,
                    ):
                        pass

    def test_int64_forward_device_mismatch_is_rejected(self) -> None:
        mismatched = SimpleNamespace(
            latents=torch.empty(1, 42, 4, device="meta")
        )
        with mock.patch.object(
            anchor, "_validate_full_source_branch", return_value=mismatched
        ):
            with self.assertRaisesRegex(
                anchor.SAICSourceAnchorError, "share the forward device"
            ):
                with self.handle.route(
                    branch=self.branch,
                    scheduler=self.scheduler,
                    timestep=torch.tensor([117], dtype=torch.int64),
                ):
                    pass


@unittest.skipUnless(_TORCH_AVAILABLE, "torch runtime is required")
class NativeRuntimeTimestepDeviceTests(unittest.TestCase):
    def test_shared_int64_timestep_must_match_noisy_forward_device(self) -> None:
        patch = runtime.SAICSourceAnchorNativeRuntimePatch.__new__(
            runtime.SAICSourceAnchorNativeRuntimePatch
        )
        patch._schedule_index = lambda _value, *, expected: expected
        patch._validate_prompt = lambda value, *, label: value
        values = {
            "model_id": "transformer_1",
            "noisy_latents": torch.zeros(1, 42, 1536),
            "timesteps": torch.empty(1, dtype=torch.int64, device="meta"),
            "cond_embeds": torch.zeros(1),
            "rotary_embs": torch.zeros(1, 1, 42, 4),
        }
        with mock.patch.object(
            runtime,
            "_detached_finite_tensor",
            side_effect=lambda value, *, label: value,
        ):
            with self.assertRaisesRegex(
                runtime.SAICSourceAnchorNativeRuntimeError,
                "noisy forward device",
            ):
                patch._validate_shared_common(values, schedule_index=39)


@unittest.skipUnless(_TORCH_AVAILABLE, "torch runtime is required")
class StrictSafetensorsLoaderTests(unittest.TestCase):
    def _handle(self, initial: torch.Tensor):
        handle = anchor.SAICSourceAnchorHandle.__new__(anchor.SAICSourceAnchorHandle)
        handle.transformer = object()
        handle.restored = False
        handle.receipt = lambda: _handle_receipt()
        parameter = nn.Parameter(initial.clone().float())
        state = {"blocks.23.attn1.to_q.state_down.weight": parameter.detach().clone()}

        def named():
            return (("blocks.23.attn1.to_q.state_down.weight", parameter),)

        def saved():
            return {
                name: value.detach().float().cpu().contiguous().clone()
                for name, value in named()
            }

        def load(values):
            with torch.no_grad():
                parameter.copy_(values[next(iter(values))])
            value = {"loaded": True}
            return {**value, "digest": runtime._object_sha256(value)}  # noqa: SLF001

        handle.trainable_named_parameters = named
        handle.state_dict_for_save = saved
        handle.load_trainable_state_dict = load
        return handle, parameter, state

    def _metadata(self, state):
        return {
            "schema_version": runtime.SAFETENSORS_SCHEMA_VERSION,
            "adapter_schema_version": anchor.SCHEMA_VERSION,
            "adapter_contract_digest": "a" * 64,
            "state_tensor_sha256": anchor.trainable_state_digest(state),
            "state_key_sha256": runtime._object_sha256(sorted(state)),  # noqa: SLF001
            "optimizer_updates": "32",
            "heldout_gate_digest": "b" * 64,
            "source_anchor_only": "true",
            "semantic_action_success": "false",
        }

    def _safe_module(self, metadata, state):
        class Opened:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def metadata(self):
                return dict(metadata)

            def keys(self):
                return tuple(state)

            def get_tensor(self, name):
                return state[name]

        module = ModuleType("safetensors")
        module.safe_open = lambda *_args, **_kwargs: Opened()
        return module

    def test_strict_registered_file_metadata_and_state_load(self) -> None:
        handle, parameter, _ = self._handle(torch.zeros(2, 3))
        loaded = {"blocks.23.attn1.to_q.state_down.weight": torch.ones(2, 3)}
        metadata = self._metadata(loaded)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "adapter.safetensors"
            path.write_bytes(b"strict-safe-test")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with mock.patch.dict(
                sys.modules,
                {"safetensors": self._safe_module(metadata, loaded)},
            ):
                receipt = runtime.load_saic_source_anchor_safetensors(
                    handle,
                    path,
                    expected_file_sha256=digest,
                    expected_metadata=metadata,
                )
        self.assertTrue(torch.equal(parameter, torch.ones_like(parameter)))
        self.assertTrue(receipt["metadata_exact_registration_match"])
        self.assertEqual(receipt["optimizer_updates"], 32)

    def test_wrong_metadata_is_rejected_before_parameter_mutation(self) -> None:
        handle, parameter, _ = self._handle(torch.zeros(2, 3))
        loaded = {"blocks.23.attn1.to_q.state_down.weight": torch.ones(2, 3)}
        metadata = self._metadata(loaded)
        wrong = dict(metadata)
        wrong["optimizer_updates"] = "31"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "adapter.safetensors"
            path.write_bytes(b"strict-safe-test")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(
                runtime.SAICSourceAnchorNativeRuntimeError,
                "metadata values differ",
            ):
                runtime.load_saic_source_anchor_safetensors(
                    handle,
                    path,
                    expected_file_sha256=digest,
                    expected_metadata=wrong,
                )
        self.assertEqual(int(torch.count_nonzero(parameter)), 0)

    def test_failed_parameter_copy_rolls_back_preload_state(self) -> None:
        handle, parameter, _ = self._handle(torch.zeros(2, 3))
        loaded = {"blocks.23.attn1.to_q.state_down.weight": torch.ones(2, 3)}
        metadata = self._metadata(loaded)
        original_loader = handle.load_trainable_state_dict
        calls = 0

        def fail_once(values):
            nonlocal calls
            calls += 1
            receipt = original_loader(values)
            if calls == 1:
                raise RuntimeError("injected device copy failure")
            return receipt

        handle.load_trainable_state_dict = fail_once
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "adapter.safetensors"
            path.write_bytes(b"strict-safe-test")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with mock.patch.dict(
                sys.modules,
                {"safetensors": self._safe_module(metadata, loaded)},
            ):
                with self.assertRaisesRegex(
                    runtime.SAICSourceAnchorNativeRuntimeError,
                    "state load failed",
                ):
                    runtime.load_saic_source_anchor_safetensors(
                        handle,
                        path,
                        expected_file_sha256=digest,
                        expected_metadata=metadata,
                    )
        self.assertEqual(calls, 2)
        self.assertEqual(int(torch.count_nonzero(parameter)), 0)


if __name__ == "__main__":
    unittest.main()
