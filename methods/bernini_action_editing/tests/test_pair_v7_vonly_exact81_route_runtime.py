#!/usr/bin/env python3

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib.util
import sys
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import inference_sigma_strata as sigma_strata  # noqa: E402
import pair_v7_vonly_exact81_route_runtime as runtime  # noqa: E402


class _Device:
    def __init__(self, kind: str = "cpu") -> None:
        self.type = kind

    def __str__(self) -> str:
        return self.type


class _FakeTensor:
    """Small explicit tensor double used by the model-free hook tests."""

    def __init__(
        self,
        shape,
        *,
        dtype="torch.float32",
        device="cpu",
        values=None,
        payload=0.0,
        finite=True,
    ) -> None:
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.dtype = dtype
        self.device = _Device(device)
        self.requires_grad = False
        self.grad_fn = None
        self._values = values
        self.payload = float(payload)
        self._finite = bool(finite)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numel(self):
        result = 1
        for item in self.shape:
            result *= item
        return result

    def item(self):
        if self.numel() != 1:
            raise ValueError("not scalar")
        if isinstance(self._values, (list, tuple)):
            return self._values[0]
        return self._values

    def tolist(self):
        if self.ndim != 1 or not isinstance(self._values, (list, tuple)):
            raise ValueError("not a vector")
        return list(self._values)

    def scalar_at(self, index: int):
        if self.ndim != 1 or not isinstance(self._values, (list, tuple)):
            raise ValueError("not a vector")
        return _FakeTensor(
            (),
            dtype=self.dtype,
            device=self.device.type,
            values=self._values[index],
        )

    def expanded_scalar(self):
        return _FakeTensor(
            (1,),
            dtype=self.dtype,
            device=self.device.type,
            values=[self.item()],
        )

    def isfinite_all(self):
        return self._finite

    def pair_v7_tensor_sha256(self):
        material = repr(
            (
                self.shape,
                self.dtype,
                self.device.type,
                self._values,
                self.payload,
                self._finite,
            )
        ).encode("utf-8")
        return hashlib.sha256(material).hexdigest()


def _make_action_module():
    module = SimpleNamespace()
    module.current = None
    module.route_enters = []
    module.route_exits = []

    def sigma_gate(index):
        if type(index) is not int or not 0 <= index < 40:
            raise RuntimeError("index outside exact40")
        if index < 33:
            return "high", 1.0
        if index < 38:
            return "mid", 0.5
        return "low_base_only", 0.0

    class PairV5ActionRoute:
        def __init__(
            self,
            *,
            total_tokens,
            condition_tokens,
            sequence_parallel_rank,
            sequence_parallel_size,
            branch_name,
            sigma_schedule_index,
            enabled=True,
        ) -> None:
            self.total_tokens = total_tokens
            self.condition_tokens = condition_tokens
            self.sequence_parallel_rank = sequence_parallel_rank
            self.sequence_parallel_size = sequence_parallel_size
            self.branch_name = branch_name
            self.sigma_schedule_index = sigma_schedule_index
            self.enabled = enabled

        @property
        def gate_name(self):
            return sigma_gate(self.sigma_schedule_index)[0]

        @property
        def gate_weight(self):
            return sigma_gate(self.sigma_schedule_index)[1] if self.enabled else 0.0

        @property
        def adapter_active(self):
            return self.enabled and self.gate_weight > 0.0

        def local_target_selector(self, *, device):
            del device
            local_length = (
                self.total_tokens + self.sequence_parallel_size - 1
            ) // self.sequence_parallel_size
            selector = (
                [False] * self.condition_tokens
                + [True] * (self.total_tokens - self.condition_tokens)
            )
            selector += [False] * (
                local_length * self.sequence_parallel_size - self.total_tokens
            )
            start = self.sequence_parallel_rank * local_length
            return tuple(selector[start : start + local_length])

    class PairV5ActionAdapterHandle:
        def __init__(self, transformer) -> None:
            self.transformer = transformer
            self.restored = False

        @contextmanager
        def route(self, route):
            if module.current is not None:
                raise RuntimeError("nested fake route")
            module.current = route
            module.route_enters.append(route)
            try:
                yield
            finally:
                module.route_exits.append(route)
                module.current = None

    def active_route():
        return module.current

    module.PairV5ActionRoute = PairV5ActionRoute
    module.PairV5ActionAdapterHandle = PairV5ActionAdapterHandle
    module.sigma_gate = sigma_gate
    module.active_route = active_route
    return module


class _FakeScheduler:
    def __init__(self, *, corrupt_sigma=False) -> None:
        self.config = SimpleNamespace(
            _class_name="UniPCMultistepScheduler",
            num_train_timesteps=1000,
            flow_shift=5.0,
            prediction_type="flow_prediction",
            predict_x0=True,
            use_flow_sigmas=True,
            thresholding=False,
            solver_order=2,
            solver_type="bh2",
            final_sigmas_type="zero",
        )
        sigmas = list(sigma_strata.PINNED_POSITIVE_SIGMAS) + [0.0]
        if corrupt_sigma:
            sigmas[13] += 0.001
        self.timesteps = _FakeTensor(
            (40,), dtype="torch.int64", values=sigma_strata.PINNED_TIMESTEPS
        )
        self.sigmas = _FakeTensor(
            (41,), dtype="torch.float32", values=sigmas
        )
        self.step_index = 0
        self.set_timesteps_calls = 0
        self.call_count = 0
        self.received_model_outputs = []

    def set_timesteps(self, steps):
        if steps != 40:
            raise ValueError("fake is exact40 only")
        self.set_timesteps_calls += 1
        self.step_index = 0

    def step(self, model_output, timestep, sample, return_dict=False):
        del timestep, return_dict
        self.received_model_outputs.append(model_output)
        self.call_count += 1
        self.step_index += 1
        return (sample,)


class _FakeTransformer:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            num_attention_heads=12,
            attention_head_dim=128,
            in_channels=16,
            out_channels=16,
            patch_size=[1, 2, 2],
            text_dim=4096,
        )
        self.gradient_checkpointing = False
        self.is_gradient_checkpointing = False


class _FakeDiffusion:
    use_unipc = True
    transformer_2 = None
    switch_dit_boundary = 0

    def __init__(
        self,
        *,
        action_module,
        zero_adapter=False,
        corrupt_sigma=False,
        fail_at=None,
        wrong_model_timestep_index=None,
    ) -> None:
        self.action_module = action_module
        self.zero_adapter = bool(zero_adapter)
        self.transformer = _FakeTransformer()
        self.scheduler = _FakeScheduler(corrupt_sigma=corrupt_sigma)
        self.fail_at = fail_at
        self.wrong_model_timestep_index = wrong_model_timestep_index
        self.original_sample_calls = 0
        self.original_shared_calls = 0
        self.shared_attempts = 0
        self.forward_log = []

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
        del model_id, rotary_embs, batch_vae_seqlen, batch_text_seqlen
        route = self.action_module.active_route()
        index = sigma_strata.PINNED_TIMESTEPS.index(int(timesteps.item()))
        role = "negative" if cond_embeds.payload < 0.0 else "action"
        self.shared_attempts += 1
        if self.fail_at == (index, role):
            raise RuntimeError("injected official shared_step failure")
        self.original_shared_calls += 1
        base_payload = float(index * 10 + (0 if role == "negative" else 1))
        gate = 0.0 if route is None else float(route.gate_weight)
        delta = 0.0 if self.zero_adapter else 100.0 * gate
        result = _FakeTensor(
            (1, noisy_latents.shape[1], 64),
            payload=base_payload + delta,
        )
        self.forward_log.append(
            {
                "index": index,
                "role": role,
                "route": route,
                "base_payload": base_payload,
                "result_payload": result.payload,
            }
        )
        return result

    def sample(
        self,
        prompt_embeds=None,
        prompt_embeds_t2=None,
        uncond_prompt_embeds=None,
        uncond_embeds_t2=None,
        num_frames=1,
        width=16,
        height=16,
        image_vae_latents=None,
        multi_video_vae_latents=None,
        multi_image_vae_latents=None,
        num_inference_steps=50,
        guidance_mode="rv2v",
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
        reference_video=None,
        target_video=None,
        **extra,
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
            eta,
            norm_threshold,
            momentum,
            mask,
            reference_video,
            target_video,
            extra,
        )
        self.original_sample_calls += 1
        self.scheduler.set_timesteps(num_inference_steps)
        total_payload = 0.0
        for index in range(num_inference_steps):
            scalar_timestep = self.scheduler.timesteps.scalar_at(index)
            model_timestep = scalar_timestep.expanded_scalar()
            if self.wrong_model_timestep_index == index:
                model_timestep = self.scheduler.timesteps.scalar_at(index + 1).expanded_scalar()
            noisy = _FakeTensor((1, 42, 1536), payload=float(index))
            rotary = _FakeTensor((1, 1, 42, 4), payload=float(index))

            def forward(prompt):
                return self.shared_step(
                    model_id="transformer_1",
                    noisy_latents=noisy,
                    timesteps=model_timestep,
                    cond_embeds=prompt,
                    rotary_embs=rotary,
                    batch_vae_seqlen=[42],
                    batch_text_seqlen=[512],
                )

            negative = forward(uncond_prompt_embeds)
            action = forward(prompt_embeds)
            total_payload += negative.payload + action.payload
            self.scheduler.step(action, scalar_timestep, noisy, return_dict=False)
        return _FakeTensor(
            (1, 16, 21, 2, 2), dtype="torch.float32", payload=total_payload
        )


class PairV7VOnlyRouteRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.action_module = _make_action_module()
        self.action = _FakeTensor((1, 512, 4096), payload=0.25)
        self.negative = _FakeTensor((1, 512, 4096), payload=-0.25)
        self.source = _FakeTensor((1, 16, 21, 2, 2), payload=0.1)
        self.distributed_patch = mock.patch.object(
            runtime,
            "_validate_live_distributed_route",
            return_value={
                "world_size": 8,
                "global_rank": 7,
                "all_gather_object_consensus": True,
                "dp_major_coordinate_set_complete": True,
            },
        )
        self.distributed_patch.start()
        self.addCleanup(self.distributed_patch.stop)

    def _parallel(self, *, dp_rank=1, sp_rank=3, global_rank=7):
        return runtime.PairV7DPSPRouteMetadata(
            data_parallel_rank=dp_rank,
            data_parallel_size=2,
            sequence_parallel_rank=sp_rank,
            sequence_parallel_size=4,
            global_rank=global_rank,
            world_size=8,
        )

    def _config(self, **kwargs):
        values = {
            "target_latent_shape": (1, 16, 21, 2, 2),
            "parallel": self._parallel(),
            "expected_seed": 7,
            "expected_source_latent_sha256": self.source.pair_v7_tensor_sha256(),
            "expected_action_prompt_sha256": self.action.pair_v7_tensor_sha256(),
            "expected_negative_prompt_sha256": self.negative.pair_v7_tensor_sha256(),
        }
        values.update(kwargs)
        return runtime.PairV7VOnlyExact81RouteConfig(**values)

    def _kwargs(self, **changes):
        values = {
            "prompt_embeds": self.action,
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
            "omega_vid": 1.25,
            "omega_img": 0.0,
            "omega_txt": 4.0,
            "omega_scale": 0.8,
            "flow_shift": 5.0,
            "seed": 7,
            "device": "cpu",
            "eta": 0.5,
            "norm_threshold": (50.0, 50.0),
            "momentum": 0.0,
        }
        values.update(changes)
        return values

    def _objects(self, **diffusion_kwargs):
        diffusion = _FakeDiffusion(
            action_module=self.action_module, **diffusion_kwargs
        )
        handle = self.action_module.PairV5ActionAdapterHandle(diffusion.transformer)
        return diffusion, handle

    def _run(self, *, diffusion_kwargs=None, sample_kwargs=None):
        diffusion, handle = self._objects(**(diffusion_kwargs or {}))
        before = (
            diffusion.sample.__func__,
            diffusion.shared_step.__func__,
            diffusion.scheduler.step.__func__,
        )
        with mock.patch.object(
            runtime,
            "_load_action_adapter_module",
            return_value=self.action_module,
        ):
            with runtime.pair_v7_vonly_exact81_route_hook(
                diffusion,
                action_handle=handle,
                config=self._config(),
            ) as patch:
                result = diffusion.sample(**(sample_kwargs or self._kwargs()))
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

    def test_exact40_routes_two_official_forwards_and_one_scheduler_in_order(self):
        diffusion, _, receipt, result = self._run()
        self.assertEqual(diffusion.original_sample_calls, 1)
        self.assertEqual(diffusion.original_shared_calls, 80)
        self.assertEqual(diffusion.scheduler.call_count, 40)
        self.assertEqual(receipt["official_sample_calls"], 1)
        self.assertEqual(receipt["official_shared_step_calls"], 80)
        self.assertEqual(receipt["official_scheduler_step_calls"], 40)
        self.assertEqual(
            receipt["registered_inputs"],
            {
                "seed": 7,
                "width": 16,
                "height": 16,
                "source_latent_sha256": self.source.pair_v7_tensor_sha256(),
                "action_prompt_sha256": self.action.pair_v7_tensor_sha256(),
                "negative_prompt_sha256": self.negative.pair_v7_tensor_sha256(),
            },
        )
        self.assertEqual(
            receipt["official_output_latent_sha256"],
            result.pair_v7_tensor_sha256(),
        )
        self.assertEqual(receipt["switch_dit_boundary"], 0.0)
        unsigned = dict(receipt)
        digest = unsigned.pop("receipt_digest")
        self.assertEqual(digest, runtime._object_sha256(unsigned))
        self.assertTrue(
            receipt["live_distributed_route_audit"][
                "dp_major_coordinate_set_complete"
            ]
        )
        self.assertEqual(result.shape, (1, 16, 21, 2, 2))
        self.assertEqual(len(self.action_module.route_enters), 80)
        self.assertEqual(
            [route.sigma_schedule_index for route in self.action_module.route_enters],
            [index for index in range(40) for _ in range(2)],
        )
        self.assertEqual(
            [(row["index"], row["role"]) for row in diffusion.forward_log],
            [
                (index, role)
                for index in range(40)
                for role in runtime.FORWARD_ORDER
            ],
        )
        self.assertIsNone(self.action_module.active_route())
        self.assertEqual(
            self.action_module.route_enters, self.action_module.route_exits
        )

    def test_registered_gate_selector_and_direct_base_partition(self):
        diffusion, _, receipt, _ = self._run()
        self.assertEqual(receipt["active_schedule_indices"], list(range(38)))
        self.assertEqual(receipt["direct_base_schedule_indices"], [38, 39])
        self.assertEqual(receipt["registered_gate_weights"][:33], [1.0] * 33)
        self.assertEqual(receipt["registered_gate_weights"][33:38], [0.5] * 5)
        self.assertEqual(receipt["registered_gate_weights"][38:], [0.0] * 2)
        for route in self.action_module.route_enters:
            self.assertEqual(route.branch_name, "V")
            self.assertEqual(route.total_tokens, 42)
            self.assertEqual(route.condition_tokens, 21)
            self.assertEqual(route.sequence_parallel_rank, 3)
            self.assertEqual(route.sequence_parallel_size, 4)
            # SP rank 3 receives global rows 33..41 plus two append-padding rows.
            self.assertEqual(
                route.local_target_selector(device=_Device()),
                (True,) * 9 + (False,) * 2,
            )
        for row in diffusion.forward_log:
            if row["index"] < 33:
                self.assertEqual(row["result_payload"] - row["base_payload"], 100.0)
            elif row["index"] < 38:
                self.assertEqual(row["result_payload"] - row["base_payload"], 50.0)
            else:
                self.assertEqual(row["result_payload"], row["base_payload"])

    def test_zero_adapter_is_exact_official_output_parity(self):
        baseline_module = _make_action_module()
        baseline = _FakeDiffusion(
            action_module=baseline_module, zero_adapter=True
        )
        baseline_result = baseline.sample(**self._kwargs())
        diffusion, _, receipt, routed_result = self._run(
            diffusion_kwargs={"zero_adapter": True}
        )
        self.assertEqual(routed_result.payload, baseline_result.payload)
        self.assertTrue(
            all(
                row["result_payload"] == row["base_payload"]
                for row in diffusion.forward_log
            )
        )
        self.assertFalse(receipt["official_sampler_arguments_mutated"])

    def test_shared_exception_restores_all_hooks_and_active_route(self):
        diffusion, handle = self._objects(fail_at=(7, "action"))
        before = (
            diffusion.sample.__func__,
            diffusion.shared_step.__func__,
            diffusion.scheduler.step.__func__,
        )
        with mock.patch.object(
            runtime,
            "_load_action_adapter_module",
            return_value=self.action_module,
        ):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                with runtime.pair_v7_vonly_exact81_route_hook(
                    diffusion, action_handle=handle, config=self._config()
                ):
                    diffusion.sample(**self._kwargs())
        after = (
            diffusion.sample.__func__,
            diffusion.shared_step.__func__,
            diffusion.scheduler.step.__func__,
        )
        self.assertEqual(before, after)
        self.assertIsNone(self.action_module.active_route())
        self.assertNotIn("sample", vars(diffusion))
        self.assertNotIn("shared_step", vars(diffusion))
        self.assertNotIn("step", vars(diffusion.scheduler))
        self.assertEqual(diffusion.original_sample_calls, 1)
        self.assertEqual(len(self.action_module.route_enters), 16)
        self.assertEqual(
            self.action_module.route_enters, self.action_module.route_exits
        )

    def test_rejects_images_masks_and_extra_visual_branches_before_sample(self):
        forbidden_cases = (
            {"image_vae_latents": _FakeTensor((1, 1, 1))},
            {"multi_image_vae_latents": [_FakeTensor((1, 1, 1))]},
            {"mask": _FakeTensor((1, 1, 1))},
            {"reference_video": _FakeTensor((1, 16, 21, 2, 2))},
            {"target_video": _FakeTensor((1, 16, 21, 2, 2))},
            {"multi_video_vae_latents": [self.source, self.source]},
            {"pose_condition": _FakeTensor((1, 81, 3))},
            {"donor_video_latent": _FakeTensor((1, 16, 21, 2, 2))},
        )
        for changes in forbidden_cases:
            with self.subTest(changes=tuple(changes)):
                action_module = _make_action_module()
                diffusion = _FakeDiffusion(action_module=action_module)
                handle = action_module.PairV5ActionAdapterHandle(
                    diffusion.transformer
                )
                with mock.patch.object(
                    runtime,
                    "_load_action_adapter_module",
                    return_value=action_module,
                ):
                    with runtime.pair_v7_vonly_exact81_route_hook(
                        diffusion, action_handle=handle, config=self._config()
                    ):
                        with self.assertRaises(runtime.PairV7VOnlyRouteRuntimeError):
                            diffusion.sample(**self._kwargs(**changes))
                self.assertEqual(diffusion.original_sample_calls, 0)
                self.assertNotIn("sample", vars(diffusion))
                self.assertNotIn("shared_step", vars(diffusion))
                self.assertNotIn("step", vars(diffusion.scheduler))
                self.assertIsNone(action_module.active_route())

    def test_full_schedule_and_each_model_timestep_fail_closed(self):
        action_module = _make_action_module()
        diffusion = _FakeDiffusion(
            action_module=action_module, corrupt_sigma=True
        )
        handle = action_module.PairV5ActionAdapterHandle(diffusion.transformer)
        with mock.patch.object(
            runtime,
            "_load_action_adapter_module",
            return_value=action_module,
        ):
            with runtime.pair_v7_vonly_exact81_route_hook(
                diffusion, action_handle=handle, config=self._config()
            ):
                with self.assertRaisesRegex(
                    runtime.PairV7VOnlyRouteRuntimeError,
                    "live exact40 shift-5 schedule",
                ):
                    diffusion.sample(**self._kwargs())
        self.assertEqual(diffusion.original_shared_calls, 0)
        self.assertEqual(diffusion.scheduler.call_count, 0)

        action_module = _make_action_module()
        diffusion = _FakeDiffusion(
            action_module=action_module, wrong_model_timestep_index=3
        )
        handle = action_module.PairV5ActionAdapterHandle(diffusion.transformer)
        with mock.patch.object(
            runtime,
            "_load_action_adapter_module",
            return_value=action_module,
        ):
            with runtime.pair_v7_vonly_exact81_route_hook(
                diffusion, action_handle=handle, config=self._config()
            ):
                with self.assertRaisesRegex(
                    runtime.PairV7VOnlyRouteRuntimeError,
                    "call order differs",
                ):
                    diffusion.sample(**self._kwargs())
        self.assertEqual(diffusion.original_shared_calls, 6)
        self.assertEqual(diffusion.scheduler.call_count, 3)
        self.assertIsNone(action_module.active_route())

    def test_exact81_and_dp2_sp4_metadata_are_mandatory(self):
        invalid_parallel = (
            runtime.PairV7DPSPRouteMetadata(0, 1, 0, 4, 0, 4),
            runtime.PairV7DPSPRouteMetadata(0, 2, 0, 2, 0, 4),
            runtime.PairV7DPSPRouteMetadata(1, 2, 3, 4, 3, 8),
        )
        for parallel in invalid_parallel:
            with self.subTest(parallel=parallel):
                with self.assertRaises(runtime.PairV7VOnlyRouteRuntimeError):
                    runtime.PairV7VOnlyExact81RouteConfig(
                        target_latent_shape=(1, 16, 21, 2, 2),
                        parallel=parallel,
                        expected_seed=7,
                        expected_source_latent_sha256="0" * 64,
                        expected_action_prompt_sha256="1" * 64,
                        expected_negative_prompt_sha256="2" * 64,
                    ).validate()
        with self.assertRaisesRegex(
            runtime.PairV7VOnlyRouteRuntimeError, "exact81"
        ):
            runtime.PairV7VOnlyExact81RouteConfig(
                target_latent_shape=(1, 16, 11, 2, 2),
                parallel=self._parallel(),
                expected_seed=7,
                expected_source_latent_sha256="0" * 64,
                expected_action_prompt_sha256="1" * 64,
                expected_negative_prompt_sha256="2" * 64,
            ).validate()

    def test_registered_seed_spatial_geometry_and_tensor_digests_fail_closed(self):
        cases = (
            ({"seed": 8}, {}),
            ({"width": 32}, {}),
            ({"height": 32}, {}),
            ({"multi_video_vae_latents": [_FakeTensor((1, 16, 21, 2, 2), payload=0.2)]}, {}),
            ({"prompt_embeds": _FakeTensor((1, 512, 4096), payload=0.5)}, {}),
            ({"uncond_prompt_embeds": _FakeTensor((1, 512, 4096), payload=-0.5)}, {}),
            ({}, {"expected_seed": 8}),
            ({}, {"expected_source_latent_sha256": "f" * 64}),
        )
        for sample_changes, config_changes in cases:
            with self.subTest(sample=sample_changes, config=config_changes):
                diffusion, handle = self._objects()
                with mock.patch.object(
                    runtime,
                    "_load_action_adapter_module",
                    return_value=self.action_module,
                ):
                    with runtime.pair_v7_vonly_exact81_route_hook(
                        diffusion,
                        action_handle=handle,
                        config=self._config(**config_changes),
                    ):
                        with self.assertRaises(runtime.PairV7VOnlyRouteRuntimeError):
                            diffusion.sample(**self._kwargs(**sample_changes))
                self.assertEqual(diffusion.original_sample_calls, 0)

    def test_live_world8_route_metadata_is_collectively_bound(self):
        self.distributed_patch.stop()
        with mock.patch.object(
            runtime.importlib,
            "import_module",
            side_effect=ModuleNotFoundError("no live torch.distributed"),
        ):
            with self.assertRaisesRegex(
                runtime.PairV7VOnlyRouteRuntimeError,
                "cannot establish live WORLD8",
            ):
                runtime._validate_live_distributed_route(self._parallel())
        self.distributed_patch.start()


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None,
    "PyTorch is required for real PairV5 Action-LoRA parity",
)
class RealPairV5LowSigmaParityTests(unittest.TestCase):
    def test_existing_wrapper_low_route_is_direct_base(self):
        import torch
        from torch import nn
        import pair_v5_action_adapter as action_adapter

        torch.manual_seed(19)
        base = nn.Linear(8, 8, bias=True)
        wrapper = action_adapter.PairV5TargetRowActionLoRA(base, projection="to_q")
        with torch.no_grad():
            wrapper.action_lora_a.weight.fill_(0.25)
            wrapper.action_lora_b.weight.fill_(0.5)
        hidden = torch.randn(1, 11, 8)
        route = action_adapter.PairV5ActionRoute(
            total_tokens=42,
            condition_tokens=21,
            sequence_parallel_rank=3,
            sequence_parallel_size=4,
            branch_name="V",
            sigma_schedule_index=38,
            enabled=True,
        )
        with action_adapter.activate_route(route):
            actual = wrapper(hidden)
        self.assertTrue(torch.equal(actual, base(hidden)))


if __name__ == "__main__":
    unittest.main()
