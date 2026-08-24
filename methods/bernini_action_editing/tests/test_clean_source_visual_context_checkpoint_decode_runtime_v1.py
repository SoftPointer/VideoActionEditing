#!/usr/bin/env python3

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
import unittest

import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import clean_source_visual_context_adapter_v1 as visual  # noqa: E402
import clean_source_visual_context_checkpoint_decode_runtime_v1 as runtime  # noqa: E402


SHA_A = "a" * 64
SHA_B = "b" * 64


class _Scheduler:
    def resolve_sigma(self, timestep: object) -> float:
        value = float(timestep)
        return value / 100.0


class _Handle:
    def __init__(self) -> None:
        self.build_calls = 0
        self.routes: list[visual.VisualContextRoute] = []

    def build_memory(
        self,
        memory_input_latent: torch.Tensor,
        *,
        source_video_sha256: str,
        memory_input_latent_sha256: str,
        input_kind: str,
    ) -> visual.CleanSourceVisualMemory:
        self.build_calls += 1
        return visual.CleanSourceVisualMemory(
            tokens=torch.zeros((1, 21, 8), dtype=torch.float32),
            source_video_sha256=source_video_sha256,
            memory_input_latent_sha256=memory_input_latent_sha256,
            latent_shape=tuple(int(value) for value in memory_input_latent.shape),
            patch_grid=(21, 1, 1),
            pooled_grid=(21, 1, 1),
            input_kind=input_kind,
            construction_digest=(f"{self.build_calls:064x}"[-64:]),
        )

    @contextmanager
    def route(self, route: visual.VisualContextRoute):
        self.routes.append(route)
        with visual.activate_route(route):
            yield


class _Diffusion:
    def __init__(self) -> None:
        self.observed: list[visual.VisualContextRoute] = []

    def shared_step(
        self,
        *,
        model_id: str,
        noisy_latents: object,
        timesteps: object,
        cond_embeds: object,
        rotary_embs: object,
        batch_vae_seqlen: object,
        batch_text_seqlen: object,
    ) -> object:
        route = visual.active_route()
        if route is None:
            raise AssertionError("route was not active")
        self.observed.append(route)
        return noisy_latents


def _latent(offset: float = 0.0) -> torch.Tensor:
    return (
        torch.arange(1 * 16 * 21 * 2 * 2, dtype=torch.float32)
        .reshape(1, 16, 21, 2, 2)
        .add(offset)
        .contiguous()
    )


def _run_exact40(hook: runtime.BranchAwareVisualContextRouteHook) -> None:
    lengths = (8, 16, 24, 24)
    with hook.sample():
        for step in range(40):
            for total in lengths:
                hook.diffusion.shared_step(
                    model_id="transformer_1",
                    noisy_latents=object(),
                    timesteps=float(40 - step),
                    cond_embeds=object(),
                    rotary_embs=object(),
                    batch_vae_seqlen=[total],
                    batch_text_seqlen=[1],
                )


class DecodeRuntimeTests(unittest.TestCase):
    def test_clean_memory_is_built_once_and_shared_across_exact40_four_branches(self) -> None:
        handle = _Handle()
        provider = runtime.VisualMemoryProvider(
            handle=handle,
            source_latent=_latent(),
            source_video_sha256=SHA_A,
            memory_input_kind="clean_source",
            scheduler=_Scheduler(),
        )
        diffusion = _Diffusion()
        hook = runtime.BranchAwareVisualContextRouteHook(
            diffusion,
            handle=handle,
            target_tokens=8,
            sequence_parallel_rank=0,
            sequence_parallel_size=4,
            source_control_arm="correct",
            target_source_video_sha256=SHA_A,
            memory_provider=provider,
        )
        hook.install()
        try:
            _run_exact40(hook)
        finally:
            hook.restore()
        self.assertEqual(handle.build_calls, 1)
        self.assertEqual(len(handle.routes), 160)
        self.assertEqual(len({id(route.memory) for route in handle.routes}), 1)
        self.assertEqual(
            [row["branch"] for row in hook.trace["calls"][:4]],
            list(runtime.NATIVE_BRANCH_ORDER),
        )
        self.assertEqual(
            [row["condition_tokens"] for row in hook.trace["calls"][:4]],
            [0, 8, 16, 16],
        )
        self.assertTrue(hook.trace["exact40"])
        self.assertFalse(hook.trace["native_guidance_changed"])
        self.assertEqual(hook.trace["memory_build_count"], 1)

    def test_same_noise_memory_is_built_once_per_step_not_once_per_branch(self) -> None:
        handle = _Handle()
        provider = runtime.VisualMemoryProvider(
            handle=handle,
            source_latent=_latent(),
            source_video_sha256=SHA_A,
            memory_input_kind="same_noise_forward_noised_source",
            scheduler=_Scheduler(),
        )
        epsilon = _latent(100.0)
        provider.bind_official_initial_gaussian(epsilon)
        diffusion = _Diffusion()
        hook = runtime.BranchAwareVisualContextRouteHook(
            diffusion,
            handle=handle,
            target_tokens=8,
            sequence_parallel_rank=3,
            sequence_parallel_size=4,
            source_control_arm="correct",
            target_source_video_sha256=SHA_A,
            memory_provider=provider,
        )
        hook.install()
        try:
            _run_exact40(hook)
        finally:
            hook.restore()
        self.assertEqual(handle.build_calls, 40)
        for start in range(0, 160, 4):
            memories = handle.routes[start : start + 4]
            self.assertEqual(len({id(route.memory) for route in memories}), 1)
        self.assertEqual(len({id(route.memory) for route in handle.routes}), 40)
        self.assertEqual(hook.trace["memory_build_count"], 40)
        self.assertEqual(provider.official_initial_gaussian_sha256, runtime._tensor_sha256(
            epsilon, label="test epsilon"
        ))

    def test_carrier_off_routes_all_branches_but_never_builds_memory(self) -> None:
        handle = _Handle()
        diffusion = _Diffusion()
        hook = runtime.BranchAwareVisualContextRouteHook(
            diffusion,
            handle=handle,
            target_tokens=8,
            sequence_parallel_rank=1,
            sequence_parallel_size=4,
            source_control_arm="carrier-off",
            target_source_video_sha256=SHA_A,
            memory_provider=None,
        )
        hook.install()
        try:
            _run_exact40(hook)
        finally:
            hook.restore()
        self.assertTrue(all(not route.enabled and route.memory is None for route in handle.routes))
        self.assertEqual(handle.build_calls, 0)
        self.assertEqual(hook.trace["memory_build_count"], 0)

    def test_wrong_owner_and_order_labels_are_bound_to_actual_provider(self) -> None:
        handle = _Handle()
        identity_wrong = runtime.VisualMemoryProvider(
            handle=handle,
            source_latent=_latent(1.0),
            source_video_sha256=SHA_B,
            memory_input_kind="clean_source",
            scheduler=_Scheduler(),
        )
        runtime.BranchAwareVisualContextRouteHook(
            _Diffusion(),
            handle=handle,
            target_tokens=8,
            sequence_parallel_rank=0,
            sequence_parallel_size=4,
            source_control_arm="wrong-owner",
            target_source_video_sha256=SHA_A,
            memory_provider=identity_wrong,
        )
        with self.assertRaisesRegex(runtime.CleanSourceVisualContextDecodeError, "distinct"):
            runtime.BranchAwareVisualContextRouteHook(
                _Diffusion(),
                handle=handle,
                target_tokens=8,
                sequence_parallel_rank=0,
                sequence_parallel_size=4,
                source_control_arm="wrong-owner",
                target_source_video_sha256=SHA_A,
                memory_provider=runtime.VisualMemoryProvider(
                    handle=handle,
                    source_latent=_latent(),
                    source_video_sha256=SHA_A,
                    memory_input_kind="clean_source",
                    scheduler=_Scheduler(),
                ),
            )
        reversed_latent = runtime.reverse_latent_phase_order(_latent())
        order = runtime.VisualMemoryProvider(
            handle=handle,
            source_latent=reversed_latent,
            source_video_sha256=SHA_A,
            memory_input_kind="clean_source",
            scheduler=_Scheduler(),
            memory_transform="reverse-phase-order-20-to-0",
        )
        runtime.BranchAwareVisualContextRouteHook(
            _Diffusion(),
            handle=handle,
            target_tokens=8,
            sequence_parallel_rank=0,
            sequence_parallel_size=4,
            source_control_arm="order-permutation",
            target_source_video_sha256=SHA_A,
            memory_provider=order,
        )
        with self.assertRaisesRegex(runtime.CleanSourceVisualContextDecodeError, "reverse"):
            runtime.BranchAwareVisualContextRouteHook(
                _Diffusion(),
                handle=handle,
                target_tokens=8,
                sequence_parallel_rank=0,
                sequence_parallel_size=4,
                source_control_arm="order-permutation",
                target_source_video_sha256=SHA_A,
                memory_provider=identity_wrong,
            )

    def test_rejects_wrong_branch_order_and_restores_instance(self) -> None:
        handle = _Handle()
        provider = runtime.VisualMemoryProvider(
            handle=handle,
            source_latent=_latent(),
            source_video_sha256=SHA_A,
            memory_input_kind="clean_source",
            scheduler=_Scheduler(),
        )
        diffusion = _Diffusion()
        hook = runtime.BranchAwareVisualContextRouteHook(
            diffusion,
            handle=handle,
            target_tokens=8,
            sequence_parallel_rank=0,
            sequence_parallel_size=4,
            source_control_arm="correct",
            target_source_video_sha256=SHA_A,
            memory_provider=provider,
        )
        hook.install()
        with self.assertRaisesRegex(runtime.CleanSourceVisualContextDecodeError, "branch order"):
            with hook.sample():
                diffusion.shared_step(
                    model_id="transformer_1",
                    noisy_latents=object(),
                    timesteps=40.0,
                    cond_embeds=object(),
                    rotary_embs=object(),
                    batch_vae_seqlen=[16],
                    batch_text_seqlen=[1],
                )
        hook.restore()
        self.assertNotIn("shared_step", vars(diffusion))


if __name__ == "__main__":
    unittest.main()
