#!/usr/bin/env python3

from __future__ import annotations

import ast
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import stat
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch required")
class GraftPhaseANativeGPUCanaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch
        import graft_phase_a_native_training_closure_v1 as phase_core
        import identity_rebinder_v1 as rebinder
        import inference_sigma_strata as sigma_strata
        import run_graft_phase_a_native_gpu_canary_v1 as runner

        torch.set_num_threads(1)
        cls.torch = torch
        cls.phase_core = phase_core
        cls.rebinder = rebinder
        cls.sigma_strata = sigma_strata
        cls.runner = runner

    class MomentumBuffer:
        def __init__(self, momentum):
            self.momentum = momentum
            self.running_average = 0

        def update(self, update_value):
            self.running_average = (
                update_value + self.momentum * self.running_average
            )

    @staticmethod
    def normalized_guidance(
        pred_cond,
        pred_uncond,
        guidance_scale,
        momentum_buffer=None,
        eta=1.0,
        norm_threshold=0.0,
    ):
        import torch
        import torch.nn.functional as functional

        difference = pred_cond - pred_uncond
        if momentum_buffer is not None:
            momentum_buffer.update(difference)
            difference = momentum_buffer.running_average
        if norm_threshold > 0:
            ones = torch.ones_like(difference)
            norm = difference.norm(p=2, dim=[-1, -2, -4], keepdim=True)
            difference = difference * torch.minimum(ones, norm_threshold / norm)
        projected, base = difference.double(), pred_cond.double()
        base = functional.normalize(base, dim=[-1, -2, -4])
        parallel = (projected * base).sum(
            dim=[-1, -2, -4], keepdim=True
        ) * base
        orthogonal = projected - parallel
        normalized = orthogonal.to(difference.dtype) + eta * parallel.to(
            difference.dtype
        )
        return pred_uncond + guidance_scale * normalized

    def _fake_classes(self):
        torch = self.torch
        rebinder = self.rebinder
        runner = self.runner

        class FakeTransformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.adapter = torch.nn.Parameter(torch.tensor(0.0))
                self.dtype = torch.bfloat16
                self.patch_log = []

            def patch_vae_latent(self, hidden_states, source_id=None):
                self.patch_log.append((float(source_id), id(hidden_states)))
                batch, channels, phases, height, width = hidden_states.shape
                patches = (
                    hidden_states.reshape(
                        batch,
                        channels,
                        phases,
                        height // 2,
                        2,
                        width // 2,
                        2,
                    )
                    .permute(0, 2, 3, 5, 4, 6, 1)
                    .reshape(
                        batch,
                        phases * (height // 2) * (width // 2),
                        64,
                    )
                )
                seed = patches.mean(dim=-1, keepdim=True)
                tokens = seed.expand(batch, seed.shape[1], 1536).to(
                    torch.bfloat16
                ).contiguous()
                rotary = torch.full(
                    (batch, 1, seed.shape[1], 8),
                    float(source_id),
                    dtype=torch.float32,
                    device=hidden_states.device,
                )
                return tokens, rotary

        class FakeAtlasEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.25))
                self.calls = 0

            def build_atlas(self, source_frames, *, source_video_sha256):
                self.calls += 1
                value = (source_frames.mean() + 1.5) * self.scale
                tokens = value.reshape(1, 1, 1).expand(1, 2, 1536).contiguous()
                construction = runner.object_sha256(
                    {
                        "schema_version": "cpu-fake-atlas-v1",
                        "source_video_sha256": source_video_sha256,
                    }
                )
                return rebinder.IdentityAtlas(
                    tokens=tokens,
                    source_video_sha256=source_video_sha256,
                    source_frame_count=int(source_frames.shape[1]),
                    construction_digest=construction,
                )

        class FakeHandle:
            def __init__(self, transformer):
                self.transformer = transformer
                self.atlas_encoder = FakeAtlasEncoder().eval()

            def build_atlas(self, source_frames, *, source_video_sha256):
                return self.atlas_encoder.build_atlas(
                    source_frames,
                    source_video_sha256=source_video_sha256,
                )

            @contextmanager
            def route(self, route):
                with rebinder.activate_route(route):
                    yield

            def trainable_named_parameters(self):
                return (
                    ("atlas_encoder.scale", self.atlas_encoder.scale),
                    (
                        "blocks.8.attn1.to_out.0.identity_rebinder.output.weight",
                        self.transformer.adapter,
                    ),
                )

        class ExplodingScheduler:
            def __getattribute__(self, name):
                if name.startswith("__"):
                    return object.__getattribute__(self, name)
                raise AssertionError("native closure must not access a scheduler")

        class FakeDiffusion(torch.nn.Module):
            def __init__(self, transformer):
                super().__init__()
                self.transformer = transformer
                self.transformer_2 = None
                self.scheduler = ExplodingScheduler()
                self.call_log = []
                self.detach_pack_replay = False
                self.detach_adapter_replay = False
                self.force_adapter_on_zero_target = False

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
                route = rebinder.active_route()
                if route is None or route.branch_name != "V":
                    raise AssertionError("authenticated V route is absent")
                local_target_rows = int(
                    torch.count_nonzero(
                        route.local_target_selector(device=noisy_latents.device)
                    ).item()
                )
                self.call_log.append(
                    {
                        "model_id": model_id,
                        "visual_id": id(noisy_latents),
                        "timestep_id": id(timesteps),
                        "rotary_id": id(rotary_embs),
                        "condition_id": id(cond_embeds),
                        "atlas_id": id(route.atlas.tokens),
                        "atlas_graph": route.atlas.tokens.requires_grad,
                        "target_tokens": route.target_tokens,
                        "local_target_rows": local_target_rows,
                    }
                )
                replay = len(self.call_log) > 2
                pack = (
                    noisy_latents.detach()
                    if replay and self.detach_pack_replay
                    else noisy_latents
                )
                base = pack[..., :64].float()
                text = cond_embeds[:, :1, :1].float()
                atlas_value = route.atlas.tokens.float().mean()
                selector = torch.cat(
                    (
                        torch.zeros(route.condition_tokens),
                        torch.ones(route.target_tokens),
                    )
                ).reshape(1, -1, 1)
                selector = selector.to(device=base.device, dtype=base.dtype)
                owns_target = local_target_rows > 0
                use_adapter = owns_target or self.force_adapter_on_zero_target
                adapter = self.transformer.adapter
                if not use_adapter or (replay and self.detach_adapter_replay):
                    adapter = adapter.detach()
                value = (
                    base
                    + text * 0.01
                    + selector * adapter * (text + atlas_value)
                )
                return value.to(torch.bfloat16)

        return FakeTransformer, FakeDiffusion, FakeHandle

    def _fixture(self, *, sp_rank=0, sp_size=1):
        torch = self.torch
        runner = self.runner
        phase_core = self.phase_core
        FakeTransformer, FakeDiffusion, FakeHandle = self._fake_classes()
        transformer = FakeTransformer().eval()
        diffusion = FakeDiffusion(transformer).eval()
        handle = FakeHandle(transformer)
        source_sha = hashlib.sha256(b"authenticated-exact81-source").hexdigest()
        source_frames = torch.linspace(
            -1.0,
            1.0,
            steps=81 * 3 * 16 * 16,
            dtype=torch.float32,
        ).reshape(1, 81, 3, 16, 16).contiguous()
        route_factory = runner.FreshAtlasRouteFactory(
            handle=handle,
            source_frames=source_frames,
            source_video_sha256=source_sha,
            sequence_parallel_rank=sp_rank,
            sequence_parallel_size=sp_size,
        )
        bindings = phase_core.authenticate_cpu_test_fakes(
            diffusion=diffusion,
            transformer=transformer,
            vendor_normalized_guidance=self.normalized_guidance,
            momentum_buffer_factory=self.MomentumBuffer,
            named_trainable_parameters=handle.trainable_named_parameters(),
            external_trainable_owner_modules={
                "atlas_encoder": handle.atlas_encoder
            },
            test_name="cpu_fake:phase_a_native_gpu_runner",
            forward_context_factory=route_factory,
        )
        generator = torch.Generator(device="cpu").manual_seed(20260810)
        spatial_width = 4 if sp_size == 4 else 2
        source_latent = torch.randn(
            (1, 16, 21, 2, spatial_width),
            generator=generator,
            dtype=torch.float32,
        )
        noise = runner.keyed_fresh_gaussian(
            shape=source_latent.shape,
            device="cpu",
            source_video_sha256=source_sha,
            cell_id="dog",
            base_seed=913,
        )
        negative = torch.full((1, 2, 4), -1.0, dtype=torch.bfloat16)
        positive = torch.full((1, 2, 4), 2.0, dtype=torch.bfloat16)
        sigma = torch.tensor(
            self.sigma_strata.PINNED_POSITIVE_SIGMAS[33],
            dtype=torch.float32,
        )
        timestep = torch.tensor(
            [self.sigma_strata.PINNED_TIMESTEPS[33]], dtype=torch.int64
        )
        return {
            "transformer": transformer,
            "diffusion": diffusion,
            "handle": handle,
            "bindings": bindings,
            "route_factory": route_factory,
            "source_latent": source_latent,
            "noise": noise,
            "negative": negative,
            "positive": positive,
            "sigma": sigma,
            "timestep": timestep,
        }

    def test_authenticated_cpu_fake_executes_four_fresh_atlas_forwards(self) -> None:
        values = self._fixture()
        before = self.runner.parameter_registry_digest(
            values["bindings"].named_trainable_parameters
        )
        pre_backward_rows = []

        def observe_pre_backward(receipt):
            self.assertTrue(
                all(
                    parameter.grad is None
                    for _, parameter in values[
                        "bindings"
                    ].named_trainable_parameters
                )
            )
            pre_backward_rows.append(receipt)

        result = self.runner.execute_authenticated_local_cell(
            bindings=values["bindings"],
            route_factory=values["route_factory"],
            source_latent=values["source_latent"],
            epsilon=values["noise"].epsilon,
            negative_condition=values["negative"],
            positive_condition=values["positive"],
            sigma=values["sigma"],
            timestep=values["timestep"],
            noise_receipt=values["noise"].receipt,
            pre_backward_observer=observe_pre_backward,
        )
        after = self.runner.parameter_registry_digest(
            values["bindings"].named_trainable_parameters
        )
        self.assertEqual(before, after)
        self.assertEqual(len(pre_backward_rows), 1)
        self.assertFalse(pre_backward_rows[0]["backward_started"])
        self.assertEqual(values["handle"].atlas_encoder.calls, 4)
        self.assertEqual(len({row["atlas_id"] for row in values["diffusion"].call_log}), 4)
        self.assertEqual(
            [row["atlas_graph"] for row in values["diffusion"].call_log],
            [False, False, True, True],
        )
        self.assertEqual(
            result.receipt["phase_core_receipt"]["call_trace"],
            [
                ["measurement", "negative"],
                ["measurement", "positive"],
                ["replay", "negative"],
                ["replay", "positive"],
            ],
        )
        self.assertEqual(
            result.receipt["phase_core_receipt"]["patch_source_ids"],
            [1.0, 0.0],
        )
        self.assertEqual(
            result.receipt["phase_core_receipt"]["pack_layout"],
            "source_id_1_prefix_then_noisy_target_id_0_suffix",
        )
        self.assertEqual(
            result.receipt["phase_core_receipt"]["apg_input_kind"],
            "fresh_detached_fp32_clean_leaves",
        )
        self.assertEqual(
            result.receipt["phase_core_receipt"]["phase_a_objective"],
            "same_source_noop_velocity_mean_mse",
        )
        self.assertTrue(
            result.receipt["phase_core_receipt"]["per_branch_raw_replay_exact"]
            == [True, True]
        )
        self.assertEqual(
            result.receipt["phase_core_receipt"][
                "external_trainable_owner_names"
            ],
            ["atlas_encoder"],
        )
        self.assertIsNotNone(values["transformer"].adapter.grad)
        self.assertIsNotNone(values["handle"].atlas_encoder.scale.grad)
        self.assertNotEqual(float(values["transformer"].adapter.grad.item()), 0.0)
        self.assertEqual(float(values["handle"].atlas_encoder.scale.grad.item()), 0.0)
        self.assertEqual(
            result.receipt["route_factory"]["call_trace"],
            [
                ["measurement", "negative"],
                ["measurement", "positive"],
                ["replay", "negative"],
                ["replay", "positive"],
            ],
        )
        self.assertTrue(
            result.receipt["route_factory"]["replay_atlases_graph_bearing"]
        )
        self.assertFalse(result.receipt["source_retelling_used"])
        self.assertFalse(result.receipt["proposal_selection_used"])
        self.assertTrue(result.receipt["phase_b_only"])
        self.assertEqual(
            result.receipt["phase_b_deferred_features"][
                "source_retelling_paired_captions"
            ],
            "reserved_not_executed",
        )
        for denied in (
            "semantic_success",
            "action_success",
            "quality_success",
            "semantic_action_success",
            "visual_quality_success",
            "beneficial_training_evidence",
            "training_positive",
            "training_run",
            "optimizer_step",
            "parameters_updated",
        ):
            self.assertFalse(result.receipt["authority"][denied])
        owned, encoded = self.runner.own_and_verify_receipt(result.receipt)
        self.assertEqual(owned, result.receipt)
        self.assertEqual(
            hashlib.sha256(encoded).hexdigest(),
            hashlib.sha256(self.runner.canonical_json_bytes(result.receipt)).hexdigest(),
        )

    def _execute_fixture(self, values, *, pre_backward_observer=None):
        return self.runner.execute_authenticated_local_cell(
            bindings=values["bindings"],
            route_factory=values["route_factory"],
            source_latent=values["source_latent"],
            epsilon=values["noise"].epsilon,
            negative_condition=values["negative"],
            positive_condition=values["positive"],
            sigma=values["sigma"],
            timestep=values["timestep"],
            noise_receipt=values["noise"].receipt,
            pre_backward_observer=pre_backward_observer,
        )

    def test_cpu_sp4_ownership_is_exact_0_0_N_N_with_pack_leaf_backward(self) -> None:
        rank_rows = []
        pre_rows = []
        observed_gradients = []
        for rank in range(4):
            values = self._fixture(sp_rank=rank, sp_size=4)
            captured = []
            result = self._execute_fixture(
                values,
                pre_backward_observer=captured.append,
            )
            self.assertEqual(len(captured), 1)
            pre_rows.append(
                {
                    "global_rank": rank,
                    "pre_backward_context": captured[0],
                }
            )
            receipt, encoded = self.runner.own_and_verify_receipt(result.receipt)
            rank_rows.append(
                {
                    "global_rank": rank,
                    "local_receipt": receipt,
                    "canonical_bytes_sha256": hashlib.sha256(encoded).hexdigest(),
                }
            )
            adapter_gradient = values["transformer"].adapter.grad
            observed_gradients.append(
                0.0
                if adapter_gradient is None
                else float(adapter_gradient.detach().abs().sum().item())
            )
            self.assertEqual(
                [
                    row["gradient_finite_nonzero"]
                    for row in receipt["phase_core_receipt"][
                        "per_branch_replay_pack_leaf_gradient"
                    ]
                ],
                [True, True],
            )
        pre_backward = self.runner.validate_world4_pre_backward_contexts(
            pre_rows, cell_id="dog"
        )
        locality = self.runner.validate_world4_locality_receipts(
            rank_rows,
            cell_id="dog",
            pre_backward_world4=pre_backward,
        )
        self.assertEqual(locality["observed_local_target_rows"], [0, 0, 21, 21])
        self.assertEqual(observed_gradients[:2], [0.0, 0.0])
        self.assertTrue(all(value > 0.0 for value in observed_gradients[2:]))
        self.assertTrue(locality["replay_native_pack_leaf_all_ranks"])
        self.assertTrue(locality["zero_target_ranks_adapter_absent_or_zero"])
        self.assertTrue(
            locality["target_ranks_output_projection_local_grad_nonzero"]
        )

    def test_cpu_sp4_attacks_on_pack_leaf_and_adapter_locality_fail_closed(self) -> None:
        zero_owner = self._fixture(sp_rank=0, sp_size=4)
        zero_owner["diffusion"].force_adapter_on_zero_target = True
        with self.assertRaisesRegex(
            self.phase_core.GraftPhaseANativeTrainingClosureError,
            "zero-target rank produced an adapter gradient",
        ):
            self._execute_fixture(zero_owner)

        target_owner = self._fixture(sp_rank=2, sp_size=4)
        target_owner["diffusion"].detach_adapter_replay = True
        with self.assertRaisesRegex(
            self.phase_core.GraftPhaseANativeTrainingClosureError,
            "output-only gradient gate",
        ):
            self._execute_fixture(target_owner)

        detached_pack = self._fixture(sp_rank=2, sp_size=4)
        detached_pack["diffusion"].detach_pack_replay = True
        with self.assertRaisesRegex(
            self.phase_core.GraftPhaseANativeTrainingClosureError,
            "visual-pack leaf gradient",
        ):
            self._execute_fixture(detached_pack)

    def test_world4_locality_receipt_tampering_fails_closed(self) -> None:
        rank_rows = []
        pre_rows = []
        for rank in range(4):
            values = self._fixture(sp_rank=rank, sp_size=4)
            captured = []
            result = self._execute_fixture(
                values,
                pre_backward_observer=captured.append,
            )
            pre_rows.append(
                {
                    "global_rank": rank,
                    "pre_backward_context": captured[0],
                }
            )
            receipt, encoded = self.runner.own_and_verify_receipt(result.receipt)
            rank_rows.append(
                {
                    "global_rank": rank,
                    "local_receipt": receipt,
                    "canonical_bytes_sha256": hashlib.sha256(encoded).hexdigest(),
                }
            )
        tampered = [dict(row) for row in rank_rows]
        tampered_receipt = dict(tampered[0]["local_receipt"])
        tampered_receipt["local_target_rows"] = 1
        tampered[0]["local_receipt"] = tampered_receipt
        pre_backward = self.runner.validate_world4_pre_backward_contexts(
            pre_rows, cell_id="human"
        )
        with self.assertRaisesRegex(
            self.runner.GraftPhaseANativeGPUCanaryError,
            r"\[0,0,N,N\]",
        ):
            self.runner.validate_world4_locality_receipts(
                tampered,
                cell_id="human",
                pre_backward_world4=pre_backward,
            )

    def test_keyed_gaussian_is_deterministic_separated_and_not_source_derived(self) -> None:
        source_sha = hashlib.sha256(b"source").hexdigest()
        first = self.runner.keyed_fresh_gaussian(
            shape=(1, 16, 21, 2, 2),
            device="cpu",
            source_video_sha256=source_sha,
            cell_id="dog",
            base_seed=7,
        )
        second = self.runner.keyed_fresh_gaussian(
            shape=(1, 16, 21, 2, 2),
            device="cpu",
            source_video_sha256=source_sha,
            cell_id="dog",
            base_seed=7,
        )
        separated = self.runner.keyed_fresh_gaussian(
            shape=(1, 16, 21, 2, 2),
            device="cpu",
            source_video_sha256=source_sha,
            cell_id="human",
            base_seed=7,
        )
        self.assertTrue(self.torch.equal(first.epsilon, second.epsilon))
        self.assertEqual(first.receipt, second.receipt)
        self.assertFalse(self.torch.equal(first.epsilon, separated.epsilon))
        self.assertFalse(first.receipt["source_or_target_derived"])
        self.assertEqual(first.receipt["schedule_index"], 33)
        with self.assertRaisesRegex(
            self.runner.GraftPhaseANativeGPUCanaryError, "key fields"
        ):
            self.runner.keyed_fresh_gaussian(
                shape=(1, 16, 21, 2, 2),
                device="cpu",
                source_video_sha256=source_sha,
                cell_id="dog",
                base_seed=7,
                schedule_index=32,
            )

    def test_noisy_target_is_exact_fp32_formula_and_disjoint(self) -> None:
        torch = self.torch
        source = torch.linspace(
            -2.0, 2.0, steps=1 * 16 * 21 * 2 * 2, dtype=torch.float32
        ).reshape(1, 16, 21, 2, 2)
        epsilon = torch.linspace(
            2.0, -2.0, steps=source.numel(), dtype=torch.float32
        ).reshape_as(source)
        sigma = torch.tensor(
            self.sigma_strata.PINNED_POSITIVE_SIGMAS[33], dtype=torch.float32
        )
        noisy, receipt = self.runner.build_noisy_target(
            source, epsilon, sigma=sigma
        )
        expected = ((1.0 - sigma) * source + sigma * epsilon).contiguous()
        self.assertTrue(torch.equal(noisy, expected))
        self.assertFalse(noisy.requires_grad)
        self.assertTrue(receipt["source_epsilon_storage_disjoint"])
        self.assertTrue(receipt["source_noisy_storage_disjoint"])
        self.assertFalse(receipt["target_video_used"])

    def test_canonical_noop_r2v_prompt_pair_is_hash_bound(self) -> None:
        positive, negative, receipt = self.runner.canonical_noop_prompt_contract(
            prompt_cleaner=lambda value: value
        )
        self.assertTrue(positive.endswith(self.runner.route_batches.EXACT_NOOP_INSTRUCTION))
        self.assertEqual(negative, self.runner.legacy.DEFAULT_NEGATIVE_PROMPT)
        self.assertEqual(
            receipt["noop_instruction_utf8_sha256"],
            self.runner.route_batches.EXACT_NOOP_INSTRUCTION_SHA256,
        )
        self.assertEqual(receipt["guidance_mode"], "v2v_apg")
        self.assertTrue(receipt["source_only"])
        self.assertFalse(receipt["action_instruction_used"])

    def test_index33_scheduler_binding_uses_cpu_schedule_and_device_int64(self) -> None:
        torch = self.torch
        strata = self.sigma_strata

        class Scheduler:
            def __init__(self):
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
                self.calls = []
                self.step_index = None

            def set_timesteps(self, steps):
                self.calls.append(steps)
                self.timesteps = torch.tensor(
                    strata.PINNED_TIMESTEPS, dtype=torch.int64, device="cpu"
                )
                self.sigmas = torch.tensor(
                    (*strata.PINNED_POSITIVE_SIGMAS, 0.0),
                    dtype=torch.float32,
                    device="cpu",
                )

        scheduler = Scheduler()
        coordinate = self.runner.bind_active_index33_coordinate(
            scheduler, device="cpu"
        )
        self.assertEqual(scheduler.calls, [40])
        self.assertEqual(coordinate.schedule_index, 33)
        self.assertEqual(coordinate.timestep.dtype, torch.int64)
        self.assertEqual(coordinate.timestep.device.type, "cpu")
        self.assertEqual(
            int(coordinate.timestep.item()), strata.PINNED_TIMESTEPS[33]
        )
        self.assertFalse(coordinate.coordinate_receipt["scheduler_step_called"])
        self.runner.assert_scheduler_unchanged(scheduler, coordinate)
        scheduler.sigmas[33] += 1.0
        with self.assertRaisesRegex(
            self.runner.GraftPhaseANativeGPUCanaryError, "state changed"
        ):
            self.runner.assert_scheduler_unchanged(scheduler, coordinate)

    def test_receipt_digest_is_immediately_owned_and_tampering_fails(self) -> None:
        receipt = self.runner.seal_mapping(
            {"schema_version": "test-receipt-v1", "pass": True}
        )
        owned, encoded = self.runner.own_and_verify_receipt(receipt)
        self.assertEqual(json.loads(encoded.decode("ascii")), owned)
        unsigned = dict(owned)
        digest = unsigned.pop("digest")
        self.assertEqual(digest, self.runner.object_sha256(unsigned))
        tampered = dict(owned)
        tampered["pass"] = False
        with self.assertRaisesRegex(
            self.runner.GraftPhaseANativeGPUCanaryError, "digest differs"
        ):
            self.runner.own_and_verify_receipt(tampered)

    def test_receipt_publication_is_create_only_mode_0444(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "phase-a-canary"
            descriptor, identity = self.runner.create_output_directory(output)
            try:
                receipt = self.runner.seal_mapping(
                    {"schema_version": "publication-test-v1", "pass": True}
                )
                path = output / "receipt.json"
                self.runner.write_receipt_create_only(
                    path,
                    receipt,
                    directory_fd=descriptor,
                    expected_directory_identity=identity,
                )
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o444)
                self.assertEqual(
                    path.read_bytes(),
                    self.runner.canonical_json_bytes(receipt) + b"\n",
                )
                with self.assertRaises(FileExistsError):
                    self.runner.write_receipt_create_only(
                        path,
                        receipt,
                        directory_fd=descriptor,
                        expected_directory_identity=identity,
                    )
            finally:
                os.close(descriptor)

    def test_world4_parity_helper_is_fail_closed(self) -> None:
        row = {"tensor_sha256": "a" * 64, "receipt_digest": "b" * 64}
        self.assertEqual(
            self.runner.require_equal_rows([row, row, row, row], label="test"),
            [row, row, row, row],
        )
        with self.assertRaisesRegex(
            self.runner.GraftPhaseANativeGPUCanaryError,
            "differs across ranks",
        ):
            self.runner.require_equal_rows(
                [row, row, row, {**row, "tensor_sha256": "c" * 64}],
                label="test",
            )

    def test_zero_initialized_gradient_gate_is_output_only(self) -> None:
        receipt = {
            "rows": [
                {
                    "name": "atlas_encoder.patchifier.weight",
                    "l2_float64_hex": 0.0.hex(),
                },
                {
                    "name": (
                        "blocks.8.attn1.to_out.0.identity_rebinder.query.weight"
                    ),
                    "l2_float64_hex": 0.0.hex(),
                },
                {
                    "name": (
                        "blocks.8.attn1.to_out.0.identity_rebinder.output.weight"
                    ),
                    "l2_float64_hex": 0.75.hex(),
                },
            ]
        }
        gate = self.runner.zero_initialized_gradient_gate(receipt)
        self.assertEqual(gate["gate"], "output_projection_only_nonzero")
        self.assertTrue(gate["external_atlas_encoder_exact_zero"])
        bad = {"rows": [dict(row) for row in receipt["rows"]]}
        bad["rows"][0]["l2_float64_hex"] = 0.25.hex()
        with self.assertRaisesRegex(
            self.runner.GraftPhaseANativeGPUCanaryError,
            "output-only",
        ):
            self.runner.zero_initialized_gradient_gate(bad)

    def test_runner_surface_is_independent_and_has_no_training_controls(self) -> None:
        path = Path(self.runner.__file__).resolve()
        source = path.read_text(encoding="utf-8")
        self.assertEqual(source.count('"checkpoint tree differs"'), 1)
        tree = ast.parse(source)
        imported = set()
        called_attributes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                called_attributes.append(node.func.attr)
        self.assertNotIn("run_identity_rebinder_gpu_structural_canary_v1", imported)
        self.assertFalse(any("ssft" in name.lower() for name in imported))
        self.assertNotIn("step", called_attributes)
        destinations = {action.dest for action in self.runner.build_parser()._actions}
        self.assertNotIn("schedule_index", destinations)
        self.assertNotIn("optimizer", destinations)
        self.assertNotIn("learning_rate", destinations)
        acknowledgement = next(
            action
            for action in self.runner.build_parser()._actions
            if action.dest == "ack_wiring_fm_gradient_only_no_training_claim"
        )
        self.assertFalse(acknowledgement.default)
        route = self.runner.route_capability_receipt()
        self.assertEqual(route["route_kind"], "identity_rebinder_v1")
        self.assertEqual(route["branch_name"], "V")
        self.assertEqual(
            route["phase_a_active_schedule_indices"],
            list(self.phase_core.PHASE_A_ACTIVE_SCHEDULE_INDICES),
        )
        self.assertEqual(route["canary_executed_schedule_indices"], [33])
        self.assertFalse(route["other_active_envelope_indices_executed"])


if __name__ == "__main__":
    unittest.main()
