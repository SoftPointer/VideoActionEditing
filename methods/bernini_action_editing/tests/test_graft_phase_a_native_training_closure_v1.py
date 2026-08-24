#!/usr/bin/env python3

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import importlib.util
import inspect
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import graft_phase_a_native_training_closure_v1 as closure  # noqa: E402


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch required")
class GraftPhaseANativeTrainingClosureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch
        torch.set_num_threads(1)
        cls.torch = torch

    class MomentumBuffer:
        def __init__(self, momentum):
            self.momentum = momentum
            self.running_average = 0

        def update(self, update_value):
            self.running_average = update_value + self.momentum * self.running_average

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

        diff = pred_cond - pred_uncond
        if momentum_buffer is not None:
            momentum_buffer.update(diff)
            diff = momentum_buffer.running_average
        if norm_threshold > 0:
            ones = torch.ones_like(diff)
            diff_norm = diff.norm(p=2, dim=[-1, -2, -4], keepdim=True)
            diff = diff * torch.minimum(ones, norm_threshold / diff_norm)
        projected, base = diff.double(), pred_cond.double()
        base = functional.normalize(base, dim=[-1, -2, -4])
        parallel = (projected * base).sum(
            dim=[-1, -2, -4], keepdim=True
        ) * base
        orthogonal = projected - parallel
        normalized = orthogonal.to(diff.dtype) + eta * parallel.to(diff.dtype)
        return pred_uncond + guidance_scale * normalized

    def _fake_classes(self):
        torch = self.torch

        class FakeTransformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.adapter = torch.nn.Parameter(torch.tensor(0.125))
                self.dtype = torch.bfloat16
                self.patch_log = []
                self.bad_patch_dtype = False
                self.route_active = True
                self.zero_negative_gradient = False

            def patch_vae_latent(self, hidden_states, source_id=None):
                self.patch_log.append((float(source_id), id(hidden_states)))
                batch, channels, phases, height, width = hidden_states.shape
                patches = (
                    hidden_states.reshape(
                        batch, channels, phases, height // 2, 2, width // 2, 2
                    )
                    .permute(0, 2, 3, 5, 4, 6, 1)
                    .reshape(batch, phases * (height // 2) * (width // 2), 64)
                )
                token_seed = patches.mean(dim=-1, keepdim=True)
                tokens = token_seed.expand(batch, token_seed.shape[1], 1536).contiguous()
                if self.bad_patch_dtype:
                    tokens = tokens.float()
                rotary = torch.full(
                    (batch, 1, token_seed.shape[1], 8),
                    float(source_id),
                    dtype=torch.float32,
                    device=hidden_states.device,
                )
                return tokens, rotary

        class FakeDiffusion(torch.nn.Module):
            def __init__(self, transformer):
                super().__init__()
                self.transformer = transformer
                self.transformer_2 = None
                self.call_log = []
                self.bad_measurement_dtype = False
                self.perturb_replay = False
                self.detach_replay = False
                self.mutate_pack = False
                self.mutate_parameter = False
                self.scheduler = ExplodingScheduler()

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
                self.call_log.append(
                    {
                        "model_id": model_id,
                        "visual_id": id(noisy_latents),
                        "timestep_id": id(timesteps),
                        "rotary_id": id(rotary_embs),
                        "condition_id": id(cond_embeds),
                        "vae_seqlen": tuple(batch_vae_seqlen),
                        "text_seqlen": tuple(batch_text_seqlen),
                    }
                )
                call_index = len(self.call_log) - 1
                if self.mutate_pack:
                    with torch.no_grad():
                        noisy_latents.add_(1)
                if self.mutate_parameter:
                    with torch.no_grad():
                        self.transformer.adapter.add_(0.25)
                base = noisy_latents[..., :64].float()
                text = cond_embeds[:, :1, :1].float()
                adapter = self.transformer.adapter
                if not self.transformer.route_active:
                    adapter = adapter.detach()
                elif self.transformer.zero_negative_gradient and float(text.item()) < 0:
                    adapter = adapter.detach() + adapter * 0.0
                value = base * (1.0 + adapter) + (text * adapter)
                if self.perturb_replay and call_index >= 2:
                    value = value + 1.0
                value = value.to(torch.bfloat16)
                if self.bad_measurement_dtype and call_index < 2:
                    value = value.float()
                if self.detach_replay and call_index >= 2:
                    value = value.detach()
                return value

        class ExplodingScheduler:
            def __getattribute__(self, name):
                if name.startswith("__"):
                    return object.__getattribute__(self, name)
                raise AssertionError("closure must not access a scheduler")

        return FakeTransformer, FakeDiffusion

    def _fixture(
        self,
        *,
        schedule_index=33,
        forward_context_factory=None,
        spatial_width=2,
    ):
        torch = self.torch
        FakeTransformer, FakeDiffusion = self._fake_classes()
        transformer = FakeTransformer().eval()
        diffusion = FakeDiffusion(transformer).eval()
        bindings = closure.authenticate_cpu_test_fakes(
            diffusion=diffusion,
            transformer=transformer,
            vendor_normalized_guidance=self.normalized_guidance,
            momentum_buffer_factory=self.MomentumBuffer,
            named_trainable_parameters=(("adapter", transformer.adapter),),
            external_trainable_owner_modules={},
            test_name="cpu_fake:unittest",
            forward_context_factory=forward_context_factory,
        )
        generator = torch.Generator(device="cpu").manual_seed(814)
        source = torch.randn(
            (1, 16, 21, 2, spatial_width),
            generator=generator,
            dtype=torch.float32,
        )
        noisy = torch.randn(
            (1, 16, 21, 2, spatial_width),
            generator=generator,
            dtype=torch.float32,
        )
        negative = torch.full(
            (1, 2, 4), -1.0, dtype=torch.bfloat16
        )
        positive = torch.full(
            (1, 2, 4), 2.0, dtype=torch.bfloat16
        )
        sigma = torch.tensor(
            closure.sigma_strata.PINNED_POSITIVE_SIGMAS[schedule_index],
            dtype=torch.float32,
        )
        timestep = torch.tensor(
            [closure.sigma_strata.PINNED_TIMESTEPS[schedule_index]],
            dtype=torch.int64,
        )
        session = closure.PhaseANativeTrainingClosure(
            bindings=bindings,
            source_video=source,
            noisy_target=noisy,
            negative_condition=negative,
            positive_condition=positive,
            schedule_index=schedule_index,
            sigma=sigma,
            timestep=timestep,
        )
        return {
            "transformer": transformer,
            "diffusion": diffusion,
            "bindings": bindings,
            "source": source,
            "noisy": noisy,
            "negative": negative,
            "positive": positive,
            "sigma": sigma,
            "timestep": timestep,
            "session": session,
        }

    @staticmethod
    def _pack_reference(spatial):
        batch, channels, phases, height, width = spatial.shape
        return (
            spatial.reshape(batch, channels, phases, 1, height // 2, 2, width // 2, 2)
            .permute(0, 2, 4, 6, 3, 5, 7, 1)
            .reshape(batch, phases * (height // 2) * (width // 2), 64)
            .contiguous()
        )

    def test_happy_path_closes_exact_four_call_chain(self) -> None:
        values = self._fixture()
        session = values["session"]
        measurement = session.measure()
        self.assertEqual(measurement.negative_full_raw.dtype, self.torch.bfloat16)
        self.assertEqual(tuple(measurement.negative_full_raw.shape), (1, 42, 64))
        pre_backward = session.forward_context_observation_receipt()
        self.assertTrue(pre_backward["measurement_complete"])
        self.assertFalse(pre_backward["backward_started"])
        self.assertFalse(pre_backward["adapter_graph_bearing"])
        self.assertEqual(pre_backward["local_target_rows"], 21)
        vjp = session.derive_phase_a_flow_matching_vjp()
        self.assertEqual(vjp.guided_clean.dtype, self.torch.float32)
        result = session.replay_and_backward()
        self.assertEqual(session.phase, "closed")
        self.assertEqual(
            session.call_trace,
            (
                ("measurement", "negative"),
                ("measurement", "positive"),
                ("replay", "negative"),
                ("replay", "positive"),
            ),
        )
        self.assertIsNotNone(values["transformer"].adapter.grad)
        self.assertNotEqual(float(values["transformer"].adapter.grad.item()), 0.0)
        receipt = result.receipt
        self.assertTrue(receipt["same_pack_timestep_rotary_objects_all_four_forwards"])
        self.assertEqual(receipt["patch_source_ids"], [1.0, 0.0])
        self.assertEqual(receipt["raw_output_dtype"], "torch.bfloat16")
        self.assertEqual(receipt["per_branch_raw_replay_exact"], [True, True])
        self.assertTrue(receipt["replay_visual_pack_detached_leaf"])
        self.assertTrue(receipt["replay_pack_gradient_cleared_after_each_branch"])
        self.assertEqual(
            [
                row["gradient_finite_nonzero"]
                for row in receipt["per_branch_replay_pack_leaf_gradient"]
            ],
            [True, True],
        )
        self.assertFalse(receipt["scheduler_step_called"])
        self.assertFalse(receipt["packed_raw_to_apg_registry_chain_verified_by_this_core"])
        self.assertFalse(receipt["official_cuda_closure_verified_by_this_core"])
        self.assertFalse(receipt["training_quality_claim_authorized"])
        self.assertEqual(
            receipt["phase_a_objective"],
            "same_source_noop_velocity_mean_mse",
        )
        self.assertEqual(
            receipt["flow_matching_loss_formula"],
            "mean((v_pred-v_target)**2)",
        )
        self.assertFalse(receipt["external_guided_clean_cotangent_accepted"])
        self.assertTrue(receipt["oracle_inputs_absent_by_public_api"])
        self.assertEqual(receipt["forbidden_oracle_api_parameters_present"], [])
        self.assertEqual(
            receipt["objective_derive_api_parameters"], ["self"]
        )
        self.assertEqual(
            set(receipt["input_tensor_sha256"]),
            {
                "source_video",
                "noisy_target",
                "negative_condition",
                "positive_condition",
                "sigma",
                "timestep",
            },
        )
        for key in (
            "guided_clean_sha256",
            "flow_matching_loss_sha256",
            "guided_clean_cotangent_sha256",
            "negative_clean_cotangent_sha256",
            "positive_clean_cotangent_sha256",
            "negative_raw_cotangent_sha256",
            "positive_raw_cotangent_sha256",
        ):
            self.assertEqual(len(receipt[key]), 64)

        logs = values["diffusion"].call_log
        self.assertEqual(len(logs), 4)
        self.assertEqual(len({row["visual_id"] for row in logs}), 1)
        self.assertEqual(len({row["rotary_id"] for row in logs}), 1)
        self.assertEqual(len({row["timestep_id"] for row in logs}), 1)
        self.assertEqual(logs[0]["condition_id"], logs[2]["condition_id"])
        self.assertEqual(logs[1]["condition_id"], logs[3]["condition_id"])
        self.assertEqual(
            [row[0] for row in values["transformer"].patch_log], [1.0, 0.0]
        )

    def test_inactive_high_sigma_cell_is_exact_zero_update_not_trained(self) -> None:
        values = self._fixture(schedule_index=0)
        values["transformer"].route_active = False
        values["session"].measure()
        values["session"].derive_phase_a_flow_matching_vjp()
        result = values["session"].replay_and_backward()
        self.assertIsNone(values["transformer"].adapter.grad)
        self.assertFalse(result.receipt["schedule_cell_active_for_training"])
        self.assertFalse(result.receipt["schedule_cell_counted_as_trained"])
        self.assertFalse(result.receipt["replay_backward_applied"])
        self.assertEqual(
            result.receipt["per_branch_gradient_delta_l2"],
            {"negative": 0.0, "positive": 0.0},
        )

    def test_each_target_owning_replay_requires_output_gradient(self) -> None:
        values = self._fixture(schedule_index=33)
        values["transformer"].zero_negative_gradient = True
        values["session"].measure()
        values["session"].derive_phase_a_flow_matching_vjp()
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError,
            "output-only gradient gate",
        ):
            values["session"].replay_and_backward()
        self.assertIsNone(values["transformer"].adapter.grad)

    def test_convenience_entrypoint_runs_the_same_closed_protocol(self) -> None:
        values = self._fixture(schedule_index=33)
        result = closure.execute_phase_a_native_training_closure(
            bindings=values["bindings"],
            source_video=values["source"],
            noisy_target=values["noisy"],
            negative_condition=values["negative"],
            positive_condition=values["positive"],
            schedule_index=33,
            sigma=values["sigma"],
            timestep=values["timestep"],
        )
        self.assertEqual(result.receipt["schedule_index"], 33)
        self.assertEqual(result.receipt["call_trace"][0], ["measurement", "negative"])

    def test_authenticated_route_receives_the_exact_native_coordinate(self) -> None:
        rows = []

        @contextmanager
        def route(*, request):
            rows.append((request, self.torch.is_grad_enabled()))
            selector = self.torch.cat(
                (
                    self.torch.zeros(
                        request.condition_tokens, dtype=self.torch.bool
                    ),
                    self.torch.ones(
                        request.target_tokens, dtype=self.torch.bool
                    ),
                )
            )
            yield closure.build_native_forward_context_observation(
                request=request,
                sequence_parallel_rank=0,
                sequence_parallel_size=1,
                local_target_selector=selector,
                route_gate=1.0,
                adapter_graph_bearing=(
                    request.phase == "replay" and self.torch.is_grad_enabled()
                ),
            )

        values = self._fixture(forward_context_factory=route)
        values["session"].measure()
        values["session"].derive_phase_a_flow_matching_vjp()
        result = values["session"].replay_and_backward()
        self.assertEqual(
            [(row.phase, row.role) for row, _ in rows],
            [
                ("measurement", "negative"),
                ("measurement", "positive"),
                ("replay", "negative"),
                ("replay", "positive"),
            ],
        )
        self.assertEqual(len({id(row.visual_pack) for row, _ in rows}), 1)
        self.assertEqual(len({id(row.rotary_pack) for row, _ in rows}), 1)
        self.assertEqual(len({id(row.timestep) for row, _ in rows}), 1)
        self.assertTrue(
            all(row.condition_tokens == row.target_tokens == 21 for row, _ in rows)
        )
        self.assertEqual([enabled for _, enabled in rows], [False, False, True, True])
        self.assertTrue(result.receipt["forward_route_context_opened_per_forward"])

    def test_cpu_sp4_context_and_pack_leaf_close_0_0_N_N_locality(self) -> None:
        ownership = []
        selector_digests = []
        gradient_nonzero = []
        for rank in range(4):
            @contextmanager
            def route(*, request, _rank=rank):
                local_rows = (request.total_tokens + 3) // 4
                global_selector = self.torch.cat(
                    (
                        self.torch.zeros(
                            request.condition_tokens, dtype=self.torch.bool
                        ),
                        self.torch.ones(
                            request.target_tokens, dtype=self.torch.bool
                        ),
                        self.torch.zeros(
                            local_rows * 4 - request.total_tokens,
                            dtype=self.torch.bool,
                        ),
                    )
                )
                selector = global_selector[
                    _rank * local_rows : (_rank + 1) * local_rows
                ].contiguous()
                local_targets = int(self.torch.count_nonzero(selector).item())
                yield closure.build_native_forward_context_observation(
                    request=request,
                    sequence_parallel_rank=_rank,
                    sequence_parallel_size=4,
                    local_target_selector=selector,
                    route_gate=1.0,
                    adapter_graph_bearing=(
                        request.phase == "replay" and local_targets > 0
                    ),
                )

            values = self._fixture(
                forward_context_factory=route,
                spatial_width=4,
            )
            values["transformer"].route_active = rank >= 2
            result = closure.execute_phase_a_native_training_closure(
                bindings=values["bindings"],
                source_video=values["source"],
                noisy_target=values["noisy"],
                negative_condition=values["negative"],
                positive_condition=values["positive"],
                schedule_index=33,
                sigma=values["sigma"],
                timestep=values["timestep"],
            )
            receipt = result.receipt
            ownership.append(receipt["local_target_rows"])
            selector_digests.append(
                receipt["forward_context_observations"][0][
                    "local_target_selector_sha256"
                ]
            )
            gradient_nonzero.append(
                receipt["trainable_registry_final_gradient_nonzero"]
            )
            self.assertEqual(
                [
                    row["gradient_finite_nonzero"]
                    for row in receipt["per_branch_replay_pack_leaf_gradient"]
                ],
                [True, True],
            )
            self.assertTrue(
                receipt["replay_pack_gradient_cleared_after_each_branch"]
            )
        self.assertEqual(ownership, [0, 0, 21, 21])
        self.assertEqual(gradient_nonzero, [False, False, True, True])
        self.assertEqual(selector_digests[0], selector_digests[1])
        self.assertEqual(selector_digests[2], selector_digests[3])
        self.assertNotEqual(selector_digests[0], selector_digests[2])

    def test_forward_context_observation_forgery_fails_closed(self) -> None:
        @contextmanager
        def forged_route(*, request):
            selector = self.torch.cat(
                (
                    self.torch.zeros(
                        request.condition_tokens, dtype=self.torch.bool
                    ),
                    self.torch.ones(
                        request.target_tokens, dtype=self.torch.bool
                    ),
                )
            )
            valid = closure.build_native_forward_context_observation(
                request=request,
                sequence_parallel_rank=0,
                sequence_parallel_size=1,
                local_target_selector=selector,
                route_gate=1.0,
                adapter_graph_bearing=False,
            )
            yield replace(valid, local_target_selector_sha256="0" * 64)

        values = self._fixture(forward_context_factory=forged_route)
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError,
            "fields differ from recomputation",
        ):
            values["session"].measure()

    def test_cpu_sp4_adapter_locality_attacks_fail_closed(self) -> None:
        def fixture_for_rank(rank, *, route_active):
            @contextmanager
            def route(*, request):
                local_rows = (request.total_tokens + 3) // 4
                selector = self.torch.cat(
                    (
                        self.torch.zeros(
                            request.condition_tokens, dtype=self.torch.bool
                        ),
                        self.torch.ones(
                            request.target_tokens, dtype=self.torch.bool
                        ),
                    )
                )[rank * local_rows : (rank + 1) * local_rows].contiguous()
                local_targets = int(self.torch.count_nonzero(selector).item())
                yield closure.build_native_forward_context_observation(
                    request=request,
                    sequence_parallel_rank=rank,
                    sequence_parallel_size=4,
                    local_target_selector=selector,
                    route_gate=1.0,
                    adapter_graph_bearing=(
                        request.phase == "replay" and local_targets > 0
                    ),
                )

            values = self._fixture(
                forward_context_factory=route,
                spatial_width=4,
            )
            values["transformer"].route_active = route_active
            return values

        zero_target_attack = fixture_for_rank(0, route_active=True)
        zero_target_attack["session"].measure()
        zero_target_attack["session"].derive_phase_a_flow_matching_vjp()
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError,
            "zero-target rank produced an adapter gradient",
        ):
            zero_target_attack["session"].replay_and_backward()

        target_detach_attack = fixture_for_rank(2, route_active=False)
        target_detach_attack["session"].measure()
        target_detach_attack["session"].derive_phase_a_flow_matching_vjp()
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError,
            "output-only gradient gate",
        ):
            target_detach_attack["session"].replay_and_backward()

    def test_wan_unpack_is_exact_roundtrip_and_preserves_graph(self) -> None:
        torch = self.torch
        spatial_leaf = torch.arange(
            1 * 16 * 21 * 4 * 6, dtype=torch.float32, requires_grad=True
        ).reshape(1, 16, 21, 4, 6)
        packed = self._pack_reference(spatial_leaf)
        restored = closure.unpack_wan_target_velocity(
            packed, spatial_shape=spatial_leaf.shape
        )
        self.assertTrue(torch.equal(restored, spatial_leaf))
        restored.sum().backward()
        self.assertIsNotNone(spatial_leaf.grad_fn)

    def test_fixed_fm_objective_matches_direct_full_graph_bit_exactly(self) -> None:
        torch = self.torch
        values = self._fixture(schedule_index=20)
        measured = values["session"].measure()
        observed = values["session"].derive_phase_a_flow_matching_vjp()
        # The pinned vendor order multiplies the BF16 raw velocity by its
        # FP32 scalar sigma *before* subtracting from FP32 x_t.  Keep the raw
        # leaves BF16 here and compare the effective cotangent delivered to a
        # BF16 replay, not an artificial all-FP32 rewrite.
        negative_raw = measured.negative_spatial_raw.clone().requires_grad_(True)
        positive_raw = measured.positive_spatial_raw.clone().requires_grad_(True)
        momentum = self.MomentumBuffer(0.0)
        expected_guided = self.normalized_guidance(
            pred_cond=values["noisy"] - values["sigma"] * positive_raw,
            pred_uncond=values["noisy"] - values["sigma"] * negative_raw,
            guidance_scale=4.0,
            momentum_buffer=momentum,
            eta=0.5,
            norm_threshold=50.0,
        )
        expected_predicted_velocity = (
            values["noisy"] - expected_guided
        ) / values["sigma"]
        expected_target_velocity = (
            values["noisy"] - values["source"]
        ) / values["sigma"]
        expected_loss = torch.mean(
            (expected_predicted_velocity - expected_target_velocity) ** 2
        )
        (expected_guided_cotangent,) = torch.autograd.grad(
            expected_loss, (expected_guided,), retain_graph=True
        )
        expected = torch.autograd.grad(
            expected_guided,
            (negative_raw, positive_raw),
            grad_outputs=expected_guided_cotangent,
        )
        torch.testing.assert_close(
            observed.guided_clean, expected_guided.detach(), atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            observed.predicted_velocity,
            expected_predicted_velocity.detach(),
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            observed.same_source_target_velocity,
            expected_target_velocity,
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            observed.flow_matching_loss,
            expected_loss.detach(),
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            observed.guided_clean_cotangent,
            expected_guided_cotangent,
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            observed.negative_raw_cotangent.to(torch.bfloat16),
            expected[0],
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            observed.positive_raw_cotangent.to(torch.bfloat16),
            expected[1],
            atol=0.0,
            rtol=0.0,
        )

    def test_returned_measurement_and_vjp_are_non_authoritative_copies(self) -> None:
        values = self._fixture()
        exported_measurement = values["session"].measure()
        exported_measurement.negative_clean.add_(1000.0)
        exported_vjp = values["session"].derive_phase_a_flow_matching_vjp()
        exported_vjp.negative_raw_cotangent.zero_()
        result = values["session"].replay_and_backward()
        self.assertTrue(result.receipt["trainable_registry_final_gradient_nonzero"])
        self.assertEqual(values["session"].phase, "closed")

    def test_test_binding_is_permanently_nonofficial(self) -> None:
        values = self._fixture()
        receipt = values["bindings"].receipt()
        self.assertTrue(receipt["test_only"])
        self.assertFalse(receipt["official_pinned_code"])
        self.assertFalse(receipt["file_hashes_verified"])
        self.assertFalse(receipt["scientific_claim_authorized"])
        self.assertEqual(receipt["trainable_scope_claim"], "exact_registry_closure")
        self.assertFalse(receipt["python_same_process_security_boundary"])

    def test_real_handle_style_external_owner_registry_is_exactly_closed(self) -> None:
        values = self._fixture()
        torch = self.torch

        class FakeHandle:
            def __init__(self, transformer):
                self.transformer = transformer
                self.atlas_encoder = torch.nn.Linear(1, 1, bias=False).eval()

            def trainable_named_parameters(self):
                return (
                    ("atlas_encoder.proj.weight", self.atlas_encoder.weight),
                    (
                        "blocks.8.attn1.to_out.0.identity_rebinder.output.weight",
                        self.transformer.adapter,
                    ),
                )

        handle = FakeHandle(values["transformer"])
        bindings = closure.authenticate_cpu_test_fakes(
            diffusion=values["diffusion"],
            transformer=values["transformer"],
            vendor_normalized_guidance=self.normalized_guidance,
            momentum_buffer_factory=self.MomentumBuffer,
            named_trainable_parameters=handle.trainable_named_parameters(),
            external_trainable_owner_modules={
                "atlas_encoder": handle.atlas_encoder
            },
            test_name="cpu_fake:real_handle_registry",
        )
        bindings.assert_live()
        self.assertEqual(
            [name for name, _ in bindings.external_trainable_owner_modules],
            ["atlas_encoder"],
        )
        self.assertEqual(
            bindings.receipt()["trainable_owner_names"],
            ["diffusion", "transformer", "atlas_encoder"],
        )
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError, "trainable scope"
        ):
            closure.authenticate_cpu_test_fakes(
                diffusion=values["diffusion"],
                transformer=values["transformer"],
                vendor_normalized_guidance=self.normalized_guidance,
                momentum_buffer_factory=self.MomentumBuffer,
                named_trainable_parameters=handle.trainable_named_parameters(),
                external_trainable_owner_modules={},
                test_name="cpu_fake:missing_external_owner",
            )
        handle.atlas_encoder.train()
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError, "callable/eval"
        ):
            bindings.assert_live()

    def test_binding_public_construction_replace_and_forged_token_fail(self) -> None:
        values = self._fixture()
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError, "no public constructor"
        ):
            closure.AuthenticatedNativeBindings()
        with self.assertRaises(
            (TypeError, ValueError, closure.GraftPhaseANativeTrainingClosureError)
        ):
            replace(values["bindings"], binding_label="forged")
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError, "mint token differs"
        ):
            closure.AuthenticatedNativeBindings._mint(token=object())  # noqa: SLF001

        object.__setattr__(values["bindings"], "binding_label", "cpu_fake:forged")
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError, "receipt fields"
        ):
            values["bindings"].assert_live()

    def test_live_guidance_momentum_and_route_receipt_are_bound(self) -> None:
        values = self._fixture()
        object.__setattr__(
            values["bindings"], "vendor_normalized_guidance", lambda **_kwargs: None
        )
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError, "callable/eval"
        ):
            values["bindings"].assert_live()

        values = self._fixture()
        object.__setattr__(
            values["bindings"], "momentum_buffer_factory", lambda momentum: None
        )
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError, "callable/eval"
        ):
            values["bindings"].assert_live()

        values = self._fixture()
        bad_route_receipt = closure._seal(  # noqa: SLF001 - hostile regression
            {
                "schema_version": closure.FORWARD_ROUTE_SCHEMA_VERSION,
                "route_kind": "cpu_fake_global_context",
                "phase_a_active_schedule_indices": [39],
                "inactive_schedule_policy": "exact_zero_update_not_trained",
                "target_queries_only": True,
                "condition_rows_written": False,
                "external_oracle_inputs": False,
            }
        )
        object.__setattr__(
            values["bindings"], "forward_route_receipt", bad_route_receipt
        )
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError, "critical fields"
        ):
            values["bindings"].assert_live()

        arbitrary_digest = dict(bad_route_receipt)
        arbitrary_digest["digest"] = "0" * 64
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError, "digest differs"
        ):
            closure._validated_route_receipt(  # noqa: SLF001
                arbitrary_digest, expected_route_kind="cpu_fake_global_context"
            )

    def test_tensor_hash_and_full_closure_do_not_require_numpy_bridge(self) -> None:
        def forbidden_numpy(_tensor):
            raise AssertionError("tensor.numpy() must not be called")

        with mock.patch.object(self.torch.Tensor, "numpy", new=forbidden_numpy):
            first = closure._tensor_bytes_sha256(  # noqa: SLF001 - regression
                self.torch.tensor(0.125, dtype=self.torch.float32)
            )
            second = closure._tensor_bytes_sha256(  # noqa: SLF001 - regression
                self.torch.tensor(0.125, dtype=self.torch.float32)
            )
            self.assertEqual(first, second)
            values = self._fixture()
            result = closure.execute_phase_a_native_training_closure(
                bindings=values["bindings"],
                source_video=values["source"],
                noisy_target=values["noisy"],
                negative_condition=values["negative"],
                positive_condition=values["positive"],
                schedule_index=33,
                sigma=values["sigma"],
                timestep=values["timestep"],
            )
        self.assertEqual(result.receipt["raw_output_dtype"], "torch.bfloat16")

    def test_tensor_hash_rejects_tensor_subclass_detach_and_uses_no_native_pointer(self) -> None:
        class TensorSubclass(self.torch.Tensor):
            pass

        value = self.torch.tensor([1.0], dtype=self.torch.float32).as_subclass(
            TensorSubclass
        )
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError, "exact torch.Tensor"
        ):
            closure._tensor_bytes_sha256(value)  # noqa: SLF001
        source = Path(closure.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import ctypes", source)
        self.assertNotIn("ctypes.string_at", source)
        self.assertIn("bytes(storage)", source)

    def test_public_surface_has_no_forbidden_oracle_or_solver_inputs(self) -> None:
        names = set(inspect.signature(closure.execute_phase_a_native_training_closure).parameters)
        for forbidden in (
            "target_video",
            "mask",
            "pose",
            "track",
            "flow",
            "donor",
            "scheduler",
            "solver_state",
            "guided_clean_cotangent",
        ):
            self.assertNotIn(forbidden, names)
        derive_names = set(
            inspect.signature(
                closure.PhaseANativeTrainingClosure.derive_phase_a_flow_matching_vjp
            ).parameters
        )
        self.assertEqual(derive_names, {"self"})
        self.assertFalse(
            hasattr(closure.PhaseANativeTrainingClosure, "derive_apg_leaf_vjp")
        )
        production_auth_names = set(
            inspect.signature(closure.authenticate_pinned_native_bindings).parameters
        )
        self.assertIn("forward_route_receipt", production_auth_names)
        self.assertNotIn("forward_route_receipt_digest", production_auth_names)
        self.assertIn("external_trainable_owner_modules", production_auth_names)

    def test_wrong_exact81_phase_count_fails_before_patch(self) -> None:
        values = self._fixture()
        bad = self.torch.zeros((1, 16, 11, 2, 2), dtype=self.torch.float32)
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError, "exact81"
        ):
            closure.PhaseANativeTrainingClosure(
                bindings=values["bindings"],
                source_video=bad,
                noisy_target=bad.clone(),
                negative_condition=values["negative"],
                positive_condition=values["positive"],
                schedule_index=33,
                sigma=values["sigma"],
                timestep=values["timestep"],
            )

    def test_wrong_sigma_or_timestep_fails_closed(self) -> None:
        values = self._fixture()
        for sigma, timestep in (
            (values["sigma"] + 1.0e-3, values["timestep"]),
            (values["sigma"], values["timestep"] + 1),
        ):
            with self.subTest(sigma=float(sigma.item()), timestep=int(timestep.item())):
                with self.assertRaisesRegex(
                    closure.GraftPhaseANativeTrainingClosureError, "same pinned exact40"
                ):
                    closure.PhaseANativeTrainingClosure(
                        bindings=values["bindings"],
                        source_video=values["source"],
                        noisy_target=values["noisy"],
                        negative_condition=values["negative"],
                        positive_condition=values["positive"],
                        schedule_index=33,
                        sigma=sigma,
                        timestep=timestep,
                    )

    def test_non_bf16_patch_or_measurement_fails_closed(self) -> None:
        values = self._fixture()
        values["transformer"].bad_patch_dtype = True
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError, "BF16 geometry"
        ):
            closure.PhaseANativeTrainingClosure(
                bindings=values["bindings"],
                source_video=values["source"],
                noisy_target=values["noisy"],
                negative_condition=values["negative"],
                positive_condition=values["positive"],
                schedule_index=33,
                sigma=values["sigma"],
                timestep=values["timestep"],
            )

        values = self._fixture()
        values["diffusion"].bad_measurement_dtype = True
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError, "packed BF16"
        ):
            values["session"].measure()
        self.assertEqual(values["session"].phase, "failed")

    def test_replay_raw_parity_failure_clears_partial_gradients(self) -> None:
        values = self._fixture()
        values["session"].measure()
        values["session"].derive_phase_a_flow_matching_vjp()
        values["diffusion"].perturb_replay = True
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError, "bytes differ"
        ):
            values["session"].replay_and_backward()
        self.assertEqual(values["session"].phase, "failed")
        self.assertIsNone(values["transformer"].adapter.grad)
        self.assertEqual(float(values["transformer"].adapter.item()), 0.125)

    def test_detached_replay_fails_and_clears_gradients(self) -> None:
        values = self._fixture()
        values["session"].measure()
        values["session"].derive_phase_a_flow_matching_vjp()
        values["diffusion"].detach_replay = True
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError,
            "detached from the native pack/route graph",
        ):
            values["session"].replay_and_backward()
        self.assertIsNone(values["transformer"].adapter.grad)
        self.assertEqual(float(values["transformer"].adapter.item()), 0.125)

    def test_visual_pack_mutation_is_detected(self) -> None:
        values = self._fixture()
        values["diffusion"].mutate_pack = True
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError, "pack mutated"
        ):
            values["session"].measure()
        self.assertEqual(values["session"].phase, "failed")

    def test_parameter_value_mutation_is_detected(self) -> None:
        values = self._fixture()
        values["diffusion"].mutate_parameter = True
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError, "parameter values changed"
        ):
            values["session"].measure()
        self.assertIsNone(values["transformer"].adapter.grad)
        self.assertEqual(float(values["transformer"].adapter.item()), 0.125)

    def test_replay_precheck_failure_poison_clears_grad_and_restores_snapshot(self) -> None:
        values = self._fixture()
        values["session"].measure()
        values["session"].derive_phase_a_flow_matching_vjp()
        with self.torch.no_grad():
            values["transformer"].adapter.add_(3.0)
        values["transformer"].adapter.grad = self.torch.ones_like(
            values["transformer"].adapter
        )
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError,
            "gradients appeared before serial replay",
        ):
            values["session"].replay_and_backward()
        self.assertEqual(values["session"].phase, "failed")
        self.assertIsNone(values["transformer"].adapter.grad)
        self.assertEqual(float(values["transformer"].adapter.item()), 0.125)
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError,
            "requires one completed",
        ):
            values["session"].replay_and_backward()

    def test_constructor_precheck_failure_also_clears_existing_gradient(self) -> None:
        values = self._fixture()
        values["transformer"].adapter.grad = self.torch.ones_like(
            values["transformer"].adapter
        )
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError,
            "gradients must be empty",
        ):
            closure.PhaseANativeTrainingClosure(
                bindings=values["bindings"],
                source_video=values["source"],
                noisy_target=values["noisy"],
                negative_condition=values["negative"],
                positive_condition=values["positive"],
                schedule_index=33,
                sigma=values["sigma"],
                timestep=values["timestep"],
            )
        self.assertIsNone(values["transformer"].adapter.grad)
        self.assertEqual(float(values["transformer"].adapter.item()), 0.125)

    def test_state_machine_rejects_out_of_order_and_repeat_calls(self) -> None:
        values = self._fixture()
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError, "requires one completed"
        ):
            values["session"].derive_phase_a_flow_matching_vjp()
        self.assertEqual(values["session"].phase, "failed")
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError, "first phase"
        ):
            values["session"].measure()
        self.assertEqual(values["session"].phase, "failed")

        values = self._fixture()
        values["session"].measure()
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError, "single-use"
        ):
            values["session"].measure()
        self.assertEqual(values["session"].phase, "failed")

    def test_bad_momentum_factory_fails_before_replay(self) -> None:
        values = self._fixture()

        class BadMomentum:
            def __init__(self, momentum):
                self.momentum = 0.9
                self.running_average = 0

        bindings = closure.authenticate_cpu_test_fakes(
            diffusion=values["diffusion"],
            transformer=values["transformer"],
            vendor_normalized_guidance=self.normalized_guidance,
            momentum_buffer_factory=BadMomentum,
            named_trainable_parameters=(("adapter", values["transformer"].adapter),),
            external_trainable_owner_modules={},
            test_name="cpu_fake:bad_momentum",
        )
        session = closure.PhaseANativeTrainingClosure(
            bindings=bindings,
            source_video=values["source"],
            noisy_target=values["noisy"],
            negative_condition=values["negative"],
            positive_condition=values["positive"],
            schedule_index=33,
            sigma=values["sigma"],
            timestep=values["timestep"],
        )
        session.measure()
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError, "momentum=0"
        ):
            session.derive_phase_a_flow_matching_vjp()

    def test_unregistered_trainable_parameter_fails_authentication(self) -> None:
        values = self._fixture()
        extra = self.torch.nn.Parameter(self.torch.tensor(1.0))
        values["transformer"].register_parameter("extra", extra)
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError, "trainable scope"
        ):
            closure.authenticate_cpu_test_fakes(
                diffusion=values["diffusion"],
                transformer=values["transformer"],
                vendor_normalized_guidance=self.normalized_guidance,
                momentum_buffer_factory=self.MomentumBuffer,
                named_trainable_parameters=(("adapter", values["transformer"].adapter),),
                external_trainable_owner_modules={},
                test_name="cpu_fake:scope",
            )

    def test_post_auth_callable_or_eval_mutation_is_rejected(self) -> None:
        values = self._fixture()
        values["diffusion"].train()
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError, "callable/eval"
        ):
            values["session"].measure()

        values = self._fixture()
        original = values["diffusion"].shared_step

        def replacement(*args, **kwargs):
            return original(*args, **kwargs)

        values["diffusion"].shared_step = replacement
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError, "callable/eval"
        ):
            values["session"].measure()

    def test_production_authenticator_rejects_unpinned_source_hashes(self) -> None:
        values = self._fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wan = root / "wan_diffusion.py"
            transformer = root / "transformer_wan.py"
            wan.write_text("not pinned\n", encoding="utf-8")
            transformer.write_text("not pinned\n", encoding="utf-8")
            with self.assertRaisesRegex(
                closure.GraftPhaseANativeTrainingClosureError, "source hash differs"
            ):
                closure.authenticate_pinned_native_bindings(
                    diffusion=values["diffusion"],
                    transformer=values["transformer"],
                    named_trainable_parameters=(
                        ("adapter", values["transformer"].adapter),
                    ),
                    external_trainable_owner_modules={},
                    wan_diffusion_path=wan,
                    transformer_wan_path=transformer,
                    bernini_commit=closure.PINNED_BERNINI_COMMIT,
                )


if __name__ == "__main__":
    unittest.main()
