#!/usr/bin/env python3

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import inspect
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import graft_phase_a_native_training_closure_v1 as v1  # noqa: E402
import graft_phase_a_native_training_closure_v2 as closure  # noqa: E402


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch required")
class GraftPhaseANativeTrainingClosureV2Tests(unittest.TestCase):
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

        class FakeAtlas(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Parameter(torch.tensor(0.19))

        class FakeTransformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.query = torch.nn.Parameter(torch.tensor(0.23))
                self.key = torch.nn.Parameter(torch.tensor(-0.31))
                self.value = torch.nn.Parameter(torch.tensor(0.41))
                self.output = torch.nn.Parameter(torch.zeros(2))
                self.frozen_base = torch.nn.Parameter(
                    torch.tensor(1.25), requires_grad=False
                )
                self.dtype = torch.bfloat16
                self.route_active = True
                self.detach_replay_category = None

            def patch_vae_latent(self, hidden_states, source_id=None):
                batch, channels, phases, height, width = hidden_states.shape
                patches = (
                    hidden_states.reshape(
                        batch, channels, phases, height // 2, 2, width // 2, 2
                    )
                    .permute(0, 2, 3, 5, 4, 6, 1)
                    .reshape(batch, phases * (height // 2) * (width // 2), 64)
                )
                seed = patches.mean(dim=-1, keepdim=True)
                tokens = seed.expand(batch, seed.shape[1], 1536).contiguous()
                rotary = torch.full(
                    (batch, 1, seed.shape[1], 8),
                    float(source_id),
                    dtype=torch.float32,
                    device=hidden_states.device,
                )
                return tokens, rotary

        class FakeDiffusion(torch.nn.Module):
            def __init__(self, transformer, atlas):
                super().__init__()
                self.transformer = transformer
                self.transformer_2 = None
                # Keep this registered in the diffusion tree while also naming
                # its explicit external owner in the authenticated registry.
                self.atlas = atlas
                self.call_count = 0
                self.mutate_output_during_replay = False
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
                del model_id, timesteps, rotary_embs, batch_vae_seqlen
                del batch_text_seqlen, kwargs
                call_index = self.call_count
                self.call_count += 1
                if self.mutate_output_during_replay and call_index >= 2:
                    with torch.no_grad():
                        self.transformer.output.add_(0.5)
                base = noisy_latents[..., :64].float()
                text = cond_embeds.float().mean().reshape(1, 1, 1)
                values = {
                    "atlas_encoder": self.atlas.proj,
                    "query_projection": self.transformer.query,
                    "key_projection": self.transformer.key,
                    "value_projection": self.transformer.value,
                    "output_projection": self.transformer.output,
                }
                if not self.transformer.route_active:
                    values = {name: value.detach() for name, value in values.items()}
                elif call_index >= 2 and self.transformer.detach_replay_category:
                    category = self.transformer.detach_replay_category
                    values[category] = values[category].detach()
                feature = (
                    values["query_projection"] * (base + 0.17)
                    + values["key_projection"] * (text + 0.29)
                    + values["value_projection"] * (base * text + 0.37)
                    + values["atlas_encoder"] * (base.square() + text + 0.43)
                )
                raw = (
                    base * (1.0 + 0.03125 * text)
                    + 0.0078125 * self.transformer.frozen_base
                    + values["output_projection"].mean() * feature
                )
                return raw.to(torch.bfloat16)

        class ExplodingScheduler:
            def __getattribute__(self, name):
                if name.startswith("__"):
                    return object.__getattribute__(self, name)
                raise AssertionError("v2 closure must not access a scheduler")

        return FakeAtlas, FakeTransformer, FakeDiffusion

    def _fixture(
        self,
        *,
        output_value=0.0,
        rank=None,
        spatial_width=2,
    ):
        torch = self.torch
        FakeAtlas, FakeTransformer, FakeDiffusion = self._fake_classes()
        atlas = FakeAtlas().eval()
        transformer = FakeTransformer().eval()
        with torch.no_grad():
            transformer.output.fill_(output_value)
        diffusion = FakeDiffusion(transformer, atlas).eval()
        route = None
        if rank is not None:
            @contextmanager
            def route(*, request):
                local_rows = (request.total_tokens + 3) // 4
                padded = local_rows * 4
                selector = torch.cat(
                    (
                        torch.zeros(request.condition_tokens, dtype=torch.bool),
                        torch.ones(request.target_tokens, dtype=torch.bool),
                        torch.zeros(
                            padded - request.total_tokens, dtype=torch.bool
                        ),
                    )
                )[rank * local_rows : (rank + 1) * local_rows].contiguous()
                targets = int(torch.count_nonzero(selector).item())
                yield v1.build_native_forward_context_observation(
                    request=request,
                    sequence_parallel_rank=rank,
                    sequence_parallel_size=4,
                    local_target_selector=selector,
                    route_gate=1.0,
                    adapter_graph_bearing=(
                        request.phase == "replay" and targets > 0
                    ),
                )

        names = (
            ("atlas_encoder.proj.weight", atlas.proj),
            (
                "blocks.8.attn1.to_out.0.identity_rebinder.query.weight",
                transformer.query,
            ),
            (
                "blocks.8.attn1.to_out.0.identity_rebinder.key.weight",
                transformer.key,
            ),
            (
                "blocks.8.attn1.to_out.0.identity_rebinder.value.weight",
                transformer.value,
            ),
            (
                "blocks.8.attn1.to_out.0.identity_rebinder.output.weight",
                transformer.output,
            ),
        )
        bindings = v1.authenticate_cpu_test_fakes(
            diffusion=diffusion,
            transformer=transformer,
            vendor_normalized_guidance=self.normalized_guidance,
            momentum_buffer_factory=self.MomentumBuffer,
            named_trainable_parameters=names,
            external_trainable_owner_modules={"atlas_encoder": atlas},
            test_name="cpu_fake:v2_post_bootstrap",
            forward_context_factory=route,
        )
        generator = torch.Generator(device="cpu").manual_seed(20260810)
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
        negative = torch.full((1, 2, 4), -1.0, dtype=torch.bfloat16)
        positive = torch.full((1, 2, 4), 2.0, dtype=torch.bfloat16)
        schedule_index = 33
        sigma = torch.tensor(
            v1.sigma_strata.PINNED_POSITIVE_SIGMAS[schedule_index],
            dtype=torch.float32,
        )
        timestep = torch.tensor(
            [v1.sigma_strata.PINNED_TIMESTEPS[schedule_index]],
            dtype=torch.int64,
        )
        values = {
            "atlas": atlas,
            "transformer": transformer,
            "diffusion": diffusion,
            "bindings": bindings,
            "names": names,
            "source": source,
            "noisy": noisy,
            "negative": negative,
            "positive": positive,
            "schedule_index": schedule_index,
            "sigma": sigma,
            "timestep": timestep,
        }
        values["session"] = self._new_session(values)
        return values

    @staticmethod
    def _new_session(values):
        return closure.PhaseANativeTrainingClosure(
            bindings=values["bindings"],
            source_video=values["source"],
            noisy_target=values["noisy"],
            negative_condition=values["negative"],
            positive_condition=values["positive"],
            schedule_index=values["schedule_index"],
            sigma=values["sigma"],
            timestep=values["timestep"],
        )

    @staticmethod
    def _run(session):
        session.measure()
        session.derive_phase_a_flow_matching_vjp()
        return session.replay_and_backward()

    def _parameter_snapshots(self, values):
        return {
            name: parameter.detach().clone() for name, parameter in values["names"]
        }

    def _assert_restored_and_clear(self, values, snapshots):
        for name, parameter in values["names"]:
            self.assertTrue(self.torch.equal(parameter.detach(), snapshots[name]))
            self.assertIsNone(parameter.grad)

    def test_bootstrap_is_live_byte_derived_and_keeps_output_only_gate(self) -> None:
        values = self._fixture()
        self.assertEqual(values["session"].training_regime, "bootstrap")
        result = self._run(values["session"])
        receipt = result.receipt
        self.assertEqual(receipt["schema_version"], closure.SCHEMA_VERSION)
        self.assertEqual(receipt["training_regime"], "bootstrap")
        self.assertTrue(receipt["bootstrap_output_only_gate_verified"])
        self.assertFalse(receipt["training_regime_caller_or_cli_input_accepted"])
        self.assertTrue(
            receipt["live_output_weight_state"]["output_raw_bytes_all_zero"]
        )
        for row in receipt["per_branch_local_trainable_gradient_gate"]:
            self.assertEqual(
                row["finite_nonzero_categories"], ["output_projection"]
            )
            self.assertEqual(
                row["gate"],
                "bootstrap_target_rows_output_projection_only_nonzero",
            )
        self.assertIsNotNone(values["transformer"].output.grad)
        self.assertTrue(
            bool(self.torch.count_nonzero(values["transformer"].output.grad).item())
        )
        for parameter in (
            values["atlas"].proj,
            values["transformer"].query,
            values["transformer"].key,
            values["transformer"].value,
        ):
            self.assertTrue(
                parameter.grad is None
                or not bool(self.torch.count_nonzero(parameter.grad).item())
            )
        self.assertIsNone(values["transformer"].frozen_base.grad)
        self.assertTrue(
            receipt["exclusive_trainable_scope_is_exact_authenticated_registry"]
        )

    def test_two_consecutive_cells_switch_automatically_to_post_bootstrap(self) -> None:
        values = self._fixture()
        first = self._run(values["session"])
        self.assertEqual(first.receipt["training_regime"], "bootstrap")
        with self.torch.no_grad():
            values["transformer"].output.add_(
                -0.05 * values["transformer"].output.grad
            )
        self.assertTrue(
            bool(self.torch.count_nonzero(values["transformer"].output).item())
        )
        for _, parameter in values["names"]:
            parameter.grad = None
        second_session = self._new_session(values)
        self.assertEqual(second_session.training_regime, "post_bootstrap")
        second = self._run(second_session)
        self.assertTrue(
            second.receipt["post_bootstrap_five_category_local_gate_verified"]
        )
        for row in second.receipt["per_branch_local_trainable_gradient_gate"]:
            self.assertEqual(
                row["finite_nonzero_categories"],
                list(closure.GRADIENT_CATEGORIES),
            )

    def test_stale_gradient_between_steps_is_poisoned_not_reused(self) -> None:
        values = self._fixture()
        self._run(values["session"])
        with self.torch.no_grad():
            values["transformer"].output.add_(
                -0.05 * values["transformer"].output.grad
            )
        post_step = self._parameter_snapshots(values)
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError,
            "gradients must be empty",
        ):
            self._new_session(values)
        self._assert_restored_and_clear(values, post_step)

    def test_caller_cannot_report_or_force_a_regime(self) -> None:
        constructor = inspect.signature(closure.PhaseANativeTrainingClosure)
        execute = inspect.signature(closure.execute_phase_a_native_training_closure)
        for forbidden in ("mode", "regime", "bootstrap", "training_regime"):
            self.assertNotIn(forbidden, constructor.parameters)
            self.assertNotIn(forbidden, execute.parameters)
        values = self._fixture(output_value=0.125)
        kwargs = {
            "bindings": values["bindings"],
            "source_video": values["source"],
            "noisy_target": values["noisy"],
            "negative_condition": values["negative"],
            "positive_condition": values["positive"],
            "schedule_index": values["schedule_index"],
            "sigma": values["sigma"],
            "timestep": values["timestep"],
            "training_regime": "bootstrap",
        }
        with self.assertRaises(TypeError):
            closure.PhaseANativeTrainingClosure(**kwargs)
        self.assertEqual(values["session"].training_regime, "post_bootstrap")

    def test_noncanonical_negative_zero_cannot_forge_post_bootstrap(self) -> None:
        values = self._fixture()
        # The fixture already opened a valid session; mutate only after it so
        # the next construction exercises raw-byte versus numerical zero.
        with self.torch.no_grad():
            values["transformer"].output.fill_(-0.0)
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError,
            "signed negative-zero",
        ):
            self._new_session(values)

    def test_mixed_nonzero_and_negative_zero_is_rejected_and_stale_grad_cleared(self) -> None:
        values = self._fixture(output_value=0.125)
        with self.torch.no_grad():
            values["transformer"].output[1] = -0.0
        values["transformer"].query.grad = self.torch.ones_like(
            values["transformer"].query
        )
        invalid_snapshot = values["transformer"].output.detach().clone()
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError,
            "signed negative-zero",
        ):
            self._new_session(values)
        self.assertIsNone(values["transformer"].query.grad)
        self.assertTrue(
            self.torch.equal(values["transformer"].output, invalid_snapshot)
        )

    def test_post_bootstrap_partial_category_attack_fails_and_rolls_back(self) -> None:
        values = self._fixture(output_value=0.125)
        snapshots = self._parameter_snapshots(values)
        values["transformer"].detach_replay_category = "key_projection"
        values["session"].measure()
        values["session"].derive_phase_a_flow_matching_vjp()
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError,
            "post-bootstrap five-category gradient gate",
        ):
            values["session"].replay_and_backward()
        self.assertEqual(values["session"].phase, "failed")
        self._assert_restored_and_clear(values, snapshots)

    def test_post_bootstrap_nonfinite_gradient_poison_clears_and_restores(self) -> None:
        values = self._fixture(output_value=0.125)
        snapshots = self._parameter_snapshots(values)
        handle = values["transformer"].key.register_hook(
            lambda gradient: gradient * self.torch.tensor(float("nan"))
        )
        try:
            values["session"].measure()
            values["session"].derive_phase_a_flow_matching_vjp()
            with self.assertRaisesRegex(
                closure.GraftPhaseANativeTrainingClosureError,
                "non-finite",
            ):
                values["session"].replay_and_backward()
        finally:
            handle.remove()
        self.assertEqual(values["session"].phase, "failed")
        self._assert_restored_and_clear(values, snapshots)

    def test_parameter_mutation_during_replay_restores_post_bootstrap_snapshot(self) -> None:
        values = self._fixture(output_value=0.125)
        snapshots = self._parameter_snapshots(values)
        values["diffusion"].mutate_output_during_replay = True
        values["session"].measure()
        values["session"].derive_phase_a_flow_matching_vjp()
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError,
            "parameter values changed",
        ):
            values["session"].replay_and_backward()
        self._assert_restored_and_clear(values, snapshots)

    def test_post_bootstrap_source_only_sp_rank_requires_all_five_exact_zero(self) -> None:
        values = self._fixture(output_value=0.125, rank=0, spatial_width=4)
        values["transformer"].route_active = False
        result = self._run(values["session"])
        receipt = result.receipt
        self.assertEqual(receipt["training_regime"], "post_bootstrap")
        self.assertEqual(receipt["local_target_rows"], 0)
        self.assertTrue(
            receipt["source_only_sp_all_five_categories_exact_zero_verified"]
        )
        for row in receipt["per_branch_local_trainable_gradient_gate"]:
            self.assertEqual(row["finite_nonzero_categories"], [])
            self.assertEqual(
                row["gate"],
                "source_only_sp_rank_all_five_categories_exact_zero",
            )
        for _, parameter in values["names"]:
            self.assertIsNone(parameter.grad)

    def test_output_state_tamper_after_construction_is_poisoned_and_restored(self) -> None:
        values = self._fixture()
        snapshots = self._parameter_snapshots(values)
        with self.torch.no_grad():
            values["transformer"].output.fill_(0.25)
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError,
            "parameter values changed",
        ):
            values["session"].measure()
        self._assert_restored_and_clear(values, snapshots)

    def test_v1_kernel_and_underlying_receipt_are_digest_bound(self) -> None:
        source = Path(v1.__file__).read_bytes()
        self.assertEqual(
            hashlib.sha256(source).hexdigest(), closure.PINNED_V1_SOURCE_SHA256
        )
        values = self._fixture()
        result = self._run(values["session"])
        self.assertEqual(
            result.receipt["wrapped_v1_source_sha256"],
            closure.PINNED_V1_SOURCE_SHA256,
        )
        self.assertEqual(len(result.receipt["wrapped_v1_receipt_digest"]), 64)
        with mock.patch.object(
            v1.PhaseANativeTrainingClosure,
            "measure",
            new=lambda self: None,
        ):
            with self.assertRaisesRegex(
                closure.GraftPhaseANativeTrainingClosureError,
                "fixed gradient-gate seam differs",
            ):
                self._new_session(values)

        values = self._fixture()
        with mock.patch.object(
            v1.PhaseANativeTrainingClosure,
            "_shared_forward",
            new=lambda self, **_kwargs: None,
        ):
            with self.assertRaisesRegex(
                closure.GraftPhaseANativeTrainingClosureError,
                "fixed gradient-gate seam differs",
            ):
                self._new_session(values)

        values = self._fixture()
        with mock.patch.object(
            v1,
            "_tensor_bytes_sha256",
            new=lambda _value: "0" * 64,
        ):
            with self.assertRaisesRegex(
                closure.GraftPhaseANativeTrainingClosureError,
                "live v1 execution namespace differs",
            ):
                self._new_session(values)

    def test_preimport_v1_patch_cannot_become_the_v2_runtime_baseline(self) -> None:
        module_path = Path(closure.__file__).resolve()
        module_name = "_graft_phase_a_closure_v2_preimport_attack"
        with mock.patch.object(
            v1.PhaseANativeTrainingClosure,
            "measure",
            new=lambda self: None,
        ):
            specification = importlib.util.spec_from_file_location(
                module_name, module_path
            )
            self.assertIsNotNone(specification)
            attacked = importlib.util.module_from_spec(specification)
            self.assertIsNotNone(specification.loader)
            specification.loader.exec_module(attacked)
            with self.assertRaisesRegex(
                attacked.GraftPhaseANativeTrainingClosureError,
                "live v1 execution namespace differs",
            ):
                attacked._assert_pinned_v1_kernel()  # noqa: SLF001

    def test_runtime_contract_is_hardcoded_for_all_supported_interpreters(self) -> None:
        expected_v1 = closure.PINNED_V1_RUNTIME_NAMESPACE_SHA256
        expected_sigma = closure.PINNED_SIGMA_RUNTIME_NAMESPACE_SHA256
        supported = {(3, 8), (3, 10), (3, 12)}
        self.assertEqual(set(expected_v1), supported)
        self.assertEqual(set(expected_sigma), supported)
        for digest in tuple(expected_v1.values()) + tuple(expected_sigma.values()):
            self.assertRegex(digest, r"[0-9a-f]{64}\Z")
            self.assertNotIn("PENDING", digest)
        python_minor = (sys.version_info.major, sys.version_info.minor)
        self.assertEqual(
            closure._v1_runtime_namespace_sha256(),  # noqa: SLF001
            expected_v1[python_minor],
        )
        self.assertEqual(
            closure._sigma_runtime_namespace_sha256(),  # noqa: SLF001
            expected_sigma[python_minor],
        )

    def test_auh_python312_runtime_contract_is_exact_when_live(self) -> None:
        self.assertEqual(
            closure.PINNED_V1_RUNTIME_NAMESPACE_SHA256[(3, 12)],
            "1af7b8ec856fc50adc0b2d1b938eb5b60650809f60cdabe08fdbf040b0731913",
        )
        self.assertEqual(
            closure.PINNED_SIGMA_RUNTIME_NAMESPACE_SHA256[(3, 12)],
            "a8ee8f4b5b4ad1d15e2d340c71a15e2e8808d0143661ebb51b41bc3a6189c50c",
        )
        if (sys.version_info.major, sys.version_info.minor) != (3, 12):
            self.skipTest("live AUH runtime-positive assertion requires Python 3.12")
        closure._assert_pinned_v1_kernel()  # noqa: SLF001
        self.assertEqual(
            closure._v1_runtime_namespace_sha256(),  # noqa: SLF001
            closure.PINNED_V1_RUNTIME_NAMESPACE_SHA256[(3, 12)],
        )
        self.assertEqual(
            closure._sigma_runtime_namespace_sha256(),  # noqa: SLF001
            closure.PINNED_SIGMA_RUNTIME_NAMESPACE_SHA256[(3, 12)],
        )

    def test_unsupported_python_minor_fails_closed(self) -> None:
        unsupported = types.SimpleNamespace(major=3, minor=11)
        with mock.patch.object(closure.sys, "version_info", unsupported):
            for assertion in (
                closure._assert_pinned_v1_kernel,  # noqa: SLF001
                closure._assert_pinned_sigma_runtime,  # noqa: SLF001
            ):
                with self.subTest(assertion=assertion.__name__):
                    with self.assertRaisesRegex(
                        closure.GraftPhaseANativeTrainingClosureError,
                        r"unsupported Python runtime minor.*3\.11",
                    ):
                        assertion()

    def test_subclass_cannot_override_the_post_bootstrap_gate(self) -> None:
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError,
            "does not permit subclassing",
        ):
            class ForgedClosure(closure.PhaseANativeTrainingClosure):
                def _local_trainable_delta_receipt(self, **_kwargs):
                    return v1._seal(  # noqa: SLF001 - hostile fake gate
                        {
                            "schema_version": (
                                closure.LOCAL_GRADIENT_SCHEMA_VERSION
                            ),
                            "role": "negative",
                            "training_regime": "post_bootstrap",
                            "gate": (
                                "post_bootstrap_target_rows_all_five_categories_"
                                "finite_nonzero"
                            ),
                        }
                    )

    def test_live_v2_gate_patch_before_replay_is_poisoned(self) -> None:
        values = self._fixture(output_value=0.125)
        snapshots = self._parameter_snapshots(values)
        values["session"].measure()
        values["session"].derive_phase_a_flow_matching_vjp()
        with mock.patch.object(
            closure.PhaseANativeTrainingClosure,
            "_local_trainable_delta_receipt",
            new=lambda self, **_kwargs: v1._seal({"forged": True}),
        ):
            with self.assertRaisesRegex(
                closure.GraftPhaseANativeTrainingClosureError,
                "fixed gradient-gate seam differs",
            ):
                values["session"].replay_and_backward()
        self._assert_restored_and_clear(values, snapshots)

    def test_exact_class_inherited_gradient_snapshot_shadow_fails_before_use(self) -> None:
        values = self._fixture(output_value=0.125)
        snapshots = self._parameter_snapshots(values)
        values["transformer"].detach_replay_category = "key_projection"
        values["session"].measure()
        values["session"].derive_phase_a_flow_matching_vjp()
        original = v1.PhaseANativeTrainingClosure._gradient_snapshot  # noqa: SLF001
        calls = []

        def stateful_forge(session):
            calls.append(len(calls))
            forged = original(session)
            key = "blocks.8.attn1.to_out.0.identity_rebinder.key.weight"
            forged[key] = self.torch.ones_like(forged[key]) * len(calls)
            return forged

        with mock.patch.object(
            closure.PhaseANativeTrainingClosure,
            "_gradient_snapshot",
            new=stateful_forge,
            create=True,
        ):
            with self.assertRaisesRegex(
                closure.GraftPhaseANativeTrainingClosureError,
                "fixed gradient-gate seam differs",
            ):
                values["session"].replay_and_backward()
        self.assertEqual(calls, [])
        self._assert_restored_and_clear(values, snapshots)

    def test_exact_instance_inherited_gradient_snapshot_shadow_is_poisoned(self) -> None:
        values = self._fixture(output_value=0.125)
        snapshots = self._parameter_snapshots(values)
        values["transformer"].detach_replay_category = "key_projection"
        values["session"].measure()
        values["session"].derive_phase_a_flow_matching_vjp()
        original = v1.PhaseANativeTrainingClosure._gradient_snapshot  # noqa: SLF001
        calls = []

        def stateful_forge(session):
            calls.append(len(calls))
            forged = original(session)
            key = "blocks.8.attn1.to_out.0.identity_rebinder.key.weight"
            forged[key] = self.torch.ones_like(forged[key]) * len(calls)
            return forged

        values["transformer"].query.grad = self.torch.ones_like(
            values["transformer"].query
        )
        with self.torch.no_grad():
            values["transformer"].query.add_(3.0)
        object.__setattr__(
            values["session"],
            "_gradient_snapshot",
            stateful_forge,
        )
        self.assertIs(type(values["session"]), closure.PhaseANativeTrainingClosure)
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError,
            "fixed gradient-gate seam differs",
        ):
            values["session"].replay_and_backward()
        self.assertEqual(calls, [])
        self.assertNotIn("_gradient_snapshot", vars(values["session"]))
        self.assertEqual(values["session"].phase, "failed")
        self._assert_restored_and_clear(values, snapshots)

    def test_exact_instance_preconstruction_execution_shadow_is_rejected(self) -> None:
        values = self._fixture(output_value=0.125)
        values.pop("session")
        snapshots = self._parameter_snapshots(values)
        calls = []

        def forged_measure():
            calls.append(True)

        raw = object.__new__(closure.PhaseANativeTrainingClosure)
        object.__setattr__(raw, "measure", forged_measure)
        values["transformer"].query.grad = self.torch.ones_like(
            values["transformer"].query
        )
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError,
            "fixed gradient-gate seam differs",
        ):
            closure.PhaseANativeTrainingClosure.__init__(
                raw,
                bindings=values["bindings"],
                source_video=values["source"],
                noisy_target=values["noisy"],
                negative_condition=values["negative"],
                positive_condition=values["positive"],
                schedule_index=values["schedule_index"],
                sigma=values["sigma"],
                timestep=values["timestep"],
            )
        self.assertEqual(calls, [])
        self.assertNotIn("measure", vars(raw))
        self.assertEqual(raw.phase, "failed")
        self._assert_restored_and_clear(values, snapshots)

    def test_prebound_replay_rechecks_later_instance_execution_shadow(self) -> None:
        values = self._fixture(output_value=0.125)
        snapshots = self._parameter_snapshots(values)
        values["session"].measure()
        values["session"].derive_phase_a_flow_matching_vjp()
        replay = values["session"].replay_and_backward
        calls = []

        def forged_snapshot():
            calls.append(True)

        object.__setattr__(
            values["session"],
            "_gradient_snapshot",
            forged_snapshot,
        )
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError,
            "fixed gradient-gate seam differs",
        ):
            replay()
        self.assertEqual(calls, [])
        self.assertNotIn("_gradient_snapshot", vars(values["session"]))
        self.assertEqual(values["session"].phase, "failed")
        self._assert_restored_and_clear(values, snapshots)

    def test_instance_shadow_guard_covers_every_execution_descriptor(self) -> None:
        expected = frozenset(
            name
            for inventory in (
                closure._PINNED_V2_CLASS_DICT_IDENTITIES,  # noqa: SLF001
                closure._PINNED_V1_EXECUTION_DESCRIPTOR_IDENTITIES,  # noqa: SLF001
            )
            for name, descriptor in inventory.items()
            if inspect.isfunction(descriptor)
            or isinstance(
                descriptor,
                (classmethod, staticmethod, property),
            )
        )
        self.assertEqual(
            closure._PINNED_INSTANCE_PROTECTED_EXECUTION_DESCRIPTOR_NAMES,  # noqa: SLF001
            expected,
        )
        self.assertIn("_gradient_snapshot", expected)
        self.assertIn("measure", expected)
        self.assertIn("replay_and_backward", expected)

    def test_same_name_module_proxies_and_external_class_clone_fail_and_cleanup(self) -> None:
        sigma_proxy = types.ModuleType(v1.sigma_strata.__name__)
        sigma_proxy.__dict__.update(vars(v1.sigma_strata))
        sigma_proxy.SCHEDULE_SHA256 = "0" * 64
        torch_proxy = types.ModuleType(v1.torch.__name__)
        torch_proxy.__dict__.update(vars(v1.torch))
        fake_path = type("Path", (), {})
        fake_path.__module__ = "pathlib"
        fake_path.__qualname__ = "Path"
        for attribute, replacement in (
            ("sigma_strata", sigma_proxy),
            ("torch", torch_proxy),
            ("Path", fake_path),
        ):
            with self.subTest(attribute=attribute):
                values = self._fixture(output_value=0.125)
                snapshots = self._parameter_snapshots(values)
                values["transformer"].query.grad = self.torch.ones_like(
                    values["transformer"].query
                )
                with mock.patch.object(v1, attribute, replacement):
                    with self.assertRaisesRegex(
                        closure.GraftPhaseANativeTrainingClosureError,
                        "execution import identity differs",
                    ):
                        self._new_session(values)
                self._assert_restored_and_clear(values, snapshots)

    def test_local_sigma_runtime_mutation_fails_and_cleans_constructor_grad(self) -> None:
        values = self._fixture(output_value=0.125)
        snapshots = self._parameter_snapshots(values)
        values["transformer"].query.grad = self.torch.ones_like(
            values["transformer"].query
        )
        with mock.patch.object(
            closure.pinned_sigma_strata, "SCHEDULE_SHA256", "0" * 64
        ):
            with self.assertRaisesRegex(
                closure.GraftPhaseANativeTrainingClosureError,
                "sigma source/runtime/export/schedule contract differs",
            ):
                self._new_session(values)
        self._assert_restored_and_clear(values, snapshots)

    def test_wrapped_and_branch_receipts_require_canonical_lowercase_digests(self) -> None:
        valid = dict(v1._seal({"schema_version": "attack-fixture", "value": 1}))
        uppercase = dict(valid)
        uppercase["digest"] = uppercase["digest"].upper()
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError,
            "canonical lowercase SHA256",
        ):
            closure._validated_sealed_mapping(  # noqa: SLF001
                uppercase, label="uppercase attack"
            )
        changed = dict(valid)
        changed["value"] = 2
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError,
            "canonical lowercase SHA256",
        ):
            closure._validated_sealed_mapping(  # noqa: SLF001
                changed, label="payload attack"
            )

        values = self._fixture(output_value=0.125)
        result = self._run(values["session"])
        self.assertEqual(
            result.receipt["wrapped_v1_receipt_digest"],
            result.receipt["wrapped_v1_receipt_digest"].lower(),
        )
        tampered = dict(result.receipt)
        branch = dict(tampered["per_branch_local_trainable_gradient_gate"][0])
        branch.pop("digest")
        branch["finite_nonzero_categories"] = ["output_projection"]
        tampered["per_branch_local_trainable_gradient_gate"] = [
            v1._seal(branch),
            tampered["per_branch_local_trainable_gradient_gate"][1],
        ]
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError,
            "critical fields differ",
        ):
            closure._verified_branch_gradient_gates(  # noqa: SLF001
                base_receipt=tampered,
                bindings=values["bindings"],
                cell_active=True,
                training_regime="post_bootstrap",
                output_weight_state=values["session"]._output_weight_state,  # noqa: SLF001
            )

        row_attack = dict(result.receipt)
        branch = dict(row_attack["per_branch_local_trainable_gradient_gate"][0])
        branch.pop("digest")
        rows = [dict(row) for row in branch["rows"]]
        nonzero_index = next(
            index for index, row in enumerate(rows) if row["delta_nonzero"]
        )
        rows[nonzero_index]["gradient_present_after"] = False
        rows[nonzero_index]["gradient_after_absent_or_exact_zero"] = True
        branch["rows"] = rows
        row_attack["per_branch_local_trainable_gradient_gate"] = [
            v1._seal(branch),
            row_attack["per_branch_local_trainable_gradient_gate"][1],
        ]
        with self.assertRaisesRegex(
            closure.GraftPhaseANativeTrainingClosureError,
            "gradient row differs",
        ):
            closure._verified_branch_gradient_gates(  # noqa: SLF001
                base_receipt=row_attack,
                bindings=values["bindings"],
                cell_active=True,
                training_regime="post_bootstrap",
                output_weight_state=values["session"]._output_weight_state,  # noqa: SLF001
            )

    def test_gpu_runner_facing_v1_surface_is_reexported_without_reimplementation(self) -> None:
        for name in (
            "AuthenticatedNativeBindings",
            "FLOW_MATCHING_OBJECTIVE",
            "FORWARD_ROUTE_SCHEMA_VERSION",
            "GUIDANCE_MODE",
            "NativeForwardContextRequest",
            "PHASE_A_ACTIVE_SCHEDULE_INDICES",
            "authenticate_pinned_native_bindings",
            "build_native_forward_context_observation",
        ):
            self.assertTrue(hasattr(closure, name), name)
        self.assertIs(
            closure.authenticate_pinned_native_bindings,
            v1.authenticate_pinned_native_bindings,
        )
        self.assertIs(
            closure.build_native_forward_context_observation,
            v1.build_native_forward_context_observation,
        )

    def test_receipt_authority_is_not_upgraded_and_gpu_assumptions_are_explicit(self) -> None:
        values = self._fixture(output_value=0.125)
        receipt = self._run(values["session"]).receipt
        for key in (
            "official_cuda_closure_verified_by_this_core",
            "forward_route_semantics_verified_by_this_core",
            "packed_raw_to_apg_registry_chain_verified_by_this_core",
            "sp4_collective_parity_verified",
            "training_quality_claim_authorized",
            "scientific_action_editing_claim_authorized",
            "optimizer_step_verified_by_this_core",
            "two_consecutive_steps_verified_by_this_core",
            "post_bootstrap_cuda_short_training_verified_by_this_core",
            "short_training_claim_authorized",
        ):
            self.assertIs(receipt[key], False)
        self.assertEqual(
            receipt["remaining_gpu_assumptions"],
            list(closure.REMAINING_GPU_ASSUMPTIONS),
        )
        self.assertTrue(receipt["test_only_binding"])
        self.assertFalse(receipt["official_pinned_code"])
        self.assertIsNone(values["transformer"].frozen_base.grad)


if __name__ == "__main__":
    unittest.main()
