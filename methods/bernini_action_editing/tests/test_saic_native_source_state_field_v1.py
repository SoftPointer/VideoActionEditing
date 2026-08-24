#!/usr/bin/env python3

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest import mock

import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import saic_native_source_state_field_v1 as native  # noqa: E402
import saic_source_state_flow_transport_v1 as transport  # noqa: E402


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def digest_tensor(value: torch.Tensor) -> str:
    return native._tensor_bytes_sha256(value)


def independent_zero_momentum_normalize(
    diff: torch.Tensor, base_pred: torch.Tensor
) -> torch.Tensor:
    norm = diff.norm(p=2, dim=[-1, -2, -4], keepdim=True)
    clipped = diff * torch.minimum(torch.ones_like(diff), 50.0 / norm)
    axis = torch.nn.functional.normalize(
        base_pred.double(), dim=[-1, -2, -4]
    )
    parallel = (clipped.double() * axis).sum(
        dim=[-1, -2, -4], keepdim=True
    ) * axis
    return (clipped.double() - parallel).to(clipped.dtype) + 0.5 * parallel.to(
        clipped.dtype
    )


def independent_native_r2v_chain(
    pred_uncond: torch.Tensor,
    image_negative: torch.Tensor,
    image_role: torch.Tensor,
) -> torch.Tensor:
    image_axis = independent_zero_momentum_normalize(
        image_negative - pred_uncond, image_negative
    )
    text_axis = independent_zero_momentum_normalize(
        image_role - image_negative, image_role
    )
    return pred_uncond + 4.5 * image_axis + 4.0 * text_axis


class FakeTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_parameter(
            "seal", torch.nn.Parameter(torch.tensor([7.0]), requires_grad=False)
        )
        self.dtype = torch.bfloat16
        self.source_ids: list[int] = []
        self.patch_input_digests: list[str] = []
        self.eval()

    def patch_vae_latent(self, *, hidden_states: torch.Tensor, source_id: float):
        self.source_ids.append(source_id)
        self.patch_input_digests.append(digest_tensor(hidden_states))
        phases = int(hidden_states.shape[2])
        tokens = phases * (int(hidden_states.shape[3]) // 2) * (
            int(hidden_states.shape[4]) // 2
        )
        token_tensor = torch.full(
            (1, tokens, 1536), float(source_id + 1), dtype=self.dtype
        )
        rotary = torch.full(
            (1, 1, tokens, 64), complex(source_id + 1, 0), dtype=torch.complex128
        )
        return token_tensor, rotary


class FakeDiffusion(torch.nn.Module):
    def __init__(self, transformer: FakeTransformer) -> None:
        super().__init__()
        self.transformer = transformer
        self.transformer_2 = None
        self.use_unipc = True
        self.register_buffer("seal", torch.tensor([11.0]))
        self.calls: list[dict[str, object]] = []
        self.eval()

    def shared_step(self, **kwargs):
        self.calls.append(
            {
                "model_id": kwargs["model_id"],
                "tokens_id": id(kwargs["noisy_latents"]),
                "rotary_id": id(kwargs["rotary_embs"]),
                "timestep_id": id(kwargs["timesteps"]),
                "total": int(kwargs["noisy_latents"].shape[1]),
                "batch_vae_seqlen": tuple(kwargs["batch_vae_seqlen"]),
                "condition_id": id(kwargs["cond_embeds"]),
                "visual_first": float(kwargs["noisy_latents"][0, 0, 0].item()),
                "visual_last": float(kwargs["noisy_latents"][0, -1, 0].item()),
                "rotary_first": complex(kwargs["rotary_embs"][0, 0, 0, 0].item()),
                "rotary_last": complex(kwargs["rotary_embs"][0, 0, -1, 0].item()),
            }
        )
        value = float(kwargs["cond_embeds"].reshape(-1)[0].item())
        if int(kwargs["noisy_latents"].shape[1]) == 22:
            value += 0.25
        result = torch.full(
            (1, int(kwargs["noisy_latents"].shape[1]), 64),
            value,
            dtype=torch.bfloat16,
        )
        # In the source-I0 fixture Nref=1 and N=21.  Poisoning the prefix
        # proves the adapter slices the direct target tail before unpacking.
        if int(result.shape[1]) == 22:
            result[:, :1].fill_(99.0)
        return result


class NativeFieldFixture:
    def __init__(self, regime: str, *, k1: bool = False) -> None:
        self.regime = regime
        self.transformer = FakeTransformer()
        self.diffusion = FakeDiffusion(self.transformer)
        self.conditions = {
            "negative": torch.full((1, 512, 4096), 1.0, dtype=torch.bfloat16),
            "target": torch.full((1, 512, 4096), 3.0, dtype=torch.bfloat16),
            "source": torch.full((1, 512, 4096), 2.0, dtype=torch.bfloat16),
        }
        self.captions = {"target": "the dog sits down", "source": "the dog stands"}
        schedule = torch.linspace(1.0, 0.0, 41, dtype=torch.float32)
        self.sigmas = tuple(schedule[index] for index in range(40))
        self.next_sigmas = tuple(float(schedule[index + 1].item()) for index in range(40))
        self.timesteps = tuple(
            value.reshape(1)
            for value in torch.arange(999, 959, -1, dtype=torch.int64)
        )
        self.candidate_schedule = (
            native.REGISTERED_K1_SCHEDULE
            if k1
            else native.REGISTERED_K5_EARLY_SCHEDULE
        )
        self.aggregation_mode = "uniform" if k1 else "source_similarity_softmax"
        self.temperature = None if k1 else 0.01
        self.sigma_schedule = tuple(float(value.item()) for value in schedule)
        self.checkpoint = "1" * 64
        self.negative_prompt = "2" * 64
        self.guidance_contract = "3" * 64
        self.reference_encoder = "0" * 64 if regime == "t2v_apg" else "4" * 64
        self.reference = (
            None
            if regime == "t2v_apg"
            else torch.arange(16 * 1 * 2 * 2, dtype=torch.float32).reshape(1, 16, 1, 2, 2)
        )
        self.provenance = native.NativeFieldProvenance(
            model_id="transformer_1",
            checkpoint_sha256=self.checkpoint,
            model_receipt_sha256="5" * 64,
            guidance_contract_sha256=self.guidance_contract,
            negative_prompt_sha256=self.negative_prompt,
            native_schedule_sha256=native.native_schedule_sha256(
                self.sigmas,
                self.next_sigmas,
                self.timesteps,
                self.candidate_schedule,
                self.aggregation_mode,
                self.temperature,
            ),
            noise_generator_id="torch.cpu.generator",
            master_seed=7,
            noise_bank_sha256="6" * 64,
            reference_encoder_sha256=self.reference_encoder,
            reference_frame0_latent_sha256=(
                "0" * 64 if self.reference is None else digest_tensor(self.reference)
            ),
            prompt_utf8_sha256_by_role={
                key: digest_text(value) for key, value in self.captions.items()
            },
            prompt_condition_sha256_by_key={
                key: digest_tensor(value) for key, value in self.conditions.items()
            },
        )

    def adapter(self) -> native.NativeSourceStateFieldAdapter:
        return native.NativeSourceStateFieldAdapter(
            diffusion=self.diffusion,
            transformer=self.transformer,
            field_regime=self.regime,
            conditions=self.conditions,
            captions=self.captions,
            sigma_scalars=self.sigmas,
            next_sigmas=self.next_sigmas,
            timestep_tensors=self.timesteps,
            candidate_schedule=self.candidate_schedule,
            aggregation_mode=self.aggregation_mode,
            temperature=self.temperature,
            provenance=self.provenance,
            reference_frame0_latent=self.reference,
        )

    def request(
        self,
        *,
        role: str,
        candidate: int,
        state: torch.Tensor,
        regime: str | None = None,
        step_index: int = 0,
    ) -> transport.VelocityQueryRequest:
        field_regime = regime or self.regime
        guidance_mode = "t2v_apg" if field_regime == "t2v_apg" else "r2v_apg"
        is_r2v = field_regime == "r2v_apg_source_i0"
        raw_per_candidate = 6 if is_r2v else 4
        binding = transport.NativeGuidanceBinding(
            model_id="transformer_1",
            checkpoint_sha256=self.checkpoint,
            negative_prompt_sha256=self.negative_prompt,
            field_regime=field_regime,
            guidance_mode=guidance_mode,
            guidance_contract_sha256=self.guidance_contract,
            image_guidance_scale=4.5 if is_r2v else 0.0,
            guidance_chain_scales=(4.5, 4.0) if is_r2v else (4.0,),
            apg_norm_thresholds=(50.0, 50.0) if is_r2v else (50.0,),
            apg_momenta=(0.0, 0.0) if is_r2v else (0.0,),
            branch_order=(
                transport.EXPECTED_R2V_I0_BRANCH_ORDER
                if is_r2v
                else transport.EXPECTED_T2V_V2V_BRANCH_ORDER
            ),
            raw_transformer_forwards_per_candidate=raw_per_candidate,
        )
        step = transport.FlowTransportStepBinding(
            step_index=step_index,
            sigma=float(self.sigmas[step_index].item()),
            next_sigma=self.next_sigmas[step_index],
            time=float(self.sigmas[step_index].item()),
            next_time=self.next_sigmas[step_index],
            candidate_count=self.candidate_schedule[step_index],
            anc_enabled=True,
            anc_retention=0.0,
            candidate_continuation="candidate_zero",
            candidate_schedule=self.candidate_schedule,
            aggregation_mode=self.aggregation_mode,
            temperature=self.temperature,
            sigma_schedule=self.sigma_schedule,
            sigma_schedule_sha256=transport.sigma_schedule_sha256(
                self.sigma_schedule
            ),
            native=binding,
            noise_generator_id="torch.cpu.generator",
            master_seed=7,
            noise_bank_sha256="6" * 64,
            raw_transformer_forwards_per_candidate=raw_per_candidate,
        )
        return transport.VelocityQueryRequest(
            state=state,
            caption=self.captions[role],
            role=role,
            candidate_index=candidate,
            step=step,
            state_sha256=digest_tensor(state),
            expected_raw_transformer_forwards=(3 if is_r2v else 2),
        )


class NativeSourceStateFieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def setUp(self) -> None:
        self._saved_vendor = sys.modules.get(native.VENDOR_APG_MODULE)
        vendor = ModuleType(native.VENDOR_APG_MODULE)

        class MomentumBuffer:
            def __init__(self, momentum):
                self.momentum = momentum
                self.running_average = 0

            def update(self, update_value):
                self.running_average = (
                    update_value + self.momentum * self.running_average
                )

        MomentumBuffer.__module__ = native.VENDOR_APG_MODULE

        def normalize_diff(diff, base_pred, momentum_buffer, eta, norm_threshold):
            if momentum_buffer is not None:
                momentum_buffer.update(diff)
                diff = momentum_buffer.running_average
            if norm_threshold > 0:
                norm = diff.norm(p=2, dim=[-1, -2, -4], keepdim=True)
                diff = diff * torch.minimum(
                    torch.ones_like(diff), norm_threshold / norm
                )
            base = torch.nn.functional.normalize(
                base_pred.double(), dim=[-1, -2, -4]
            )
            parallel = (diff.double() * base).sum(
                dim=[-1, -2, -4], keepdim=True
            ) * base
            orthogonal = diff.double() - parallel
            return orthogonal.to(diff.dtype) + eta * parallel.to(diff.dtype)

        def normalized_guidance(
            pred_cond,
            pred_uncond,
            guidance_scale,
            momentum_buffer=None,
            eta=1.0,
            norm_threshold=0.0,
        ):
            vendor.single_calls.append(
                (guidance_scale, momentum_buffer, eta, norm_threshold)
            )
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
            vendor.chain_calls.append(
                (
                    tuple(scales),
                    tuple(momentum_buffers),
                    eta,
                    tuple(norm_thresholds),
                )
            )
            bases = [pred_uncond] + list(preds)
            result = pred_uncond
            for index, pred in enumerate(preds):
                normalized = normalize_diff(
                    pred - bases[index],
                    pred,
                    momentum_buffers[index],
                    eta,
                    norm_thresholds[index],
                )
                result = result + scales[index] * normalized
            return result

        normalized_guidance.__module__ = native.VENDOR_APG_MODULE
        normalized_guidance_chain.__module__ = native.VENDOR_APG_MODULE
        vendor.MomentumBuffer = MomentumBuffer
        vendor.normalized_guidance = normalized_guidance
        vendor.normalized_guidance_chain = normalized_guidance_chain
        vendor.single_calls = []
        vendor.chain_calls = []
        sys.modules[native.VENDOR_APG_MODULE] = vendor
        self.vendor = vendor

    def tearDown(self) -> None:
        if self._saved_vendor is None:
            sys.modules.pop(native.VENDOR_APG_MODULE, None)
        else:
            sys.modules[native.VENDOR_APG_MODULE] = self._saved_vendor

    def test_t2v_two_raw_forwards_reuse_exact_objects_and_apg_arithmetic(self):
        fixture = NativeFieldFixture("t2v_apg")
        adapter = fixture.adapter()
        state = torch.zeros((1, 16, 21, 2, 2), dtype=torch.float32)
        result = adapter(fixture.request(role="target", candidate=0, state=state))

        self.assertEqual(len(fixture.diffusion.calls), 2)
        first, second = fixture.diffusion.calls
        self.assertEqual(first["total"], 21)
        self.assertEqual(first["batch_vae_seqlen"], (21,))
        self.assertEqual(
            (first["tokens_id"], first["rotary_id"], first["timestep_id"]),
            (second["tokens_id"], second["rotary_id"], second["timestep_id"]),
        )
        self.assertEqual(fixture.transformer.source_ids, [0])
        negative_clean = state - fixture.sigmas[0] * torch.ones_like(state)
        condition_clean = state - fixture.sigmas[0] * torch.full_like(state, 3.0)
        expected_clean = negative_clean + 4.0 * independent_zero_momentum_normalize(
            condition_clean - negative_clean, condition_clean
        )
        expected = (state - expected_clean) / fixture.sigmas[0]
        torch.testing.assert_close(result, expected.float(), rtol=0.0, atol=0.0)
        self.assertEqual(result.dtype, torch.float32)
        self.assertFalse(result.requires_grad)
        self.assertNotEqual(result.untyped_storage().data_ptr(), state.untyped_storage().data_ptr())
        self.assertEqual(adapter.diagnostics.raw_transformer_forward_count, 2)
        self.assertEqual(adapter.diagnostics.patch_query_count, 1)
        self.assertEqual(adapter.diagnostics.vendor_single_attempt_count, 1)
        self.assertEqual(adapter.diagnostics.vendor_single_success_count, 1)
        self.assertEqual(adapter.diagnostics.vendor_chain_attempt_count, 0)
        self.assertEqual(len(self.vendor.single_calls), 1)
        self.assertEqual(
            (
                self.vendor.single_calls[0][0],
                self.vendor.single_calls[0][2],
                self.vendor.single_calls[0][3],
            ),
            (4.0, 0.5, 50.0),
        )
        self.assertFalse(adapter.diagnostics.optimizer_step_allowed)

    def test_r2v_i0_has_nref_plus_n_and_direct_target_only_output(self):
        fixture = NativeFieldFixture("r2v_apg_source_i0")
        adapter = fixture.adapter()
        target_state = torch.zeros((1, 16, 21, 2, 2), dtype=torch.float32)
        source_state = torch.ones((1, 16, 21, 2, 2), dtype=torch.float32)
        target_result = adapter(
            fixture.request(role="target", candidate=0, state=target_state)
        )
        source_result = adapter(
            fixture.request(role="source", candidate=0, state=source_state)
        )

        self.assertEqual(tuple(target_result.shape), tuple(target_state.shape))
        self.assertEqual(tuple(source_result.shape), tuple(source_state.shape))
        self.assertLess(float(target_result.abs().max().item()), 20.0)
        self.assertLess(float(source_result.abs().max().item()), 20.0)
        self.assertEqual(len(fixture.diffusion.calls), 6)
        self.assertEqual(
            [call["total"] for call in fixture.diffusion.calls],
            [21, 22, 22, 21, 22, 22],
        )
        self.assertEqual(
            [call["batch_vae_seqlen"] for call in fixture.diffusion.calls],
            [(21,), (22,), (22,), (21,), (22,), (22,)],
        )
        for start in (0, 3):
            none_call, image_negative_call, image_role_call = (
                fixture.diffusion.calls[start : start + 3]
            )
            self.assertNotEqual(
                none_call["tokens_id"], image_negative_call["tokens_id"]
            )
            self.assertEqual(
                image_negative_call["tokens_id"], image_role_call["tokens_id"]
            )
            self.assertEqual(
                image_negative_call["rotary_id"], image_role_call["rotary_id"]
            )
            self.assertEqual(
                none_call["timestep_id"], image_negative_call["timestep_id"]
            )
            self.assertEqual(
                image_negative_call["timestep_id"], image_role_call["timestep_id"]
            )
        self.assertEqual(
            [call["visual_first"] for call in fixture.diffusion.calls],
            [1.0, 2.0, 2.0, 1.0, 2.0, 2.0],
        )
        self.assertTrue(all(call["visual_last"] == 1.0 for call in fixture.diffusion.calls))
        self.assertEqual(
            [call["rotary_first"] for call in fixture.diffusion.calls],
            [1.0 + 0.0j, 2.0 + 0.0j, 2.0 + 0.0j,
             1.0 + 0.0j, 2.0 + 0.0j, 2.0 + 0.0j],
        )
        self.assertTrue(all(call["rotary_last"] == 1.0 + 0.0j for call in fixture.diffusion.calls))
        self.assertEqual(fixture.transformer.source_ids, [1.0, 0.0, 1.0, 0.0])
        self.assertEqual(
            fixture.transformer.patch_input_digests[0],
            fixture.transformer.patch_input_digests[2],
        )
        self.assertEqual(
            [call["condition_id"] for call in fixture.diffusion.calls],
            [
                id(fixture.conditions["negative"]),
                id(fixture.conditions["negative"]),
                id(fixture.conditions["target"]),
                id(fixture.conditions["negative"]),
                id(fixture.conditions["negative"]),
                id(fixture.conditions["source"]),
            ],
        )
        self.assertEqual(adapter.diagnostics.patch_reference_count, 2)
        self.assertEqual(adapter.diagnostics.patch_query_count, 2)
        self.assertEqual(adapter.diagnostics.raw_transformer_forward_count, 6)
        self.assertEqual(adapter.diagnostics.raw_transformer_forward_success_count, 6)
        self.assertEqual(adapter.diagnostics.vendor_chain_attempt_count, 2)
        self.assertEqual(adapter.diagnostics.vendor_chain_success_count, 2)
        self.assertEqual(adapter.diagnostics.vendor_single_attempt_count, 0)
        self.assertEqual(
            adapter.diagnostics.expected_raw_transformer_forward_count, 312
        )
        self.assertEqual(len(self.vendor.chain_calls), 2)
        self.assertTrue(
            all(call[0] == (4.5, 4.0) for call in self.vendor.chain_calls)
        )
        self.assertTrue(all(call[2:] == (0.5, (50.0, 50.0)) for call in self.vendor.chain_calls))
        sigma = fixture.sigmas[0]
        for state, role_value, observed in (
            (target_state, 3.25, target_result),
            (source_state, 2.25, source_result),
        ):
            no_visual_negative = state - sigma * torch.full_like(state, 1.0)
            image_negative = state - sigma * torch.full_like(state, 1.25)
            image_role = state - sigma * torch.full_like(state, role_value)
            expected_clean = independent_native_r2v_chain(
                no_visual_negative, image_negative, image_role
            )
            expected_velocity = ((state - expected_clean) / sigma).float()
            torch.testing.assert_close(
                observed, expected_velocity, rtol=0.0, atol=0.0
            )
        self.assertEqual(adapter.diagnostics.guided_query_count, 2)
        self.assertEqual(adapter.diagnostics.next_candidate_index, 1)
        self.assertEqual(adapter.diagnostics.next_role, "target")

    def test_call_order_drift_fails_before_a_forward(self):
        fixture = NativeFieldFixture("t2v_apg")
        adapter = fixture.adapter()
        state = torch.zeros((1, 16, 21, 2, 2), dtype=torch.float32)
        with self.assertRaisesRegex(native.SAICNativeSourceStateFieldError, "call-order drift"):
            adapter(fixture.request(role="source", candidate=0, state=state))
        self.assertEqual(len(fixture.diffusion.calls), 0)

    def test_mixed_regime_fails_closed(self):
        fixture = NativeFieldFixture("t2v_apg")
        adapter = fixture.adapter()
        state = torch.zeros((1, 16, 21, 2, 2), dtype=torch.float32)
        with self.assertRaisesRegex(native.SAICNativeSourceStateFieldError, "regime closure"):
            adapter(
                fixture.request(
                    role="target",
                    candidate=0,
                    state=state,
                    regime="r2v_apg_source_i0",
                )
            )

    def test_trainable_or_mutated_model_fails(self):
        fixture = NativeFieldFixture("t2v_apg")
        fixture.transformer.seal.requires_grad_(True)
        with self.assertRaisesRegex(native.SAICNativeSourceStateFieldError, "trainable"):
            fixture.adapter()

        fixture = NativeFieldFixture("t2v_apg")
        adapter = fixture.adapter()
        fixture.transformer.seal.data.add_(1.0)
        with self.assertRaisesRegex(native.SAICNativeSourceStateFieldError, "content was mutated"):
            adapter.finalize()

    def test_fp32_prompt_condition_is_rejected_as_non_native(self):
        fixture = NativeFieldFixture("t2v_apg")
        fixture.conditions["target"] = fixture.conditions["target"].float()
        fixture.provenance.prompt_condition_sha256_by_key["target"] = digest_tensor(
            fixture.conditions["target"]
        )
        with self.assertRaisesRegex(native.SAICNativeSourceStateFieldError, "BF16"):
            fixture.adapter()

    def test_per_query_audit_does_not_read_full_model_content(self):
        fixture = NativeFieldFixture("t2v_apg")
        adapter = fixture.adapter()
        state = torch.zeros((1, 16, 21, 2, 2), dtype=torch.float32)
        with mock.patch.object(
            native,
            "_module_content_seal",
            side_effect=AssertionError("full seal called inside query"),
        ):
            result = adapter(
                fixture.request(role="target", candidate=0, state=state)
            )
        self.assertEqual(tuple(result.shape), tuple(state.shape))
        self.assertFalse(adapter.diagnostics.final_full_model_content_audit)
        self.assertFalse(adapter.diagnostics.rollout_complete)

    def test_k1_uniform_arm_binds_80_guided_and_160_raw_expected_calls(self):
        fixture = NativeFieldFixture("t2v_apg", k1=True)
        adapter = fixture.adapter()
        diagnostics = adapter.diagnostics
        self.assertEqual(diagnostics.expected_guided_query_count, 80)
        self.assertEqual(diagnostics.expected_raw_transformer_forward_count, 160)
        self.assertEqual(diagnostics.final_model_content_seal_sha256_by_module, ())
        self.assertFalse(diagnostics.model_checkpoint_use_verified)
        state = torch.zeros((1, 16, 21, 2, 2), dtype=torch.float32)
        adapter(fixture.request(role="target", candidate=0, state=state))
        adapter(fixture.request(role="source", candidate=0, state=state.clone()))
        self.assertEqual(adapter.diagnostics.next_step_index, 1)

    def test_reference_contract_rejects_wrong_count_geometry_and_alias(self):
        fixture = NativeFieldFixture("r2v_apg_source_i0")
        fixture.reference = torch.zeros((1, 16, 2, 2, 2), dtype=torch.float32)
        with self.assertRaisesRegex(native.SAICNativeSourceStateFieldError, "geometry"):
            fixture.adapter()

        fixture = NativeFieldFixture("t2v_apg")
        fixture.reference = torch.zeros((1, 16, 1, 2, 2), dtype=torch.float32)
        with self.assertRaisesRegex(native.SAICNativeSourceStateFieldError, "cannot consume"):
            fixture.adapter()

        fixture = NativeFieldFixture("r2v_apg_source_i0")
        fixture.reference = fixture.reference.clone().add_(1.0)
        with self.assertRaisesRegex(native.SAICNativeSourceStateFieldError, "bytes differ"):
            fixture.adapter()

    def test_shared_step_mutation_of_packed_query_is_detected(self):
        class MutatingDiffusion(FakeDiffusion):
            def shared_step(self, **kwargs):
                result = super().shared_step(**kwargs)
                kwargs["noisy_latents"].add_(1)
                return result

        fixture = NativeFieldFixture("t2v_apg")
        fixture.diffusion = MutatingDiffusion(fixture.transformer)
        adapter = fixture.adapter()
        state = torch.zeros((1, 16, 21, 2, 2), dtype=torch.float32)
        with self.assertRaisesRegex(native.SAICNativeSourceStateFieldError, "bytes or objects changed"):
            adapter(fixture.request(role="target", candidate=0, state=state))

    def test_replaced_native_callable_fails_before_dispatch_and_poison_is_permanent(self):
        fixture = NativeFieldFixture("t2v_apg")
        adapter = fixture.adapter()

        def forged_shared_step(**kwargs):
            raise AssertionError("replacement must never execute")

        fixture.diffusion.shared_step = forged_shared_step
        state = torch.zeros((1, 16, 21, 2, 2), dtype=torch.float32)
        request = fixture.request(role="target", candidate=0, state=state)
        with self.assertRaisesRegex(
            native.SAICNativeSourceStateFieldError, "callable identity"
        ):
            adapter(request)
        self.assertEqual(adapter.diagnostics.raw_transformer_forward_count, 0)
        self.assertTrue(adapter.diagnostics.adapter_failed)
        with self.assertRaisesRegex(
            native.SAICNativeSourceStateFieldError, "cannot be resumed"
        ):
            adapter(request)

    def test_live_adapter_owner_and_regime_properties_are_read_only(self):
        fixture = NativeFieldFixture("t2v_apg")
        adapter = fixture.adapter()
        with self.assertRaises(AttributeError):
            adapter.diffusion = FakeDiffusion(fixture.transformer)
        with self.assertRaises(AttributeError):
            adapter.transformer = FakeTransformer()
        with self.assertRaises(AttributeError):
            adapter.field_regime = "r2v_apg_source_i0"

    def test_class_descriptor_patch_rebinding_is_detected_and_never_called(self):
        fixture = NativeFieldFixture("t2v_apg")
        adapter = fixture.adapter()
        original = FakeTransformer.patch_vae_latent

        def forged_patch(self, *, hidden_states, source_id):
            raise AssertionError("rebound class descriptor must never execute")

        FakeTransformer.patch_vae_latent = forged_patch
        try:
            state = torch.zeros((1, 16, 21, 2, 2), dtype=torch.float32)
            with self.assertRaisesRegex(
                native.SAICNativeSourceStateFieldError, "callable identity"
            ):
                adapter(fixture.request(role="target", candidate=0, state=state))
            self.assertEqual(adapter.diagnostics.patch_query_attempt_count, 0)
        finally:
            FakeTransformer.patch_vae_latent = original

    def test_vendor_chain_rebinding_is_detected_before_any_native_call(self):
        fixture = NativeFieldFixture("r2v_apg_source_i0")
        adapter = fixture.adapter()

        def forged_chain(
            pred_uncond,
            preds,
            scales,
            momentum_buffers,
            eta,
            norm_thresholds,
        ):
            raise AssertionError("rebound vendor chain must never execute")

        self.vendor.normalized_guidance_chain = forged_chain
        state = torch.zeros((1, 16, 21, 2, 2), dtype=torch.float32)
        with self.assertRaisesRegex(
            native.SAICNativeSourceStateFieldError, "callable identity"
        ):
            adapter(fixture.request(role="target", candidate=0, state=state))
        diagnostics = adapter.diagnostics
        self.assertEqual(diagnostics.patch_reference_attempt_count, 0)
        self.assertEqual(diagnostics.patch_query_attempt_count, 0)
        self.assertEqual(diagnostics.raw_transformer_forward_attempt_count, 0)
        self.assertEqual(diagnostics.vendor_chain_attempt_count, 0)

    def test_vendor_chain_exception_after_triplet_is_counted_and_poisoned(self):
        def failing_chain(
            pred_uncond,
            preds,
            scales,
            momentum_buffers,
            eta,
            norm_thresholds,
        ):
            raise RuntimeError("synthetic authenticated-chain failure")

        failing_chain.__name__ = "normalized_guidance_chain"
        failing_chain.__module__ = native.VENDOR_APG_MODULE
        self.vendor.normalized_guidance_chain = failing_chain
        fixture = NativeFieldFixture("r2v_apg_source_i0")
        adapter = fixture.adapter()
        state = torch.zeros((1, 16, 21, 2, 2), dtype=torch.float32)
        request = fixture.request(role="target", candidate=0, state=state)
        with self.assertRaisesRegex(
            native.SAICNativeSourceStateFieldError, "permanently poisoned"
        ):
            adapter(request)
        diagnostics = adapter.diagnostics
        self.assertEqual(diagnostics.raw_transformer_forward_attempt_count, 3)
        self.assertEqual(diagnostics.raw_transformer_forward_success_count, 3)
        self.assertEqual(diagnostics.vendor_chain_attempt_count, 1)
        self.assertEqual(diagnostics.vendor_chain_success_count, 0)
        self.assertEqual(diagnostics.guided_query_attempt_count, 1)
        self.assertEqual(diagnostics.guided_query_success_count, 0)
        with self.assertRaisesRegex(
            native.SAICNativeSourceStateFieldError, "cannot be resumed"
        ):
            adapter(request)

    def test_nested_training_mode_drift_is_detected_before_dispatch(self):
        class NestedTransformer(FakeTransformer):
            def __init__(self):
                super().__init__()
                self.nested = torch.nn.Dropout(p=0.5)
                self.eval()

        fixture = NativeFieldFixture("t2v_apg")
        fixture.transformer = NestedTransformer()
        fixture.diffusion = FakeDiffusion(fixture.transformer)
        adapter = fixture.adapter()
        fixture.transformer.nested.train()
        state = torch.zeros((1, 16, 21, 2, 2), dtype=torch.float32)
        with self.assertRaisesRegex(
            native.SAICNativeSourceStateFieldError, "must remain in eval mode"
        ):
            adapter(fixture.request(role="target", candidate=0, state=state))
        self.assertEqual(adapter.diagnostics.raw_transformer_forward_attempt_count, 0)

    def test_second_raw_forward_exception_counts_attempt_and_poison_blocks_retry(self):
        class SecondForwardFails(FakeDiffusion):
            def shared_step(self, **kwargs):
                result = super().shared_step(**kwargs)
                if len(self.calls) == 2:
                    raise RuntimeError("synthetic second-forward failure")
                return result

        fixture = NativeFieldFixture("t2v_apg")
        fixture.diffusion = SecondForwardFails(fixture.transformer)
        adapter = fixture.adapter()
        state = torch.zeros((1, 16, 21, 2, 2), dtype=torch.float32)
        request = fixture.request(role="target", candidate=0, state=state)
        with self.assertRaisesRegex(
            native.SAICNativeSourceStateFieldError, "permanently poisoned"
        ):
            adapter(request)
        self.assertEqual(len(fixture.diffusion.calls), 2)
        self.assertEqual(adapter.diagnostics.raw_transformer_forward_count, 2)
        self.assertEqual(adapter.diagnostics.raw_transformer_forward_attempt_count, 2)
        self.assertEqual(adapter.diagnostics.raw_transformer_forward_success_count, 1)
        self.assertEqual(adapter.diagnostics.guided_query_attempt_count, 1)
        self.assertEqual(adapter.diagnostics.guided_query_success_count, 0)
        self.assertEqual(adapter.diagnostics.guided_query_count, 0)
        with self.assertRaisesRegex(
            native.SAICNativeSourceStateFieldError, "cannot be resumed"
        ):
            adapter(request)
        self.assertEqual(len(fixture.diffusion.calls), 2)
        with self.assertRaisesRegex(
            native.SAICNativeSourceStateFieldError, "cannot be finalized"
        ):
            adapter.finalize()

    def test_second_reference_patch_failure_preserves_exact_native_attempt_counts(self):
        class SecondReferenceFails(FakeTransformer):
            def patch_vae_latent(self, *, hidden_states, source_id):
                result = super().patch_vae_latent(
                    hidden_states=hidden_states, source_id=source_id
                )
                if source_id == 1.0 and self.source_ids.count(1.0) == 2:
                    raise RuntimeError("synthetic second-I0 patch failure")
                return result

        fixture = NativeFieldFixture("r2v_apg_source_i0")
        fixture.transformer = SecondReferenceFails()
        fixture.diffusion = FakeDiffusion(fixture.transformer)
        adapter = fixture.adapter()
        target_state = torch.zeros((1, 16, 21, 2, 2), dtype=torch.float32)
        source_state = torch.ones((1, 16, 21, 2, 2), dtype=torch.float32)
        adapter(fixture.request(role="target", candidate=0, state=target_state))
        source_request = fixture.request(
            role="source", candidate=0, state=source_state
        )
        with self.assertRaisesRegex(
            native.SAICNativeSourceStateFieldError, "permanently poisoned"
        ):
            adapter(source_request)
        diagnostics = adapter.diagnostics
        self.assertEqual(diagnostics.guided_query_attempt_count, 2)
        self.assertEqual(diagnostics.guided_query_success_count, 1)
        self.assertEqual(diagnostics.patch_reference_attempt_count, 2)
        self.assertEqual(diagnostics.patch_reference_success_count, 1)
        self.assertEqual(diagnostics.patch_query_attempt_count, 1)
        self.assertEqual(diagnostics.patch_query_success_count, 1)
        self.assertEqual(diagnostics.raw_transformer_forward_attempt_count, 3)
        self.assertEqual(diagnostics.raw_transformer_forward_success_count, 3)
        self.assertEqual(
            fixture.transformer.source_ids, [1.0, 0.0, 1.0]
        )
        with self.assertRaisesRegex(
            native.SAICNativeSourceStateFieldError, "cannot be resumed"
        ):
            adapter(source_request)

    def test_public_constant_and_receipt_class_rebinding_cannot_change_live_adapter(self):
        fixture = NativeFieldFixture("t2v_apg")
        adapter = fixture.adapter()
        state = torch.zeros((1, 16, 21, 2, 2), dtype=torch.float32)
        with mock.patch.multiple(
            native,
            GUIDANCE_SCALE=400.0,
            IMAGE_GUIDANCE_SCALE=900.0,
            APG_ETA=9.0,
            APG_NORM_THRESHOLD=0.0,
            PATCH_CHANNELS=1,
            TEXT_TOKENS=1,
            TEXT_DIM=1,
            REGISTERED_K5_EARLY_SCHEDULE=(9,) * 40,
            NativeFieldDiagnostics=object,
        ), mock.patch.multiple(
            native.runtime_contract,
            PINNED_INNER_DIM=1,
            PINNED_PATCH_DIM=1,
            PINNED_ROPE_DIM=1,
        ):
            result = adapter(
                fixture.request(role="target", candidate=0, state=state)
            )
            diagnostics = adapter.diagnostics
        self.assertEqual(tuple(result.shape), tuple(state.shape))
        self.assertIs(type(diagnostics), native._NATIVE_FIELD_DIAGNOSTICS_TYPE)
        self.assertLess(float(result.abs().max().item()), 100.0)

    def test_caller_mapping_replacement_after_init_is_not_consumed(self):
        fixture = NativeFieldFixture("t2v_apg")
        adapter = fixture.adapter()
        original_target = fixture.conditions["target"]
        state = torch.zeros((1, 16, 21, 2, 2), dtype=torch.float32)
        request = fixture.request(role="target", candidate=0, state=state)
        fixture.conditions["target"] = torch.full_like(original_target, 17.0)
        fixture.captions["target"] = "forged caption"
        adapter(request)
        self.assertEqual(
            fixture.diffusion.calls[1]["condition_id"], id(original_target)
        )

    def test_inference_mode_state_and_independent_apg_reference(self):
        fixture = NativeFieldFixture("t2v_apg")
        adapter = fixture.adapter()
        with torch.inference_mode():
            state = torch.linspace(
                -1.0, 1.0, 1 * 16 * 21 * 2 * 2, dtype=torch.float32
            ).reshape(1, 16, 21, 2, 2)
        result = adapter(fixture.request(role="target", candidate=0, state=state))

        negative_clean = state - fixture.sigmas[0] * torch.ones_like(state)
        condition_clean = state - fixture.sigmas[0] * torch.full_like(state, 3.0)
        diff = condition_clean - negative_clean
        norm = diff.norm(p=2, dim=[-1, -2, -4], keepdim=True)
        clipped = diff * torch.minimum(torch.ones_like(diff), 50.0 / norm)
        axis = torch.nn.functional.normalize(
            condition_clean.double(), dim=[-1, -2, -4]
        )
        parallel = (clipped.double() * axis).sum(
            dim=[-1, -2, -4], keepdim=True
        ) * axis
        normalized = (clipped.double() - parallel).to(clipped.dtype) + 0.5 * parallel.to(
            clipped.dtype
        )
        expected_clean = negative_clean + 4.0 * normalized
        expected = ((state - expected_clean) / fixture.sigmas[0]).float()
        torch.testing.assert_close(result, expected, rtol=0.0, atol=0.0)

    def test_non_bf16_transformer_and_patch_tokens_are_rejected(self):
        fixture = NativeFieldFixture("t2v_apg")
        fixture.transformer.dtype = torch.float32
        with self.assertRaisesRegex(
            native.SAICNativeSourceStateFieldError, "exactly torch.bfloat16"
        ):
            fixture.adapter()

        class FP32PatchTransformer(FakeTransformer):
            def patch_vae_latent(self, *, hidden_states, source_id):
                tokens, rotary = super().patch_vae_latent(
                    hidden_states=hidden_states, source_id=source_id
                )
                return tokens.float(), rotary

        fixture = NativeFieldFixture("t2v_apg")
        fixture.transformer = FP32PatchTransformer()
        fixture.diffusion = FakeDiffusion(fixture.transformer)
        adapter = fixture.adapter()
        state = torch.zeros((1, 16, 21, 2, 2), dtype=torch.float32)
        with self.assertRaisesRegex(
            native.SAICNativeSourceStateFieldError, "token geometry"
        ):
            adapter(fixture.request(role="target", candidate=0, state=state))
        self.assertEqual(adapter.diagnostics.patch_query_count, 1)
        self.assertTrue(adapter.diagnostics.adapter_failed)

    def test_premature_finalize_poison_is_irreversible_and_binds_provenance(self):
        fixture = NativeFieldFixture("t2v_apg", k1=True)
        adapter = fixture.adapter()
        before = adapter.diagnostics
        self.assertEqual(before.model_receipt_sha256, "5" * 64)
        self.assertEqual(before.native_schedule_sha256, fixture.provenance.native_schedule_sha256)
        self.assertRegex(before.provenance_seal_sha256, r"^[0-9a-f]{64}$")
        with self.assertRaisesRegex(
            native.SAICNativeSourceStateFieldError, "cannot finalize before"
        ):
            adapter.finalize()
        self.assertTrue(adapter.diagnostics.adapter_failed)
        with self.assertRaisesRegex(
            native.SAICNativeSourceStateFieldError, "cannot be finalized"
        ):
            adapter.finalize()

    def test_complete_k1_t2v_finalize_is_idempotent_and_counts_exact_successes(self):
        fixture = NativeFieldFixture("t2v_apg", k1=True)
        adapter = fixture.adapter()
        state = torch.zeros((1, 16, 21, 2, 2), dtype=torch.float32)
        for step_index in range(40):
            adapter(
                fixture.request(
                    role="target",
                    candidate=0,
                    state=state,
                    step_index=step_index,
                )
            )
            adapter(
                fixture.request(
                    role="source",
                    candidate=0,
                    state=state,
                    step_index=step_index,
                )
            )
        receipt = adapter.finalize()
        repeated = adapter.finalize()
        self.assertIs(receipt, repeated)
        self.assertTrue(receipt.rollout_complete)
        self.assertFalse(receipt.adapter_failed)
        self.assertEqual(receipt.guided_query_attempt_count, 80)
        self.assertEqual(receipt.guided_query_success_count, 80)
        self.assertEqual(receipt.raw_transformer_forward_attempt_count, 160)
        self.assertEqual(receipt.raw_transformer_forward_success_count, 160)
        self.assertEqual(receipt.patch_query_attempt_count, 80)
        self.assertEqual(receipt.patch_query_success_count, 80)
        self.assertEqual(receipt.patch_reference_attempt_count, 0)
        self.assertEqual(receipt.vendor_single_attempt_count, 80)
        self.assertEqual(receipt.vendor_single_success_count, 80)
        self.assertEqual(receipt.vendor_chain_attempt_count, 0)
        self.assertTrue(receipt.final_full_model_content_audit)
        self.assertTrue(receipt.raw_transformer_forward_count_verified)
        self.assertTrue(receipt.native_request_execution_verified)
        self.assertTrue(receipt.vendor_apg_execution_verified)
        self.assertFalse(receipt.model_checkpoint_use_verified)
        self.assertEqual(len(fixture.diffusion.calls), 160)

    def test_complete_k1_r2v_finalize_binds_three_forward_chain_and_patch_order(self):
        fixture = NativeFieldFixture("r2v_apg_source_i0", k1=True)
        adapter = fixture.adapter()
        state = torch.zeros((1, 16, 21, 2, 2), dtype=torch.float32)
        for step_index in range(40):
            for role in ("target", "source"):
                adapter(
                    fixture.request(
                        role=role,
                        candidate=0,
                        state=state,
                        step_index=step_index,
                    )
                )
        receipt = adapter.finalize()
        self.assertTrue(receipt.rollout_complete)
        self.assertEqual(receipt.guided_query_attempt_count, 80)
        self.assertEqual(receipt.guided_query_success_count, 80)
        self.assertEqual(receipt.raw_transformer_forward_attempt_count, 240)
        self.assertEqual(receipt.raw_transformer_forward_success_count, 240)
        self.assertEqual(receipt.expected_raw_transformer_forward_count, 240)
        self.assertEqual(receipt.patch_query_attempt_count, 80)
        self.assertEqual(receipt.patch_query_success_count, 80)
        self.assertEqual(receipt.patch_reference_attempt_count, 80)
        self.assertEqual(receipt.patch_reference_success_count, 80)
        self.assertEqual(receipt.vendor_chain_attempt_count, 80)
        self.assertEqual(receipt.vendor_chain_success_count, 80)
        self.assertEqual(receipt.vendor_single_attempt_count, 0)
        self.assertEqual(
            fixture.transformer.source_ids,
            [value for _ in range(80) for value in (1.0, 0.0)],
        )
        self.assertEqual(len(fixture.diffusion.calls), 240)


if __name__ == "__main__":
    unittest.main()
