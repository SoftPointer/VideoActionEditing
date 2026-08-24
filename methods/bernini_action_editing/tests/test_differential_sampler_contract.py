from __future__ import annotations

import inspect
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import differential_sampler as sampler  # noqa: E402


class _ArrayLike:
    def __init__(self, values, *, shape=(1, 2, 3), dtype="bf16"):
        self.values = values
        self.shape = shape
        self.dtype = dtype


class _Diffusion:
    scheduler = object()
    transformer = object()

    def shared_step(self):
        raise AssertionError("not called")


class _Renderer:
    def __init__(self):
        self.diff_dec = _Diffusion()


class DifferentialSamplerContractTests(unittest.TestCase):
    def test_contract_is_source_prompt_only_and_has_no_anc(self) -> None:
        contract = sampler.sampler_contract()
        self.assertEqual(
            contract["inference_conditions"],
            [
                "clean_source_vae_latent",
                "action_prompt_embedding",
                "noop_prompt_embedding",
            ],
        )
        self.assertEqual(contract["noise_policy"], "one_fixed_shared_gaussian")
        self.assertIs(contract["fresh_per_step_probe_noise"], False)
        self.assertIs(contract["anc"], False)
        self.assertEqual(contract["source_id"], 1.0)
        self.assertEqual(contract["query_id"], 0.0)
        self.assertEqual(contract["ulysses_world_size_tested"], 4)
        self.assertEqual(contract["sequence_parallel_owner"], "official_bernini_transformer")

        parameters = set(inspect.signature(sampler.sample_differential_flow).parameters)
        self.assertTrue(
            {"source_latent", "action_prompt_embeds", "noop_prompt_embeds"} <= parameters
        )
        forbidden = {
            "target",
            "target_video",
            "mask",
            "track",
            "tube",
            "pose",
            "trajectory",
            "flow",
            "anchor",
        }
        self.assertTrue(parameters.isdisjoint(forbidden))

    def test_config_fails_closed(self) -> None:
        self.assertIsInstance(sampler.DifferentialFlowConfig().validate(), sampler.DifferentialFlowConfig)
        for kwargs in (
            {"num_inference_steps": 0},
            {"num_inference_steps": True},
            {"seed": -1},
            {"flow_shift": 0.0},
            {"motion_scale": -0.1},
            {"motion_scale": float("nan")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(
                sampler.DifferentialSamplerContractError
            ):
                sampler.DifferentialFlowConfig(**kwargs).validate()

    def test_latent_layout_matches_wan_1x2x2_patch_order_geometry(self) -> None:
        layout = sampler.validate_latent_shape((1, 16, 21, 60, 104))
        self.assertEqual(layout.tokens, 21 * 30 * 52)
        self.assertEqual(layout.packed_channels, 64)
        for shape in (
            (2, 16, 21, 60, 104),
            (1, 16, 21, 61, 104),
            (1, 16, 21, 60),
            (1, 16, 0, 60, 104),
        ):
            with self.subTest(shape=shape), self.assertRaises(
                sampler.DifferentialSamplerContractError
            ):
                sampler.validate_latent_shape(shape)

    def test_sigma_intervals_support_unipc_and_flowmatch_forms(self) -> None:
        unipc = sampler.descending_sigma_intervals(
            [1.0, 0.75, 0.2, 0.0], expected_steps=3
        )
        self.assertEqual(unipc, ((1.0, 0.75), (0.75, 0.2), (0.2, 0.0)))
        flowmatch = sampler.descending_sigma_intervals(
            [1.0, 0.5, 0.01], expected_steps=3
        )
        self.assertEqual(flowmatch[-1], (0.01, 0.0))
        for values in (
            [1.0, 0.4],
            [0.9, 1.0, 0.0, 0.0],
            [1.1, 0.5, 0.0, 0.0],
            [1.0, 0.5, 0.1, 0.01],
        ):
            with self.subTest(values=values), self.assertRaises(
                sampler.DifferentialSamplerContractError
            ):
                sampler.descending_sigma_intervals(values, expected_steps=3)

    def test_prompt_identity_is_exact_not_semantic_or_approximate(self) -> None:
        same = _ArrayLike([1, 2])
        self.assertTrue(sampler.prompts_are_exactly_identical(same, same))
        left = _ArrayLike([1, 2])
        right = _ArrayLike([1, 2])
        self.assertTrue(
            sampler.prompts_are_exactly_identical(
                left, right, tensor_equal=lambda a, b: a.values == b.values
            )
        )
        self.assertFalse(
            sampler.prompts_are_exactly_identical(
                left,
                _ArrayLike([1, 2], shape=(1, 4, 3)),
                tensor_equal=lambda a, b: True,
            )
        )
        self.assertFalse(
            sampler.prompts_are_exactly_identical(
                left,
                _ArrayLike([1, 2], dtype="fp32"),
                tensor_equal=lambda a, b: True,
            )
        )

    def test_exact_noop_bypasses_before_torch_or_model_work(self) -> None:
        source = object()
        prompt = object()
        result = sampler.sample_differential_flow(
            object(),
            source_latent=source,
            action_prompt_embeds=prompt,
            noop_prompt_embeds=prompt,
        )
        self.assertIs(result, source)
        traced, trace = sampler.sample_differential_flow(
            object(),
            source_latent=source,
            action_prompt_embeds=object(),
            noop_prompt_embeds=object(),
            config=sampler.DifferentialFlowConfig(motion_scale=0.0),
            return_trace=True,
        )
        self.assertIs(traced, source)
        self.assertTrue(trace.identity_bypassed)
        self.assertEqual(trace.sigmas, ())

    def test_renderer_and_peft_style_resolution(self) -> None:
        renderer = _Renderer()
        self.assertIs(sampler.resolve_diffusion_core(renderer), renderer.diff_dec)

        class Wrapper:
            def get_base_model(self):
                return renderer

        self.assertIs(sampler.resolve_diffusion_core(Wrapper()), renderer.diff_dec)
        with self.assertRaises(sampler.DifferentialSamplerContractError):
            sampler.resolve_diffusion_core(object())

    def test_torch_pack_round_trip_if_available(self) -> None:
        try:
            import torch
        except Exception as error:  # pragma: no cover - minimal local environment
            self.skipTest(f"torch unavailable: {error}")
        latent = torch.arange(1 * 3 * 2 * 4 * 6, dtype=torch.float32).reshape(1, 3, 2, 4, 6)
        layout = sampler.validate_latent_shape(tuple(latent.shape))
        packed = sampler._pack_spatial_latent(latent, layout)
        self.assertEqual(tuple(packed.shape), (1, 2 * 2 * 3, 3 * 4))
        restored = sampler._unpack_spatial_latent(packed, layout)
        self.assertTrue(torch.equal(restored, latent))

    def test_torch_fake_bernini_assembles_source_and_query_and_integrates_sign(self) -> None:
        try:
            import torch
        except Exception as error:  # pragma: no cover - minimal local environment
            self.skipTest(f"torch unavailable: {error}")

        class Config:
            in_channels = 1
            text_dim = 1

        class Transformer:
            config = Config()
            dtype = torch.float32

            def patch_vae_latent(self, latent, source_id):
                packed = sampler._pack_spatial_latent(latent)
                tokens = packed.mean(dim=2, keepdim=True)
                rotary = torch.full(
                    (1, 1, tokens.shape[1], 1), float(source_id), dtype=torch.float32
                )
                return tokens, rotary

        class SchedulerConfig:
            flow_shift = 5.0

        class Scheduler:
            config = SchedulerConfig()

            def set_timesteps(self, steps):
                self.timesteps = torch.tensor([1000.0, 500.0])
                self.sigmas = torch.tensor([1.0, 0.5, 0.0])

        class Diffusion:
            use_unipc = True
            transformer_2 = None

            def __init__(self):
                self.transformer = Transformer()
                self.scheduler = Scheduler()
                self.calls = []

            def shared_step(
                self,
                *,
                noisy_latents,
                timesteps,
                cond_embeds,
                rotary_embs,
                batch_vae_seqlen,
                batch_text_seqlen,
                **kwargs,
            ):
                self.calls.append(
                    {
                        "sequence": noisy_latents.shape[1],
                        "rotary_sequence": rotary_embs.shape[2],
                        "vae_len": batch_vae_seqlen,
                        "text_len": batch_text_seqlen,
                    }
                )
                # Perfect toy fields: v_action=-1 and v_noop=0.  Because sigma
                # descends by a total of -1, Euler(delta_v=-1) must yield +1.
                value = -float(cond_embeds[0, 0, 0].item())
                return torch.full((1, noisy_latents.shape[1], 4), value)

        diffusion = Diffusion()
        source = torch.zeros(1, 1, 1, 2, 2)
        action = torch.ones(1, 1, 1)
        noop = torch.zeros(1, 1, 1)
        result = sampler.sample_differential_flow(
            diffusion,
            source_latent=source,
            action_prompt_embeds=action,
            noop_prompt_embeds=noop,
            config=sampler.DifferentialFlowConfig(
                num_inference_steps=2, flow_shift=5.0, seed=9
            ),
        )
        self.assertTrue(torch.equal(result, torch.ones_like(result)))
        self.assertEqual(len(diffusion.calls), 4)
        for call in diffusion.calls:
            self.assertEqual(call["sequence"], 2)  # source token + query token
            self.assertEqual(call["rotary_sequence"], 2)
            self.assertEqual(call["vae_len"], [2])
            self.assertEqual(call["text_len"], [1])


if __name__ == "__main__":
    unittest.main()
