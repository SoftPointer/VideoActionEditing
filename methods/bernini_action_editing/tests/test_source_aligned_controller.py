from __future__ import annotations

import inspect
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import source_aligned_controller as sac  # noqa: E402


class SourceAlignedControllerContractTests(unittest.TestCase):
    def test_contract_is_source_instruction_only_and_honest(self) -> None:
        contract = sac.controller_contract()
        self.assertEqual(contract["user_inputs"], ["source_video", "edit_instruction"])
        self.assertEqual(contract["clip_geometry"], {"rgb_frames": 81, "wan_vae_phases": 21})
        self.assertTrue(contract["sga"]["enabled"])
        self.assertTrue(contract["anc"]["enabled"])
        self.assertFalse(contract["anc"]["fixed_noise"])
        self.assertIn("not_official_reproduction", contract["status"])
        self.assertIn("approximation", contract["candidate_chain_collapse"])
        forbidden = set(contract["forbidden_conditions"])
        self.assertTrue(
            {"target_video", "mask", "track", "swept_tube", "pose", "trajectory",
             "optical_flow", "first_frame_anchor"} <= forbidden
        )
        parameters = set(inspect.signature(sac.sample_source_aligned_controller).parameters)
        self.assertTrue(
            {"source_latent", "source_rgb_frames", "action_prompt_embeds", "noop_prompt_embeds"}
            <= parameters
        )
        self.assertTrue(parameters.isdisjoint(forbidden))

    def test_defaults_are_40_step_three_by_five_sga_and_anc(self) -> None:
        config = sac.SourceAlignedControllerConfig().validate()
        self.assertEqual(config.num_inference_steps, 40)
        self.assertEqual(config.sga_steps, 3)
        self.assertEqual(config.sga_candidates, 5)
        self.assertEqual(config.anc_lock_sigma, 0.25)

    def test_config_fails_closed(self) -> None:
        for kwargs in (
            {"num_inference_steps": 0},
            {"motion_scale": -0.1},
            {"sga_steps": -1},
            {"sga_steps": 41},
            {"sga_candidates": 1},
            {"sga_candidates": True},
            {"sga_temperature": 0.0},
            {"sga_temperature": float("nan")},
            {"anc_lock_sigma": -0.1},
            {"anc_lock_sigma": 1.0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(sac.SourceAlignedControllerError):
                sac.SourceAlignedControllerConfig(**kwargs).validate()

    def test_anc_schedule_is_variance_not_mislabeled_pearson(self) -> None:
        values = [sac.anc_retained_variance(sigma) for sigma in (1.0, 0.625, 0.25, 0.0)]
        self.assertEqual(values, [0.0, 0.5, 1.0, 1.0])
        self.assertAlmostEqual(values[1] ** 0.5, 2 ** -0.5)
        for sigma in (-0.1, 1.1, float("nan")):
            with self.subTest(sigma=sigma), self.assertRaises(sac.SourceAlignedControllerError):
                sac.anc_retained_variance(sigma)
        with self.assertRaises(sac.SourceAlignedControllerError):
            sac.anc_retained_variance(0.5, lock_sigma=False)

    def test_identity_bypasses_before_torch_but_still_requires_81_frames(self) -> None:
        source = object()
        prompt = object()
        result, trace = sac.sample_source_aligned_controller(
            object(),
            source_latent=source,
            source_rgb_frames=81,
            action_prompt_embeds=prompt,
            noop_prompt_embeds=prompt,
            return_trace=True,
        )
        self.assertIs(result, source)
        self.assertTrue(trace.identity_bypassed)
        self.assertEqual(trace.fresh_noise_draws, 0)
        with self.assertRaises(sac.SourceAlignedControllerError):
            sac.sample_source_aligned_controller(
                object(),
                source_latent=source,
                source_rgb_frames=41,
                action_prompt_embeds=prompt,
                noop_prompt_embeds=prompt,
            )


class SourceAlignedControllerTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
        except Exception as error:  # pragma: no cover
            raise unittest.SkipTest(f"torch unavailable: {error}")
        cls.torch = torch

    def test_flowedit_state_is_exactly_source_relative(self) -> None:
        torch = self.torch
        source = torch.tensor([[2.0, 4.0]])
        edit = torch.tensor([[3.0, 1.0]])
        noise = torch.tensor([[10.0, -2.0]])
        source_state, target_state = sac.flowedit_source_target_states(
            source, edit, noise, sigma=0.25
        )
        self.assertTrue(torch.allclose(source_state, 0.75 * source + 0.25 * noise))
        self.assertTrue(torch.allclose(target_state - source_state, edit - source))
        at_full_noise = sac.flowedit_source_target_states(source, source, noise, sigma=1.0)
        self.assertTrue(torch.equal(at_full_noise[0], noise))
        self.assertTrue(torch.equal(at_full_noise[1], noise))

    def test_anc_uses_fresh_noise_early_and_locks_late(self) -> None:
        torch = self.torch
        previous = torch.tensor([1.0, -1.0])
        fresh = torch.tensor([-1.0, 1.0])
        independent = sac.advance_anc_noise(previous, fresh, retained_variance=0.0)
        locked = sac.advance_anc_noise(previous, fresh, retained_variance=1.0)
        middle = sac.advance_anc_noise(previous, fresh, retained_variance=0.25)
        self.assertTrue(torch.equal(independent, fresh))
        self.assertTrue(torch.equal(locked, previous))
        self.assertTrue(torch.allclose(middle, 0.5 * previous + (0.75 ** 0.5) * fresh))

    def test_sga_soft_selects_more_source_aligned_projection(self) -> None:
        torch = self.torch
        source = torch.tensor([[1.0, 0.0]])
        edit = source.clone()
        # Projected endpoints are [1,0] and [0,1] at sigma=1.
        deltas = torch.stack(
            [torch.tensor([[0.0, 0.0]]), torch.tensor([[1.0, -1.0]])], dim=0
        )
        aggregate, weights, scores, projected = sac.similarity_guided_aggregate(
            source=source,
            edit=edit,
            candidate_deltas=deltas,
            sigma=1.0,
            temperature=0.01,
        )
        self.assertGreater(float(scores[0, 0]), float(scores[1, 0]))
        self.assertGreater(float(weights[0, 0]), 0.999)
        self.assertTrue(torch.allclose(projected[0], source))
        self.assertLess(float(aggregate.abs().max()), 1.0e-4)

    def test_noise_chain_collapse_has_unit_variance_normalisation(self) -> None:
        torch = self.torch
        noise = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
        weights = torch.tensor([[0.5], [0.5]])
        collapsed = sac.collapse_sga_noise_chains(noise, weights)
        self.assertTrue(torch.allclose(collapsed, torch.tensor([[2 ** -0.5, 2 ** -0.5]])))

    def test_fake_bernini_runs_real_sga_and_anc_on_exact_21_phases(self) -> None:
        torch = self.torch
        import differential_sampler as cdf

        class Config:
            in_channels = 1
            text_dim = 1

        class Transformer:
            config = Config()
            dtype = torch.float32

            def patch_vae_latent(self, latent, source_id):
                packed = cdf._pack_spatial_latent(latent)
                tokens = packed.mean(dim=2, keepdim=True)
                rotary = torch.full((1, 1, tokens.shape[1], 1), float(source_id))
                return tokens, rotary

        class SchedulerConfig:
            flow_shift = 5.0

        class Scheduler:
            config = SchedulerConfig()

            def set_timesteps(self, steps):
                self.timesteps = torch.tensor([1000.0, 625.0, 250.0])
                self.sigmas = torch.tensor([1.0, 0.625, 0.25, 0.0])

        class Diffusion:
            use_unipc = True
            transformer_2 = None

            def __init__(self):
                self.transformer = Transformer()
                self.scheduler = Scheduler()
                self.calls = 0

            def shared_step(self, *, noisy_latents, cond_embeds, **kwargs):
                self.calls += 1
                value = -float(cond_embeds[0, 0, 0])
                return torch.full((1, noisy_latents.shape[1], 4), value)

        diffusion = Diffusion()
        # Nonzero source is required because SGA's published cosine is undefined at zero.
        source = torch.ones(1, 1, 21, 2, 2)
        result, trace = sac.sample_source_aligned_controller(
            diffusion,
            source_latent=source,
            source_rgb_frames=81,
            action_prompt_embeds=torch.ones(1, 1, 1),
            noop_prompt_embeds=torch.zeros(1, 1, 1),
            config=sac.SourceAlignedControllerConfig(
                num_inference_steps=3,
                sga_steps=2,
                sga_candidates=3,
                sga_temperature=0.1,
                seed=7,
            ),
            return_trace=True,
        )
        self.assertTrue(torch.equal(result, torch.full_like(result, 2.0)))
        self.assertEqual(trace.candidate_counts, (3, 3, 1))
        self.assertEqual(trace.anc_retained_variance, (0.0, 0.5, 1.0))
        self.assertEqual(trace.fresh_noise_draws, 7)
        self.assertEqual(diffusion.calls, 2 * 7)
        self.assertEqual(len(trace.sga_weights[0]), 3)
        self.assertEqual(trace.sga_weights[-1], (1.0,))
        self.assertGreater(trace.noise_state_change_rms[0], 0.0)
        self.assertEqual(trace.noise_state_change_rms[-1], 0.0)

    def test_non_21_phase_latent_fails_closed(self) -> None:
        torch = self.torch

        class Config:
            in_channels = 1
            text_dim = 1

        class Transformer:
            config = Config()
            dtype = torch.float32

        class Diffusion:
            scheduler = object()
            transformer_2 = None
            transformer = Transformer()

            def shared_step(self):
                raise AssertionError("not reached")

        with self.assertRaises(sac.SourceAlignedControllerError):
            sac.sample_source_aligned_controller(
                Diffusion(),
                source_latent=torch.ones(1, 1, 11, 2, 2),
                source_rgb_frames=81,
                action_prompt_embeds=torch.ones(1, 1, 1),
                noop_prompt_embeds=torch.zeros(1, 1, 1),
            )


if __name__ == "__main__":
    unittest.main()
