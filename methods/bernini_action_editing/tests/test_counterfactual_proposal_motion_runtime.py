from __future__ import annotations

from pathlib import Path
import sys
import unittest

try:
    import torch
except ImportError as error:  # pragma: no cover
    raise unittest.SkipTest("torch unavailable") from error


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import counterfactual_proposal_motion_branch as branch
import counterfactual_proposal_motion_runtime as runtime


class _FakeDiffusion:
    transformer_2 = None

    def __init__(self, owner_token: object):
        self.transformer = object()
        self.owner_token = owner_token
        self.observed = []
        self.positive_first = False

    def shared_step(
        self,
        model_id,
        noisy_latents,
        timesteps,
        cond_embeds,
        rotary_embs,
        batch_vae_seqlen,
    ):
        invocation = branch.current_cpmr_motion_invocation()
        if invocation is None:
            raise AssertionError("runtime did not install an invocation")
        self.observed.append((invocation.polarity, invocation.routes_motion))
        if invocation.routes_motion:
            internal = cond_embeds.clone()
            invocation.conditioned_encoder_binding.observe(
                self.owner_token, 0, internal
            )
        return noisy_latents

    def sample(
        self,
        prompt_embeds,
        uncond_prompt_embeds,
        image_vae_latents=None,
        multi_video_vae_latents=None,
        multi_image_vae_latents=None,
        num_frames=81,
        width=480,
        height=496,
        num_inference_steps=40,
        guidance_mode="v2v_apg",
        omega_vid=3.0,
        omega_img=3.0,
        omega_txt=4.0,
        omega_scale=0.75,
        flow_shift=5.0,
        seed=2027,
        device="cpu",
        eta=0.5,
        norm_threshold=(50.0, 50.0),
        momentum=-0.5,
    ):
        del width, height, omega_vid, omega_img, omega_txt, omega_scale
        del seed, device, eta, norm_threshold, momentum
        noisy = torch.zeros(1, 39_060, 2)
        rotary = torch.zeros(1, 1, 39_060, 1)
        for step in range(num_inference_steps):
            timestep = torch.tensor(step, dtype=torch.int64)
            kwargs = dict(
                model_id="transformer_1",
                noisy_latents=noisy,
                timesteps=timestep,
                rotary_embs=rotary,
                batch_vae_seqlen=[39_060],
            )
            ordered = (
                (prompt_embeds, uncond_prompt_embeds)
                if self.positive_first
                else (uncond_prompt_embeds, prompt_embeds)
            )
            for condition in ordered:
                self.shared_step(cond_embeds=condition, **kwargs)
        return noisy


class CPMRRuntimeTests(unittest.TestCase):
    def _hook(self, *, gate=0.1):
        owner_token = object()
        diffusion = _FakeDiffusion(owner_token)
        carrier = torch.zeros(1, 1_344, 1_536, dtype=torch.bfloat16)
        carrier[:, 64:] = 1
        activity = torch.ones(1, 21, dtype=torch.bool)
        activity[:, 0] = False

        def binding_factory():
            return branch.CPMRConditionedEncoderBinding((0,), owner_token)

        hook = runtime.InstalledCPMRFinalRenderHook(
            diffusion,
            patch_handle=object(),
            carrier=carrier,
            activity=activity,
            gate=gate,
            binding_factory=binding_factory,
        )
        return diffusion, hook

    def test_exact_40_step_identity_routing_and_restore(self):
        diffusion, hook = self._hook()
        positive = torch.randn(1, 8, 16)
        negative = torch.randn(1, 8, 16)
        original_sample = diffusion.sample
        original_shared = diffusion.shared_step
        with hook:
            result = diffusion.sample(
                prompt_embeds=positive,
                uncond_prompt_embeds=negative,
                multi_video_vae_latents=[torch.zeros(1)],
            )
        self.assertEqual(tuple(result.shape), (1, 39_060, 2))
        self.assertTrue(hook.restored)
        self.assertEqual(diffusion.sample, original_sample)
        self.assertEqual(diffusion.shared_step, original_shared)
        self.assertEqual(diffusion.observed, [("unconditional", False), ("positive", True)] * 40)
        receipt = hook.trace.receipt()
        self.assertEqual(receipt["sample_calls"], 1)
        self.assertEqual(receipt["shared_step_calls"], 80)
        self.assertEqual(receipt["completed_steps"], 40)
        self.assertTrue(receipt["all_prompt_identity_exact"])
        self.assertTrue(receipt["all_paired_state_identity_exact"])
        self.assertTrue(receipt["all_bindings_complete"])

    def test_rejects_wrong_order_without_publishing_a_step(self):
        diffusion, hook = self._hook()
        diffusion.positive_first = True
        with self.assertRaises(runtime.CPMRRuntimeContractError):
            with hook:
                diffusion.sample(
                    prompt_embeds=torch.randn(1, 8, 16),
                    uncond_prompt_embeds=torch.randn(1, 8, 16),
                    multi_video_vae_latents=[torch.zeros(1)],
                )
        self.assertEqual(hook.trace.records, [])
        self.assertTrue(hook.restored)

    def test_rejects_noncanonical_gate_and_geometry(self):
        with self.assertRaises(runtime.CPMRRuntimeContractError):
            self._hook(gate=0.11)
        diffusion, hook = self._hook()
        with hook:
            with self.assertRaises(runtime.CPMRRuntimeContractError):
                diffusion.sample(
                    prompt_embeds=torch.randn(1, 8, 16),
                    uncond_prompt_embeds=torch.randn(1, 8, 16),
                    multi_video_vae_latents=[torch.zeros(1)],
                    num_frames=41,
                )

    def test_static_contract_forbids_prompt_text_matching(self):
        source = (METHOD_ROOT / "counterfactual_proposal_motion_runtime.py").read_text()
        self.assertNotIn("instruction", source.lower())
        self.assertNotIn("prompt_clean", source)
        self.assertIn("prompt is not state.negative_prompt", source)
        self.assertIn("prompt is not state.positive_prompt", source)


if __name__ == "__main__":
    unittest.main()
