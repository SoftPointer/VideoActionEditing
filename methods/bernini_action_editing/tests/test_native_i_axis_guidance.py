from __future__ import annotations

from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import native_i_axis_guidance as subject


class NativeIAxisPureContractTests(unittest.TestCase):
    def test_formula_is_raw_velocity_and_exact(self) -> None:
        v0, v_v, v_i, v_vi_u, v_vi_c = 1.0, 2.0, 7.0, 5.0, 9.0
        native = subject.native_rv2v_velocity(v0, v_v, v_vi_u, v_vi_c)
        self.assertEqual(native, 31.75)
        gated = subject.gated_native_i_velocity(
            v0, v_v, v_i, v_vi_u, v_vi_c, gate=0.25
        )
        self.assertEqual(gated, native + 4.5 * 0.25 * ((v_i - v0) - (v_vi_u - v_v)))
        self.assertEqual(
            subject.gated_native_i_velocity(
                v0, v_v, v_i, v_vi_u, v_vi_c, gate=0.0
            ),
            native,
        )
        receipt = subject.hook_contract()
        self.assertFalse(receipt["apg"])
        self.assertEqual(
            receipt["gated_formula"], "vG=vN+4.5*g*((vI-v0)-(vVIu-vV))"
        )
        self.assertEqual(receipt["coordinate"], "raw_velocity_before_original_unipc_step")

    def test_gate_is_only_33_through_37_and_final_two_are_zero(self) -> None:
        active = [index for index in range(40) if subject.sigma_gate(index) > 0.0]
        self.assertEqual(active, [33, 34, 35, 36, 37])
        self.assertEqual(subject.sigma_gate(38), 0.0)
        self.assertEqual(subject.sigma_gate(39), 0.0)
        with self.assertRaises(subject.NativeIAxisGuidanceError):
            subject.sigma_gate(40)
        with self.assertRaises(subject.NativeIAxisGuidanceError):
            subject.sigma_gate(True)

    def test_seven_arm_controls_are_closed(self) -> None:
        plan = subject.arm_plan()
        self.assertEqual(tuple(row["arm"] for row in plan), subject.ARM_ORDER)
        self.assertEqual(subject.ARM_ORDER, ("N-C", "N-W", "G-C", "G-W", "G-P", "G-D", "G-S"))
        self.assertEqual(
            subject.arm_reference_contract("G-P")["reference_indices_in_list_order"],
            [27, 53, 80, 0],
        )
        self.assertFalse(subject.arm_reference_contract("G-P")["chronological_shuffle_claimed"])
        self.assertEqual(subject.arm_reference_contract("G-D")["reference_count"], 0)
        self.assertEqual(
            subject.arm_reference_contract("G-S")["reference_indices_in_list_order"],
            [10, 30, 50, 70],
        )

    def test_permutation_reuses_same_objects_without_copy(self) -> None:
        objects = tuple(object() for _ in range(4))
        result = subject.permute_reference_objects(objects)
        self.assertEqual(result, (objects[1], objects[2], objects[3], objects[0]))
        self.assertEqual(sorted(map(id, result)), sorted(map(id, objects)))


try:
    import torch

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:
    class _FakeScheduler:
        def __init__(self) -> None:
            self.sigmas = torch.linspace(1.0, 0.1, 40, dtype=torch.float32)
            self.step_index = 0
            self.received = []

        def step(self, model_output, timestep, sample, return_dict=False):
            self.received.append(model_output.detach().clone())
            self.step_index += 1
            return (sample - 0.001 * model_output.float(),)


    class _FakeTransformer:
        dtype = torch.float32

        def patch_vae_latent(self, value, source_id=None):
            sid = float(source_id)
            if getattr(value, "ndim", None) == 3:
                tokens = int(value.shape[1])
                payload = value.clone()
            else:
                tokens = 1 if int(value.shape[2]) == 1 else 4
                payload = torch.full((1, tokens, 4), float(value.mean()) + sid)
            rotary = torch.full((1, 2, tokens, 3), sid)
            return payload, rotary


    class _FakeDiffusion:
        use_unipc = True
        transformer_2 = None

        def __init__(self) -> None:
            self.transformer = _FakeTransformer()
            self.scheduler = _FakeScheduler()
            self.shared_call_count = 0

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
            self.shared_call_count += 1
            context = noisy_latents.float().mean(dim=1, keepdim=True)
            text = cond_embeds.float().mean()
            return noisy_latents + context.to(noisy_latents.dtype) + text.to(noisy_latents.dtype)

        def sample(
            self,
            prompt_embeds,
            uncond_prompt_embeds,
            image_vae_latents,
            multi_video_vae_latents,
            multi_image_vae_latents,
            width,
            height,
            num_frames,
            num_inference_steps,
            guidance_mode,
            omega_vid,
            omega_img,
            omega_txt,
            omega_scale,
            flow_shift,
            seed,
            eta,
            norm_threshold,
            momentum,
            device,
        ):
            sample = torch.zeros((1, 4, 4), dtype=torch.float32)
            refs = tuple(multi_image_vae_latents or ())
            for index in range(num_inference_steps):
                timestep = torch.tensor(float(999 - index), dtype=torch.float32)
                video = self.transformer.patch_vae_latent(
                    multi_video_vae_latents[0], source_id=1.0
                )
                vi_refs, i_refs = [], []
                for ref_index, ref in enumerate(refs):
                    vi_refs.append(
                        self.transformer.patch_vae_latent(
                            ref, source_id=float(ref_index + 2)
                        )
                    )
                    i_refs.append(
                        self.transformer.patch_vae_latent(
                            ref, source_id=float(ref_index + 1)
                        )
                    )
                target = self.transformer.patch_vae_latent(sample, source_id=0.0)

                def assembled(parts):
                    latents = torch.cat([part[0] for part in parts] + [target[0]], dim=1)
                    rotary = torch.cat([part[1] for part in parts] + [target[1]], dim=2)
                    return latents, rotary

                none = assembled(())
                video_branch = assembled((video,))
                vi = assembled((video, *vi_refs))

                def forward(branch, text):
                    return self.shared_step(
                        model_id="transformer_1",
                        noisy_latents=branch[0],
                        timesteps=timestep.reshape(1),
                        cond_embeds=text,
                        rotary_embs=branch[1],
                        batch_vae_seqlen=[int(branch[0].shape[1])],
                        batch_text_seqlen=[int(text.shape[1])],
                    )[:, -sample.shape[1] :, :]

                v0 = forward(none, uncond_prompt_embeds)
                v_v = forward(video_branch, uncond_prompt_embeds)
                v_vi_u = forward(vi, uncond_prompt_embeds)
                v_vi_c = forward(vi, prompt_embeds)
                native = subject.native_rv2v_velocity(v0, v_v, v_vi_u, v_vi_c)
                sample = self.scheduler.step(
                    native, timestep, sample, return_dict=False
                )[0]
            return sample.contiguous()


def _run_fake_arm(arm: str):
    diffusion = _FakeDiffusion()
    hook = subject.NativeIAxisGuidanceHook(
        diffusion,
        arm=arm,
        expected_steps=40,
        expected_bernini_commit=subject.PINNED_BERNINI_COMMIT,
        observed_wan_diffusion_sha256=subject.PINNED_WAN_DIFFUSION_SHA256,
    )
    refs = [] if arm == "G-D" else [torch.full((1, 16, 1, 1, 1), float(i + 1)) for i in range(4)]
    hook.install()
    try:
        result = diffusion.sample(
            prompt_embeds=torch.ones((1, 2, 3)),
            uncond_prompt_embeds=torch.zeros((1, 2, 3)),
            image_vae_latents=None,
            multi_video_vae_latents=[torch.ones((1, 16, 21, 1, 1))],
            multi_image_vae_latents=refs or None,
            width=16,
            height=16,
            num_frames=81,
            num_inference_steps=40,
            guidance_mode="rv2v",
            omega_vid=1.25,
            omega_img=4.5,
            omega_txt=4.0,
            omega_scale=0.8,
            flow_shift=5.0,
            seed=7,
            eta=0.5,
            norm_threshold=(50.0, 50.0),
            momentum=0.0,
            device="cpu",
        )
    finally:
        hook.restore()
    return diffusion, hook, result


@unittest.skipUnless(_TORCH_AVAILABLE, "PyTorch runtime is required")
class NativeIAxisHookTests(unittest.TestCase):
    def test_gated_hook_has_five_forwards_and_registered_gate(self) -> None:
        diffusion, hook, result = _run_fake_arm("G-C")
        self.assertEqual(tuple(result.shape), (1, 4, 4))
        self.assertEqual(hook.trace["observed_transformer_forwards"], 200)
        self.assertEqual(diffusion.shared_call_count, 200)
        active = [row["step_index"] for row in hook.trace["steps"] if row["gate_active"]]
        self.assertEqual(active, [33, 34, 35, 36, 37])
        self.assertTrue(
            all(
                hook.trace["steps"][index]["scheduler_received_original_model_output_object"]
                for index in (38, 39)
            )
        )
        self.assertTrue(any(row["correction_rms"] > 0 for row in hook.trace["steps"][33:38]))
        self.assertNotIn("sample", vars(diffusion))
        self.assertNotIn("step", vars(diffusion.scheduler))

    def test_native_and_refdrop_controls_are_bit_exact(self) -> None:
        native_diffusion, native_hook, _ = _run_fake_arm("N-C")
        self.assertEqual(native_diffusion.shared_call_count, 160)
        self.assertEqual(native_hook.trace["observed_transformer_forwards"], 160)
        self.assertTrue(
            all(
                row["native_velocity_raw_sha256"] == row["executed_velocity_raw_sha256"]
                for row in native_hook.trace["steps"]
            )
        )
        drop_diffusion, drop_hook, _ = _run_fake_arm("G-D")
        self.assertEqual(drop_diffusion.shared_call_count, 160)
        self.assertEqual(drop_hook.trace["observed_transformer_forwards"], 160)
        self.assertEqual(
            [row["step_index"] for row in drop_hook.trace["steps"] if row["gate_active"]],
            [33, 34, 35, 36, 37],
        )
        self.assertTrue(all(row["i_axis_degenerate_alias_none"] for row in drop_hook.trace["steps"]))
        self.assertTrue(
            all(
                row["native_velocity_raw_sha256"] == row["executed_velocity_raw_sha256"]
                for row in drop_hook.trace["steps"]
            )
        )


if __name__ == "__main__":
    unittest.main()
