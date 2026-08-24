from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_native_action_scaffold_identity_rebinding_canary as runtime  # noqa: E402
import infer_native_multivideo_motion_donor_oracle as donor  # noqa: E402


class _FakeTransformer:
    dtype = "bf16"

    def patch_vae_latent(self, hidden_states, source_id=None):
        return (hidden_states, source_id), ("rotary", source_id)


class _FakeScheduler:
    def step(self, model_output, timestep, sample, return_dict=False):
        return (model_output, timestep, sample, return_dict)


class _FakeDiffusion:
    def __init__(self, *, bad_source_id: bool = False) -> None:
        self.transformer = _FakeTransformer()
        self.transformer_2 = None
        self.scheduler = _FakeScheduler()
        self.config = SimpleNamespace(interpolate_src_id=True, max_trained_src_id=5)
        self.bad_source_id = bad_source_id
        self.returned = object()

    def shared_step(
        self, model_id, noisy_latents, timesteps, cond_embeds,
        rotary_embs, batch_vae_seqlen, batch_text_seqlen,
    ):
        return noisy_latents

    def sample(
        self,
        prompt_embeds=None,
        uncond_prompt_embeds=None,
        prompt_embeds_t2=None,
        uncond_embeds_t2=None,
        num_frames=1,
        width=832,
        height=480,
        image_vae_latents=None,
        multi_video_vae_latents=None,
        multi_image_vae_latents=None,
        num_inference_steps=50,
        guidance_mode="rv2v",
        omega_vid=3.0,
        omega_img=3.0,
        omega_txt=4.0,
        omega_scale=0.75,
        flow_shift=5.0,
        seed=42,
        device="cuda",
        eta=1.0,
        norm_threshold=(50.0, 50.0),
        momentum=0.0,
    ):
        del prompt_embeds_t2, uncond_embeds_t2, num_frames, width, height
        del image_vae_latents, omega_vid, omega_img, omega_txt, omega_scale
        del flow_shift, seed, device, eta, norm_threshold, momentum
        self.assert_rv2v = guidance_mode
        videos = multi_video_vae_latents
        refs = multi_image_vae_latents or []
        vi_tokens = (
            runtime.PATCH_TOKENS
            + len(videos) * runtime.PATCH_TOKENS
            + len(refs) * runtime.REFERENCE_PATCH_TOKENS
        )
        for step in range(num_inference_steps):
            vi_sids = runtime._native_source_ids(len(videos) + len(refs))
            i_sids = runtime._native_source_ids(len(refs))
            patch_index = 0
            for condition, source_id in zip(videos, vi_sids[:len(videos)]):
                patch_index += 1
                self.transformer.patch_vae_latent(condition, source_id=source_id)
            for offset, condition in enumerate(refs):
                for source_id in (vi_sids[len(videos) + offset], i_sids[offset]):
                    patch_index += 1
                    if self.bad_source_id and step == 0 and patch_index == 3:
                        source_id = 99.0
                    self.transformer.patch_vae_latent(condition, source_id=source_id)
            self.transformer.patch_vae_latent(object(), source_id=0)
            lengths = [
                runtime.PATCH_TOKENS,
                2 * runtime.PATCH_TOKENS,
                vi_tokens,
                vi_tokens,
            ]
            prompts = [
                uncond_prompt_embeds,
                uncond_prompt_embeds,
                uncond_prompt_embeds,
                prompt_embeds,
            ]
            visuals = [object(), object(), object()]
            rotaries = [object(), object(), object()]
            visual_objects = [visuals[0], visuals[1], visuals[2], visuals[2]]
            rotary_objects = [rotaries[0], rotaries[1], rotaries[2], rotaries[2]]
            for length, prompt, visual, rotary in zip(
                lengths, prompts, visual_objects, rotary_objects
            ):
                self.shared_step(
                    model_id="transformer_1",
                    noisy_latents=visual,
                    timesteps=step,
                    cond_embeds=prompt,
                    rotary_embs=rotary,
                    batch_vae_seqlen=[length],
                    batch_text_seqlen=[1],
                )
            self.scheduler.step(object(), step, object(), return_dict=False)
        return self.returned


def _kwargs(spec, videos, refs, prompt, negative, *, steps):
    return {
        "prompt_embeds": prompt,
        "uncond_prompt_embeds": negative,
        "image_vae_latents": None,
        "multi_video_vae_latents": videos,
        "multi_image_vae_latents": refs if refs else None,
        "width": runtime.WIDTH,
        "height": runtime.HEIGHT,
        "device": "cuda",
        **runtime.native.native_sampling_contract(
            "rv2v", steps=steps, seed=runtime.TARGET_SEED
        ),
    }


class RoleRebindingCanaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_path = Path(runtime.__file__).resolve()
        cls.source = cls.source_path.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_exact_eight_arm_design_and_dual4_split(self) -> None:
        self.assertEqual(
            runtime.ARM_ORDER,
            (
                "source-video-source-refs",
                "action-donor-source-refs",
                "noop-donor-source-refs",
                "reverse-donor-source-refs",
                "action-donor-only",
                "source-action-source-refs",
                "action-source-source-refs",
                "action-donor-wrong-refs",
            ),
        )
        self.assertEqual(runtime.ARM_GROUPS["group-a"], runtime.ARM_ORDER[:4])
        self.assertEqual(runtime.ARM_GROUPS["group-b"], runtime.ARM_ORDER[4:])
        self.assertEqual(runtime.FACTOR_EXECUTION_GROUP, "sp4-a")
        self.assertEqual(runtime.REFERENCE_INDICES, (0, 27, 53, 80))
        self.assertEqual(runtime.LATENT_SHAPE, (1, 16, 21, 62, 60))
        self.assertEqual(runtime.REFERENCE_SHAPE, (1, 16, 1, 62, 60))
        self.assertEqual(
            runtime.CDF_DOG_WRONG_SOURCE_SHA256,
            "da7e3efa6f4fabac1f1c57b9376667366ca2ad43d4710adea5892eb313cc5e7a",
        )

    def test_source_id_contract_matches_native_dual_axis_interpolation(self) -> None:
        single = runtime.condition_source_id_contract(runtime.ARM_SPECS[1])
        self.assertEqual(single["vi_video_source_ids"], [1.0])
        self.assertEqual(single["vi_reference_source_ids"], [2.0, 3.0, 4.0, 5.0])
        self.assertEqual(single["image_only_reference_source_ids"], [1.0, 2.0, 3.0, 4.0])
        self.assertFalse(single["native_source_id_interpolation_used"])
        pair = runtime.condition_source_id_contract(runtime.ARM_SPECS[6])
        self.assertEqual(
            pair["vi_video_source_ids"],
            [1.0, 1.7999999523162842],
        )
        self.assertEqual(
            pair["vi_reference_source_ids"],
            [2.5999999046325684, 3.4000000953674316, 4.199999809265137, 5.0],
        )
        self.assertEqual(pair["image_only_reference_source_ids"], [1.0, 2.0, 3.0, 4.0])
        self.assertTrue(pair["native_source_id_interpolation_used"])
        self.assertTrue(pair["all_patch_source_ids_within_trained_interval_0_through_5"])
        self.assertFalse(pair["conditioning_source_id_extrapolation_used"])
        wrong = runtime.condition_source_id_contract(runtime.ARM_SPECS[7])
        self.assertEqual(wrong["vi_video_source_ids"], [1.0])
        self.assertEqual(wrong["vi_reference_source_ids"], [2.0, 3.0, 4.0, 5.0])
        self.assertFalse(wrong["native_source_id_interpolation_used"])
        self.assertEqual(
            {
                row.get("reference_video_role")
                for row in wrong["patch_vae_latent_calls_in_order"]
                if row.get("kind") == "independently_encoded_rgb_frame"
            },
            {"wrong_source_video"},
        )

    def _run_audit(self, spec, *, steps=1):
        diffusion = _FakeDiffusion()
        videos = [object() for _ in spec.video_roles]
        refs = [object() for _ in spec.source_reference_indices]
        prompt, negative = object(), object()
        audit = runtime.NativeRoleRebindingConditionAudit(
            diffusion,
            spec=spec,
            video_conditions=videos,
            image_references=refs,
            expected_steps=steps,
            prompt_embeds=prompt,
            uncond_prompt_embeds=negative,
        )
        originals = (
            diffusion.sample,
            diffusion.shared_step,
            diffusion.transformer.patch_vae_latent,
            diffusion.scheduler.step,
        )
        audit.install()
        result = diffusion.sample(**_kwargs(
            spec, videos, refs, prompt, negative, steps=steps
        ))
        audit.restore()
        self.assertIs(result, diffusion.returned)
        self.assertEqual(
            (
                diffusion.sample,
                diffusion.shared_step,
                diffusion.transformer.patch_vae_latent,
                diffusion.scheduler.step,
            ),
            originals,
        )
        self.assertTrue(audit.restored)
        self.assertFalse(audit.trace["observer_modified_numerics"])
        return audit.trace

    def test_one_video_four_refs_exact_native_sequence(self) -> None:
        trace = self._run_audit(runtime.ARM_SPECS[1])
        self.assertEqual(
            trace["source_id_order_per_step"],
            [1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 0.0],
        )
        self.assertEqual(
            trace["shared_visual_token_lengths_per_step"],
            [19_530, 39_060, 42_780, 42_780],
        )

    def test_two_video_four_refs_exact40_native_sequence(self) -> None:
        trace = self._run_audit(runtime.ARM_SPECS[6], steps=40)
        self.assertEqual(
            trace["source_id_order_per_step"],
            [
                1.0, 1.7999999523162842,
                2.5999999046325684, 1.0,
                3.4000000953674316, 2.0,
                4.199999809265137, 3.0,
                5.0, 4.0, 0.0,
            ],
        )
        self.assertEqual(
            trace["shared_visual_token_lengths_per_step"],
            [19_530, 39_060, 62_310, 62_310],
        )
        self.assertEqual(len(trace["step_records"]), 40)

    def test_action_donor_only_passes_native_none_for_refs(self) -> None:
        trace = self._run_audit(runtime.ARM_SPECS[4])
        self.assertEqual(trace["source_id_order_per_step"], [1.0, 0.0])
        self.assertEqual(
            trace["shared_visual_token_lengths_per_step"],
            [19_530, 39_060, 39_060, 39_060],
        )

    def test_wrong_reference_arm_selects_only_wrong_rgb_reference_bank(self) -> None:
        spec = runtime.ARM_SPECS[7]
        source = object()
        proposal = object()
        source_refs = {index: object() for index in runtime.REFERENCE_INDICES}
        wrong_refs = {index: object() for index in runtime.REFERENCE_INDICES}
        videos, refs = runtime._condition_lists(
            spec,
            source=source,
            donors={"full_action": proposal},
            source_references=source_refs,
            wrong_source_references=wrong_refs,
        )
        self.assertEqual(videos, [proposal])
        self.assertEqual(refs, [wrong_refs[index] for index in runtime.REFERENCE_INDICES])
        self.assertTrue(all(value not in refs for value in source_refs.values()))
        trace = self._run_audit(spec)
        self.assertEqual(
            trace["source_id_order_per_step"],
            [1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 0.0],
        )

    def test_audit_fails_closed_and_restores_on_wrong_source_id(self) -> None:
        spec = runtime.ARM_SPECS[6]
        diffusion = _FakeDiffusion(bad_source_id=True)
        videos = [object(), object()]
        refs = [object()] * 4
        prompt, negative = object(), object()
        audit = runtime.NativeRoleRebindingConditionAudit(
            diffusion, spec=spec, video_conditions=videos,
            image_references=refs, expected_steps=1,
            prompt_embeds=prompt, uncond_prompt_embeds=negative,
        )
        original = diffusion.sample
        audit.install()
        try:
            with self.assertRaises(runtime.RoleRebindingCanaryError):
                diffusion.sample(**_kwargs(
                    spec, videos, refs, prompt, negative, steps=1
                ))
        finally:
            audit.restore()
        self.assertEqual(diffusion.sample, original)
        self.assertTrue(audit.restored)

    def test_reuses_sealed_predecode_loader_and_never_reads_proposal_mp4(self) -> None:
        self.assertIs(runtime.donor.load_registered_clean_donor, donor.load_registered_clean_donor)
        self.assertIn("donor.load_registered_clean_donor", self.source)
        self.assertNotIn("video_path\"]", self.source)
        self.assertNotIn("_vae_encode(vae, proposal", self.source)
        self.assertIn('"proposal_mp4_consumed": False', self.source)
        self.assertIn('"predecode_fp32_latents_only": True', self.source)

    def test_runtime_is_frozen_target_free_and_mask_free(self) -> None:
        self.assertNotIn("torch.optim", self.source)
        self.assertNotIn(".backward(", self.source)
        self.assertNotIn("loss.backward", self.source)
        self.assertIn('"training_performed": False', self.source)
        self.assertIn('"target_video": False', self.source)
        for name in ("mask", "flow", "pose", "track", "trajectory", "optimization"):
            self.assertIn(f'"{name}": False', self.source)
        self.assertIn("prepare_hashed_source_snapshot", self.source)
        self.assertIn("source_pixels[:, :, index:index + 1", self.source)
        self.assertIn("wrong_source_pixels[:, :, index:index + 1", self.source)
        self.assertNotIn("_vae_encode(vae, wrong_source_pixels).contiguous()", self.source)
        self.assertIn('parser.add_argument("--wrong-source-video", required=True)', self.source)
        self.assertIn('"paired_target_accessed": False', self.source)
        self.assertIn('"paired_parquet_accessed": False', self.source)
        self.assertIn('"precomputed_latent_accessed": False', self.source)
        self.assertIn('"references_sliced_from_full_video_latent": False', self.source)
        self.assertIn("donor._output_staging_directory(output_dir)", self.source)
        self.assertIn("donor._commit_output_transaction", self.source)

    def test_prompt_is_source_specific_and_same_for_every_arm(self) -> None:
        prompt = runtime.RENDERER_BODY.lower()
        for phrase in (
            "locked overhead camera", "muscular tan-and-white pit bull",
            "black collar", "begins seated", "long bone", "gray concrete",
            "lowers its head", "grips the bone",
            "lifts it upward", "holds it", "do not copy donor identity",
        ):
            self.assertIn(phrase, prompt)
        self.assertNotEqual(
            runtime.RENDERER_BODY,
            runtime.donor.bind_registered_donors,
        )
        self.assertIn('"same_prompt_all_eight_arms": True', self.source)
        self.assertIn('"factor_bank_dual_dog_prompt_reused": False', self.source)

    def test_original_randn_observer_forwards_the_same_tensor_object(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is available on AUH vace runtime")
        produced = []
        received = []

        def canonical(shape, *, generator, device, dtype):
            value = torch.randn(shape, generator=generator, device=device, dtype=dtype)
            produced.append(value)
            return value

        module = SimpleNamespace(randn_tensor=canonical)

        def sample_fn():
            generator = torch.Generator(device="cpu").manual_seed(runtime.TARGET_SEED)
            value = module.randn_tensor(
                (1, 2, 3), generator=generator, device=torch.device("cpu"),
                dtype=torch.float32,
            )
            received.append(value)
            return "native-result"

        result, capture = runtime.native._sample_with_native_initial_noise_observer(
            sample_fn=sample_fn,
            wan_diffusion_module=module,
            expected_shape=(1, 2, 3),
            expected_device=torch.device("cpu"),
            expected_seed=runtime.TARGET_SEED,
            canonical_randn_tensor=canonical,
        )
        self.assertEqual(result, "native-result")
        self.assertIs(received[0], produced[0])
        self.assertIs(module.randn_tensor, canonical)
        self.assertEqual(capture.call_count, 1)


if __name__ == "__main__":
    unittest.main()
