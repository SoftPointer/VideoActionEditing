from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import inspect
import os
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import tri_branch_unipc as tri  # noqa: E402


def _scheduler_config() -> SimpleNamespace:
    return SimpleNamespace(
        _class_name="UniPCMultistepScheduler",
        flow_shift=5.0,
        prediction_type="flow_prediction",
        predict_x0=True,
        use_flow_sigmas=True,
        thresholding=False,
        solver_order=2,
        solver_type="bh2",
    )


class _Prompt:
    def __init__(self, name: str, length: int = 3) -> None:
        self.name = name
        self.shape = (1, length, 4)


class _Prediction:
    def __init__(
        self,
        label: str,
        noisy: object = None,
        *,
        shape: tuple[int, int, int] | None = None,
    ) -> None:
        self.label = label
        self.noisy = noisy
        self.shape = shape

    def __getitem__(self, key):
        if self.shape is None:
            raise TypeError("shape-less fake prediction is not sliceable")
        # The adapter only performs ``[:, -target_tokens:, :]``.  Materialize
        # its resulting symbolic shape without allocating a huge fake tensor.
        target_slice = key[1]
        selected = self.shape[1]
        if isinstance(target_slice, slice) and target_slice.start is not None:
            selected = abs(int(target_slice.start))
        return _Prediction(
            self.label,
            self.noisy,
            shape=(self.shape[0], selected, self.shape[2]),
        )


class _Scalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def __float__(self) -> float:
        return float(self.value)


class _Scheduler:
    def __init__(self) -> None:
        self.config = _scheduler_config()
        self.step_index = None
        self.calls = []

    def set_timesteps(self, steps: int) -> None:
        self.timesteps = [_Scalar(1000.0 - 500.0 * index) for index in range(steps)]
        self.sigmas = [1.0 - 0.5 * index for index in range(steps)] + [0.0]
        self.step_index = None

    def index_for_timestep(self, timestep: _Scalar) -> int:
        query = float(timestep)
        return [float(value) for value in self.timesteps].index(query)

    def step(self, model_output, timestep, sample, return_dict=False):
        self.calls.append(
            {
                "model_output": model_output,
                "timestep": timestep,
                "sample": sample,
                "return_dict": return_dict,
            }
        )
        if self.step_index is None:
            self.step_index = 0
        self.step_index += 1
        return (sample,)


class _Diffusion:
    use_unipc = True

    def __init__(self, *, mismatch_action_state: bool = False) -> None:
        self.scheduler = _Scheduler()
        self.forward_calls = []
        self.mismatch_action_state = mismatch_action_state

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
        self.forward_calls.append(
            {
                "model_id": model_id,
                "noisy": noisy_latents,
                "timestep": timesteps,
                "prompt": cond_embeds,
                "rotary": rotary_embs,
                "vae_len": batch_vae_seqlen,
                "text_len": batch_text_seqlen,
            }
        )
        return _Prediction(
            cond_embeds.name,
            noisy_latents,
            shape=(1, 21 * 30 * 52 + 9, 64),
        )

    def sample(
        self,
        prompt_embeds=None,
        prompt_embeds_t2=None,
        uncond_prompt_embeds=None,
        uncond_embeds_t2=None,
        num_inference_steps=2,
        guidance_mode="v2v_apg",
        omega_txt=4.0,
        omega_scale=0.75,
        flow_shift=5.0,
        eta=1.0,
        norm_threshold=(50.0, 50.0),
        momentum=0.0,
    ):
        self.scheduler.set_timesteps(num_inference_steps)
        noisy_sample = _Prediction("initial-noise")
        for timestep in self.scheduler.timesteps:
            vi_inp = object()
            rotary = object()
            negative = self.shared_step(
                model_id="transformer_1",
                noisy_latents=vi_inp,
                timesteps=timestep,
                cond_embeds=uncond_prompt_embeds,
                rotary_embs=rotary,
                batch_vae_seqlen=[17],
                batch_text_seqlen=[uncond_prompt_embeds.shape[1]],
            )
            action_vi_inp = object() if self.mismatch_action_state else vi_inp
            action = self.shared_step(
                model_id="transformer_1",
                noisy_latents=action_vi_inp,
                timesteps=timestep,
                cond_embeds=prompt_embeds,
                rotary_embs=rotary,
                batch_vae_seqlen=[17],
                batch_text_seqlen=[prompt_embeds.shape[1]],
            )
            # The fake does not reproduce APG numerics; the hook intentionally
            # replaces this placeholder at the pre-UniPC boundary.
            official = _Prediction(f"official:{negative.label}:{action.label}")
            noisy_sample = self.scheduler.step(
                official, timestep, noisy_sample, return_dict=False
            )[0]
        return noisy_sample


class _Renderer:
    def __init__(self, diffusion: _Diffusion) -> None:
        self.diff_dec = diffusion


class TriBranchUniPCHookTests(unittest.TestCase):
    def setUp(self) -> None:
        # Unit fakes still exercise actual file hashing.  Production retains
        # the immutable official hash; the test-local replacement is restored
        # after every case.
        self._official_vendor_hash = tri.PINNED_WAN_DIFFUSION_SHA256
        self.vendor_path = Path(tri.__file__).resolve()
        tri.PINNED_WAN_DIFFUSION_SHA256 = tri._file_sha256(self.vendor_path)

    def tearDown(self) -> None:
        tri.PINNED_WAN_DIFFUSION_SHA256 = self._official_vendor_hash

    def _run_fake(
        self,
        *,
        action: _Prompt | None = None,
        noop: _Prompt | None = None,
        mismatch_action_state: bool = False,
    ):
        diffusion = _Diffusion(mismatch_action_state=mismatch_action_state)
        action = action or _Prompt("action", length=7)
        negative = _Prompt("negative", length=5)
        noop = noop or _Prompt("noop", length=3)
        callback_inputs = []
        projector_calls = []
        buffer_ids = []

        def callback(fields):
            callback_inputs.append(fields)
            return _Prediction("callback-clean")

        def projector(
            raw,
            *,
            action_momentum,
            noop_momentum,
            clean_field_callback,
        ):
            projector_calls.append(raw)
            buffer_ids.append((id(action_momentum), id(noop_momentum)))
            self.assertIsNot(action_momentum, noop_momentum)
            action_momentum.update(1.0)
            noop_momentum.update(2.0)
            result = clean_field_callback(SimpleNamespace(raw=raw))
            parity = raw.action_velocity_packed.label == raw.noop_velocity_packed.label
            return tri.ProjectedVelocity(
                model_output=_Prediction("executed", raw.sample_packed),
                correction_rms=0.25,
                raw_action_noop_delta_rms=0.0 if parity else 1.25,
                guided_action_noop_delta_rms=0.0 if parity else 0.75,
                guided_action_noop_delta_l2=0.0 if parity else 12.5,
                action_noop_exact_parity=parity,
            )

        before = {
            "sample": diffusion.sample.__func__,
            "shared": diffusion.shared_step.__func__,
            "scheduler": diffusion.scheduler.step.__func__,
        }
        with tri.tri_branch_unipc_hook(
            _Renderer(diffusion),
            noop_prompt_embeds=noop,
            latent_shape=(1, 16, 21, 60, 104),
            clean_field_callback=callback,
            bernini_commit=tri.PINNED_BERNINI_COMMIT,
            wan_diffusion_path=self.vendor_path,
            expected_steps=2,
            projector=projector,
        ) as trace:
            result = diffusion.sample(
                prompt_embeds=action,
                uncond_prompt_embeds=negative,
                num_inference_steps=2,
                guidance_mode="v2v_apg",
                omega_txt=4.0,
                omega_scale=0.75,
                flow_shift=5.0,
                eta=1.0,
                norm_threshold=(50.0, 50.0),
                momentum=0.5,
            )
            self.assertEqual(result.label, "initial-noise")
        after = {
            "sample": diffusion.sample.__func__,
            "shared": diffusion.shared_step.__func__,
            "scheduler": diffusion.scheduler.step.__func__,
        }
        self.assertEqual(before, after)
        self.assertNotIn("sample", vars(diffusion))
        self.assertNotIn("shared_step", vars(diffusion))
        self.assertNotIn("step", vars(diffusion.scheduler))
        return diffusion, trace, callback_inputs, projector_calls, buffer_ids

    def test_contract_has_three_forwards_one_unipc_and_no_external_localizer(self) -> None:
        contract = tri.sampler_contract()
        self.assertEqual(
            contract["external_inference_conditions"],
            ["source_video", "action_instruction"],
        )
        self.assertIn("noop_instruction", contract["internal_fixed_controls"])
        self.assertEqual(
            contract["per_step_transformer_forwards"],
            ["shared_negative", "action", "noop"],
        )
        self.assertEqual(
            contract["integrator"],
            "one_original_unipc_scheduler_step_per_diffusion_step",
        )
        self.assertIs(contract["custom_euler_integrator"], False)
        self.assertEqual(contract["expected_transformer_cost_vs_official_v2v_apg"], 1.5)
        self.assertIn("mask", contract["forbidden_inference_conditions"])
        parameters = set(inspect.signature(tri.tri_branch_unipc_hook).parameters)
        self.assertTrue(
            {
                "noop_prompt_embeds",
                "latent_shape",
                "clean_field_callback",
                "wan_diffusion_path",
            }
            <= parameters
        )
        self.assertTrue(
            parameters.isdisjoint(
                {"target_video", "mask", "track", "pose", "flow", "trajectory"}
            )
        )

    def test_fake_pinned_loop_runs_shared_negative_action_noop_then_one_scheduler(self) -> None:
        diffusion, trace, callbacks, projectors, buffer_ids = self._run_fake()
        self.assertEqual(len(diffusion.forward_calls), 6)
        self.assertEqual(
            [call["prompt"].name for call in diffusion.forward_calls],
            ["negative", "action", "noop", "negative", "action", "noop"],
        )
        for offset in (0, 3):
            negative, action, noop = diffusion.forward_calls[offset : offset + 3]
            self.assertIs(negative["noisy"], action["noisy"])
            self.assertIs(action["noisy"], noop["noisy"])
            self.assertIs(negative["rotary"], action["rotary"])
            self.assertIs(action["rotary"], noop["rotary"])
            self.assertIs(negative["timestep"], action["timestep"])
            self.assertIs(action["timestep"], noop["timestep"])
            self.assertEqual(noop["text_len"], [3])
        self.assertEqual(len(diffusion.scheduler.calls), 2)
        self.assertTrue(
            all(call["model_output"].label == "executed" for call in diffusion.scheduler.calls)
        )
        self.assertEqual(len(projectors), 2)
        self.assertEqual(len(callbacks), 2)
        self.assertEqual(buffer_ids[0], buffer_ids[1])
        self.assertNotEqual(buffer_ids[0][0], buffer_ids[0][1])
        self.assertEqual(trace.sample_calls, 1)
        self.assertEqual(len(trace.records), 2)
        for record in trace.records:
            self.assertEqual(record.transformer_forwards, 3)
            self.assertEqual(record.shared_negative_forwards, 1)
            self.assertEqual(record.action_forwards, 1)
            self.assertEqual(record.noop_forwards, 1)
            self.assertEqual(record.original_scheduler_calls, 1)
            self.assertEqual(record.raw_action_noop_delta_rms, 1.25)
            self.assertEqual(record.guided_action_noop_delta_rms, 0.75)
            self.assertEqual(record.guided_action_noop_delta_l2, 12.5)
            self.assertIs(record.action_noop_exact_parity, False)

    def test_action_equals_noop_records_exact_parity(self) -> None:
        prompt = _Prompt("same", length=4)
        diffusion, trace, _, _, _ = self._run_fake(action=prompt, noop=prompt)
        self.assertEqual(
            [call["prompt"].name for call in diffusion.forward_calls],
            ["negative", "same", "same", "negative", "same", "same"],
        )
        self.assertTrue(all(record.action_noop_exact_parity for record in trace.records))
        self.assertTrue(
            all(record.raw_action_noop_delta_rms == 0.0 for record in trace.records)
        )
        self.assertTrue(
            all(record.guided_action_noop_delta_rms == 0.0 for record in trace.records)
        )

    def test_mismatched_vi_state_fails_before_noop_and_scheduler(self) -> None:
        diffusion = _Diffusion(mismatch_action_state=True)
        with tri.tri_branch_unipc_hook(
            diffusion,
            noop_prompt_embeds=_Prompt("noop"),
            latent_shape=(1, 16, 21, 60, 104),
            clean_field_callback=lambda fields: fields,
            bernini_commit=tri.PINNED_BERNINI_COMMIT,
            wan_diffusion_path=self.vendor_path,
            expected_steps=2,
            projector=lambda *args, **kwargs: self.fail("projector must not run"),
        ):
            with self.assertRaisesRegex(tri.TriBranchHookError, "exact same object"):
                diffusion.sample(
                    prompt_embeds=_Prompt("action"),
                    uncond_prompt_embeds=_Prompt("negative"),
                    num_inference_steps=2,
                )
        self.assertEqual(len(diffusion.forward_calls), 1)
        self.assertEqual(diffusion.scheduler.calls, [])

    def test_wrong_source_identity_and_guidance_fail_before_model_work(self) -> None:
        diffusion = _Diffusion()
        with self.assertRaisesRegex(tri.TriBranchHookError, "revision differs"):
            with tri.tri_branch_unipc_hook(
                diffusion,
                noop_prompt_embeds=_Prompt("noop"),
                latent_shape=(1, 16, 21, 60, 104),
                clean_field_callback=lambda fields: fields,
                bernini_commit="wrong",
                wan_diffusion_path=self.vendor_path,
                expected_steps=2,
            ):
                self.fail("installation must not succeed")
        self.assertNotIn("sample", vars(diffusion))

        with self.assertRaisesRegex(tri.TriBranchHookError, "differs"):
            with tri.tri_branch_unipc_hook(
                diffusion,
                noop_prompt_embeds=_Prompt("noop"),
                latent_shape=(1, 16, 21, 60, 104),
                clean_field_callback=lambda fields: fields,
                bernini_commit=tri.PINNED_BERNINI_COMMIT,
                wan_diffusion_path=Path(__file__).resolve(),
                expected_steps=2,
            ):
                self.fail("installation must not accept caller-declared source identity")

        with tri.tri_branch_unipc_hook(
            diffusion,
            noop_prompt_embeds=_Prompt("noop"),
            latent_shape=(1, 16, 21, 60, 104),
            clean_field_callback=lambda fields: fields,
            bernini_commit=tri.PINNED_BERNINI_COMMIT,
            wan_diffusion_path=self.vendor_path,
            expected_steps=2,
        ):
            with self.assertRaisesRegex(tri.TriBranchHookError, "guidance_mode"):
                diffusion.sample(
                    prompt_embeds=_Prompt("action"),
                    uncond_prompt_embeds=_Prompt("negative"),
                    num_inference_steps=2,
                    guidance_mode="v2v",
                )
        self.assertEqual(diffusion.forward_calls, [])
        self.assertEqual(diffusion.scheduler.calls, [])

    def test_scheduler_config_mismatch_fails_before_installation(self) -> None:
        for field, invalid in (
            ("prediction_type", "epsilon"),
            ("predict_x0", False),
            ("use_flow_sigmas", False),
            ("thresholding", True),
            ("solver_order", 3),
            ("solver_type", "midpoint"),
        ):
            with self.subTest(field=field):
                diffusion = _Diffusion()
                setattr(diffusion.scheduler.config, field, invalid)
                with self.assertRaisesRegex(tri.TriBranchHookError, field):
                    with tri.tri_branch_unipc_hook(
                        diffusion,
                        noop_prompt_embeds=_Prompt("noop"),
                        latent_shape=(1, 16, 21, 60, 104),
                        clean_field_callback=lambda fields: fields,
                        bernini_commit=tri.PINNED_BERNINI_COMMIT,
                        wan_diffusion_path=self.vendor_path,
                        expected_steps=2,
                    ):
                        self.fail("invalid scheduler configuration must fail")
                self.assertNotIn("sample", vars(diffusion))

    def test_sigma_lookup_moves_gpu_timestep_to_scheduler_timeline_device(self) -> None:
        class DeviceScalar:
            def __init__(self, value: float, device: str) -> None:
                self.value = value
                self.device = device

            def __float__(self) -> float:
                return float(self.value)

            def to(self, *, device):
                return DeviceScalar(self.value, str(device))

        class DeviceStrictScheduler:
            def __init__(self) -> None:
                self.step_index = None
                self.timesteps = SimpleNamespace(device="cpu")
                self.sigmas = [0.8, 0.0]
                self.lookup_device = None

            def index_for_timestep(self, timestep) -> int:
                self.lookup_device = timestep.device
                if timestep.device != "cpu":
                    raise RuntimeError("timeline/timestep device mismatch")
                return 0

        scheduler = DeviceStrictScheduler()
        original = DeviceScalar(1000.0, "cuda:3")
        index, sigma, sigma_float = tri._resolve_sigma(scheduler, original)
        self.assertEqual(index, 0)
        self.assertEqual(sigma, 0.8)
        self.assertEqual(sigma_float, 0.8)
        self.assertEqual(scheduler.lookup_device, "cpu")
        self.assertEqual(original.device, "cuda:3")

    def test_projector_failure_never_calls_original_scheduler_and_restores(self) -> None:
        diffusion = _Diffusion()

        def fail_projector(*args, **kwargs):
            raise RuntimeError("diagnostic failure")

        with tri.tri_branch_unipc_hook(
            diffusion,
            noop_prompt_embeds=_Prompt("noop"),
            latent_shape=(1, 16, 21, 60, 104),
            clean_field_callback=lambda fields: fields,
            bernini_commit=tri.PINNED_BERNINI_COMMIT,
            wan_diffusion_path=self.vendor_path,
            expected_steps=2,
            projector=fail_projector,
        ):
            with self.assertRaisesRegex(tri.TriBranchHookError, "projector failed"):
                diffusion.sample(
                    prompt_embeds=_Prompt("action"),
                    uncond_prompt_embeds=_Prompt("negative"),
                    num_inference_steps=2,
                )
        self.assertEqual(diffusion.scheduler.calls, [])
        self.assertNotIn("sample", vars(diffusion))
        self.assertNotIn("shared_step", vars(diffusion))
        self.assertNotIn("step", vars(diffusion.scheduler))

    def test_tensor_clean_projection_and_passthrough_if_torch_available(self) -> None:
        try:
            import torch
        except Exception as error:  # pragma: no cover - lightweight local environment
            self.skipTest(f"torch unavailable: {error}")

        layout = tri.PackedLatentLayout.from_spatial_shape((1, 1, 1, 2, 2))
        sample = torch.full(layout.packed_shape, 2.0)
        negative = torch.zeros(layout.packed_shape, dtype=torch.bfloat16)
        action = torch.full(layout.packed_shape, 2.0, dtype=torch.bfloat16)
        noop = torch.zeros(layout.packed_shape, dtype=torch.bfloat16)
        raw = tri.RawTriBranchStep(
            step_index=0,
            timestep=torch.tensor(1000.0),
            timestep_float=1000.0,
            sigma=torch.tensor(0.5),
            sigma_float=0.5,
            model_id="transformer_1",
            sample_packed=sample,
            official_model_output=action,
            negative_velocity_packed=negative,
            action_velocity_packed=action,
            noop_velocity_packed=noop,
            apg=tri.APGParameters(
                guidance_scale=1.0,
                omega_scale=1.0,
                scale_transformer_2=False,
                eta=1.0,
                norm_threshold=0.0,
                momentum=0.0,
            ),
            layout=layout,
        )
        action_momentum = tri._MomentumBuffer(0.0, branch="action")
        noop_momentum = tri._MomentumBuffer(0.0, branch="noop")
        projected = tri.project_clean_fields(
            raw,
            action_momentum=action_momentum,
            noop_momentum=noop_momentum,
            clean_field_callback=tri.scaled_action_delta(0.5),
        )
        self.assertTrue(
            torch.allclose(projected.model_output, torch.ones_like(action))
        )
        self.assertAlmostEqual(projected.correction_rms, 1.0, places=6)
        self.assertAlmostEqual(projected.raw_action_noop_delta_rms, 2.0, places=6)
        self.assertAlmostEqual(projected.guided_action_noop_delta_rms, 1.0, places=6)
        self.assertAlmostEqual(projected.guided_action_noop_delta_l2, 2.0, places=6)
        self.assertIs(projected.action_noop_exact_parity, False)

        action_momentum = tri._MomentumBuffer(0.0, branch="action")
        noop_momentum = tri._MomentumBuffer(0.0, branch="noop")
        passthrough = tri.project_clean_fields(
            raw,
            action_momentum=action_momentum,
            noop_momentum=noop_momentum,
            clean_field_callback=tri.action_clean_passthrough,
        )
        self.assertTrue(torch.equal(passthrough.model_output, action))
        self.assertAlmostEqual(passthrough.correction_rms, 0.0, places=6)

    def test_bf16_dual_expert_action_certificate_uses_effective_scale(self) -> None:
        try:
            import torch
        except Exception as error:  # pragma: no cover - lightweight local environment
            self.skipTest(f"torch unavailable: {error}")

        torch.manual_seed(19)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        layout = tri.PackedLatentLayout.from_spatial_shape((1, 2, 2, 2, 2))
        apg = tri.APGParameters(
            guidance_scale=4.0,
            omega_scale=0.5,
            scale_transformer_2=True,
            eta=0.7,
            norm_threshold=3.0,
            momentum=0.25,
        )
        vendor_action_momentum = tri._MomentumBuffer(0.25, branch="vendor-action")
        local_action_momentum = tri._MomentumBuffer(0.25, branch="local-action")
        local_noop_momentum = tri._MomentumBuffer(0.25, branch="local-noop")

        def official_output(sample, negative, action, sigma, model_id):
            sample_s = tri._packed_to_spatial(sample, layout)
            negative_s = tri._packed_to_spatial(negative, layout)
            action_s = tri._packed_to_spatial(action, layout)
            # Match pinned Diffusers/Bernini exactly: UniPC sigmas remain CPU
            # 0-d tensors while branch tensors can live on the accelerator.
            self.assertEqual(sigma.device.type, "cpu")
            negative_clean = sample_s - sigma * negative_s
            action_clean = sample_s - sigma * action_s
            guided = tri._normalized_guidance(
                action_clean,
                negative_clean,
                apg.guidance_scale_for(model_id),
                vendor_action_momentum,
                apg.eta,
                apg.norm_threshold,
            )
            return tri._spatial_to_packed((sample_s - guided) / sigma, layout)

        scales = []
        for step_index, model_id in enumerate(("transformer_1", "transformer_2")):
            sample = torch.randn(
                layout.packed_shape, dtype=torch.float32, device=device
            )
            negative = torch.randn(layout.packed_shape, device=device).to(
                torch.bfloat16
            )
            action = torch.randn(layout.packed_shape, device=device).to(
                torch.bfloat16
            )
            noop = torch.randn(layout.packed_shape, device=device).to(torch.bfloat16)
            sigma = torch.tensor(0.8 - 0.2 * step_index, dtype=torch.float32)
            official = official_output(sample, negative, action, sigma, model_id)
            raw = tri.RawTriBranchStep(
                step_index=step_index,
                timestep=torch.tensor(900.0 - 500.0 * step_index),
                timestep_float=900.0 - 500.0 * step_index,
                sigma=sigma,
                sigma_float=float(sigma),
                model_id=model_id,
                sample_packed=sample,
                official_model_output=official,
                negative_velocity_packed=negative,
                action_velocity_packed=action,
                noop_velocity_packed=noop,
                apg=apg,
                layout=layout,
            )
            projected = tri.project_clean_fields(
                raw,
                action_momentum=local_action_momentum,
                noop_momentum=local_noop_momentum,
                clean_field_callback=tri.action_clean_passthrough,
            )
            self.assertTrue(torch.equal(projected.model_output, official))
            self.assertTrue(projected.official_action_exact_parity)
            self.assertEqual(projected.official_action_parity_max_abs_error, 0.0)
            self.assertEqual(projected.official_action_parity_rms_error, 0.0)
            self.assertEqual(projected.branch_velocity_dtype, "torch.bfloat16")
            self.assertEqual(projected.sample_dtype, "torch.float32")
            scales.append(projected.effective_guidance_scale)
            if device.type == "cuda":
                with self.assertRaisesRegex(
                    tri.TriBranchHookError, "CPU fp32 scalar"
                ):
                    tri.project_clean_fields(
                        replace(raw, sigma=sigma.to(device)),
                        action_momentum=tri._MomentumBuffer(
                            0.25, branch="wrong-device-action"
                        ),
                        noop_momentum=tri._MomentumBuffer(
                            0.25, branch="wrong-device-noop"
                        ),
                        clean_field_callback=tri.action_clean_passthrough,
                    )
        self.assertEqual(scales, [4.0, 2.0])

    def test_exact_pinned_gen_sample_loop_with_fake_core_if_available(self) -> None:
        """Exercise the actual audited ``GEN_Wanx22.sample`` without weights."""

        tri.PINNED_WAN_DIFFUSION_SHA256 = self._official_vendor_hash
        bernini_root = os.environ.get("BERNINI_OFFICIAL_ROOT")
        veomni_root = os.environ.get("BERNINI_VEOMNI_ROOT")
        if not bernini_root or not veomni_root:
            self.skipTest("pinned Bernini/VeOmni roots are not configured")
        try:
            import torch
            import torch.nn as nn
        except Exception as error:  # pragma: no cover - lightweight local environment
            self.skipTest(f"torch unavailable: {error}")
        sys.path.insert(0, veomni_root)
        sys.path.insert(0, bernini_root)
        try:
            from bernini.models.wan_diffusion import GEN_Wanx22
        except Exception as error:  # pragma: no cover - AUH dependency audit
            self.skipTest(f"pinned Bernini import unavailable: {error}")

        layout = tri.PackedLatentLayout.from_spatial_shape((1, 1, 1, 2, 2))

        class PinnedScheduler:
            def __init__(self):
                self.config = _scheduler_config()
                self.num_train_timesteps = 1000
                self.step_index = None
                self.calls = []

            def set_timesteps(self, steps):
                self.timesteps = torch.linspace(1000.0, 500.0, steps)
                self.sigmas = torch.linspace(1.0, 0.0, steps + 1)
                self.step_index = None

            def index_for_timestep(self, timestep):
                indices = (self.timesteps == timestep).nonzero().reshape(-1)
                return int(indices[0].item())

            def step(self, model_output, timestep, sample, return_dict=False):
                self.calls.append(model_output.detach().clone())
                if self.step_index is None:
                    self.step_index = 0
                self.step_index += 1
                return (sample,)

        class FakeTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(in_channels=1)
                self.dtype = torch.bfloat16

            def patch_vae_latent(self, latent, source_id):
                packed = tri._spatial_to_packed(latent, layout)
                rotary = torch.full(
                    (1, 1, layout.tokens, 1),
                    float(source_id),
                    device=latent.device,
                    dtype=latent.dtype,
                )
                return packed, rotary

        class FakePinnedGEN(GEN_Wanx22):
            def __init__(self):
                nn.Module.__init__(self)
                self.config = SimpleNamespace(
                    interpolate_src_id=True,
                    max_trained_src_id=5,
                )
                self.switch_dit_boundary = 0.0
                self.transformer = FakeTransformer()
                self.transformer_2 = None
                self.scheduler = PinnedScheduler()
                self.use_unipc = True
                self.vae_scale_factor_temporal = 4
                self.vae_scale_factor_spatial = 8
                self.raw_calls = []

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
                self.raw_calls.append(
                    (model_id, noisy_latents, timesteps, cond_embeds, rotary_embs)
                )
                value = float(cond_embeds[0, 0, 0].item())
                return torch.full_like(noisy_latents, value)

        diffusion = FakePinnedGEN()
        action = torch.ones(1, 2, 1)
        negative = torch.zeros(1, 3, 1)
        noop = torch.full((1, 4, 1), 0.25)
        source = torch.zeros(1, 1, 1, 2, 2)
        with tri.tri_branch_unipc_hook(
            diffusion,
            noop_prompt_embeds=noop,
            latent_shape=tuple(source.shape),
            clean_field_callback=tri.action_clean_passthrough,
            bernini_commit=tri.PINNED_BERNINI_COMMIT,
            wan_diffusion_path=Path(bernini_root) / "bernini/models/wan_diffusion.py",
            expected_steps=2,
        ) as trace:
            result = diffusion.sample(
                prompt_embeds=action,
                uncond_prompt_embeds=negative,
                multi_video_vae_latents=[source],
                num_frames=1,
                width=16,
                height=16,
                num_inference_steps=2,
                guidance_mode="v2v_apg",
                omega_txt=2.0,
                flow_shift=5.0,
                device="cpu",
                eta=0.7,
                norm_threshold=0.0,
                momentum=0.3,
            )
        self.assertEqual(tuple(result.shape), tuple(source.shape))
        self.assertEqual(len(diffusion.raw_calls), 6)
        self.assertEqual(len(diffusion.scheduler.calls), 2)
        self.assertEqual(len(trace.records), 2)
        # action_clean_passthrough must reconstruct the action APG already
        # computed inside pinned sample; replacement therefore has zero error.
        for record in trace.records:
            self.assertEqual(record.transformer_forwards, 3)
            self.assertEqual(record.original_scheduler_calls, 1)
            self.assertAlmostEqual(record.callback_correction_rms, 0.0, places=6)
            self.assertGreater(record.raw_action_noop_delta_rms, 0.0)
            self.assertIs(record.action_noop_exact_parity, False)


if __name__ == "__main__":
    unittest.main()
