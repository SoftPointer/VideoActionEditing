from __future__ import annotations

import inspect
import hashlib
from pathlib import Path
import struct
import sys
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import guided_source_aligned_controller as guided  # noqa: E402


class GuidedContractTests(unittest.TestCase):
    def test_exact_guided_arm_and_apg_contract(self) -> None:
        self.assertEqual(
            guided.GUIDED_ARMS,
            ("FIID1G", "FANC1G", "FAVG5G", "FSGA5G"),
        )
        expected = {
            "FIID1G": (40, 160),
            "FANC1G": (40, 160),
            "FAVG5G": (52, 208),
            "FSGA5G": (52, 208),
        }
        for arm, (evaluations, calls) in expected.items():
            with self.subTest(arm=arm):
                config = guided.GuidedSourceAlignedConfig(arm=arm).validate()
                self.assertEqual(config.expected_candidate_evaluations, evaluations)
                self.assertEqual(config.expected_shared_step_calls, calls)
                self.assertEqual(config.apg_guidance_scale, 4.0)
                self.assertEqual(config.apg_eta, 0.5)
                self.assertEqual(config.apg_norm_threshold, 50.0)
                self.assertEqual(config.apg_momentum, 0.0)
                self.assertEqual(config.sga_temperature, 1.0)

    def test_apg_and_schedule_parameters_fail_closed(self) -> None:
        cases = (
            {"apg_guidance_scale": 4.1},
            {"apg_eta": 0.0},
            {"apg_norm_threshold": 49.0},
            {"apg_momentum": 0.1},
            {"sga_temperature": 0.01},
            {"num_inference_steps": 41},
            {"seed": 7},
        )
        for values in cases:
            with self.subTest(values=values), self.assertRaises(
                guided.GuidedSourceAlignedControllerError
            ):
                guided.GuidedSourceAlignedConfig(arm="FSGA5G", **values).validate()

    def test_pinned_unipc_start_sigma_regression(self) -> None:
        start = guided.PINNED_UNIPC_START_SIGMA
        values = [start * (1.0 - index / 40.0) for index in range(41)]
        intervals = tuple(zip(values, values[1:]))
        actual = guided.validate_pinned_sigma_intervals(intervals)
        self.assertEqual(actual[0][0], start)
        first_retention = (1.0 - start) / (1.0 - guided.ANC_LOCK_SIGMA)
        self.assertGreater(first_retention, 0.0)
        self.assertAlmostEqual(first_retention, 1.3510386149088542e-6)
        wrong = list(intervals)
        wrong[0] = (1.0, wrong[0][1])
        with self.assertRaises(guided.GuidedSourceAlignedControllerError):
            guided.validate_pinned_sigma_intervals(wrong)

    def test_keyed_noise_pairing_is_arm_independent(self) -> None:
        bank = guided.noise_bank_pairing_contract(seed=2027)
        again = guided.noise_bank_pairing_contract(seed=2027)
        self.assertEqual(bank, again)
        self.assertTrue(bank["iid_anc_candidate0_exact_pairing"])
        self.assertTrue(bank["avg_sga_full_early_bank_exact_pairing"])
        self.assertEqual(len(bank["full_bank_digest"]), 64)
        self.assertEqual(len(bank["candidate0_bank_digest"]), 64)
        self.assertNotEqual(
            guided.keyed_noise_seed(2027, 0, 0),
            guided.keyed_noise_seed(2027, 0, 1),
        )
        self.assertEqual(
            guided.keyed_noise_seed(2027, 7, 3),
            guided.keyed_noise_seed(2027, 7, 3),
        )

    def test_public_signature_has_no_privileged_inputs(self) -> None:
        parameters = set(
            inspect.signature(
                guided.sample_guided_source_aligned_controller
            ).parameters
        )
        self.assertEqual(
            parameters,
            {
                "renderer_or_diffusion",
                "source_latent",
                "source_rgb_frames",
                "action_prompt_embeds",
                "noop_prompt_embeds",
                "negative_prompt_embeds",
                "config",
                "return_trace",
            },
        )
        forbidden = set(guided.guided_controller_contract()["forbidden_conditions"])
        self.assertTrue(parameters.isdisjoint(forbidden))
        self.assertEqual(
            guided.guided_controller_contract()["user_inputs"],
            ["source_video", "edit_instruction"],
        )

    def test_static_apg_path_uses_direct_sigma_and_keeps_guided_fp32(self) -> None:
        source = Path(guided.__file__).read_text()
        self.assertIn("sigma=scheduler_sigma_scalars[index]", source)
        self.assertIn("sigma_tensor = _validate_sigma_cpu_fp32(sigma)", source)
        self.assertNotIn("def _sigma_cpu_fp32", source)
        self.assertNotIn("guided_velocity_fp32, layout\n    ).to(dtype=torch.bfloat16)", source)


class GuidedFakeBerniniTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
        except Exception as error:  # pragma: no cover
            raise unittest.SkipTest(f"torch unavailable: {error}")
        cls.torch = torch

    def _diffusion(self):
        torch = self.torch
        import differential_sampler as cdf

        class Config:
            in_channels = 1
            text_dim = 1

        class Transformer:
            config = Config()
            dtype = torch.bfloat16

            def patch_vae_latent(self, latent, source_id):
                packed = cdf._pack_spatial_latent(latent)
                tokens = packed.to(torch.bfloat16)
                rotary = torch.full(
                    (1, 1, tokens.shape[1], 1),
                    float(source_id),
                    dtype=torch.float32,
                )
                return tokens, rotary

        class SchedulerConfig:
            flow_shift = 5.0

        class Scheduler:
            config = SchedulerConfig()

            def set_timesteps(self, steps):
                start = guided.PINNED_UNIPC_START_SIGMA
                values = [start * (1.0 - index / steps) for index in range(steps)]
                self.timesteps = torch.tensor(
                    [value * 1000.0 for value in values], dtype=torch.float32
                )
                self.sigmas = torch.tensor(values + [0.0], dtype=torch.float32)

        class Diffusion:
            use_unipc = True
            transformer_2 = None

            def __init__(self):
                self.transformer = Transformer()
                self.scheduler = Scheduler()
                self.calls = []

            def shared_step(self, *, noisy_latents, cond_embeds, **kwargs):
                prompt_value = float(cond_embeds[0, 0, 0])
                self.calls.append(prompt_value)
                # Depend weakly on the query so different noise candidates are
                # not algebraically forced to have identical SGA scores.
                state = noisy_latents.to(torch.float32)
                mean = state.mean(dim=-1, keepdim=True)
                output = mean.repeat(1, 1, 4) * 0.01 - prompt_value
                return output.to(torch.bfloat16)

        return Diffusion()

    def _run(self, arm: str):
        torch = self.torch
        diffusion = self._diffusion()
        source = torch.ones(1, 1, 21, 2, 2, dtype=torch.float32)
        diffusion.scheduler.set_timesteps(40)
        values = tuple(float(value) for value in diffusion.scheduler.sigmas.tolist())
        payload = {
            "timesteps": [
                float(value) for value in diffusion.scheduler.timesteps.tolist()
            ],
            "sigmas": list(values),
            "flow_shift": 5.0,
            "steps": 40,
        }
        fake_digest = guided._object_sha256(payload)
        fake_sigma_digest = hashlib.sha256(
            b"".join(
                struct.pack(">f", float(diffusion.scheduler.sigmas[index].item()))
                for index in range(40)
            )
        ).hexdigest()
        with mock.patch.object(
            guided, "PINNED_UNIPC_SCHEDULE_DIGEST", fake_digest
        ), mock.patch.object(
            guided, "PINNED_UNIPC_SIGMA_FP32_DIGEST", fake_sigma_digest
        ):
            result, trace = guided.sample_guided_source_aligned_controller(
                diffusion,
                source_latent=source,
                source_rgb_frames=81,
                action_prompt_embeds=torch.ones(1, 1, 1, dtype=torch.bfloat16),
                noop_prompt_embeds=torch.zeros(1, 1, 1, dtype=torch.bfloat16),
                negative_prompt_embeds=-torch.ones(1, 1, 1, dtype=torch.bfloat16),
                config=guided.GuidedSourceAlignedConfig(arm=arm),
                return_trace=True,
            )
        return diffusion, result, trace

    def test_scheduler_sigma_capture_is_a_direct_bit_exact_storage_view(self) -> None:
        diffusion = self._diffusion()
        diffusion.scheduler.set_timesteps(40)
        raw = diffusion.scheduler.sigmas
        values = tuple(float(value) for value in raw.tolist())
        intervals = guided.cdf.descending_sigma_intervals(values, expected_steps=40)
        scalars, digest = guided.capture_pinned_scheduler_sigma_scalars(
            diffusion, intervals
        )
        self.assertEqual(len(scalars), 40)
        for index, scalar in enumerate(scalars):
            self.assertEqual(scalar.untyped_storage().data_ptr(), raw.untyped_storage().data_ptr())
            self.assertEqual(scalar.storage_offset(), index)
            self.assertEqual(float(scalar.item()), intervals[index][0])
        expected = hashlib.sha256(
            b"".join(struct.pack(">f", float(raw[index].item())) for index in range(40))
        ).hexdigest()
        self.assertEqual(digest, expected)

        diffusion.scheduler.sigmas = raw[:-1].clone()
        with self.assertRaises(guided.GuidedSourceAlignedControllerError):
            guided.capture_pinned_scheduler_sigma_scalars(diffusion, intervals)

        diffusion.scheduler.sigmas = raw.clone()
        diffusion.scheduler.sigmas[-1] = -0.0
        with self.assertRaises(guided.GuidedSourceAlignedControllerError):
            guided.capture_pinned_scheduler_sigma_scalars(diffusion, intervals)

    def test_schedule_drift_fails_before_any_transformer_forward(self) -> None:
        torch = self.torch
        diffusion = self._diffusion()
        source = torch.ones(1, 1, 21, 2, 2, dtype=torch.float32)
        with self.assertRaises(guided.GuidedSourceAlignedControllerError):
            guided.sample_guided_source_aligned_controller(
                diffusion,
                source_latent=source,
                source_rgb_frames=81,
                action_prompt_embeds=torch.ones(1, 1, 1, dtype=torch.bfloat16),
                noop_prompt_embeds=torch.zeros(1, 1, 1, dtype=torch.bfloat16),
                negative_prompt_embeds=-torch.ones(1, 1, 1, dtype=torch.bfloat16),
                config=guided.GuidedSourceAlignedConfig(arm="FANC1G"),
                return_trace=True,
            )
        self.assertEqual(diffusion.calls, [])

    def test_each_arm_executes_exact_guided_branch_count_and_order(self) -> None:
        expected = {
            "FIID1G": (160, 40),
            "FANC1G": (160, 40),
            "FAVG5G": (208, 52),
            "FSGA5G": (208, 52),
        }
        traces = {}
        for arm, (calls, draws) in expected.items():
            with self.subTest(arm=arm):
                diffusion, result, trace = self._run(arm)
                traces[arm] = trace
                self.assertEqual(len(diffusion.calls), calls)
                self.assertEqual(trace.total_shared_step_calls, calls)
                self.assertEqual(trace.branch_counts, (draws,) * 4)
                self.assertEqual(trace.fresh_noise_draws, draws)
                self.assertEqual(trace.branch_order, guided.BRANCH_ORDER)
                self.assertEqual(
                    [diffusion.calls[index : index + 4] for index in range(0, calls, 4)],
                    [[-1.0, 1.0, -1.0, 0.0]] * draws,
                )
                self.assertEqual(str(result.dtype), "torch.float32")
                self.assertEqual(trace.raw_velocity_dtype, "torch.bfloat16")
                self.assertEqual(trace.guided_velocity_dtype, "torch.float32")
                self.assertEqual(trace.apg_clean_dtype, "torch.float32")
                self.assertEqual(trace.delta_dtype, "torch.float32")
                self.assertEqual(trace.edit_state_dtype, "torch.float32")
                self.assertTrue(trace.target_branch_query_parity)
                self.assertTrue(trace.source_branch_query_parity)
                self.assertFalse(trace.weighted_noise_collapse_used)
                self.assertEqual(trace.candidate_continuation, "candidate_0")
                self.assertEqual(
                    trace.anc_initial_predecessor_policy,
                    "zero_initialized_per_dynaedit_pseudocode",
                )
                self.assertTrue(trace.scheduler_sigma_direct_views)
                self.assertEqual(trace.scheduler_sigma_dtype, "torch.float32")
                self.assertEqual(trace.scheduler_sigma_device, "cpu")
                self.assertEqual(len(trace.scheduler_sigma_fp32_digest), 64)
                self.assertEqual(len(trace.used_fresh_noise_content_digest), 64)
                self.assertEqual(len(trace.candidate0_fresh_noise_content_digest), 64)
                self.assertEqual(len(trace.sga_entropy), 40)
                self.assertEqual(len(trace.sga_top1_margin), 40)
                self.assertEqual(
                    dict(trace.apg_parameters),
                    {
                        "guidance_mode": "v2v_apg",
                        "guidance_scale": 4.0,
                        "eta": 0.5,
                        "norm_threshold": 50.0,
                        "momentum": 0.0,
                    },
                )
                self.assertAlmostEqual(
                    trace.sigmas[0], guided.PINNED_UNIPC_START_SIGMA, places=7
                )
                if arm == "FIID1G":
                    self.assertEqual(trace.anc_retained_variance[0], 0.0)
                else:
                    self.assertGreater(trace.anc_retained_variance[0], 0.0)
        self.assertEqual(
            traces["FIID1G"].used_noise_key_digest,
            traces["FANC1G"].used_noise_key_digest,
        )
        self.assertEqual(
            traces["FIID1G"].used_fresh_noise_content_digest,
            traces["FANC1G"].used_fresh_noise_content_digest,
        )
        self.assertEqual(
            traces["FAVG5G"].used_noise_key_digest,
            traces["FSGA5G"].used_noise_key_digest,
        )
        self.assertEqual(
            traces["FAVG5G"].used_fresh_noise_content_digest,
            traces["FSGA5G"].used_fresh_noise_content_digest,
        )
        self.assertEqual(
            len(
                {
                    trace.candidate0_fresh_noise_content_digest
                    for trace in traces.values()
                }
            ),
            1,
        )
        self.assertEqual(
            traces["FIID1G"].candidate0_noise_bank_digest,
            traces["FANC1G"].candidate0_noise_bank_digest,
        )
        self.assertEqual(
            traces["FAVG5G"].full_noise_bank_digest,
            traces["FSGA5G"].full_noise_bank_digest,
        )

    def test_avg_and_sga_share_noise_bank_and_weighting_intervention(self) -> None:
        _, _, avg = self._run("FAVG5G")
        _, _, sga = self._run("FSGA5G")
        self.assertEqual(avg.used_noise_key_digest, sga.used_noise_key_digest)
        self.assertEqual(
            avg.used_fresh_noise_content_digest,
            sga.used_fresh_noise_content_digest,
        )
        # Both arms enter step zero from the same source/edit state and see the
        # same five actual noise tensors, so their first candidate scores must
        # match exactly.  Different step-zero weights then intentionally make
        # the edit states (and therefore later scores) diverge; requiring all
        # three score rows to match would incorrectly erase the SGA effect.
        self.assertEqual(avg.sga_scores[0], sga.sga_scores[0])
        self.assertEqual(avg.candidate_counts, (5, 5, 5) + (1,) * 37)
        self.assertEqual(sga.candidate_counts, avg.candidate_counts)
        for weights in avg.sga_weights[:3]:
            for value in weights:
                self.assertAlmostEqual(value, 0.2, places=7)
        for weights in sga.sga_weights[:3]:
            self.assertAlmostEqual(sum(weights), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
