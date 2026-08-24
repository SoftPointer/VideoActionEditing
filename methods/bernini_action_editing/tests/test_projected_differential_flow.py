from __future__ import annotations

import inspect
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import projected_differential_flow as pdf  # noqa: E402
import infer_projected_delta_lora as infer_v2  # noqa: E402
import train_projected_delta_lora as train_v2  # noqa: E402


SHA1 = "1" * 40
SHA256 = "2" * 64


class ProjectedContractTests(unittest.TestCase):
    def test_contract_is_source_only_and_diagnostic(self) -> None:
        contract = pdf.sampler_contract()
        self.assertEqual(contract["inference_conditions"], ["source_video", "edit_instruction"])
        self.assertTrue(contract["train_inference_projection_identical"])
        self.assertEqual(contract["minimum_substeps_per_scheduler_interval"], 2)
        self.assertEqual(contract["status"], "diagnostic_bridge_not_final_method")
        forbidden = set(contract["forbidden_conditions"])
        self.assertTrue({"target_video", "mask", "track", "first_frame_anchor"} <= forbidden)
        parameters = set(inspect.signature(pdf.sample_differential_flow).parameters)
        self.assertTrue({"source_latent", "action_prompt_embeds", "noop_prompt_embeds"} <= parameters)
        self.assertTrue(parameters.isdisjoint({"target", "mask", "track", "pose", "anchor"}))

    def test_config_requires_real_substeps(self) -> None:
        self.assertEqual(pdf.DifferentialFlowConfig().validate().substeps, 2)
        for value in (0, 1, 17, True):
            with self.subTest(value=value), self.assertRaises(pdf.ProjectedFlowContractError):
                pdf.DifferentialFlowConfig(substeps=value).validate()

    def test_shifted_grid_is_descending_and_interval_weights_are_normalised(self) -> None:
        values = pdf.shifted_inference_sigmas(num_steps=40, flow_shift=5.0)
        self.assertEqual(len(values), 41)
        self.assertEqual(values[0], 1.0)
        self.assertEqual(values[-1], 0.0)
        self.assertTrue(all(a > b for a, b in zip(values, values[1:])))
        widths = [a - b for a, b in zip(values, values[1:])]
        self.assertGreater(widths[-1], widths[0])

    def test_noop_is_exact_bypass_without_torch(self) -> None:
        source = object()
        prompt = object()
        result, trace = pdf.sample_differential_flow(
            object(),
            source_latent=source,
            action_prompt_embeds=prompt,
            noop_prompt_embeds=prompt,
            return_trace=True,
        )
        self.assertIs(result, source)
        self.assertTrue(trace.identity_bypassed)
        self.assertEqual(trace.contribution_rms, ())


class ProjectedTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
        except Exception as error:  # pragma: no cover
            raise unittest.SkipTest(f"torch unavailable: {error}")
        cls.torch = torch

    def test_projection_is_idempotent_and_exactly_zero_dc(self) -> None:
        torch = self.torch
        field = torch.randn(2, 5 * 6, 4)
        projected = pdf.project_temporal_dc(field, latent_frames=5)
        self.assertTrue(torch.allclose(pdf.temporal_dc(projected, latent_frames=5), torch.zeros(2, 6, 4), atol=1e-6))
        self.assertTrue(torch.allclose(pdf.project_temporal_dc(projected, latent_frames=5), projected, atol=1e-6))
        static = torch.randn(2, 1, 6, 4).expand(2, 5, 6, 4).reshape(2, 30, 4)
        self.assertTrue(torch.allclose(pdf.project_temporal_dc(static, latent_frames=5), torch.zeros_like(static), atol=1e-6))

    def test_interval_weight_exposes_large_terminal_solver_interval(self) -> None:
        torch = self.torch
        weights = pdf.integration_interval_weight(
            torch.tensor([0.99, 0.01]), num_steps=40, flow_shift=5.0
        )
        self.assertGreater(float(weights[1]), float(weights[0]))

    def test_fake_solver_projects_and_records_every_substep(self) -> None:
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
                self.timesteps = torch.tensor([1000.0, 500.0])
                self.sigmas = torch.tensor([1.0, 0.5, 0.0])

        class Diffusion:
            use_unipc = True
            transformer_2 = None

            def __init__(self):
                self.transformer = Transformer()
                self.scheduler = Scheduler()
                self.calls = 0

            def shared_step(self, *, noisy_latents, cond_embeds, **kwargs):
                self.calls += 1
                result = torch.zeros((1, noisy_latents.shape[1], 4))
                if float(cond_embeds[0, 0, 0]) == 1.0:
                    # Query frames are the last two tokens.  Their mean is
                    # zero, so projection keeps this known field unchanged.
                    result[:, -2, :] = -1.0
                    result[:, -1, :] = 1.0
                return result

        diffusion = Diffusion()
        source = torch.zeros(1, 1, 2, 2, 2)
        result, trace = pdf.sample_differential_flow(
            diffusion,
            source_latent=source,
            action_prompt_embeds=torch.ones(1, 1, 1),
            noop_prompt_embeds=torch.zeros(1, 1, 1),
            config=pdf.DifferentialFlowConfig(num_inference_steps=2, substeps=2),
            return_trace=True,
        )
        self.assertEqual(diffusion.calls, 2 * 2 * 2)
        self.assertEqual(len(trace.contribution_rms), 4)
        self.assertEqual(len(trace.sigmas), 5)
        packed = cdf._pack_spatial_latent(result)
        self.assertTrue(torch.allclose(pdf.temporal_dc(packed, latent_frames=2), torch.zeros(1, 1, 4), atol=1e-6))


class TrainingCliTests(unittest.TestCase):
    def _args(self, *extra: str):
        parser = train_v2.build_parser()
        return parser.parse_args([
            "--bernini-root", "/b", "--veomni-root", "/v", "--checkpoint", "/c",
            "--preprocessed-parquet-dir", "/d", "--dataset-summary", "/s",
            "--output", "/o", "--method-source-revision", SHA1,
            "--method-source-archive-sha256", SHA256, *extra,
        ])

    def test_defaults_bind_projection_interval_weighting_and_no_anchor(self) -> None:
        args = self._args()
        train_v2.validate_cli(args)
        self.assertEqual(args.anchor_loss_weight, 0.0)
        self.assertEqual(args.branch_state_mode, "separate_clean_paths")
        self.assertEqual(args.bridge_consistency_weight, 0.0)
        self.assertEqual(args.lora_scope, "cross_q_out")
        self.assertEqual(args.integration_steps, 40)
        self.assertEqual(args.integration_flow_shift, 5.0)
        self.assertGreater(args.dc_loss_weight, 0.0)
        receipt = train_v2._supervision_receipt(args)
        self.assertTrue(receipt["train_inference_projection_identical"])
        self.assertFalse(receipt["full_target_framewise_loss_enabled"])

    def test_anchor_and_raw_objective_fail_closed(self) -> None:
        for extra in (["--anchor-loss-weight", "0.1"], ["--motion-objective", "raw_delta"]):
            with self.subTest(extra=extra), self.assertRaises(train_v2.base.DeltaTrainingError):
                train_v2.validate_cli(self._args(*extra))


class InferenceCliTests(unittest.TestCase):
    def _args(self, *extra: str):
        parser = infer_v2.build_parser()
        return parser.parse_args([
            "--bernini-root", "/b", "--veomni-root", "/v", "--checkpoint", "/c",
            "--adapter-checkpoint", "/a", "--source-video", "/source.mp4",
            "--instruction", "move", "--output", "/out.mp4",
            "--method-source-revision", SHA1,
            "--method-source-archive-sha256", SHA256, *extra,
        ])

    def test_projected_sampler_and_two_substeps_are_default(self) -> None:
        args = self._args()
        infer_v2.validate_cli(args)
        self.assertEqual(args.sampling_mode, "differential")
        self.assertEqual(args.solver_substeps, 2)

    def test_standard_sampler_and_one_substep_fail_closed(self) -> None:
        for extra in (["--sampling-mode", "standard"], ["--solver-substeps", "1"]):
            with self.subTest(extra=extra), self.assertRaises(infer_v2.base.DeltaInferenceError):
                infer_v2.validate_cli(self._args(*extra))


if __name__ == "__main__":
    unittest.main()
