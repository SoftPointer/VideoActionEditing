#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import inspect
import sys
import unittest

try:
    import torch
except ImportError:  # pragma: no cover - environment dependent
    torch = None


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import train_feasible_quotient_auh as trainer  # noqa: E402


SHA1 = "1" * 40
SHA256 = "2" * 64


def _argv() -> list[str]:
    return [
        "--bernini-root",
        "/bernini",
        "--veomni-root",
        "/veomni",
        "--checkpoint",
        "/checkpoint",
        "--preprocessed-parquet-dir",
        "/data/parquet",
        "--dataset-summary",
        "/data/summary.json",
        "--routing-jsonl",
        "/data/strict359.jsonl",
        "--output",
        "/output",
        "--method-source-revision",
        SHA1,
        "--method-source-archive-sha256",
        SHA256,
        "--inference-loader-parity-verified",
    ]


class FeasibleQuotientTrainerContractTests(unittest.TestCase):
    def test_parser_and_validation_pin_exact40_lower_lr_pilot(self):
        args = trainer.build_parser().parse_args(_argv())
        self.assertEqual(args.max_steps, 40)
        self.assertEqual(args.save_every, 40)
        self.assertEqual(args.learning_rate, 1.0e-5)
        self.assertEqual(args.teacher_mode, "paired_displacement_only")
        self.assertEqual(args.relational_auxiliary_weight, 0.0)
        trainer.validate_cli(args)

        args.max_steps = 80
        with self.assertRaises(trainer.FeasibleQuotientAUHError):
            trainer.validate_cli(args)
        args.max_steps = 40
        args.weight_decay = 0.01
        with self.assertRaises(trainer.FeasibleQuotientAUHError):
            trainer.validate_cli(args)

    def test_five_forward_cell_contains_no_generator_model_call(self):
        source = inspect.getsource(trainer._run_five_forward_cell)
        self.assertEqual(source.count("motion.renderer_velocity_prediction("), 5)
        self.assertNotIn("candidate.generator_", source)
        self.assertIn("compute_feasible_quotient_objective", source)

    def test_immutable_contract_is_source_only_at_inference(self):
        args = trainer.build_parser().parse_args(_argv())
        dataset = SimpleNamespace(signature="dataset-signature")
        summary = {"sha256": "3" * 64, "index_sha256": "4" * 64}
        router = SimpleNamespace(digest="5" * 64, file_sha256="6" * 64)
        route = SimpleNamespace(
            iid="iid-0", tier="motion_only", full_target_weight=0.0
        )
        immutable = trainer._immutable_contract(
            args=args,
            dataset=dataset,
            dataset_summary=summary,
            router=router,
            eligible_routes=[(index, route) for index in range(359)],
            target_modules=[f"target_{index:02d}" for index in range(46)],
            checkpoint=Path("/checkpoint"),
            loss_config=trainer.loss_config_from_args(args),
        )
        value = immutable["value"]
        self.assertEqual(value["forwards_per_candidate"], 5)
        self.assertEqual(value["training_generator_forwards"], 0)
        self.assertEqual(value["inference_generator_forwards"], 0)
        self.assertEqual(
            value["training_only_conditions"], ["paired_target_video"]
        )
        self.assertEqual(
            value["appearance_carrier"],
            "frozen_noop_reconstruction_section",
        )
        self.assertFalse(value["target_used_as_model_condition"])
        self.assertEqual(value["training_diffusion_query"], "target(beta=1)")
        self.assertTrue(
            value["paired_target_constructs_training_diffusion_state"]
        )
        self.assertFalse(
            value["paired_target_used_as_external_model_condition"]
        )
        self.assertEqual(
            value["target_motion_teacher"],
            "Q0(target_clean-stopgrad(frozen_noop_section))",
        )
        self.assertIn("beta1", value["target_section_reference"])
        self.assertEqual(
            value["editor_guidance"]["v8_frozen_action_section"],
            "local_fp32_apg_after_bit_exact_native_bf16_official_parity",
        )
        self.assertFalse(value["objective_contract"]["first_frame_anchor"])
        self.assertEqual(
            immutable["digest"], trainer.legacy.object_sha256(value)
        )

    def test_target_state_builder_is_installed_without_generator_batches(self):
        install_source = inspect.getsource(trainer._install_strategy)
        self.assertIn(
            "v6_runtime._prepare_candidate_cpu = _prepare_target_state_candidate_cpu",
            install_source,
        )
        self.assertIn(
            "_move_target_state_candidate_to_device",
            install_source,
        )
        source = inspect.getsource(trainer._prepare_target_state_candidate_cpu)
        self.assertIn('endpoints["target"]', source)
        self.assertNotIn("generator_action", source)


@unittest.skipUnless(torch is not None, "PyTorch is required for AdamW reset tests")
class FeasibleQuotientOptimizerBoundaryTests(unittest.TestCase):
    def test_adamw_moments_reset_exactly_before_first_zero_release_step(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer_class = trainer._build_zero_release_reset_adamw(
            torch.optim.AdamW
        )
        optimizer = optimizer_class([parameter], lr=1.0e-3, weight_decay=0.0)
        observed_state_steps = []
        for index in range(40):
            optimizer.zero_grad(set_to_none=True)
            parameter.grad = torch.tensor([1.0 if index < 31 else -1.0])
            optimizer.step()
            state_step = optimizer.state[parameter]["step"]
            observed_state_steps.append(int(state_step.item()))
        self.assertEqual(observed_state_steps[30], 31)
        self.assertEqual(observed_state_steps[31], 1)
        self.assertEqual(observed_state_steps[-1], 9)
        self.assertEqual(optimizer._v8_completed_optimizer_steps, 40)
        self.assertEqual(optimizer._v8_moment_reset_count, 1)
        payload = trainer._optimizer_payload(
            optimizer=optimizer,
            global_step=40,
            immutable={"value": {}, "digest": "0" * 64},
            parameter_names=["adapter.weight"],
            step_audit=[],
        )
        self.assertEqual(
            payload["zero_release_moment_reset"]["state_step_values"], [9]
        )
        self.assertEqual(
            payload["zero_release_moment_reset"]["state_parameter_count"], 1
        )

    def test_adamw_reset_wrapper_rejects_weight_decay(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer_class = trainer._build_zero_release_reset_adamw(
            torch.optim.AdamW
        )
        with self.assertRaises(trainer.FeasibleQuotientAUHError):
            optimizer_class([parameter], lr=1.0e-3, weight_decay=0.01)


@unittest.skipUnless(
    torch is not None and torch.cuda.is_available(),
    "AUH ROCm GPU is required for the numerical APG parity certificate",
)
class FeasibleQuotientROCmParityTests(unittest.TestCase):
    def test_gpu_packed_training_apg_is_exact_spatial_inference_apg(self):
        batch, phases, patch_h, patch_w, channels = 1, 4, 2, 3, 3
        patch_elements = 2 * 2 * channels

        def grid_to_spatial(grid):
            return (
                grid.reshape(
                    batch,
                    phases,
                    patch_h,
                    patch_w,
                    2,
                    2,
                    channels,
                )
                .permute(0, 6, 1, 2, 4, 3, 5)
                .reshape(
                    batch,
                    channels,
                    phases,
                    patch_h * 2,
                    patch_w * 2,
                )
            )

        def spatial_to_grid(spatial):
            return (
                spatial.reshape(
                    batch,
                    channels,
                    phases,
                    patch_h,
                    2,
                    patch_w,
                    2,
                )
                .permute(0, 2, 3, 5, 4, 6, 1)
                .reshape(
                    batch,
                    phases,
                    patch_h * patch_w,
                    patch_elements,
                )
            )

        count = batch * phases * patch_h * patch_w * patch_elements
        negative = (
            torch.arange(count, dtype=torch.float32, device="cuda").reshape(
                batch, phases, patch_h * patch_w, patch_elements
            )
            / 16.0
        )
        conditional = negative + (
            torch.arange(
                count - 1,
                -1,
                -1,
                dtype=torch.float32,
                device="cuda",
            ).reshape_as(negative)
            / 32.0
        )
        packed = trainer.v5._official_momentum_zero_apg(
            conditional, negative
        )
        spatial = trainer.v5.tri._normalized_guidance(
            grid_to_spatial(conditional),
            grid_to_spatial(negative),
            trainer.v5.APG_GUIDANCE_SCALE,
            trainer.v5.tri._MomentumBuffer(0.0, branch="v8-rocm-parity"),
            trainer.v5.APG_ETA,
            trainer.v5.APG_NORM_THRESHOLD,
        )
        self.assertTrue(torch.equal(packed, spatial_to_grid(spatial)))


if __name__ == "__main__":
    unittest.main()
