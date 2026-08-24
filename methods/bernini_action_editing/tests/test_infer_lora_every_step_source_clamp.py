from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = METHOD_ROOT / "scripts/auh_infer_action_quotient_one_v1.sh"
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_lora as inference  # noqa: E402


try:
    import torch
except ImportError:
    torch = None


class LauncherStaticTests(unittest.TestCase):
    def test_cli_and_auh_launcher_expose_new_policy_without_removing_old_ones(self) -> None:
        parser = inference.build_parser()
        action = next(
            item for item in parser._actions if item.dest == "source_onset_policy"
        )
        self.assertEqual(
            set(action.choices),
            {"none", "hard1", "ramp3", "hard1_every_step"},
        )
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn(
            "none|hard1|ramp3|hard1_every_step",
            source,
        )
        self.assertIn(
            '--source-onset-policy "${source_onset_policy}"',
            source,
        )


class UniPCMultistepScheduler:
    """Small flow-UniPC-shaped double; the wrapper still owns projection."""

    def __init__(self, *, predict_x0: bool = True) -> None:
        self.config = SimpleNamespace(
            _class_name="UniPCMultistepScheduler",
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            predict_x0=predict_x0,
            final_sigmas_type="zero",
            flow_shift=5.0,
        )
        self.sigmas = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)
        self.timesteps = torch.tensor([999, 500], dtype=torch.int64)
        self.step_index = None
        self.calls = []

    def step(self, model_output, timestep, sample, return_dict=True):
        index = 0 if self.step_index is None else int(self.step_index)
        self.calls.append(
            {
                "model_output": model_output.detach().clone(),
                "timestep": timestep,
                "sample": sample,
                "return_dict": return_dict,
            }
        )
        self.step_index = index + 1
        # The exact fake update is deliberately not the desired trajectory;
        # the wrapper must replace phase zero after this native call.
        return (sample + 0.125 * model_output,)


@unittest.skipIf(torch is None, "torch is unavailable")
class EveryStepSourceClampTests(unittest.TestCase):
    def _fixture(self, *, predict_x0: bool = True):
        scheduler = UniPCMultistepScheduler(predict_x0=predict_x0)
        diffusion = SimpleNamespace(use_unipc=True, scheduler=scheduler)
        source = (
            torch.arange(1 * 16 * 21 * 2 * 2, dtype=torch.float32)
            .reshape(1, 16, 21, 2, 2)
            .div(100.0)
            .contiguous()
        )
        packed = inference._pack_wan_source_latent(source)
        initial = torch.linspace(
            -1.0, 1.0, packed.numel(), dtype=torch.float32
        ).reshape_as(packed)
        return diffusion, scheduler, source, packed, initial

    def test_native_pack_is_phase_major_wan_122_order(self) -> None:
        _, _, source, packed, _ = self._fixture()
        expected_phase0 = (
            source[:, :, 0]
            .reshape(1, 16, 1, 2, 1, 2)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(1, 1, 64)
        )
        expected_phase1 = (
            source[:, :, 1]
            .reshape(1, 16, 1, 2, 1, 2)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(1, 1, 64)
        )
        self.assertTrue(torch.equal(packed[:, 0:1], expected_phase0))
        self.assertTrue(torch.equal(packed[:, 1:2], expected_phase1))

    def test_every_step_forces_velocity_and_projects_only_phase_zero(self) -> None:
        diffusion, scheduler, source, source_packed, initial = self._fixture()
        original_function = scheduler.step.__func__
        output0 = torch.full_like(initial, 3.0)
        output1 = torch.full_like(initial, -2.0)
        original0 = output0.clone()
        original1 = output1.clone()

        with inference.hard_phase0_source_trajectory_clamp(
            diffusion, source, expected_steps=2
        ) as trace:
            first = scheduler.step(
                model_output=output0,
                timestep=scheduler.timesteps[0],
                sample=initial,
                return_dict=False,
            )
            self.assertIs(type(first), tuple)
            expected_midpoint = 0.5 * source_packed[:, :1] + 0.5 * initial[:, :1]
            self.assertTrue(torch.equal(first[0][:, :1], expected_midpoint))
            native_other = initial[:, 1:] + 0.125 * output0[:, 1:]
            self.assertTrue(torch.equal(first[0][:, 1:], native_other))

            second = scheduler.step(
                output1, scheduler.timesteps[1], first[0], False
            )
            self.assertTrue(torch.equal(second[0][:, :1], source_packed[:, :1]))
            native_other = first[0][:, 1:] + 0.125 * output1[:, 1:]
            self.assertTrue(torch.equal(second[0][:, 1:], native_other))

        self.assertNotIn("step", vars(scheduler))
        self.assertIs(scheduler.step.__func__, original_function)
        self.assertTrue(torch.equal(output0, original0))
        self.assertTrue(torch.equal(output1, original1))
        expected_velocity = initial[:, :1] - source_packed[:, :1]
        self.assertTrue(
            torch.equal(scheduler.calls[0]["model_output"][:, :1], expected_velocity)
        )
        self.assertTrue(
            torch.equal(scheduler.calls[1]["model_output"][:, :1], expected_velocity)
        )
        self.assertTrue(
            torch.equal(scheduler.calls[0]["model_output"][:, 1:], output0[:, 1:])
        )
        self.assertTrue(
            torch.equal(scheduler.calls[1]["model_output"][:, 1:], output1[:, 1:])
        )
        receipt = trace.as_dict()
        self.assertEqual(receipt["step_count"], 2)
        self.assertEqual(receipt["steps"][-1]["next_sigma"], 0.0)
        self.assertTrue(receipt["initial_packed_noise_captured"])
        self.assertFalse(receipt["identity_or_background_claim"])

    def test_existing_instance_step_is_restored_after_solver_exception(self) -> None:
        diffusion, scheduler, source, source_packed, initial = self._fixture()

        def failing_step(*args, **kwargs):
            del args, kwargs
            raise LookupError("synthetic native solver failure")

        scheduler.step = failing_step
        model_output = torch.zeros_like(source_packed)
        with self.assertRaisesRegex(LookupError, "synthetic native solver failure"):
            with inference.hard_phase0_source_trajectory_clamp(
                diffusion, source, expected_steps=2
            ):
                scheduler.step(
                    model_output,
                    scheduler.timesteps[0],
                    initial,
                    return_dict=False,
                )
        self.assertIs(scheduler.step, failing_step)

    def test_bad_flow_contract_fails_before_installation(self) -> None:
        diffusion, scheduler, source, _, _ = self._fixture(predict_x0=False)
        with self.assertRaisesRegex(
            inference.InferenceContractError, "predict_x0"
        ):
            with inference.hard_phase0_source_trajectory_clamp(
                diffusion, source, expected_steps=2
            ):
                self.fail("invalid scheduler must not install")
        self.assertNotIn("step", vars(scheduler))

    def test_return_dict_true_fails_without_calling_native_step_and_restores(self) -> None:
        diffusion, scheduler, source, source_packed, initial = self._fixture()
        with self.assertRaisesRegex(
            inference.InferenceContractError, "return_dict=False"
        ):
            with inference.hard_phase0_source_trajectory_clamp(
                diffusion, source, expected_steps=2
            ):
                scheduler.step(
                    torch.zeros_like(source_packed),
                    scheduler.timesteps[0],
                    initial,
                    return_dict=True,
                )
        self.assertEqual(scheduler.calls, [])
        self.assertNotIn("step", vars(scheduler))

    def test_incomplete_normal_exit_fails_closed_and_restores(self) -> None:
        diffusion, scheduler, source, _, _ = self._fixture()
        with self.assertRaisesRegex(
            inference.InferenceContractError, "did not complete"
        ):
            with inference.hard_phase0_source_trajectory_clamp(
                diffusion, source, expected_steps=2
            ):
                pass
        self.assertNotIn("step", vars(scheduler))

    def test_every_step_policy_has_no_second_post_denoise_mutation(self) -> None:
        generated = torch.randn(1, 16, 21, 2, 2)
        source = torch.randn_like(generated)
        result = inference.apply_source_onset_policy(
            generated, source, inference.EVERY_STEP_SOURCE_ONSET_POLICY
        )
        self.assertIs(result, generated)


if __name__ == "__main__":
    unittest.main()
