from __future__ import annotations

import inspect
from pathlib import Path
import sys
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from spt_v2 import phase_transport as spt  # noqa: E402
from spt_v2 import unipc_projection as projection  # noqa: E402


class PureProjectionContractTests(unittest.TestCase):
    def test_runtime_contract_is_source_instruction_only_and_keeps_unipc(self) -> None:
        contract = projection.sampler_contract()
        self.assertEqual(
            contract["inference_conditions"], ["source_video", "edit_instruction"]
        )
        self.assertEqual(contract["required_plan_provenance"], "student")
        self.assertEqual(contract["integrator"], "original_unipc_scheduler_step")
        self.assertFalse(contract["custom_euler_integrator"])
        self.assertEqual(contract["packed_latent_phases"], 21)
        self.assertEqual(contract["default_max_generate_fraction"], 0.12)
        self.assertEqual(
            contract["generate_budget_scope"],
            "each_sample_each_latent_phase_spatial_mean",
        )
        self.assertEqual(contract["generate_budget_policy"], "fail_before_scheduler_step")
        self.assertEqual(
            contract["unbounded_generate_budget_mode"],
            "explicit_offline_oracle_ablation_only",
        )
        forbidden = set(contract["forbidden_inference_conditions"])
        self.assertTrue(
            {"target_video", "paired_oracle_plan", "mask", "track", "pose"}
            <= forbidden
        )

    def test_wrapper_api_has_no_target_or_spatial_hint_inputs(self) -> None:
        parameters = set(inspect.signature(projection.project_unipc_steps).parameters)
        self.assertTrue({"scheduler", "source_packed", "plan", "height", "width"} <= parameters)
        self.assertTrue(
            parameters.isdisjoint(
                {"target", "target_video", "mask", "track", "pose", "flow", "anchor"}
            )
        )


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is unavailable")
class TensorProjectionTests(unittest.TestCase):
    class FakeScheduler:
        def __init__(self, *, sigmas=None, timesteps=None):
            self.sigmas = torch.tensor(sigmas if sigmas is not None else [0.5, 0.25, 0.0])
            self.timesteps = torch.tensor(
                timesteps if timesteps is not None else [900.0, 500.0, 0.0]
            )
            self.step_index = None
            self.calls = []
            self.return_sentinel = object()

        def index_for_timestep(self, timestep):
            indices = (self.timesteps == timestep).nonzero().reshape(-1)
            return int(indices[0].item())

        def step(self, model_output, timestep, sample, return_dict=True):
            self.calls.append(
                {
                    "model_output": model_output,
                    "timestep": timestep,
                    "sample": sample,
                    "return_dict": return_dict,
                }
            )
            if self.step_index is None:
                self.step_index = self.index_for_timestep(timestep) + 1
            else:
                self.step_index += 1
            return self.return_sentinel

    def _inputs(self, *, gates=(1.0, 0.0, 0.0), provenance="student"):
        source = torch.randn(1, 21, 2, 3, 4)
        source_packed = spt.video_to_packed(source)
        sample = torch.randn_like(source_packed)
        model_output = torch.randn_like(source_packed)
        offsets = torch.zeros(1, 3, 21, 2, 3)
        gate_probs = torch.zeros_like(offsets)
        for index, value in enumerate(gates):
            gate_probs[:, index].fill_(value)
        plan = spt.PhasePlan(offsets, gate_probs, provenance)
        return source, source_packed, sample, model_output, plan

    def test_preserve_projection_reaches_original_step_and_return_is_unchanged(self) -> None:
        source, source_packed, sample, model_output, plan = self._inputs()
        scheduler = self.FakeScheduler()
        self.assertNotIn("step", vars(scheduler))
        original_function = scheduler.step.__func__

        with projection.project_unipc_steps(
            scheduler,
            source_packed=source_packed,
            plan=plan,
            height=2,
            width=3,
        ) as trace:
            result = scheduler.step(
                model_output=model_output,
                timestep=torch.tensor(900.0),
                sample=sample,
                return_dict=False,
            )
            self.assertIs(result, scheduler.return_sentinel)
            self.assertEqual(len(trace.records), 1)

        self.assertNotIn("step", vars(scheduler))
        self.assertIs(scheduler.step.__func__, original_function)
        expected = (sample - source_packed) / 0.5
        self.assertTrue(
            torch.allclose(
                scheduler.calls[0]["model_output"], expected, atol=2e-6, rtol=2e-6
            )
        )
        self.assertFalse(scheduler.calls[0]["return_dict"])
        record = trace.records[0]
        self.assertEqual(record.step_index, 0)
        self.assertEqual(record.sigma, 0.5)
        self.assertTrue(record.projection_applied)
        self.assertAlmostEqual(record.preserve_fraction, 1.0)
        self.assertAlmostEqual(record.transport_fraction, 0.0)
        self.assertAlmostEqual(record.generate_fraction, 0.0)
        expected_correction = torch.sqrt(torch.mean((expected - model_output) ** 2))
        self.assertAlmostEqual(record.correction_rms, float(expected_correction), places=6)

    def test_fractional_ptg_trace_and_positional_step_signature(self) -> None:
        _, source_packed, sample, model_output, plan = self._inputs(
            gates=(0.2, 0.3, 0.5)
        )
        scheduler = self.FakeScheduler()
        with projection.project_unipc_steps(
            scheduler,
            source_packed=source_packed,
            plan=plan,
            height=2,
            width=3,
            max_generate_fraction=0.6,
        ) as trace:
            scheduler.step(model_output, torch.tensor(900.0), sample, False)
        record = trace.records[0]
        self.assertAlmostEqual(record.preserve_fraction, 0.2, places=6)
        self.assertAlmostEqual(record.transport_fraction, 0.3, places=6)
        self.assertAlmostEqual(record.generate_fraction, 0.5, places=6)
        self.assertAlmostEqual(record.max_sample_generate_fraction, 0.5, places=6)
        self.assertAlmostEqual(record.max_phase_generate_fraction, 0.5, places=6)
        self.assertEqual(record.generate_budget, 0.6)
        self.assertEqual(trace.max_generate_fraction, 0.6)

    def test_zero_sigma_never_divides_and_preserves_model_output_object(self) -> None:
        _, source_packed, sample, model_output, plan = self._inputs()
        scheduler = self.FakeScheduler(sigmas=[0.0], timesteps=[0.0])
        with projection.project_unipc_steps(
            scheduler,
            source_packed=source_packed,
            plan=plan,
            height=2,
            width=3,
        ) as trace:
            result = scheduler.step(model_output, torch.tensor(0.0), sample, return_dict=False)
            self.assertIs(result, scheduler.return_sentinel)
        self.assertIs(scheduler.calls[0]["model_output"], model_output)
        self.assertFalse(trace.records[0].projection_applied)
        self.assertEqual(trace.records[0].correction_rms, 0.0)

    def test_projection_failure_is_closed_and_context_restores_original_step(self) -> None:
        _, source_packed, sample, model_output, plan = self._inputs()
        scheduler = self.FakeScheduler()
        original_function = scheduler.step.__func__
        with self.assertRaises(projection.UniPCProjectionError):
            with projection.project_unipc_steps(
                scheduler,
                source_packed=source_packed,
                plan=plan,
                height=2,
                width=3,
            ):
                scheduler.step(
                    model_output[:, :-1], torch.tensor(900.0), sample, return_dict=False
                )
        self.assertEqual(scheduler.calls, [])
        self.assertNotIn("step", vars(scheduler))

    def test_phase_local_redraw_cannot_hide_under_global_average(self) -> None:
        _, source_packed, _, _, plan = self._inputs(gates=(0.995, 0.0, 0.005))
        plan.gate_probs[:, 0, 0].fill_(0.0)
        plan.gate_probs[:, 2, 0].fill_(1.0)
        # Global generate is only about 5.2%, but one whole latent phase would
        # be regenerated.  The training oracle also budgets each phase, so the
        # deployment boundary must reject this hidden local redraw.
        self.assertLess(float(plan.gate_probs[:, 2].mean()), 0.12)
        scheduler = self.FakeScheduler()
        original_function = scheduler.step.__func__
        with self.assertRaisesRegex(
            projection.UniPCProjectionError, "per-phase generate budget"
        ):
            with projection.project_unipc_steps(
                scheduler,
                source_packed=source_packed,
                plan=plan,
                height=2,
                width=3,
            ):
                self.fail("phase-local redraw leak must not install")
        self.assertIs(scheduler.step.__func__, original_function)

    def test_paired_oracle_plan_is_rejected_before_installation(self) -> None:
        _, source_packed, _, _, plan = self._inputs(provenance="oracle_pair_proxy")
        scheduler = self.FakeScheduler()
        with self.assertRaises(projection.UniPCProjectionError):
            with projection.project_unipc_steps(
                scheduler,
                source_packed=source_packed,
                plan=plan,
                height=2,
                width=3,
            ):
                self.fail("paired oracle wrapper must not install")
        self.assertNotIn("step", vars(scheduler))

    def test_generate_redraw_leak_fails_before_step_installation(self) -> None:
        _, source_packed, _, _, plan = self._inputs(gates=(0.2, 0.2, 0.6))
        scheduler = self.FakeScheduler()
        with self.assertRaisesRegex(
            projection.UniPCProjectionError, "per-phase generate budget"
        ):
            with projection.project_unipc_steps(
                scheduler,
                source_packed=source_packed,
                plan=plan,
                height=2,
                width=3,
            ):
                self.fail("over-budget wrapper must not install")
        self.assertEqual(scheduler.calls, [])
        self.assertNotIn("step", vars(scheduler))

    def test_unbounded_budget_requires_explicit_oracle_ablation_flag(self) -> None:
        _, source_packed, _, _, plan = self._inputs(gates=(0.2, 0.2, 0.6))
        scheduler = self.FakeScheduler()
        with self.assertRaisesRegex(
            projection.UniPCProjectionError, "explicit oracle ablation"
        ):
            with projection.project_unipc_steps(
                scheduler,
                source_packed=source_packed,
                plan=plan,
                height=2,
                width=3,
                max_generate_fraction=None,
            ):
                self.fail("implicit unbounded budget must not install")

        _, _, _, _, oracle_plan = self._inputs(
            gates=(0.2, 0.2, 0.6), provenance="oracle_pair_proxy"
        )
        with projection.project_unipc_steps(
            scheduler,
            source_packed=source_packed,
            plan=oracle_plan,
            height=2,
            width=3,
            max_generate_fraction=None,
            allow_unbounded_generate_oracle_ablation=True,
        ) as trace:
            self.assertIsNone(trace.max_generate_fraction)
            self.assertTrue(trace.oracle_ablation)

    def test_begin_index_selects_sigma_without_mutating_solver_before_call(self) -> None:
        _, source_packed, sample, model_output, plan = self._inputs()
        scheduler = self.FakeScheduler()
        scheduler._begin_index = 1
        observed_before_original = []
        original = scheduler.step

        def checking_step(*args, **kwargs):
            observed_before_original.append(scheduler.step_index)
            return original(*args, **kwargs)

        scheduler.step = checking_step
        with projection.project_unipc_steps(
            scheduler,
            source_packed=source_packed,
            plan=plan,
            height=2,
            width=3,
        ) as trace:
            scheduler.step(model_output, torch.tensor(500.0), sample, return_dict=False)
        self.assertEqual(observed_before_original, [None])
        self.assertEqual(trace.records[0].step_index, 1)
        self.assertEqual(trace.records[0].sigma, 0.25)
        self.assertIs(scheduler.step, checking_step)

    def test_real_diffusers_unipc_step_keeps_native_tuple_and_state_update(self) -> None:
        try:
            from diffusers import UniPCMultistepScheduler
        except Exception as error:
            self.skipTest(f"diffusers UniPC unavailable: {error}")

        _, source_packed, sample, model_output, plan = self._inputs()
        scheduler = UniPCMultistepScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(2)
        timestep = scheduler.timesteps[0]
        original_function = scheduler.step.__func__
        with projection.project_unipc_steps(
            scheduler,
            source_packed=source_packed,
            plan=plan,
            height=2,
            width=3,
        ) as trace:
            result = scheduler.step(
                model_output, timestep, sample, return_dict=False
            )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result[0].shape), tuple(sample.shape))
        self.assertEqual(scheduler.step_index, 1)
        self.assertEqual(len(trace.records), 1)
        self.assertIs(scheduler.step.__func__, original_function)

    def test_original_scheduler_exception_still_restores_exact_instance_step(self) -> None:
        _, source_packed, sample, model_output, plan = self._inputs()
        scheduler = self.FakeScheduler()

        def failing_step(*args, **kwargs):
            raise LookupError("synthetic solver failure")

        scheduler.step = failing_step
        with self.assertRaisesRegex(LookupError, "synthetic solver failure"):
            with projection.project_unipc_steps(
                scheduler,
                source_packed=source_packed,
                plan=plan,
                height=2,
                width=3,
            ) as trace:
                scheduler.step(model_output, torch.tensor(900.0), sample, False)
        self.assertIs(scheduler.step, failing_step)
        self.assertEqual(trace.records, [])


if __name__ == "__main__":
    unittest.main()
