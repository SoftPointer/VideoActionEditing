from __future__ import annotations

import copy
import math
import os
from pathlib import Path
import sys
from typing import Any
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import torch

import t_qmosaic_bernini_unipc_runtime_adapter_v1 as runtime
import t_qmosaic_trajectory_intervention_v1 as trajectory


def _initial_state() -> torch.Tensor:
    return torch.tensor(
        [[[-0.0, 0.25], [-0.5, 1.0]]],
        dtype=torch.float32,
    ).detach()


class _FakeSchedulerCore:
    def __init__(
        self,
        *,
        corrupt_schedule_index: int | None = None,
        fail_at: int | None = None,
        result_kind: str = "tuple",
        mutate_config_after: int | None = None,
    ) -> None:
        self.config = copy.deepcopy(runtime._expected_config_snapshot())
        self.corrupt_schedule_index = corrupt_schedule_index
        self.fail_at = fail_at
        self.result_kind = result_kind
        self.mutate_config_after = mutate_config_after
        self.timesteps = torch.empty(0, dtype=torch.int64)
        self.sigmas = torch.empty(0, dtype=torch.float32)
        self.step_index: int | None = None
        self.raw_results: list[tuple[torch.Tensor]] = []
        self.set_timesteps_calls: list[int] = []

    def set_timesteps(self, num_inference_steps: int) -> None:
        self.set_timesteps_calls.append(num_inference_steps)
        self.timesteps = torch.tensor(
            trajectory.PINNED_TIMESTEPS,
            dtype=torch.int64,
        )
        self.sigmas = torch.tensor(
            [*trajectory.PINNED_SIGMAS, 0.0],
            dtype=torch.float32,
        )
        if self.corrupt_schedule_index is not None:
            self.timesteps[self.corrupt_schedule_index] += 1
        self.step_index = None
        self.raw_results.clear()

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> tuple[torch.Tensor]:
        del model_output, timestep
        index = 0 if self.step_index is None else self.step_index
        if self.fail_at is not None and index == self.fail_at:
            raise RuntimeError("synthetic original UniPC failure")
        if return_dict is not False:
            raise RuntimeError("fake official scheduler requires return_dict=False")
        state = (sample + float(index + 1) / 100.0).detach()
        result = (state,)
        self.raw_results.append(result)
        self.step_index = index + 1
        if self.mutate_config_after is not None and index == self.mutate_config_after:
            self.config["solver_type"] = "midpoint"
        if self.result_kind == "list":
            return [state]  # type: ignore[return-value]
        if self.result_kind == "pair":
            return (state, state)  # type: ignore[return-value]
        return result


UniPCMultistepScheduler = type(
    "UniPCMultistepScheduler",
    (_FakeSchedulerCore,),
    {"__module__": runtime.PINNED_SCHEDULER_CLASS[0]},
)


class _FakeDiffusionCore:
    def __init__(
        self,
        scheduler: _FakeSchedulerCore,
        *,
        sample_step_count: int = 40,
        raise_after_step: int | None = None,
        scheduler_call_form: str = "official",
    ) -> None:
        self.scheduler = scheduler
        self.use_unipc = True
        self.transformer_2 = None
        self.sample_step_count = sample_step_count
        self.raise_after_step = raise_after_step
        self.scheduler_call_form = scheduler_call_form
        self.returned_containers: list[Any] = []
        self.patch_was_live_during_sample = False

    def sample(
        self,
        prompt_embeds: Any = None,
        num_inference_steps: int = 40,
        guidance_mode: str = "v2v_apg",
        flow_shift: float = 5.0,
    ) -> torch.Tensor:
        del prompt_embeds, guidance_mode, flow_shift
        self.scheduler.set_timesteps(num_inference_steps)
        self.patch_was_live_during_sample = (
            getattr(
                self.scheduler.step,
                "_bernini_t_qmosaic_unipc_runtime_v1",
                None,
            )
            is not None
        )
        state = _initial_state()
        for index, timestep in enumerate(
            self.scheduler.timesteps[: self.sample_step_count]
        ):
            model_output = torch.zeros_like(state)
            if self.scheduler_call_form == "missing_return_dict":
                result = self.scheduler.step(model_output, timestep, state)
            elif self.scheduler_call_form == "true_return_dict":
                result = self.scheduler.step(
                    model_output,
                    timestep,
                    state,
                    return_dict=True,
                )
            else:
                if self.scheduler_call_form == "float_timestep":
                    timestep = timestep.float()
                result = self.scheduler.step(
                    model_output,
                    timestep,
                    state,
                    return_dict=False,
                )
            self.returned_containers.append(result)
            state = result[0]
            if self.raise_after_step is not None and index == self.raise_after_step:
                raise RuntimeError("synthetic Bernini sample failure")
        if self.sample_step_count > int(self.scheduler.timesteps.numel()):
            # A deliberate 41st call reuses the terminal published timestep;
            # the adapter must reject it before the original scheduler runs.
            result = self.scheduler.step(
                torch.zeros_like(state),
                self.scheduler.timesteps[-1],
                state,
                return_dict=False,
            )
            state = result[0]
        return state


GEN_Wanx22 = type(
    "GEN_Wanx22",
    (_FakeDiffusionCore,),
    {"__module__": runtime.PINNED_DIFFUSION_CLASS[0]},
)


def _fake_diffusion(**kwargs: Any) -> Any:
    scheduler_kwargs = kwargs.pop("scheduler_kwargs", {})
    scheduler = UniPCMultistepScheduler(**scheduler_kwargs)
    return GEN_Wanx22(scheduler, **kwargs)


def _vjp_rows(shape: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
    count = math.prod(shape)
    return tuple(
        torch.linspace(
            0.25 + position,
            1.25 + position,
            count,
            dtype=torch.float32,
        )
        .reshape(shape)
        .detach()
        for position in range(3)
    )


def _capture_plan() -> tuple[Any, Any, Any]:
    diffusion = _fake_diffusion()
    adapter = runtime.BerniniUniPCTrajectoryRuntimeAdapterV1(
        diffusion,
        trajectory=trajectory.ActualTrajectoryCaptureV1(),
    )
    result = adapter.run_sample(num_inference_steps=40, flow_shift=5.0)
    capture = result.trajectory_artifact
    plan = trajectory.build_trajectory_intervention_v1(
        capture=capture,
        state_vjps=_vjp_rows(capture.shape),
    )
    return capture, plan, result


class BerniniUniPCRuntimeAdapterTests(unittest.TestCase):
    def assert_step_restored(self, diffusion: Any) -> None:
        self.assertNotIn("step", vars(diffusion.scheduler))
        self.assertIs(
            diffusion.scheduler.step.__func__,
            _FakeSchedulerCore.step,
        )

    def test_capture_uses_real_call_shape_exact40_and_restores(self) -> None:
        diffusion = _fake_diffusion()
        capture_state = trajectory.ActualTrajectoryCaptureV1()
        adapter = runtime.BerniniUniPCTrajectoryRuntimeAdapterV1(
            diffusion,
            trajectory=capture_state,
        )
        result = adapter.run_sample(num_inference_steps=40, flow_shift=5.0)

        self.assertTrue(diffusion.patch_was_live_during_sample)
        self.assert_step_restored(diffusion)
        self.assertIsInstance(
            result.trajectory_artifact,
            trajectory.CapturedActualTrajectoryV1,
        )
        self.assertEqual(len(diffusion.scheduler.raw_results), 40)
        self.assertTrue(
            all(
                returned is raw
                for returned, raw in zip(
                    diffusion.returned_containers,
                    diffusion.scheduler.raw_results,
                    strict=True,
                )
            )
        )
        receipt = result.receipt
        self.assertEqual(receipt["evidence_tier"], "ENGINEERING_ONLY")
        self.assertEqual(receipt["scheduler_calls_observed"], 40)
        self.assertEqual(receipt["original_scheduler_calls_observed"], 40)
        self.assertEqual(
            receipt["schedule_sha256"],
            trajectory.PINNED_SCHEDULE_SHA256,
        )
        self.assertEqual(
            receipt["scheduler_return_contract"][
                "tuple_objects_returned_by_identity_count"
            ],
            40,
        )
        self.assertTrue(receipt["patch_lifecycle"]["restored_before_receipt"])
        for key in (
            "semantic_success_assessed",
            "scientific_claim_authorized",
            "optimizer_authorized",
            "training_update_authorized",
            "parameter_update_performed",
            "gpu_experiment_authorized",
            "deployment_authorized",
        ):
            self.assertFalse(receipt[key])

        # A caller can mutate its copy without changing the adapter's seal.
        result.receipt["scheduler_config"]["solver_type"] = "tampered"
        self.assertEqual(adapter.receipt()["scheduler_config"]["solver_type"], "bh2")
        with self.assertRaisesRegex(runtime.TQMosaicBerniniRuntimeError, "one-shot"):
            adapter.run_sample(num_inference_steps=40, flow_shift=5.0)

    def test_zero_replay_returns_all_original_tuple_and_tensor_objects(self) -> None:
        _capture, plan, _capture_result = _capture_plan()
        diffusion = _fake_diffusion()
        adapter = runtime.BerniniUniPCTrajectoryRuntimeAdapterV1(
            diffusion,
            trajectory=plan.new_replay(sign=0),
        )
        result = adapter.run_sample(num_inference_steps=40, flow_shift=5.0)

        self.assert_step_restored(diffusion)
        self.assertTrue(
            all(
                returned is raw and returned[0] is raw[0]
                for returned, raw in zip(
                    diffusion.returned_containers,
                    diffusion.scheduler.raw_results,
                    strict=True,
                )
            )
        )
        replay = result.trajectory_artifact
        self.assertTrue(replay["zero_sign_all_scheduler_outputs_returned_by_identity"])
        contract = result.receipt["scheduler_return_contract"]
        self.assertEqual(contract["tuple_objects_returned_by_identity_count"], 40)
        self.assertEqual(contract["tensor_objects_returned_by_identity_count"], 40)
        self.assertTrue(
            contract["zero_sign_all_original_tuple_objects_returned_by_identity"]
        )
        self.assertTrue(
            contract["zero_sign_all_original_tensor_objects_returned_by_identity"]
        )

    def test_nonzero_replay_replaces_only_three_post_step_tuple_objects(self) -> None:
        _capture, plan, _capture_result = _capture_plan()
        diffusion = _fake_diffusion()
        adapter = runtime.BerniniUniPCTrajectoryRuntimeAdapterV1(
            diffusion,
            trajectory=plan.new_replay(sign=1),
        )
        result = adapter.run_sample(num_inference_steps=40, flow_shift=5.0)

        replaced = [
            index
            for index, (returned, raw) in enumerate(
                zip(
                    diffusion.returned_containers,
                    diffusion.scheduler.raw_results,
                    strict=True,
                )
            )
            if returned is not raw
        ]
        self.assertEqual(replaced, list(trajectory.INJECT_AFTER_STEP_INDICES))
        self.assertEqual(
            result.receipt["scheduler_return_contract"][
                "tuple_objects_returned_by_identity_count"
            ],
            37,
        )
        self.assertIsNone(
            result.receipt["scheduler_return_contract"][
                "zero_sign_all_original_tuple_objects_returned_by_identity"
            ]
        )
        self.assert_step_restored(diffusion)

    def test_wrong_full_schedule_fails_before_original_step_and_restores(self) -> None:
        diffusion = _fake_diffusion(
            scheduler_kwargs={"corrupt_schedule_index": 17}
        )
        adapter = runtime.BerniniUniPCTrajectoryRuntimeAdapterV1(
            diffusion,
            trajectory=trajectory.ActualTrajectoryCaptureV1(),
        )
        with self.assertRaisesRegex(
            runtime.TQMosaicBerniniRuntimeError,
            "full exact40 schedule differs",
        ):
            adapter.run_sample(num_inference_steps=40, flow_shift=5.0)
        self.assertEqual(diffusion.scheduler.raw_results, [])
        self.assert_step_restored(diffusion)
        with self.assertRaisesRegex(runtime.TQMosaicBerniniRuntimeError, "no receipt"):
            adapter.receipt()

    def test_config_mutation_during_sample_fails_closed_and_restores(self) -> None:
        diffusion = _fake_diffusion(
            scheduler_kwargs={"mutate_config_after": 0}
        )
        adapter = runtime.BerniniUniPCTrajectoryRuntimeAdapterV1(
            diffusion,
            trajectory=trajectory.ActualTrajectoryCaptureV1(),
        )
        with self.assertRaisesRegex(
            runtime.TQMosaicBerniniRuntimeError,
            "scheduler config solver_type differs",
        ):
            adapter.run_sample(num_inference_steps=40, flow_shift=5.0)
        self.assertEqual(len(diffusion.scheduler.raw_results), 1)
        self.assert_step_restored(diffusion)

    def test_exactly_40_calls_is_required_on_short_and_long_samples(self) -> None:
        short = _fake_diffusion(sample_step_count=39)
        short_adapter = runtime.BerniniUniPCTrajectoryRuntimeAdapterV1(
            short,
            trajectory=trajectory.ActualTrajectoryCaptureV1(),
        )
        with self.assertRaisesRegex(
            runtime.TQMosaicBerniniRuntimeError,
            "made 39 scheduler calls",
        ):
            short_adapter.run_sample(num_inference_steps=40, flow_shift=5.0)
        self.assert_step_restored(short)

        long = _fake_diffusion(sample_step_count=41)
        long_adapter = runtime.BerniniUniPCTrajectoryRuntimeAdapterV1(
            long,
            trajectory=trajectory.ActualTrajectoryCaptureV1(),
        )
        with self.assertRaisesRegex(
            runtime.TQMosaicBerniniRuntimeError,
            "more than 40 scheduler calls",
        ):
            long_adapter.run_sample(num_inference_steps=40, flow_shift=5.0)
        self.assertEqual(len(long.scheduler.raw_results), 40)
        self.assert_step_restored(long)

    def test_original_scheduler_and_sample_exceptions_both_restore(self) -> None:
        scheduler_failure = _fake_diffusion(
            scheduler_kwargs={"fail_at": 5}
        )
        scheduler_adapter = runtime.BerniniUniPCTrajectoryRuntimeAdapterV1(
            scheduler_failure,
            trajectory=trajectory.ActualTrajectoryCaptureV1(),
        )
        with self.assertRaisesRegex(RuntimeError, "original UniPC failure"):
            scheduler_adapter.run_sample(num_inference_steps=40, flow_shift=5.0)
        self.assert_step_restored(scheduler_failure)

        sample_failure = _fake_diffusion(raise_after_step=5)
        sample_adapter = runtime.BerniniUniPCTrajectoryRuntimeAdapterV1(
            sample_failure,
            trajectory=trajectory.ActualTrajectoryCaptureV1(),
        )
        with self.assertRaisesRegex(RuntimeError, "Bernini sample failure"):
            sample_adapter.run_sample(num_inference_steps=40, flow_shift=5.0)
        self.assert_step_restored(sample_failure)

    def test_return_dict_and_one_tuple_contracts_fail_closed(self) -> None:
        for call_form, message in (
            ("missing_return_dict", "explicit return_dict=False"),
            ("true_return_dict", "explicitly use return_dict=False"),
            ("float_timestep", "materialized int64 scalar"),
        ):
            with self.subTest(call_form=call_form):
                diffusion = _fake_diffusion(scheduler_call_form=call_form)
                adapter = runtime.BerniniUniPCTrajectoryRuntimeAdapterV1(
                    diffusion,
                    trajectory=trajectory.ActualTrajectoryCaptureV1(),
                )
                with self.assertRaisesRegex(
                    runtime.TQMosaicBerniniRuntimeError,
                    message,
                ):
                    adapter.run_sample(num_inference_steps=40, flow_shift=5.0)
                self.assert_step_restored(diffusion)

        for result_kind in ("list", "pair"):
            with self.subTest(result_kind=result_kind):
                diffusion = _fake_diffusion(
                    scheduler_kwargs={"result_kind": result_kind}
                )
                adapter = runtime.BerniniUniPCTrajectoryRuntimeAdapterV1(
                    diffusion,
                    trajectory=trajectory.ActualTrajectoryCaptureV1(),
                )
                with self.assertRaisesRegex(
                    runtime.TQMosaicBerniniRuntimeError,
                    "one built-in tuple",
                ):
                    adapter.run_sample(num_inference_steps=40, flow_shift=5.0)
                self.assert_step_restored(diffusion)

    def test_static_class_config_patch_and_sampling_inputs_fail_before_run(self) -> None:
        wrong_config = _fake_diffusion()
        wrong_config.scheduler.config["flow_shift"] = 3.0
        with self.assertRaisesRegex(
            runtime.TQMosaicBerniniRuntimeError,
            "config flow_shift differs",
        ):
            runtime.BerniniUniPCTrajectoryRuntimeAdapterV1(
                wrong_config,
                trajectory=trajectory.ActualTrajectoryCaptureV1(),
            )

        missing_none_field = _fake_diffusion()
        del missing_none_field.scheduler.config["sigma_min"]
        with self.assertRaisesRegex(
            runtime.TQMosaicBerniniRuntimeError,
            "missing required field sigma_min",
        ):
            runtime.BerniniUniPCTrajectoryRuntimeAdapterV1(
                missing_none_field,
                trajectory=trajectory.ActualTrajectoryCaptureV1(),
            )

        wrong_diffusion_class = _FakeDiffusionCore(UniPCMultistepScheduler())
        with self.assertRaisesRegex(
            runtime.TQMosaicBerniniRuntimeError,
            "diffusion class is not pinned",
        ):
            runtime.BerniniUniPCTrajectoryRuntimeAdapterV1(
                wrong_diffusion_class,
                trajectory=trajectory.ActualTrajectoryCaptureV1(),
            )

        stacked = _fake_diffusion()
        stacked.scheduler.step = lambda *args, **kwargs: None
        with self.assertRaisesRegex(
            runtime.TQMosaicBerniniRuntimeError,
            "refusing to stack",
        ):
            runtime.BerniniUniPCTrajectoryRuntimeAdapterV1(
                stacked,
                trajectory=trajectory.ActualTrajectoryCaptureV1(),
            )

        for kwargs, message in (
            ({"num_inference_steps": 41}, "requires num_inference_steps=40"),
            ({"flow_shift": 3.0}, "requires flow_shift=5.0"),
        ):
            with self.subTest(kwargs=kwargs):
                diffusion = _fake_diffusion()
                adapter = runtime.BerniniUniPCTrajectoryRuntimeAdapterV1(
                    diffusion,
                    trajectory=trajectory.ActualTrajectoryCaptureV1(),
                )
                with self.assertRaisesRegex(runtime.TQMosaicBerniniRuntimeError, message):
                    adapter.run_sample(**kwargs)
                self.assertNotIn("step", vars(diffusion.scheduler))

    @unittest.skipUnless(
        os.environ.get("BERNINI_TQ_REAL_SCHEDULER_TEST") == "1"
        and bool(os.environ.get("BERNINI_TQ_CHECKPOINT_ROOT")),
        "set BERNINI_TQ_REAL_SCHEDULER_TEST=1 and BERNINI_TQ_CHECKPOINT_ROOT",
    )
    def test_real_checkpoint_unipc_capture_and_zero_replay_on_cpu(self) -> None:
        from diffusers import UniPCMultistepScheduler as RealUniPC

        checkpoint = os.environ["BERNINI_TQ_CHECKPOINT_ROOT"]

        class _RealSchedulerDiffusionCore:
            def __init__(self, scheduler: Any) -> None:
                self.scheduler = scheduler
                self.use_unipc = True
                self.transformer_2 = None
                self.returned_containers: list[Any] = []

            def sample(
                self,
                prompt_embeds: Any = None,
                num_inference_steps: int = 40,
                guidance_mode: str = "v2v_apg",
                flow_shift: float = 5.0,
            ) -> torch.Tensor:
                del prompt_embeds, guidance_mode, flow_shift
                self.scheduler.set_timesteps(num_inference_steps)
                state = _initial_state()
                for timestep in self.scheduler.timesteps:
                    result = self.scheduler.step(
                        torch.zeros_like(state),
                        timestep,
                        state,
                        return_dict=False,
                    )
                    self.returned_containers.append(result)
                    state = result[0]
                return state

        RealGEN = type(
            "GEN_Wanx22",
            (_RealSchedulerDiffusionCore,),
            {"__module__": runtime.PINNED_DIFFUSION_CLASS[0]},
        )

        capture_diffusion = RealGEN(
            RealUniPC.from_pretrained(
                checkpoint,
                subfolder="scheduler",
                flow_shift=5.0,
            )
        )
        capture_adapter = runtime.BerniniUniPCTrajectoryRuntimeAdapterV1(
            capture_diffusion,
            trajectory=trajectory.ActualTrajectoryCaptureV1(),
        )
        capture_result = capture_adapter.run_sample(
            num_inference_steps=40,
            flow_shift=5.0,
        )
        capture = capture_result.trajectory_artifact
        plan = trajectory.build_trajectory_intervention_v1(
            capture=capture,
            state_vjps=_vjp_rows(capture.shape),
        )

        zero_diffusion = RealGEN(
            RealUniPC.from_pretrained(
                checkpoint,
                subfolder="scheduler",
                flow_shift=5.0,
            )
        )
        zero_adapter = runtime.BerniniUniPCTrajectoryRuntimeAdapterV1(
            zero_diffusion,
            trajectory=plan.new_replay(sign=0),
        )
        zero_result = zero_adapter.run_sample(
            num_inference_steps=40,
            flow_shift=5.0,
        )
        self.assertTrue(
            zero_result.receipt["scheduler_return_contract"][
                "zero_sign_all_original_tuple_objects_returned_by_identity"
            ]
        )
        self.assertNotIn("step", vars(zero_diffusion.scheduler))
        self.assertEqual(
            type(zero_diffusion.scheduler).__module__,
            runtime.PINNED_SCHEDULER_CLASS[0],
        )


if __name__ == "__main__":
    unittest.main()
