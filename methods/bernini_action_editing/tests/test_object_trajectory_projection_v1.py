from __future__ import annotations

from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import object_trajectory_projection_v1 as projection  # noqa: E402


try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class ContractTests(unittest.TestCase):
    def test_scope_is_explicitly_oracle_tensor_core_not_runner(self) -> None:
        contract = projection.tensor_core_contract()
        self.assertEqual(contract["scope"], "zero_training_oracle_tensor_core")
        self.assertFalse(contract["production_runner_integration"])
        self.assertFalse(contract["renderer_abi_integration"])
        self.assertEqual(contract["integrator"], "original_unipc_scheduler_step")
        self.assertEqual(contract["weight_policy"], "strict_binary_0_or_1_v1")
        self.assertFalse(contract["fractional_weights_supported"])
        self.assertEqual(
            contract["inactive_step_policy"],
            "exact_native_delegate_no_argument_clone_or_replacement",
        )


@unittest.skipIf(torch is None, "torch is unavailable")
class UniPCMultistepScheduler:
    """Small stateful solver with the exact public contract pinned by v1."""

    def __init__(self, *, sigmas=(1.0, 0.75, 0.5, 0.0), fail=False):
        self.config = {
            "_class_name": "UniPCMultistepScheduler",
            "prediction_type": "flow_prediction",
            "use_flow_sigmas": True,
            "predict_x0": True,
            "final_sigmas_type": "zero",
            "flow_shift": 5.0,
        }
        self.sigmas = torch.tensor(sigmas, dtype=torch.float32)
        self.timesteps = torch.tensor(
            [1000.0, 750.0, 500.0][: len(sigmas) - 1], dtype=torch.float32
        )
        self.step_index = None
        self.calls = []
        self.fail = fail

    def step(self, model_output, timestep, sample, return_dict=True):
        if self.fail:
            raise LookupError("synthetic native UniPC failure")
        index = 0 if self.step_index is None else int(self.step_index)
        self.calls.append(
            {
                "model_output": model_output,
                "timestep": timestep,
                "sample": sample,
                "return_dict": return_dict,
                "args_ids": (
                    id(model_output),
                    id(timestep),
                    id(sample),
                    id(return_dict),
                ),
            }
        )
        # The numerical formula is intentionally irrelevant to the projection;
        # it merely makes exact native/delegated comparisons observable.
        previous = (sample.float() - (0.07 + 0.01 * index) * model_output.float()).to(
            dtype=sample.dtype
        )
        self.step_index = index + 1
        return (previous.contiguous(),)


@unittest.skipIf(torch is None, "torch is unavailable")
class TensorCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260821)

    def _tensors(self, *, tokens=6):
        clean = torch.randn(1, tokens, 64, dtype=torch.float32).contiguous()
        noise = torch.randn_like(clean).contiguous()
        weights = torch.zeros(1, tokens, 1, dtype=torch.float32)
        return clean, noise, weights

    def _run_native(self, scheduler, noise, outputs):
        sample = noise
        results = []
        for index, model_output in enumerate(outputs):
            result = scheduler.step(
                model_output,
                scheduler.timesteps[index],
                sample,
                return_dict=False,
            )
            results.append(result[0])
            sample = result[0]
        return results

    def test_all_zero_mask_is_exact_no_install_no_delegate(self) -> None:
        clean, noise, weights = self._tensors()
        outputs = [torch.randn_like(clean) for _ in range(3)]
        baseline = UniPCMultistepScheduler()
        expected = self._run_native(baseline, noise, outputs)

        candidate = UniPCMultistepScheduler()
        original_function = candidate.step.__func__
        with projection.project_single_object_trajectory_unipc_steps(
            candidate,
            clean_packed=clean,
            initial_noise=noise,
            projection_weights=weights,
            source_token_count=6,
            target_token_count=6,
            expected_steps=3,
        ) as trace:
            self.assertNotIn("step", vars(candidate))
            actual = self._run_native(candidate, noise, outputs)
            self.assertEqual(trace.records, [])
        self.assertIs(candidate.step.__func__, original_function)
        self.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected):
            self.assertTrue(torch.equal(left, right))
        receipt = trace.as_dict()
        self.assertFalse(receipt["globally_enabled"])
        self.assertFalse(receipt["wrapper_installed"])
        self.assertTrue(receipt["wrapper_restored"])
        self.assertEqual(receipt["step_count"], 0)
        self.assertEqual(receipt["dimensions"]["source_reference"], [1, 6, 64])
        self.assertEqual(receipt["dimensions"]["target_sampler"], [1, 6, 64])

    def test_origin_clear_target_tube_conserves_one_source_object(self) -> None:
        tokens = 8
        noise = torch.randn(1, tokens, 64, dtype=torch.float32).contiguous()
        bone_detail = torch.linspace(0.25, 1.0, 64, dtype=torch.float32)

        bone_clean = torch.zeros_like(noise)
        origin_token = 1
        target_tube_token = 6
        bone_clean[0, origin_token].zero_()
        bone_clean[0, target_tube_token].copy_(bone_detail)
        bone_mask = torch.zeros(1, tokens, 1, dtype=torch.bool)
        bone_mask[0, origin_token, 0] = True
        bone_mask[0, target_tube_token, 0] = True

        dog_clean = torch.zeros_like(noise)
        dog_core_token = 3
        dog_identity = torch.linspace(-1.0, -0.25, 64, dtype=torch.float32)
        dog_clean[0, dog_core_token].copy_(dog_identity)
        dog_mask = torch.zeros(1, tokens, 1, dtype=torch.bool)
        dog_mask[0, dog_core_token, 0] = True

        rows = (
            projection.ProjectionRow("bone_all_sigma", bone_clean, bone_mask),
            projection.ProjectionRow(
                "dog_core_low_mid",
                dog_clean,
                dog_mask,
                active_next_sigma_max=0.5,
            ),
        )
        scheduler = UniPCMultistepScheduler()
        outputs = [torch.randn_like(noise) for _ in range(3)]
        sample = noise
        with projection.project_object_trajectory_unipc_steps(
            scheduler,
            rows=rows,
            initial_noise=noise,
            source_token_count=tokens,
            target_token_count=tokens,
            expected_steps=3,
        ) as trace:
            for index, model_output in enumerate(outputs):
                sample = scheduler.step(
                    model_output,
                    scheduler.timesteps[index],
                    sample,
                    return_dict=False,
                )[0]

        # Terminal +0 projection clears the old support and puts the exact same
        # source detail vector at one target-tube support.
        self.assertTrue(torch.equal(sample[0, origin_token], torch.zeros(64)))
        self.assertTrue(torch.equal(sample[0, target_tube_token], bone_detail))
        matches = [
            token
            for token in range(tokens)
            if torch.equal(sample[0, token], bone_detail)
        ]
        self.assertEqual(matches, [target_tube_token])
        self.assertTrue(torch.equal(sample[0, dog_core_token], dog_identity))

        self.assertEqual(trace.records[0].active_rows, ("bone_all_sigma",))
        self.assertEqual(
            trace.records[1].active_rows,
            ("bone_all_sigma", "dog_core_low_mid"),
        )
        self.assertEqual(
            trace.records[2].active_rows,
            ("bone_all_sigma", "dog_core_low_mid"),
        )
        self.assertTrue(all(row.selected_velocity_exact for row in trace.records))
        self.assertTrue(all(row.selected_post_step_exact for row in trace.records))
        self.assertEqual(scheduler.step_index, 3)
        self.assertEqual(len(scheduler.calls), 3)
        self.assertEqual(trace.as_dict()["step_count"], 3)

    def test_inactive_step_gate_is_exact_native_delegate_without_clone(self) -> None:
        clean, noise, weights = self._tensors()
        weights[:, 2] = 1
        outputs = [torch.randn_like(clean) for _ in range(3)]
        scheduler = UniPCMultistepScheduler()
        with projection.project_single_object_trajectory_unipc_steps(
            scheduler,
            clean_packed=clean,
            initial_noise=noise,
            projection_weights=weights,
            source_token_count=6,
            target_token_count=6,
            expected_steps=3,
            step_gates=(0, 1, 1),
        ) as trace:
            first_timestep = scheduler.timesteps[0]
            first_result = scheduler.step(
                outputs[0], first_timestep, noise, return_dict=False
            )
            self.assertIs(scheduler.calls[0]["model_output"], outputs[0])
            self.assertIs(scheduler.calls[0]["timestep"], first_timestep)
            self.assertIs(scheduler.calls[0]["sample"], noise)
            sample = first_result[0]
            for index in (1, 2):
                sample = scheduler.step(
                    outputs[index],
                    scheduler.timesteps[index],
                    sample,
                    return_dict=False,
                )[0]
        first = trace.records[0]
        self.assertFalse(first.projection_applied)
        self.assertTrue(first.exact_native_delegate_no_argument_clone)
        self.assertFalse(first.initial_noise_snapshot_created_this_step)
        self.assertEqual(first.active_rows, ())
        self.assertTrue(trace.records[1].projection_applied)

    def test_lazy_noise_capture_uses_first_native_sample_without_rng(self) -> None:
        clean, noise, weights = self._tensors()
        weights[:, 0] = 1
        scheduler = UniPCMultistepScheduler()
        rng_before = torch.random.get_rng_state().clone()
        outputs = [torch.zeros_like(clean) for _ in range(3)]
        sample = noise
        with projection.project_single_object_trajectory_unipc_steps(
            scheduler,
            clean_packed=clean,
            projection_weights=weights,
            source_token_count=6,
            target_token_count=6,
            expected_steps=3,
        ) as trace:
            for index, model_output in enumerate(outputs):
                sample = scheduler.step(
                    model_output,
                    scheduler.timesteps[index],
                    sample,
                    return_dict=False,
                )[0]
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        self.assertTrue(trace.initial_noise_verified)
        self.assertTrue(trace.initial_noise_captured_from_first_native_sample)
        receipt = trace.as_dict()
        self.assertEqual(
            receipt["initial_noise_registration"],
            "lazy_capture_first_native_sample",
        )
        self.assertTrue(receipt["initial_noise_captured_from_first_native_sample"])
        self.assertTrue(torch.equal(sample[:, 0], clean[:, 0]))

    def test_lazy_capture_precedes_an_inactive_exact_delegate(self) -> None:
        clean, noise, weights = self._tensors()
        weights[:, 4] = 1
        scheduler = UniPCMultistepScheduler()
        outputs = [torch.randn_like(clean) for _ in range(3)]
        with projection.project_single_object_trajectory_unipc_steps(
            scheduler,
            clean_packed=clean,
            initial_noise=None,
            projection_weights=weights,
            source_token_count=6,
            target_token_count=6,
            expected_steps=3,
            step_gates=(0, 1, 1),
        ) as trace:
            first = scheduler.step(
                outputs[0], scheduler.timesteps[0], noise, return_dict=False
            )
            self.assertIs(scheduler.calls[0]["model_output"], outputs[0])
            self.assertIs(scheduler.calls[0]["sample"], noise)
            sample = first[0]
            for index in (1, 2):
                sample = scheduler.step(
                    outputs[index],
                    scheduler.timesteps[index],
                    sample,
                    return_dict=False,
                )[0]
        self.assertTrue(trace.initial_noise_captured_from_first_native_sample)
        self.assertTrue(trace.records[0].exact_native_delegate_no_argument_clone)
        self.assertTrue(trace.records[0].initial_noise_snapshot_created_this_step)
        self.assertTrue(torch.equal(sample[:, 4], clean[:, 4]))

    def test_per_channel_mask_changes_only_selected_channel(self) -> None:
        clean, noise, _ = self._tensors(tokens=3)
        weights = torch.zeros_like(clean)
        weights[0, 1, 7] = 1
        scheduler = UniPCMultistepScheduler(sigmas=(1.0, 0.0))
        model_output = torch.randn_like(clean)
        native_previous = (
            noise.float() - 0.07 * model_output.float()
        ).contiguous()
        with projection.project_single_object_trajectory_unipc_steps(
            scheduler,
            clean_packed=clean,
            initial_noise=noise,
            projection_weights=weights,
            source_token_count=3,
            target_token_count=3,
            expected_steps=1,
        ):
            result = scheduler.step(
                model_output,
                scheduler.timesteps[0],
                noise,
                return_dict=False,
            )[0]
        selected = torch.zeros_like(weights, dtype=torch.bool)
        selected[0, 1, 7] = True
        self.assertTrue(torch.equal(result[selected], clean[selected]))
        self.assertTrue(torch.equal(result[~selected], native_previous[~selected]))

    def test_float64_terminal_authority_is_not_rounded_through_float32(self) -> None:
        clean = torch.zeros(1, 2, 64, dtype=torch.float64)
        clean[0, 0, 0] = 1.0 + 2.0 ** -40
        noise = torch.randn_like(clean)
        weights = torch.zeros(1, 2, 1, dtype=torch.bool)
        weights[:, 0] = True
        scheduler = UniPCMultistepScheduler(sigmas=(1.0, 0.0))
        with projection.project_single_object_trajectory_unipc_steps(
            scheduler,
            clean_packed=clean,
            initial_noise=noise,
            projection_weights=weights,
            source_token_count=2,
            target_token_count=2,
            expected_steps=1,
        ):
            result = scheduler.step(
                torch.zeros_like(clean),
                scheduler.timesteps[0],
                noise,
                return_dict=False,
            )[0]
        self.assertTrue(torch.equal(result[:, 0], clean[:, 0]))
        self.assertNotEqual(float(result[0, 0, 0]), 1.0)

    def test_overlap_requires_identical_clean_values(self) -> None:
        clean, noise, weights = self._tensors()
        weights[:, 1] = 1
        other = clean.clone()
        other[:, 1] += 1
        scheduler = UniPCMultistepScheduler()
        with self.assertRaisesRegex(
            projection.ObjectTrajectoryProjectionError,
            "overlap with different clean authority",
        ):
            with projection.project_object_trajectory_unipc_steps(
                scheduler,
                rows=(
                    projection.ProjectionRow("left", clean, weights),
                    projection.ProjectionRow("right", other, weights.clone()),
                ),
                initial_noise=noise,
                source_token_count=6,
                target_token_count=6,
                expected_steps=3,
            ):
                self.fail("conflicting overlap must fail before installation")
        self.assertNotIn("step", vars(scheduler))

    def test_fractional_weights_and_input_aliases_fail_closed(self) -> None:
        clean, noise, weights = self._tensors()
        weights[:, 0] = 0.5
        scheduler = UniPCMultistepScheduler()
        with self.assertRaisesRegex(
            projection.ObjectTrajectoryProjectionError, "fractional weights"
        ):
            with projection.project_single_object_trajectory_unipc_steps(
                scheduler,
                clean_packed=clean,
                initial_noise=noise,
                projection_weights=weights,
                source_token_count=6,
                target_token_count=6,
                expected_steps=3,
            ):
                self.fail("fractional v1 weights must fail before installation")

        alias_noise = clean.view_as(clean)
        binary = torch.zeros(1, 6, 1)
        with self.assertRaisesRegex(
            projection.ObjectTrajectoryProjectionError, "storage alias"
        ):
            with projection.project_single_object_trajectory_unipc_steps(
                scheduler,
                clean_packed=clean,
                initial_noise=alias_noise,
                projection_weights=binary,
                source_token_count=6,
                target_token_count=6,
                expected_steps=3,
            ):
                self.fail("aliased authority/noise must fail before installation")
        self.assertEqual(scheduler.calls, [])
        self.assertNotIn("step", vars(scheduler))

    def test_shape_dtype_finite_and_runtime_tensor_contracts_fail_closed(self) -> None:
        clean, noise, weights = self._tensors()
        weights[:, 0] = 1

        cases = []
        nonfinite = clean.clone()
        nonfinite[0, 0, 0] = float("nan")
        cases.append(("finite", nonfinite, noise, weights, 6, 6))
        cases.append(("dtypes differ", clean, noise.double(), weights, 6, 6))
        cases.append(("shape", clean, noise, torch.ones(1, 6, 2), 6, 6))
        cases.append(("equal source and target", clean, noise, weights, 5, 6))
        for message, bad_clean, bad_noise, bad_weights, source_tokens, target_tokens in cases:
            with self.subTest(message=message):
                scheduler = UniPCMultistepScheduler()
                with self.assertRaisesRegex(
                    projection.ObjectTrajectoryProjectionError, message
                ):
                    with projection.project_single_object_trajectory_unipc_steps(
                        scheduler,
                        clean_packed=bad_clean,
                        initial_noise=bad_noise,
                        projection_weights=bad_weights,
                        source_token_count=source_tokens,
                        target_token_count=target_tokens,
                        expected_steps=3,
                    ):
                        self.fail("invalid tensor contract must fail before install")
                self.assertEqual(scheduler.calls, [])
                self.assertNotIn("step", vars(scheduler))

        scheduler = UniPCMultistepScheduler()
        original_function = scheduler.step.__func__
        with self.assertRaisesRegex(
            projection.ObjectTrajectoryProjectionError, "dtypes differ"
        ):
            with projection.project_single_object_trajectory_unipc_steps(
                scheduler,
                clean_packed=clean,
                initial_noise=noise,
                projection_weights=weights,
                source_token_count=6,
                target_token_count=6,
                expected_steps=3,
            ):
                scheduler.step(
                    torch.randn_like(clean).double().contiguous(),
                    scheduler.timesteps[0],
                    noise,
                    return_dict=False,
                )
        self.assertEqual(scheduler.calls, [])
        self.assertNotIn("step", vars(scheduler))
        self.assertIs(scheduler.step.__func__, original_function)

    def test_explicit_noise_mismatch_fails_before_native_step_and_restores(self) -> None:
        clean, noise, weights = self._tensors()
        weights[:, 0] = 1
        scheduler = UniPCMultistepScheduler()
        original_function = scheduler.step.__func__
        with self.assertRaisesRegex(
            projection.ObjectTrajectoryProjectionError,
            "differs from registered initial_noise",
        ):
            with projection.project_single_object_trajectory_unipc_steps(
                scheduler,
                clean_packed=clean,
                initial_noise=noise,
                projection_weights=weights,
                source_token_count=6,
                target_token_count=6,
                expected_steps=3,
            ):
                scheduler.step(
                    torch.randn_like(clean),
                    scheduler.timesteps[0],
                    noise + 1,
                    return_dict=False,
                )
        self.assertEqual(scheduler.calls, [])
        self.assertNotIn("step", vars(scheduler))
        self.assertIs(scheduler.step.__func__, original_function)

    def test_native_exception_restores_exact_instance_step(self) -> None:
        clean, noise, weights = self._tensors()
        weights[:, 0] = 1
        scheduler = UniPCMultistepScheduler()

        def failing_step(*args, **kwargs):
            del args, kwargs
            raise LookupError("synthetic solver failure")

        scheduler.step = failing_step
        with self.assertRaisesRegex(LookupError, "synthetic solver failure"):
            with projection.project_single_object_trajectory_unipc_steps(
                scheduler,
                clean_packed=clean,
                initial_noise=noise,
                projection_weights=weights,
                source_token_count=6,
                target_token_count=6,
                expected_steps=3,
            ):
                scheduler.step(
                    torch.randn_like(clean),
                    scheduler.timesteps[0],
                    noise,
                    return_dict=False,
                )
        self.assertIs(scheduler.step, failing_step)

    def test_incomplete_schedule_fails_finalization_and_restores(self) -> None:
        clean, noise, weights = self._tensors()
        weights[:, 0] = 1
        scheduler = UniPCMultistepScheduler()
        original_function = scheduler.step.__func__
        with self.assertRaisesRegex(
            projection.ObjectTrajectoryProjectionError,
            "full terminal-zero schedule",
        ):
            with projection.project_single_object_trajectory_unipc_steps(
                scheduler,
                clean_packed=clean,
                initial_noise=noise,
                projection_weights=weights,
                source_token_count=6,
                target_token_count=6,
                expected_steps=3,
            ):
                scheduler.step(
                    torch.randn_like(clean),
                    scheduler.timesteps[0],
                    noise,
                    return_dict=False,
                )
        self.assertNotIn("step", vars(scheduler))
        self.assertIs(scheduler.step.__func__, original_function)

    def test_exact_scheduler_config_is_required_before_install(self) -> None:
        clean, noise, weights = self._tensors()
        weights[:, 0] = 1
        scheduler = UniPCMultistepScheduler()
        scheduler.config["prediction_type"] = "epsilon"
        with self.assertRaisesRegex(
            projection.ObjectTrajectoryProjectionError, "prediction_type differs"
        ):
            with projection.project_single_object_trajectory_unipc_steps(
                scheduler,
                clean_packed=clean,
                initial_noise=noise,
                projection_weights=weights,
                source_token_count=6,
                target_token_count=6,
                expected_steps=3,
            ):
                self.fail("wrong scheduler config must fail before installation")


if __name__ == "__main__":
    unittest.main()
