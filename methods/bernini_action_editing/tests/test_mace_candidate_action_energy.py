from __future__ import annotations

import inspect
import math
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch  # noqa: E402
except ImportError:
    torch = None

if torch is not None:
    import mace_candidate_action_energy as mace  # noqa: E402
else:  # pragma: no cover - dependency-light environments
    mace = None


def _prompt_registry():
    return {
        branch: f"registered prompt for {branch}"
        for branch in mace.BRANCH_ORDER
    }


if torch is not None:

    class RecordingOffsetDenoiser(torch.nn.Module):
        def __init__(self, offsets, *, output_dtype=torch.float32):
            super().__init__()
            self.offsets = dict(offsets)
            self.output_dtype = output_dtype
            self.calls = []

        def forward(self, x_sigma, sigma, prompt):
            self.calls.append(
                {
                    "x_id": id(x_sigma),
                    "x_ptr": x_sigma.data_ptr(),
                    "x": x_sigma.detach().clone(),
                    "sigma_id": id(sigma),
                    "sigma_ptr": sigma.data_ptr(),
                    "sigma": sigma.detach().clone(),
                    "prompt": prompt,
                    "grad_enabled": torch.is_grad_enabled(),
                    "argument_count": 3,
                }
            )
            offset = self.offsets[prompt]
            # The fixtures use epsilon-clean == 4 everywhere.
            return torch.full_like(x_sigma, 4.0 + offset).to(self.output_dtype)


@unittest.skipIf(torch is None, "torch is unavailable")
class CandidateOwnCoordinateTests(unittest.TestCase):
    def _fixture(self, *, batch_size=2, output_dtype=None):
        prompts = _prompt_registry()
        offsets = {prompt: 2.0 for prompt in prompts.values()}
        offsets[prompts["action"]] = 1.0
        offsets[prompts["incomplete"]] = 0.5
        if output_dtype is None:
            output_dtype = torch.float32
        denoiser = RecordingOffsetDenoiser(
            offsets, output_dtype=output_dtype
        ).eval()
        clean = torch.tensor([[1.0, 3.0]], dtype=torch.float32).repeat(
            batch_size, 1
        )
        epsilon = clean + 4.0
        sigma = torch.tensor(0.25, dtype=torch.float32)
        return prompts, denoiser, clean, epsilon, sigma

    def test_constructs_exact_state_target_and_shared_prompt_queries(self) -> None:
        prompts, denoiser, clean, epsilon, sigma = self._fixture()
        result = mace.evaluate_candidate_action_energy(
            clean, epsilon, sigma, prompts, denoiser
        )

        expected_x = 0.75 * clean + 0.25 * epsilon
        self.assertTrue(torch.equal(result.x_sigma, expected_x))
        self.assertTrue(torch.equal(result.velocity_target, epsilon - clean))
        self.assertEqual(len(denoiser.calls), len(mace.BRANCH_ORDER))
        self.assertEqual(
            [row["prompt"] for row in denoiser.calls],
            [prompts[branch] for branch in mace.BRANCH_ORDER],
        )
        self.assertEqual(len({row["x_id"] for row in denoiser.calls}), 1)
        self.assertEqual(len({row["x_ptr"] for row in denoiser.calls}), 1)
        self.assertEqual(len({row["sigma_id"] for row in denoiser.calls}), 1)
        self.assertEqual(len({row["sigma_ptr"] for row in denoiser.calls}), 1)
        self.assertTrue(all(torch.equal(row["x"], expected_x) for row in denoiser.calls))
        self.assertTrue(
            all(
                torch.equal(row["sigma"], torch.full((2,), 0.25))
                for row in denoiser.calls
            )
        )
        self.assertTrue(all(not row["grad_enabled"] for row in denoiser.calls))
        self.assertTrue(all(row["argument_count"] == 3 for row in denoiser.calls))

    def test_reward_is_the_hardest_negative_log_energy_ratio(self) -> None:
        prompts, denoiser, clean, epsilon, sigma = self._fixture()
        result = mace.evaluate_candidate_action_energy(
            clean, epsilon, sigma, prompts, denoiser
        )

        self.assertEqual(tuple(result.branch_energies.shape), (10, 2))
        self.assertTrue(
            torch.equal(result.branch_energies[0], torch.ones(2))
        )
        incomplete_index = mace.HARD_NEGATIVE_BRANCHES.index("incomplete")
        self.assertTrue(
            torch.equal(
                result.branch_energies[1 + incomplete_index],
                torch.full((2,), 0.25),
            )
        )
        expected_reward = math.log(
            (0.25 + mace.DEFAULT_ENERGY_EPSILON)
            / (1.0 + mace.DEFAULT_ENERGY_EPSILON)
        )
        self.assertTrue(
            torch.allclose(
                result.reward,
                torch.full((2,), expected_reward),
                atol=1.0e-6,
            )
        )
        self.assertTrue(
            torch.equal(
                result.hardest_negative_index,
                torch.full((2,), incomplete_index, dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                result.reward,
                result.negative_log_energy_ratios.min(dim=0).values,
            )
        )
        action_fp64 = result.branch_energies[:1].to(torch.float64)
        expected_stable = torch.log1p(
            (
                result.branch_energies[1:].to(torch.float64)
                - action_fp64
            )
            / (action_fp64 + mace.DEFAULT_ENERGY_EPSILON)
        ).to(torch.float32)
        self.assertTrue(
            torch.equal(result.negative_log_energy_ratios, expected_stable)
        )

    def test_bfloat_prediction_still_uses_target_only_fp32_mse(self) -> None:
        prompts, denoiser, clean, epsilon, sigma = self._fixture(
            output_dtype=torch.bfloat16
        )
        result = mace.evaluate_candidate_action_energy(
            clean, epsilon, sigma, prompts, denoiser
        )
        self.assertEqual(result.branch_energies.dtype, torch.float32)
        self.assertEqual(result.negative_log_energy_ratios.dtype, torch.float32)
        self.assertEqual(result.reward.dtype, torch.float32)
        self.assertEqual(tuple(result.branch_energies.shape), (10, 2))


@unittest.skipIf(torch is None, "torch is unavailable")
class PromptAndInputClosureTests(unittest.TestCase):
    def test_negative_registry_is_exact_and_closed(self) -> None:
        expected_negatives = (
            "noop",
            "incomplete",
            "reverse",
            "shuffle",
            "wrong_actor",
            "wrong_object",
            "camera_only",
            "appearance_only",
            "generic_wrong_motion",
        )
        self.assertEqual(mace.HARD_NEGATIVE_BRANCHES, expected_negatives)
        self.assertEqual(
            mace.BRANCH_ORDER, ("action", *expected_negatives)
        )

        prompts = _prompt_registry()
        missing = dict(prompts)
        missing.pop("reverse")
        with self.assertRaisesRegex(
            mace.MACECandidateActionEnergyError, "missing=.*reverse"
        ):
            mace.validate_prompt_closure(missing)

        extra = dict(prompts)
        extra["easy_negative"] = "an unregistered easy negative"
        with self.assertRaisesRegex(
            mace.MACECandidateActionEnergyError, "extra=.*easy_negative"
        ):
            mace.validate_prompt_closure(extra)

    def test_prompt_alias_empty_and_non_string_fail_closed(self) -> None:
        prompts = _prompt_registry()
        prompts["reverse"] = prompts["action"]
        with self.assertRaisesRegex(
            mace.MACECandidateActionEnergyError, "aliases branch action"
        ):
            mace.validate_prompt_closure(prompts)

        prompts = _prompt_registry()
        prompts["noop"] = " "
        with self.assertRaisesRegex(
            mace.MACECandidateActionEnergyError, "canonical non-empty"
        ):
            mace.validate_prompt_closure(prompts)

        prompts = _prompt_registry()
        prompts["noop"] = 7
        with self.assertRaisesRegex(
            mace.MACECandidateActionEnergyError, "must be a string"
        ):
            mace.validate_prompt_closure(prompts)

    def test_public_evaluator_has_no_privileged_external_input_slot(self) -> None:
        parameters = set(
            inspect.signature(mace.evaluate_candidate_action_energy).parameters
        )
        self.assertTrue(parameters.isdisjoint(mace.FORBIDDEN_EXTERNAL_INPUT_NAMES))
        prompts = _prompt_registry()
        denoiser = RecordingOffsetDenoiser(
            {prompt: 1.0 for prompt in prompts.values()}
        ).eval()
        clean = torch.zeros(1, 2, dtype=torch.float32)
        with self.assertRaises(TypeError):
            mace.evaluate_candidate_action_energy(
                clean,
                clean.clone(),
                torch.tensor(0.5, dtype=torch.float32),
                prompts,
                denoiser,
                source_video=clean,
            )

    def test_shape_dtype_sigma_and_target_only_output_fail_closed(self) -> None:
        prompts = _prompt_registry()
        offsets = {prompt: 1.0 for prompt in prompts.values()}
        denoiser = RecordingOffsetDenoiser(offsets).eval()
        clean = torch.zeros(1, 2, dtype=torch.float32)
        epsilon = torch.ones_like(clean)
        sigma = torch.tensor(0.5, dtype=torch.float32)

        with self.assertRaisesRegex(
            mace.MACECandidateActionEnergyError, "epsilon shape"
        ):
            mace.evaluate_candidate_action_energy(
                clean, torch.ones(1, 3), sigma, prompts, denoiser
            )
        with self.assertRaisesRegex(
            mace.MACECandidateActionEnergyError, "clean_candidate must have dtype"
        ):
            mace.evaluate_candidate_action_energy(
                clean.bfloat16(), epsilon.bfloat16(), sigma, prompts, denoiser
            )
        with self.assertRaisesRegex(
            mace.MACECandidateActionEnergyError, "sigma must have dtype"
        ):
            mace.evaluate_candidate_action_energy(
                clean, epsilon, sigma.double(), prompts, denoiser
            )
        with self.assertRaisesRegex(
            mace.MACECandidateActionEnergyError, r"sigma must remain in \[0, 1\]"
        ):
            mace.evaluate_candidate_action_energy(
                clean,
                epsilon,
                torch.tensor(1.1, dtype=torch.float32),
                prompts,
                denoiser,
            )

        class PrefixOutput(torch.nn.Module):
            def forward(self, x_sigma, sigma, prompt):
                return torch.cat((x_sigma, x_sigma), dim=1)

        with self.assertRaisesRegex(
            mace.MACECandidateActionEnergyError, "exact target-only shape"
        ):
            mace.evaluate_candidate_action_energy(
                clean, epsilon, sigma, prompts, PrefixOutput().eval()
            )


@unittest.skipIf(torch is None, "torch is unavailable")
class FrozenAndGradientPolicyTests(unittest.TestCase):
    def _valid_inputs(self):
        prompts = _prompt_registry()
        clean = torch.zeros(1, 2, dtype=torch.float32)
        epsilon = clean + 4.0
        sigma = torch.tensor(0.5, dtype=torch.float32)
        offsets = {prompt: 1.0 for prompt in prompts.values()}
        return prompts, clean, epsilon, sigma, offsets

    def test_candidate_noise_and_sigma_gradients_are_forbidden(self) -> None:
        prompts, clean, epsilon, sigma, offsets = self._valid_inputs()
        denoiser = RecordingOffsetDenoiser(offsets).eval()
        for name, bad_clean, bad_noise, bad_sigma in (
            ("clean_candidate", clean.requires_grad_(), epsilon, sigma),
            ("epsilon", clean.detach(), epsilon.requires_grad_(), sigma),
            ("sigma", clean.detach(), epsilon.detach(), sigma.requires_grad_()),
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    mace.MACECandidateActionEnergyError,
                    rf"{name} must be detached",
                ):
                    mace.evaluate_candidate_action_energy(
                        bad_clean,
                        bad_noise,
                        bad_sigma,
                        prompts,
                        denoiser,
                    )

    def test_denoiser_must_be_frozen_eval_and_results_are_detached(self) -> None:
        prompts, clean, epsilon, sigma, offsets = self._valid_inputs()

        class ParameterDenoiser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, x_sigma, sigma, prompt):
                return torch.full_like(x_sigma, 4.0) * self.scale

        trainable = ParameterDenoiser().eval()
        with self.assertRaisesRegex(
            mace.MACECandidateActionEnergyError, "is trainable"
        ):
            mace.evaluate_candidate_action_energy(
                clean, epsilon, sigma, prompts, trainable
            )

        frozen = RecordingOffsetDenoiser(offsets)
        with self.assertRaisesRegex(
            mace.MACECandidateActionEnergyError, "eval mode"
        ):
            mace.evaluate_candidate_action_energy(
                clean, epsilon, sigma, prompts, frozen
            )

        frozen.eval()
        result = mace.evaluate_candidate_action_energy(
            clean, epsilon, sigma, prompts, frozen
        )
        for tensor in (
            result.x_sigma,
            result.velocity_target,
            result.branch_energies,
            result.negative_log_energy_ratios,
            result.reward,
        ):
            self.assertFalse(tensor.requires_grad)
            self.assertIsNone(tensor.grad_fn)

    def test_state_mutation_during_scoring_fails_closed(self) -> None:
        prompts, clean, epsilon, sigma, _ = self._valid_inputs()

        class MutatingDenoiser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("counter", torch.zeros((), dtype=torch.float32))

            def forward(self, x_sigma, sigma, prompt):
                self.counter.add_(1.0)
                return torch.full_like(x_sigma, 4.0)

        with self.assertRaisesRegex(
            mace.MACECandidateActionEnergyError,
            "parameters or buffers changed",
        ):
            mace.evaluate_candidate_action_energy(
                clean,
                epsilon,
                sigma,
                prompts,
                MutatingDenoiser().eval(),
            )

    def test_inference_mode_state_never_reads_missing_version_counter(self) -> None:
        prompts, clean, epsilon, sigma, offsets = self._valid_inputs()

        class InferenceStateDenoiser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                with torch.inference_mode():
                    self.register_buffer(
                        "frozen_offset", torch.tensor(0.0, dtype=torch.float32)
                    )

            def forward(self, x_sigma, sigma, prompt):
                return torch.full_like(x_sigma, 4.0 + offsets[prompt]) + self.frozen_offset

        denoiser = InferenceStateDenoiser().eval()
        self.assertTrue(torch.is_inference(denoiser.frozen_offset))
        with self.assertRaisesRegex(RuntimeError, "version counter"):
            _ = denoiser.frozen_offset._version

        result = mace.evaluate_candidate_action_energy(
            clean, epsilon, sigma, prompts, denoiser
        )
        self.assertEqual(tuple(result.branch_energies.shape), (10, 1))
        self.assertTrue(torch.isfinite(result.reward).all())

    def test_non_module_callable_is_not_accepted_as_frozen(self) -> None:
        prompts, clean, epsilon, sigma, _ = self._valid_inputs()

        def plain_function(x_sigma, sigma, prompt):
            return torch.full_like(x_sigma, 4.0)

        with self.assertRaisesRegex(
            mace.MACECandidateActionEnergyError,
            "must be a torch.nn.Module",
        ):
            mace.evaluate_candidate_action_energy(
                clean, epsilon, sigma, prompts, plain_function
            )


if __name__ == "__main__":
    unittest.main()
