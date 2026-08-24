from __future__ import annotations

import copy
import inspect
import math
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import generator_native_trajectory_controller as egntc  # noqa: E402

strata = egntc.sigma_strata


def _resign_receipt(receipt):
    value = copy.deepcopy(receipt)
    value.pop("receipt_digest", None)
    value["receipt_digest"] = egntc._object_sha256(value)
    return value


def _synthetic_complete_receipt():
    kappa_knots = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    rho_knots = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    parameters = {
        "schema_version": egntc.PARAMETER_SCHEMA_VERSION,
        "method": egntc.METHOD_NAME,
        "trainable_dimension": egntc.TRAINABLE_DIMENSION,
        "parameter_shapes": {
            "alpha_logits": [6, 4],
            "kappa_raw": [6],
            "rho_raw": [6],
        },
        "parameter_vector_sha256": "a" * 64,
        "decoded_kappa_knots": kappa_knots,
        "decoded_rho_knots": rho_knots,
        "kappa_monotone_nondecreasing": True,
        "rho_monotone_nondecreasing": True,
        "kappa_strict_upper_bound": egntc.MAX_KAPPA,
        "rho_strict_upper_bound": egntc.MAX_RHO,
        "schedule_sha256": strata.SCHEDULE_SHA256,
    }
    parameters["receipt_digest"] = egntc._object_sha256(parameters)
    steps = []
    for index in range(strata.NUM_INFERENCE_STEPS):
        timestep = strata.PINNED_TIMESTEPS[index]
        sigma = strata.PINNED_POSITIVE_SIGMAS[index]
        interpolation = egntc.sigma_interpolation(
            step_index=index, timestep=timestep, sigma=sigma
        )
        steps.append(
            {
                "step_index": index,
                "timestep": timestep,
                "sigma": sigma,
                "sigma_float32_be_hex": strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[
                    index
                ],
                "upper_knot": interpolation.upper_knot,
                "lower_knot": interpolation.lower_knot,
                "lower_weight": interpolation.lower_weight,
                "kappa": egntc._interpolate_receipt_knots(
                    kappa_knots, interpolation
                ),
                "rho": egntc._interpolate_receipt_knots(rho_knots, interpolation),
                "action_noop_exact_parity": False,
                "native_delta_rms_max": 0.2,
                "proposal_correction_rms_max": 0.15,
                "executed_correction_rms_max": 0.1,
                "trust_region_satisfied": True,
            }
        )
    payload = {
        "schema_version": egntc.SCHEMA_VERSION,
        "method": egntc.METHOD_NAME,
        "runtime_contract": egntc.controller_contract(),
        "parameters": parameters,
        "controls": {
            "phase_reverse": False,
            "sigma_shuffle": False,
            "kappa_off": False,
            "rho_off": False,
        },
        "active_controls": [],
        "state": {
            "expected_next_step": strata.NUM_INFERENCE_STEPS,
            "completed": True,
            "step_count": strata.NUM_INFERENCE_STEPS,
            "memory_present": True,
            "reset_count": 0,
        },
        "steps": steps,
    }
    return _resign_receipt(payload)


class EGNTCContractTests(unittest.TestCase):
    def test_source_instruction_only_runtime_contract(self) -> None:
        contract = egntc.controller_contract()
        self.assertEqual(
            contract["external_inference_conditions"],
            ["source_video", "action_instruction"],
        )
        forbidden = {
            "target_video",
            "mask",
            "track",
            "pose",
            "flow",
            "trajectory",
            "first_frame_anchor",
        }
        self.assertTrue(
            forbidden <= set(contract["forbidden_inference_conditions"])
        )
        self.assertEqual(
            contract["parameterization"]["trainable_dimension"], 36
        )
        self.assertEqual(
            contract["schedule"]["schedule_sha256"], strata.SCHEDULE_SHA256
        )

    def test_public_runtime_apis_cannot_receive_privileged_conditions(self) -> None:
        forbidden_fragments = (
            "target",
            "mask",
            "track",
            "pose",
            "flow",
            "trajectory",
            "anchor",
        )
        functions = (
            egntc.EGNTCCallback.__init__,
            egntc.EGNTCCallback.__call__,
            egntc.EGNTCCallback.apply_fields,
        )
        for function in functions:
            parameters = inspect.signature(function).parameters
            for name in parameters:
                self.assertFalse(
                    any(fragment in name for fragment in forbidden_fragments),
                    (function.__qualname__, name),
                )

    def test_six_knots_are_real_members_of_exact_pinned_schedule(self) -> None:
        self.assertEqual(egntc.TRAINABLE_DIMENSION, 36)
        self.assertEqual(len(egntc.PINNED_SIGMA_KNOTS), 6)
        self.assertEqual(
            egntc.PINNED_SIGMA_KNOTS,
            tuple(
                strata.PINNED_POSITIVE_SIGMAS[index]
                for index in egntc.SIGMA_KNOT_SCHEDULE_INDICES
            ),
        )
        for knot, index in zip(
            egntc.PINNED_SIGMA_KNOTS,
            egntc.SIGMA_KNOT_SCHEDULE_INDICES,
        ):
            self.assertEqual(
                egntc._float32_hex(knot),
                strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[index],
            )

    def test_schedule_and_control_validation_fail_closed(self) -> None:
        with self.assertRaisesRegex(egntc.EGNTCContractError, "step_index"):
            egntc.validate_pinned_step(
                step_index=True,
                timestep=strata.PINNED_TIMESTEPS[0],
                sigma=strata.PINNED_POSITIVE_SIGMAS[0],
            )
        with self.assertRaisesRegex(egntc.EGNTCContractError, "timestep differs"):
            egntc.validate_pinned_step(
                step_index=0,
                timestep=strata.PINNED_TIMESTEPS[0] - 1,
                sigma=strata.PINNED_POSITIVE_SIGMAS[0],
            )
        with self.assertRaisesRegex(egntc.EGNTCContractError, "sigma differs"):
            egntc.validate_pinned_step(
                step_index=0,
                timestep=strata.PINNED_TIMESTEPS[0],
                sigma=strata.PINNED_POSITIVE_SIGMAS[0] - 1e-4,
            )
        with self.assertRaisesRegex(egntc.EGNTCContractError, "unknown"):
            egntc.normalize_controls("not_a_control")
        with self.assertRaisesRegex(egntc.EGNTCContractError, "must be bool"):
            egntc.EGNTCControls(kappa_off=1).validate()

    def test_resigned_adversarial_receipts_fail_semantic_validation(self) -> None:
        valid = _synthetic_complete_receipt()
        egntc.validate_controller_receipt(valid, require_complete=True)

        mutations = {
            "method": lambda value: value.__setitem__("method", "wrong-method"),
            "parameter_receipt": lambda value: value.__setitem__(
                "parameters", {"trainable_dimension": 36}
            ),
            "numeric_sigma": lambda value: value["steps"][3].__setitem__(
                "sigma", 9999.0
            ),
            "interpolation_knots": lambda value: value["steps"][4].__setitem__(
                "upper_knot", 999
            ),
            "interpolation_weight": lambda value: value["steps"][5].__setitem__(
                "lower_weight", 42.0
            ),
            "parity_type": lambda value: value["steps"][6].__setitem__(
                "action_noop_exact_parity", "not-bool"
            ),
            "negative_rms": lambda value: value["steps"][7].__setitem__(
                "proposal_correction_rms_max", -3.0
            ),
            "false_trust_claim": lambda value: (
                value["steps"][8].__setitem__("native_delta_rms_max", 0.0),
                value["steps"][8].__setitem__(
                    "executed_correction_rms_max", 1.0e9
                ),
            ),
            "active_controls": lambda value: value.__setitem__(
                "active_controls", ["rho_off"]
            ),
            "negative_reset_count": lambda value: value["state"].__setitem__(
                "reset_count", -99
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                corrupted = copy.deepcopy(valid)
                mutate(corrupted)
                corrupted = _resign_receipt(corrupted)
                with self.assertRaises(egntc.EGNTCContractError):
                    egntc.validate_controller_receipt(
                        corrupted, require_complete=True
                    )


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is unavailable")
class EGNTCTensorTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260807)
        self.parameters = egntc.EGNTCParameters()
        self.shape = (2, 3, egntc.EXPECTED_LATENT_PHASES, 2, 2)
        self.source = torch.randn(self.shape, dtype=torch.float32)

    @staticmethod
    def schedule(index: int) -> tuple[int, float]:
        return (
            strata.PINNED_TIMESTEPS[index],
            strata.PINNED_POSITIVE_SIGMAS[index],
        )

    def make_controller(self, *, controls=None) -> egntc.EGNTCCallback:
        return egntc.EGNTCCallback(
            self.parameters, self.source, controls=controls
        )

    def apply(
        self,
        controller: egntc.EGNTCCallback,
        index: int,
        action: "torch.Tensor",
        noop: "torch.Tensor",
    ) -> "torch.Tensor":
        timestep, sigma = self.schedule(index)
        return controller.apply_fields(
            action_clean=action,
            noop_clean=noop,
            step_index=index,
            timestep=timestep,
            sigma=sigma,
        )

    def test_exact_parameter_shapes_and_vector_roundtrip(self) -> None:
        self.assertEqual(tuple(self.parameters.alpha_logits.shape), (6, 4))
        self.assertEqual(tuple(self.parameters.kappa_raw.shape), (6,))
        self.assertEqual(tuple(self.parameters.rho_raw.shape), (6,))
        self.assertEqual(
            sum(parameter.numel() for parameter in self.parameters.parameters()), 36
        )
        timestep, sigma = self.schedule(0)
        initial_alpha = self.parameters.coefficients(
            step_index=0, timestep=timestep, sigma=sigma
        ).alpha
        torch.testing.assert_close(
            initial_alpha,
            torch.full_like(initial_alpha, egntc.INITIAL_ALPHA),
            atol=2e-7,
            rtol=0,
        )
        vector = torch.linspace(-1.0, 1.0, 36)
        self.parameters.load_parameter_vector_(vector)
        torch.testing.assert_close(
            self.parameters.parameter_vector(detach=True), vector
        )
        receipt = self.parameters.receipt()
        self.assertEqual(receipt["trainable_dimension"], 36)
        self.assertTrue(receipt["kappa_monotone_nondecreasing"])
        self.assertTrue(receipt["rho_monotone_nondecreasing"])
        alpha = self.parameters.coefficients(
            step_index=0, timestep=timestep, sigma=sigma
        ).alpha
        self.assertTrue(torch.all((alpha >= 0.0) & (alpha <= 1.0)))

    def test_interpolation_uses_actual_sigma_not_normalized_step(self) -> None:
        values = torch.arange(6, dtype=torch.float32)
        index = 4  # Bracketed by schedule indices zero and eight.
        timestep, sigma = self.schedule(index)
        observed = egntc.interpolate_sigma_knots(
            values, step_index=index, timestep=timestep, sigma=sigma
        )
        high = egntc.PINNED_SIGMA_KNOTS[0]
        low = egntc.PINNED_SIGMA_KNOTS[1]
        real_sigma_weight = (high - sigma) / (high - low)
        self.assertAlmostEqual(float(observed), real_sigma_weight, places=7)
        self.assertNotAlmostEqual(real_sigma_weight, index / 8.0, places=4)
        for knot_position, schedule_index in enumerate(
            egntc.SIGMA_KNOT_SCHEDULE_INDICES
        ):
            timestep, sigma = self.schedule(schedule_index)
            exact = egntc.interpolate_sigma_knots(
                values,
                step_index=schedule_index,
                timestep=timestep,
                sigma=sigma,
            )
            self.assertEqual(float(exact), float(knot_position))

    def test_kappa_and_rho_are_monotone_over_all_real_denoising_sigmas(self) -> None:
        with torch.no_grad():
            self.parameters.kappa_raw.copy_(torch.tensor([3, -4, 1, -2, 4, -1.0]))
            self.parameters.rho_raw.copy_(torch.tensor([-3, 2, -1, 4, -2, 1.0]))
        kappas = []
        rhos = []
        for index in range(40):
            timestep, sigma = self.schedule(index)
            coefficients = self.parameters.coefficients(
                step_index=index, timestep=timestep, sigma=sigma
            )
            kappas.append(float(coefficients.kappa))
            rhos.append(float(coefficients.rho))
        self.assertTrue(all(a <= b for a, b in zip(kappas, kappas[1:])))
        self.assertTrue(all(a <= b for a, b in zip(rhos, rhos[1:])))
        self.assertGreaterEqual(min(kappas), 0.0)
        self.assertLess(max(kappas), egntc.MAX_KAPPA)
        self.assertGreaterEqual(min(rhos), 0.0)
        self.assertLess(max(rhos), egntc.MAX_RHO)

        shuffled_kappas = []
        shuffled_rhos = []
        controls = egntc.EGNTCControls(sigma_shuffle=True)
        for index in range(40):
            timestep, sigma = self.schedule(index)
            coefficients = self.parameters.coefficients(
                step_index=index,
                timestep=timestep,
                sigma=sigma,
                controls=controls,
            )
            shuffled_kappas.append(float(coefficients.kappa))
            shuffled_rhos.append(float(coefficients.rho))
        self.assertTrue(
            all(a <= b for a, b in zip(shuffled_kappas, shuffled_kappas[1:]))
        )
        self.assertTrue(
            all(a <= b for a, b in zip(shuffled_rhos, shuffled_rhos[1:]))
        )

    def test_stateful_recurrence_and_per_batch_phase_native_trust(self) -> None:
        with torch.no_grad():
            self.parameters.alpha_logits.fill_(4.0)
            self.parameters.rho_raw.fill_(4.0)
        controller = self.make_controller()
        noop0 = torch.randn(self.shape)
        action0 = noop0 + torch.randn(self.shape) * 0.05
        output0 = self.apply(controller, 0, action0, noop0)
        self.assertIsNotNone(controller.memory)
        first_memory = controller.memory.detach().clone()

        noop1 = torch.randn(self.shape)
        action1 = noop1 + torch.randn(self.shape) * 0.05
        output1 = self.apply(controller, 1, action1, noop1)
        self.assertFalse(torch.equal(first_memory, controller.memory.detach()))
        for output, noop, action in (
            (output0, noop0, action0),
            (output1, noop1, action1),
        ):
            executed_rms = egntc._per_batch_phase_rms(output - noop)
            native_rms = egntc._per_batch_phase_rms(action - noop)
            self.assertTrue(torch.all(executed_rms <= native_rms + 1e-6))

    def test_exact_action_noop_parity_is_hard_bypass_and_clears_memory(self) -> None:
        controller = self.make_controller()
        noop0 = torch.randn(self.shape)
        self.apply(controller, 0, noop0 + 0.2, noop0)
        self.assertGreater(int(torch.count_nonzero(controller.memory)), 0)

        noop1 = torch.randn(self.shape)
        output = self.apply(controller, 1, noop1, noop1)
        self.assertTrue(torch.equal(output, noop1))
        self.assertEqual(int(torch.count_nonzero(controller.memory)), 0)
        self.assertTrue(controller.receipt()["steps"][-1]["action_noop_exact_parity"])

        controller.reset()
        timestep0, sigma0 = self.schedule(0)
        fields = type(
            "Fields",
            (),
            {
                "step_index": 0,
                "timestep": timestep0,
                "sigma": sigma0,
                "action_guided_clean": noop1,
                "noop_guided_clean": noop1.clone(),
            },
        )()
        callback_output = controller(fields)
        self.assertIs(callback_output, fields.action_guided_clean)
        self.assertEqual(controller.expected_step, 1)

    def test_gradients_terminate_at_fields_and_are_finite_only_on_controller(self) -> None:
        source = self.source.clone().requires_grad_(True)
        controller = egntc.EGNTCCallback(self.parameters, source)
        noop = torch.randn(self.shape, requires_grad=True)
        action = (noop.detach() + torch.randn(self.shape) * 0.2).requires_grad_(True)
        output = self.apply(controller, 0, action, noop)
        output.square().mean().backward()
        self.assertIsNone(source.grad)
        self.assertIsNone(action.grad)
        self.assertIsNone(noop.grad)
        gradients = [parameter.grad for parameter in self.parameters.parameters()]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0.0)

    def test_all_controls_are_effective_and_keep_declared_bounds(self) -> None:
        with torch.no_grad():
            self.parameters.alpha_logits.copy_(
                torch.arange(24, dtype=torch.float32).reshape(6, 4) / 11.0 - 1.0
            )
            self.parameters.kappa_raw.copy_(torch.tensor([-4, -2, 0, 2, 4, 1.0]))
            self.parameters.rho_raw.copy_(torch.tensor([-3, -1, 1, 3, 0, 2.0]))
        index = 0
        noop = torch.randn(self.shape)
        action = noop + torch.randn(self.shape) * 0.3

        def one(control):
            return self.apply(self.make_controller(controls=control), index, action, noop)

        baseline = one(None)
        for name in ("phase_reverse", "sigma_shuffle", "kappa_off", "rho_off"):
            with self.subTest(control=name):
                controlled = one(name)
                self.assertFalse(torch.allclose(controlled, baseline))
                receipt = self.make_controller(controls=name).receipt()
                self.assertEqual(receipt["active_controls"], [name])

    def test_shape_finite_order_reset_and_receipt_validation(self) -> None:
        controller = self.make_controller()
        noop = torch.randn(self.shape)
        action = noop + 0.1
        timestep1, sigma1 = self.schedule(1)
        with self.assertRaisesRegex(egntc.EGNTCContractError, "expected denoising step"):
            controller.apply_fields(
                action_clean=action,
                noop_clean=noop,
                step_index=1,
                timestep=timestep1,
                sigma=sigma1,
            )
        bad_shape = torch.zeros((2, 3, 20, 2, 2))
        timestep0, sigma0 = self.schedule(0)
        with self.assertRaisesRegex(egntc.EGNTCContractError, "21 latent phases"):
            controller.apply_fields(
                action_clean=bad_shape,
                noop_clean=bad_shape,
                step_index=0,
                timestep=timestep0,
                sigma=sigma0,
            )
        bad_finite = action.clone()
        bad_finite[0, 0, 0, 0, 0] = math.nan
        with self.assertRaisesRegex(egntc.EGNTCContractError, "finite"):
            controller.apply_fields(
                action_clean=bad_finite,
                noop_clean=noop,
                step_index=0,
                timestep=timestep0,
                sigma=sigma0,
            )

        self.apply(controller, 0, action, noop)
        partial = controller.receipt()
        egntc.validate_controller_receipt(partial)
        with self.assertRaisesRegex(egntc.EGNTCContractError, "not a complete"):
            egntc.validate_controller_receipt(partial, require_complete=True)
        corrupted = copy.deepcopy(partial)
        corrupted["steps"][0]["trust_region_satisfied"] = False
        with self.assertRaisesRegex(egntc.EGNTCContractError, "digest differs"):
            egntc.validate_controller_receipt(corrupted)

        controller.reset()
        self.assertEqual(controller.expected_step, 0)
        self.assertIsNone(controller.memory)
        self.assertEqual(controller.receipt()["state"]["step_count"], 0)
        self.apply(controller, 0, action, noop)

    def test_complete_40_step_receipt_has_exact_order_and_schedule(self) -> None:
        controller = self.make_controller()
        noop = torch.zeros(self.shape)
        for index in range(40):
            self.apply(controller, index, noop, noop)
        receipt = controller.receipt()
        egntc.validate_controller_receipt(receipt, require_complete=True)
        self.assertTrue(receipt["state"]["completed"])
        self.assertEqual(
            [record["sigma_float32_be_hex"] for record in receipt["steps"]],
            list(strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX),
        )


if __name__ == "__main__":
    unittest.main()
