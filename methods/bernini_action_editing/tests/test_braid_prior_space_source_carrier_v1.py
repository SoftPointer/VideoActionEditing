#!/usr/bin/env python3

from __future__ import annotations

import builtins
from dataclasses import replace
import inspect
from pathlib import Path
import sys
import unittest
from unittest import mock


try:
    import torch

    TORCH_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - minimal local workspace
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

if TORCH_AVAILABLE:
    import braid_prior_space_source_carrier_v1 as carrier  # noqa: E402
else:  # pragma: no cover
    carrier = None  # type: ignore[assignment]


def _inputs(*, shape=(2, 16, 21, 3, 4)):
    source_generator = torch.Generator(device="cpu").manual_seed(73)
    gaussian_generator = torch.Generator(device="cpu").manual_seed(109)
    source = torch.randn(shape, generator=source_generator, dtype=torch.float32)
    # Give eta_source a visible low-frequency component without changing the
    # contract that this primitive does not authenticate its origin.
    source = source + torch.linspace(-0.7, 0.9, 21).reshape(1, 1, 21, 1, 1)
    gaussian = torch.randn(
        shape, generator=gaussian_generator, dtype=torch.float32
    )
    return source.contiguous(), gaussian.contiguous()


def _config(arm_id=None):
    return carrier.preregistered_config(arm_id or carrier.PRIMARY_ARM_ID)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is unavailable")
class PriorSpaceSourceCarrierTests(unittest.TestCase):
    def test_registry_is_closed_exact81_and_receipt_has_no_authority(self) -> None:
        self.assertEqual(carrier.NUM_FRAMES, 81)
        self.assertEqual(carrier.LATENT_PHASES, 21)
        self.assertEqual(
            carrier.PREREGISTERED_ARM_IDS,
            (
                carrier.PRIMARY_ARM_ID,
                carrier.RHO_ZERO_CONTROL_ARM_ID,
                carrier.RHO_ONE_CONTROL_ARM_ID,
            ),
        )
        config = _config()
        self.assertEqual(config.low_frequency_max_mode, 3)
        self.assertEqual(config.rho, 0.5)
        config_receipt = config.receipt()
        self.assertFalse(config_receipt["caller_selectable_k_or_rho"])
        self.assertTrue(config_receipt["registry_coordinate_embedded_in_source"])
        self.assertFalse(config_receipt["decoded_result_input_accepted"])
        self.assertEqual(config_receipt["low_frequency_modes"], [0, 1, 2, 3])

        source, gaussian = _inputs()
        receipt = carrier.build_prior_space_source_carrier(
            source, gaussian, config=config
        ).receipt
        self.assertEqual(receipt["classification"], "math_primitive_only/no_authority")
        claims = receipt["upstream_claims"]
        for name in (
            "eta_source_inversion_executed_by_this_primitive",
            "eta_source_inversion_authenticated",
            "official_gaussian_provenance_authenticated",
            "source_roundtrip_executed",
            "prior_statistics_gate_executed",
            "action_gate_executed",
        ):
            self.assertFalse(claims[name])
        self.assertIsNone(claims["source_roundtrip_passed"])
        self.assertIsNone(claims["prior_statistics_gate_passed"])
        self.assertIsNone(claims["action_gate_passed"])
        authority = receipt["side_effects_and_authority"]
        self.assertTrue(all(value is False for value in authority.values()))

        with self.assertRaises(carrier.BraidPriorSpaceCarrierError):
            carrier.preregistered_config("posthoc-rho0p73")
        forged = carrier.PriorSpaceSourceCarrierConfig(
            arm_id=carrier.PRIMARY_ARM_ID,
            num_frames=81,
            latent_phases=21,
            low_frequency_max_mode=2,
            rho=0.5,
            role="prospective_primary",
        )
        with self.assertRaisesRegex(
            carrier.BraidPriorSpaceCarrierError, "preregistered coordinate"
        ):
            carrier.build_prior_space_source_carrier(
                source, gaussian, config=forged
            )
        for altered in (
            replace(config, num_frames=41),
            replace(config, latent_phases=20),
            replace(config, rho=0.75),
            replace(config, role="posthoc_role"),
        ):
            with self.subTest(altered=altered):
                with self.assertRaisesRegex(
                    carrier.BraidPriorSpaceCarrierError,
                    "preregistered coordinate",
                ):
                    carrier.build_prior_space_source_carrier(
                        source, gaussian, config=altered
                    )

    def test_dct_basis_split_reconstructs_and_preserves_energy(self) -> None:
        basis = carrier.build_temporal_orthonormal_dct_basis()
        identity = torch.eye(21, dtype=torch.float64)
        self.assertTrue(
            torch.allclose(basis.T @ basis, identity, rtol=0.0, atol=2.0e-14)
        )
        source, gaussian = _inputs()
        for label, value in (("source", source), ("gaussian", gaussian)):
            with self.subTest(label=label):
                bands = carrier.temporal_dct_split(value, config=_config())
                reconstructed = bands.low + bands.high
                self.assertTrue(
                    torch.allclose(reconstructed, value, rtol=2.0e-6, atol=2.0e-6)
                )
                low64 = bands.low.double()
                high64 = bands.high.double()
                value64 = value.double()
                dot = (low64 * high64).sum().abs()
                scale = low64.norm() * high64.norm()
                self.assertLess(float(dot / scale), 2.0e-7)
                split_energy = low64.square().sum() + high64.square().sum()
                relative = (split_energy - value64.square().sum()).abs() / value64.square().sum()
                self.assertLess(float(relative), 2.0e-7)

    def test_rho_zero_keeps_source_low_and_uses_only_gaussian_high(self) -> None:
        source, gaussian = _inputs()
        config = _config(carrier.RHO_ZERO_CONTROL_ARM_ID)
        result = carrier.build_prior_space_source_carrier(
            source, gaussian, config=config
        )
        source_bands = carrier.temporal_dct_split(source, config=config)
        gaussian_bands = carrier.temporal_dct_split(gaussian, config=config)
        output_bands = carrier.temporal_dct_split(
            result.initial_noise, config=config
        )
        self.assertTrue(
            torch.allclose(output_bands.low, source_bands.low, rtol=3.0e-6, atol=3.0e-6)
        )
        self.assertTrue(
            torch.allclose(output_bands.high, gaussian_bands.high, rtol=3.0e-6, atol=3.0e-6)
        )
        self.assertFalse(torch.equal(result.initial_noise, gaussian))
        operator = result.receipt["operator"]
        self.assertEqual(operator["source_high_band_coefficient"], 0.0)
        self.assertEqual(operator["official_gaussian_high_band_coefficient"], 1.0)
        self.assertFalse(operator["both_high_bands_active"])

    def test_rho_one_is_exact_source_copy_without_alias(self) -> None:
        source, gaussian = _inputs()
        result = carrier.build_prior_space_source_carrier(
            source,
            gaussian,
            config=_config(carrier.RHO_ONE_CONTROL_ARM_ID),
        )
        self.assertTrue(torch.equal(result.initial_noise, source))
        self.assertIsNot(result.initial_noise, source)
        self.assertNotEqual(result.initial_noise.data_ptr(), source.data_ptr())
        self.assertNotEqual(result.initial_noise.data_ptr(), gaussian.data_ptr())
        operator = result.receipt["operator"]
        self.assertEqual(operator["source_high_band_coefficient"], 1.0)
        self.assertEqual(operator["official_gaussian_high_band_coefficient"], 0.0)
        self.assertFalse(operator["both_high_bands_active"])

    def test_intermediate_rho_matches_registered_band_formula(self) -> None:
        source, gaussian = _inputs()
        config = _config()
        source_bands = carrier.temporal_dct_split(source, config=config)
        gaussian_bands = carrier.temporal_dct_split(gaussian, config=config)
        expected = (
            source_bands.low.double()
            + config.rho * source_bands.high.double()
            + (1.0 - config.rho**2) ** 0.5 * gaussian_bands.high.double()
        ).float()
        actual = carrier.build_prior_space_source_carrier(
            source, gaussian, config=config
        ).initial_noise
        self.assertTrue(torch.allclose(actual, expected, rtol=3.0e-6, atol=3.0e-6))

    def test_inputs_are_immutable_and_output_owns_fresh_storage(self) -> None:
        source, gaussian = _inputs()
        source_before = source.clone()
        gaussian_before = gaussian.clone()
        result = carrier.build_prior_space_source_carrier(
            source, gaussian, config=_config()
        )
        self.assertTrue(torch.equal(source, source_before))
        self.assertTrue(torch.equal(gaussian, gaussian_before))
        self.assertNotEqual(result.initial_noise.data_ptr(), source.data_ptr())
        self.assertNotEqual(result.initial_noise.data_ptr(), gaussian.data_ptr())
        result.initial_noise.add_(1.0)
        self.assertTrue(torch.equal(source, source_before))
        self.assertTrue(torch.equal(gaussian, gaussian_before))

    def test_rejects_nonfinite_wrong_geometry_dtype_and_nonowned_views(self) -> None:
        source, gaussian = _inputs()
        bad_values = []
        nonfinite = source.clone()
        nonfinite[0, 0, 0, 0, 0] = float("nan")
        bad_values.append(nonfinite)
        infinite = source.clone()
        infinite[0, 0, 0, 0, 0] = float("inf")
        bad_values.append(infinite)
        bad_values.extend(
            (
                source[:, :, :-1].clone(),
                torch.zeros(2, 15, 21, 3, 4),
                source.double(),
                source.transpose(3, 4),
                source.view_as(source),
            )
        )
        for value in bad_values:
            with self.subTest(shape=tuple(value.shape), dtype=str(value.dtype)):
                with self.assertRaises(carrier.BraidPriorSpaceCarrierError):
                    carrier.build_prior_space_source_carrier(
                        value, gaussian, config=_config()
                    )
        with self.assertRaisesRegex(
            carrier.BraidPriorSpaceCarrierError, "share shape"
        ):
            carrier.build_prior_space_source_carrier(
                source,
                torch.zeros(1, 16, 21, 3, 4),
                config=_config(),
            )

    def test_rejects_grad_tensor_subclass_and_pair_alias(self) -> None:
        source, gaussian = _inputs()
        with self.assertRaisesRegex(
            carrier.BraidPriorSpaceCarrierError, "grad-free"
        ):
            carrier.build_prior_space_source_carrier(
                source.clone().requires_grad_(), gaussian, config=_config()
            )
        grad_marked = source.clone()
        grad_marked.grad = torch.ones_like(grad_marked)
        with self.assertRaisesRegex(
            carrier.BraidPriorSpaceCarrierError, "grad-free"
        ):
            carrier.build_prior_space_source_carrier(
                grad_marked, gaussian, config=_config()
            )

        class TensorSubclass(torch.Tensor):
            pass

        subclass = torch.Tensor._make_subclass(TensorSubclass, source, False)
        with self.assertRaisesRegex(
            carrier.BraidPriorSpaceCarrierError, "exact-type"
        ):
            carrier.build_prior_space_source_carrier(
                subclass, gaussian, config=_config()
            )
        with self.assertRaisesRegex(
            carrier.BraidPriorSpaceCarrierError, "alias"
        ):
            carrier.build_prior_space_source_carrier(
                source, source, config=_config()
            )
        shared = source.detach()
        self.assertEqual(shared.data_ptr(), source.data_ptr())
        with self.assertRaisesRegex(
            carrier.BraidPriorSpaceCarrierError, "alias"
        ):
            carrier.build_prior_space_source_carrier(
                source, shared, config=_config()
            )

    def test_train_inference_are_same_function_and_call_has_no_side_effects(self) -> None:
        self.assertIs(
            carrier.training_source_carrier,
            carrier.build_prior_space_source_carrier,
        )
        self.assertIs(
            carrier.inference_source_carrier,
            carrier.build_prior_space_source_carrier,
        )
        parameters = tuple(
            inspect.signature(carrier.build_prior_space_source_carrier).parameters
        )
        self.assertEqual(parameters, ("eta_source", "official_gaussian", "config"))
        for forbidden in (
            "media",
            "owner",
            "model",
            "prompt",
            "scheduler",
            "mask",
            "pose",
            "flow",
            "trajectory",
            "training",
            "inference",
            "stage",
        ):
            self.assertNotIn(forbidden, parameters)

        source, gaussian = _inputs(shape=(1, 16, 21, 2, 2))
        with mock.patch.object(
            builtins, "open", side_effect=AssertionError("file read/write")
        ), mock.patch.object(
            torch, "save", side_effect=AssertionError("tensor save")
        ), mock.patch.object(
            torch, "load", side_effect=AssertionError("tensor load")
        ), mock.patch.object(
            torch.optim.Optimizer,
            "__init__",
            side_effect=AssertionError("optimizer construction"),
        ):
            result = carrier.build_prior_space_source_carrier(
                source, gaussian, config=_config()
            )
        self.assertFalse(result.receipt["side_effects_and_authority"]["media_read"])
        self.assertFalse(
            result.receipt["side_effects_and_authority"]["model_loaded_or_forwarded"]
        )

        source_text = Path(carrier.__file__).read_text(encoding="utf-8")
        for forbidden_call in (
            ".backward(",
            "torch.optim.",
            "torch.save(",
            "torch.load(",
            "Autoencoder",
            "BerniniRendererModel",
            "subprocess.",
        ):
            self.assertNotIn(forbidden_call, source_text)

    def test_basis_hash_and_receipt_do_not_cross_numpy_abi(self) -> None:
        source, gaussian = _inputs(shape=(1, 16, 21, 2, 2))
        with mock.patch.object(
            torch.Tensor,
            "numpy",
            side_effect=RuntimeError("NumPy is not available"),
        ):
            result = carrier.build_prior_space_source_carrier(
                source, gaussian, config=_config()
            )
        digest = result.receipt["temporal_basis_sha256"]
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertEqual(digest, carrier._basis_sha256())
        source_text = Path(carrier.__file__).read_text(encoding="utf-8")
        self.assertNotIn(".numpy(", source_text)


if __name__ == "__main__":
    unittest.main()
