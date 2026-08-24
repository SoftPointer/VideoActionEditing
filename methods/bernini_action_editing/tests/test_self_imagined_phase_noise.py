import importlib
import inspect
import pathlib
import sys
import unittest

import torch


METHOD_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

phase = importlib.import_module("self_imagined_phase_noise")
source_bridge = importlib.import_module("source_spectral_bridge")


class SelfImaginedPhaseNoiseTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator(device="cpu").manual_seed(17)
        shape = (1, 16, 21, 10, 12)
        self.gaussian = torch.randn(shape, generator=generator).float().contiguous()
        self.action = torch.randn(shape, generator=generator).float().contiguous()
        self.source = torch.randn(shape, generator=generator).float().contiguous()

    def test_inactive_operator_returns_exact_input_object(self):
        result = phase.build_factorized_phase_noise(
            self.gaussian,
            self.action,
            self.source,
            spatial_radius=0,
            source_rho=0.0,
        )
        self.assertIs(result.initial_noise, self.gaussian)
        self.assertIs(result.action_phase_noise, self.gaussian)
        self.assertTrue(result.phase_diagnostics.radius_zero_exact_gaussian_alias)

    def test_spatial_phase_operator_preserves_realized_energy(self):
        output, diagnostics = phase.spatial_action_phase_noise(
            self.gaussian, self.action, spatial_radius=2, gamma=30.0
        )
        self.assertEqual(tuple(output.shape), tuple(self.gaussian.shape))
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output).all())
        self.assertFalse(torch.equal(output, self.gaussian))
        before = self.gaussian.double().flatten(1).norm(dim=1)
        after = output.double().flatten(1).norm(dim=1)
        self.assertTrue(torch.allclose(before, after, rtol=5e-5, atol=5e-5))
        self.assertTrue(diagnostics.energy_audit_passed)
        self.assertGreater(diagnostics.low_phase_cosine_min, 1.0 - 1.0e-10)
        self.assertLess(diagnostics.high_spectrum_max_relative_error_before_beta, 1e-12)

    def test_action_reference_changes_only_active_operator(self):
        first, _ = phase.spatial_action_phase_noise(
            self.gaussian, self.action, spatial_radius=2, gamma=30.0
        )
        second_action = torch.roll(self.action, shifts=1, dims=-1).contiguous()
        second, _ = phase.spatial_action_phase_noise(
            self.gaussian, second_action, spatial_radius=2, gamma=30.0
        )
        self.assertFalse(torch.equal(first, second))
        inactive_a, _ = phase.spatial_action_phase_noise(
            self.gaussian, self.action, spatial_radius=0
        )
        inactive_b, _ = phase.spatial_action_phase_noise(
            self.gaussian, second_action, spatial_radius=0
        )
        self.assertIs(inactive_a, inactive_b)

    def test_source_bridge_preserves_action_temporal_residual(self):
        result = phase.build_factorized_phase_noise(
            self.gaussian,
            self.action,
            self.source,
            spatial_radius=2,
            gamma=30.0,
            source_rho=0.2,
        )
        action_dc, action_residual = source_bridge.temporal_dc_residual(
            result.action_phase_noise
        )
        output_dc, output_residual = source_bridge.temporal_dc_residual(
            result.initial_noise
        )
        del action_dc, output_dc
        self.assertTrue(torch.allclose(action_residual, output_residual, rtol=5e-5, atol=5e-5))
        self.assertTrue(
            result.receipt["source_carrier"][
                "intermediate_noise_temporal_residual_preserved_by_source_bridge"
            ]
        )
        self.assertTrue(
            result.receipt["source_carrier"]["semantic_action_preservation_not_claimed"]
        )

    def test_source_carrier_is_frame_permutation_invariant(self):
        permutation = torch.randperm(21, generator=torch.Generator().manual_seed(91))
        permuted = self.source.index_select(2, permutation).contiguous()
        left = phase.build_factorized_phase_noise(
            self.gaussian,
            self.action,
            self.source,
            spatial_radius=2,
            source_rho=0.2,
        )
        right = phase.build_factorized_phase_noise(
            self.gaussian,
            self.action,
            permuted,
            spatial_radius=2,
            source_rho=0.2,
        )
        self.assertTrue(torch.equal(left.initial_noise, right.initial_noise))

    def test_receipt_refuses_gaussian_overclaim_and_target_inputs(self):
        result = phase.build_factorized_phase_noise(
            self.gaussian,
            self.action,
            self.source,
            spatial_radius=2,
            source_rho=0.2,
        )
        self.assertFalse(result.receipt["distribution_claim"]["modified_noise_proven_gaussian"])
        self.assertIn("paired_target_edit", result.receipt["forbidden_inputs"])
        parameters = set(inspect.signature(phase.build_factorized_phase_noise).parameters)
        self.assertEqual(
            parameters,
            {
                "gaussian",
                "action_reference",
                "source_normalized_latent",
                "spatial_radius",
                "gamma",
                "source_rho",
            },
        )

    def test_mask_is_conjugate_symmetric(self):
        mask = phase.spatial_low_frequency_mask(10, 12, radius=3)
        rows = (-torch.arange(10)) % 10
        cols = (-torch.arange(12)) % 12
        self.assertTrue(torch.equal(mask, mask.index_select(0, rows).index_select(1, cols)))

    def test_invalid_contracts_fail_closed(self):
        with self.assertRaisesRegex(phase.SelfImaginedPhaseNoiseError, "share Gaussian"):
            phase.build_factorized_phase_noise(
                self.gaussian, self.action[:, :, :, :, :-1].contiguous(), self.source
            )
        with self.assertRaisesRegex(phase.SelfImaginedPhaseNoiseError, "spatial_radius"):
            phase.spatial_action_phase_noise(self.gaussian, self.action, spatial_radius=99)
        with self.assertRaisesRegex(phase.SelfImaginedPhaseNoiseError, "gamma"):
            phase.spatial_action_phase_noise(self.gaussian, self.action, gamma=0.5)
        with self.assertRaisesRegex(phase.SelfImaginedPhaseNoiseError, "source_rho"):
            phase.build_factorized_phase_noise(
                self.gaussian, self.action, self.source, source_rho=1.1
            )
        grad = self.gaussian.clone().requires_grad_(True)
        with self.assertRaisesRegex(phase.SelfImaginedPhaseNoiseError, "detached"):
            phase.spatial_action_phase_noise(grad, self.action)


if __name__ == "__main__":
    unittest.main()
