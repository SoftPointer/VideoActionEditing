from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

if torch is not None:
    import source_spectral_bridge as bridge
else:  # pragma: no cover
    bridge = None


@unittest.skipIf(torch is None, "torch is unavailable")
class SourceSpectralBridgeTests(unittest.TestCase):
    @staticmethod
    def tensors(*, dtype=None):
        dtype = dtype or torch.float32
        gaussian_generator = torch.Generator(device="cpu").manual_seed(71)
        source_generator = torch.Generator(device="cpu").manual_seed(113)
        shape = (2, 16, 21, 4, 6)
        gaussian = torch.randn(shape, generator=gaussian_generator, dtype=dtype)
        source = torch.randn(shape, generator=source_generator, dtype=dtype)
        source = 0.7 * source + torch.linspace(
            -1.0, 1.0, 21, dtype=dtype
        ).reshape(1, 1, 21, 1, 1)
        return gaussian, source

    def test_rho_zero_is_exact_gaussian_object_and_honest_receipt(self) -> None:
        gaussian, source = self.tensors()
        result = bridge.source_spectral_bridge(gaussian, source, rho=0.0)
        self.assertIs(result.initial_noise, gaussian)
        self.assertTrue(torch.equal(result.initial_noise, gaussian))
        self.assertTrue(result.diagnostics.rho_zero_exact_gaussian_alias)
        self.assertFalse(result.diagnostics.non_gaussian_initial_noise)
        receipt = result.receipt
        self.assertTrue(receipt["source_only"])
        self.assertFalse(receipt["paired_target_accessed"])
        self.assertFalse(receipt["non_gaussian_initial_noise"])
        self.assertTrue(receipt["train_inference_same_contract"])
        self.assertTrue(receipt["distribution"]["exact_gaussian_baseline"])
        self.assertFalse(receipt["distribution"]["non_gaussian_initial_noise"])
        self.assertTrue(receipt["inputs"]["source_only_conditioning"])
        self.assertEqual(receipt["inputs"]["target_columns_accessed"], [])
        self.assertFalse(receipt["inputs"]["target_media_accessed"])
        self.assertFalse(receipt["inputs"]["paired_target_accessed"])

    def test_gaussian_dc_and_residual_are_orthogonal(self) -> None:
        gaussian, _ = self.tensors(dtype=torch.float64)
        dc, residual = bridge.temporal_dc_residual(gaussian)
        dot = (dc.flatten(1) * residual.flatten(1)).sum(dim=1)
        denominator = dc.flatten(1).norm(dim=1) * residual.flatten(1).norm(dim=1)
        self.assertLess(float((dot.abs() / denominator).max()), 1.0e-15)
        self.assertTrue(
            torch.allclose(
                residual.mean(dim=2),
                torch.zeros_like(residual[:, :, 0]),
                atol=1.0e-15,
                rtol=0.0,
            )
        )

    def test_primary_carrier_and_output_are_temporal_permutation_invariant(self) -> None:
        gaussian, source = self.tensors(dtype=torch.float64)
        permutation = torch.tensor(
            [7, 2, 18, 0, 20, 4, 14, 5, 11, 1, 9, 16, 3, 13, 8, 19, 6, 15, 10, 17, 12]
        )
        permuted_source = source.index_select(2, permutation)
        carrier = bridge.temporal_set_carrier(source)
        permuted_carrier = bridge.temporal_set_carrier(permuted_source)
        self.assertTrue(torch.equal(carrier, permuted_carrier))
        self.assertTrue(
            torch.equal(carrier, carrier[:, :, :1].expand_as(carrier))
        )
        left = bridge.source_spectral_bridge(
            gaussian, source, rho=0.45, arm=bridge.PRIMARY_ARM
        )
        right = bridge.source_spectral_bridge(
            gaussian, permuted_source, rho=0.45, arm=bridge.PRIMARY_ARM
        )
        self.assertTrue(torch.equal(left.initial_noise, right.initial_noise))
        self.assertTrue(left.diagnostics.temporal_order_invariant_carrier)
        self.assertFalse(left.diagnostics.ordered_source_trajectory_injected)
        self.assertFalse(left.diagnostics.source_temporal_residual_injected)
        self.assertFalse(left.diagnostics.action_leakage_risk)

    def test_primary_active_mix_preserves_realized_moments_and_residual(self) -> None:
        gaussian, source = self.tensors()
        result = bridge.source_spectral_bridge(gaussian, source, rho=0.6)
        output = result.initial_noise
        self.assertFalse(torch.equal(output, gaussian))
        gaussian_residual = gaussian - gaussian.mean(dim=2, keepdim=True)
        output_residual = output - output.mean(dim=2, keepdim=True)
        self.assertTrue(
            torch.allclose(output_residual, gaussian_residual, atol=6.0e-6, rtol=4.0e-5)
        )
        self.assertTrue(result.diagnostics.moment_norm_audit_passed)
        self.assertTrue(result.diagnostics.non_gaussian_initial_noise)
        self.assertLess(result.diagnostics.total_norm_max_relative_error, 4.0e-5)
        self.assertLess(
            result.diagnostics.centered_dc_norm_max_relative_error, 4.0e-5
        )
        self.assertLess(
            result.diagnostics.temporal_residual_norm_max_relative_error, 4.0e-5
        )
        self.assertLess(result.diagnostics.carrier_base_normalized_dot_max, 1.0e-12)
        receipt = result.receipt
        self.assertEqual(
            receipt["distribution"]["non_gaussian_initial_noise"], True
        )
        self.assertEqual(
            receipt["mix"]["gaussian_decomposition"],
            "orthogonal_temporal_dc_plus_residual",
        )
        self.assertTrue(receipt["mix"]["moment_norm_audited"])
        self.assertTrue(receipt["train_inference_contract"]["same_contract"])
        self.assertFalse(receipt["train_inference_contract"]["stage_dependent_branch"])

    def test_shuffled_ablation_is_deterministic_and_permanently_risk_marked(self) -> None:
        gaussian, source = self.tensors(dtype=torch.float64)
        kwargs = {
            "rho": 0.5,
            "arm": bridge.SHUFFLED_FRAME_ABLATION_ARM,
            "shuffle_seed": 29,
        }
        first = bridge.source_spectral_bridge(gaussian, source, **kwargs)
        second = bridge.source_spectral_bridge(gaussian, source, **kwargs)
        other_seed = bridge.source_spectral_bridge(
            gaussian,
            source,
            rho=0.5,
            arm=bridge.SHUFFLED_FRAME_ABLATION_ARM,
            shuffle_seed=31,
        )
        self.assertTrue(torch.equal(first.initial_noise, second.initial_noise))
        self.assertFalse(torch.equal(first.initial_noise, other_seed.initial_noise))
        self.assertTrue(first.diagnostics.action_leakage_risk)
        self.assertTrue(first.diagnostics.source_temporal_residual_injected)
        self.assertFalse(first.diagnostics.temporal_order_invariant_carrier)
        self.assertFalse(first.diagnostics.ordered_source_trajectory_injected)
        permutation = first.diagnostics.shuffled_frame_permutation
        self.assertEqual(len(permutation), 21)
        self.assertEqual(sorted(permutation), list(range(21)))
        self.assertNotEqual(permutation, tuple(range(21)))
        self.assertTrue(first.receipt["carrier"]["action_leakage_risk"])
        self.assertFalse(first.receipt["carrier"]["primary_arm_eligible"])
        self.assertIn("shuffled", first.receipt["carrier"]["type"])

    def test_shuffled_carrier_never_equals_ordered_source_residual(self) -> None:
        _, source = self.tensors(dtype=torch.float64)
        ordered_residual = source - source.mean(dim=2, keepdim=True)
        shuffled = bridge.deterministic_shuffled_frame_carrier(source, seed=5)
        self.assertFalse(torch.equal(shuffled, ordered_residual))
        self.assertTrue(
            torch.allclose(
                shuffled.mean(dim=2),
                torch.zeros_like(shuffled[:, :, 0]),
                atol=1.0e-15,
                rtol=0.0,
            )
        )

    def test_contract_exposes_no_ordered_carrier_or_target_path(self) -> None:
        gaussian, source = self.tensors()
        with self.assertRaisesRegex(
            bridge.SourceSpectralBridgeError, "ordered source carriers are forbidden"
        ):
            bridge.source_spectral_bridge(
                gaussian, source, rho=0.2, arm="ordered_source_trajectory"
            )
        serialized = json.dumps(
            bridge.source_spectral_bridge(gaussian, source, rho=0.2).receipt,
            sort_keys=True,
        )
        self.assertIn('"non_gaussian_initial_noise": true', serialized)
        self.assertIn('"ordered_source_trajectory_injected": false', serialized)
        self.assertIn('"same_contract": true', serialized)
        self.assertIn('"source_only_conditioning": true', serialized)
        self.assertNotIn("trainer_state", serialized)

    def test_rejects_non_exact81_mismatch_bad_rho_and_degenerate_active_carrier(self) -> None:
        gaussian, source = self.tensors()
        cases = (
            (gaussian[:, :, :-1], source[:, :, :-1], 0.2, "must be"),
            (gaussian, source[:, :, :, :-1], 0.2, "share shape"),
            (gaussian, source, -0.1, "rho"),
            (gaussian, source, 1.1, "rho"),
        )
        for candidate, candidate_source, rho, message in cases:
            with self.subTest(rho=rho, shape=tuple(candidate.shape)):
                with self.assertRaisesRegex(
                    bridge.SourceSpectralBridgeError, message
                ):
                    bridge.source_spectral_bridge(
                        candidate, candidate_source, rho=rho
                    )
        constant_source = torch.ones_like(source)
        with self.assertRaisesRegex(
            bridge.SourceSpectralBridgeError, "source carrier is degenerate"
        ):
            bridge.source_spectral_bridge(
                gaussian, constant_source, rho=0.4, arm=bridge.PRIMARY_ARM
            )
        # Degenerate source is harmless when the bridge is exactly disabled.
        off = bridge.source_spectral_bridge(gaussian, constant_source, rho=0.0)
        self.assertIs(off.initial_noise, gaussian)


if __name__ == "__main__":
    unittest.main()
