from __future__ import annotations

import inspect
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
    import fewshot_teacher_objective as fto  # noqa: E402
else:  # pragma: no cover - exercised only in dependency-light environments
    fto = None


def _clean_video(*, batch: int = 1) -> "torch.Tensor":
    return torch.zeros(batch, 16, 21, 8, 8, dtype=torch.float32)


def _centered_motion(source: "torch.Tensor") -> "torch.Tensor":
    phases = torch.arange(21, dtype=torch.float32, device=source.device)
    amplitude = torch.sin(2.0 * torch.pi * phases / 20.0)
    spatial = torch.linspace(
        -1.0, 1.0, 64, dtype=torch.float32, device=source.device
    ).reshape(1, 1, 1, 8, 8)
    channel = torch.linspace(
        0.25, 1.0, 16, dtype=torch.float32, device=source.device
    ).reshape(1, 16, 1, 1, 1)
    relative = channel * amplitude.reshape(1, 1, 21, 1, 1) * spatial
    return source + relative.expand(source.shape[0], -1, -1, -1, -1)


def _zero_gates(*, batch: int = 1) -> tuple["torch.Tensor", "torch.Tensor"]:
    return (
        torch.zeros(batch, 21, dtype=torch.float32),
        torch.zeros(batch, 16, dtype=torch.float32),
    )


@unittest.skipIf(torch is None, "torch is unavailable")
class ObjectiveContractTests(unittest.TestCase):
    def test_contract_is_motion_only_and_full_target_fm_is_impossible(self) -> None:
        contract = fto.objective_contract()
        self.assertEqual(contract["clean_video_shape"], "[B,16,21,H,W]")
        self.assertEqual(contract["feature"]["temporal_lags"], [1, 2, 4])
        self.assertEqual(contract["feature"]["spatial_pool"], [8, 8])
        self.assertEqual(contract["code"]["dimension"], 36)
        self.assertEqual(contract["weights"]["full_target_flow_matching"], 0.0)
        self.assertEqual(fto.FULL_TARGET_FLOW_MATCHING_WEIGHT, 0.0)

        parameters = tuple(inspect.signature(fto.fewshot_teacher_objective).parameters)
        self.assertEqual(
            parameters,
            (
                "source_clean",
                "predicted_clean",
                "target_clean",
                "phase_gates",
                "block_gates",
            ),
        )
        for forbidden in (
            "mask",
            "flow",
            "pose",
            "track",
            "trajectory",
            "target_velocity",
            "full_target_loss",
        ):
            self.assertNotIn(forbidden, parameters)

    def test_strict_clean_layout_dtype_and_finiteness_fail_closed(self) -> None:
        source = _clean_video()
        phase, block = _zero_gates()
        invalid_cases = (
            torch.zeros(1, 16, 20, 8, 8, dtype=torch.float32),
            torch.zeros(1, 15, 21, 8, 8, dtype=torch.float32),
            torch.zeros(1, 16, 21, 7, 8, dtype=torch.float32),
            source.to(torch.float64),
        )
        for invalid in invalid_cases:
            with self.subTest(shape=tuple(invalid.shape), dtype=invalid.dtype):
                with self.assertRaises(fto.FewShotTeacherObjectiveError):
                    fto.fewshot_teacher_objective(
                        source, invalid, source, phase, block
                    )

        nonfinite = source.clone()
        nonfinite[0, 0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(fto.FewShotTeacherObjectiveError, "NaN"):
            fto.source_relative_motion_features(nonfinite, source)

        with self.assertRaisesRegex(
            fto.FewShotTeacherObjectiveError, "gate batch"
        ):
            fto.fewshot_teacher_objective(
                source,
                source,
                source,
                torch.zeros(2, 21, dtype=torch.float32),
                torch.zeros(2, 16, dtype=torch.float32),
            )


@unittest.skipIf(torch is None, "torch is unavailable")
class MotionFeatureTests(unittest.TestCase):
    def test_feature_shapes_q0_rms_and_clip(self) -> None:
        source = _clean_video(batch=2)
        candidate = _centered_motion(source)
        candidate[0, 0, 10, 0, 0] += 1000.0
        features = fto.source_relative_motion_features(candidate, source)

        self.assertEqual(tuple(features.pooled_q0.shape), (2, 16, 21, 8, 8))
        self.assertTrue(
            torch.allclose(
                features.pooled_q0.mean(dim=2),
                torch.zeros(2, 16, 8, 8),
                atol=1.0e-5,
                rtol=0.0,
            )
        )
        for lag, normalized, phase_rms in zip(
            fto.TEMPORAL_LAGS,
            features.normalized_by_lag,
            features.phase_rms_by_lag,
        ):
            self.assertEqual(
                tuple(normalized.shape), (2, 16, 21 - lag, 8, 8)
            )
            self.assertEqual(tuple(phase_rms.shape), (2, 21 - lag))
            self.assertTrue(torch.isfinite(normalized).all().item())
            self.assertLessEqual(
                normalized.abs().max().item(), fto.NORMALIZED_FEATURE_CLIP
            )
        self.assertEqual(
            features.normalized_by_lag[0].abs().max().item(),
            fto.NORMALIZED_FEATURE_CLIP,
        )

    def test_temporal_dc_is_removed_from_F_but_penalized_separately(self) -> None:
        source = _clean_video()
        shifted = source + 0.25
        source_features = fto.source_relative_motion_features(source, source)
        shifted_features = fto.source_relative_motion_features(shifted, source)
        for baseline, changed in zip(
            source_features.normalized_by_lag,
            shifted_features.normalized_by_lag,
        ):
            self.assertTrue(torch.equal(baseline, changed))
        for baseline, changed in zip(
            source_features.phase_rms_by_lag,
            shifted_features.phase_rms_by_lag,
        ):
            self.assertTrue(torch.equal(baseline, changed))
        self.assertEqual(
            fto.temporal_dc_residual_penalty(source, source).item(), 0.0
        )
        self.assertGreater(
            fto.temporal_dc_residual_penalty(shifted, source).item(), 0.0
        )

    def test_exact_match_is_zero_and_perturbation_has_finite_gradient(self) -> None:
        source = _clean_video()
        target = _centered_motion(source)
        target_features = fto.source_relative_motion_features(target, source)
        self.assertEqual(
            fto.motion_feature_match_loss(target_features, target_features).item(),
            0.0,
        )
        self.assertEqual(
            fto.phase_rms_match_loss(target_features, target_features).item(),
            0.0,
        )

        predicted = target.clone()
        predicted[:, 0, 7, 1, 2] += 0.4
        predicted.requires_grad_(True)
        phase, block = _zero_gates()
        result = fto.fewshot_teacher_objective(
            source, predicted, target, phase, block
        )
        self.assertGreater(result.total.item(), 0.0)
        self.assertEqual(result.full_target_flow_matching_weight, 0.0)
        result.total.backward()
        self.assertIsNotNone(predicted.grad)
        self.assertTrue(torch.isfinite(predicted.grad).all().item())
        self.assertGreater(predicted.grad.abs().sum().item(), 0.0)

    def test_motionless_input_has_stable_zero_gradient(self) -> None:
        source = _clean_video()
        predicted = source.clone().requires_grad_(True)
        phase, block = _zero_gates()
        result = fto.fewshot_teacher_objective(
            source, predicted, source, phase, block
        )
        self.assertEqual(result.total.item(), 0.0)
        result.total.backward()
        self.assertTrue(torch.isfinite(predicted.grad).all().item())
        self.assertEqual(torch.count_nonzero(predicted.grad).item(), 0)

    def test_phase0_base_parity_is_explicit(self) -> None:
        source = _clean_video()
        changed = source.clone()
        changed[:, :, 0] = 0.5
        self.assertGreater(
            fto.target_phase0_base_parity_penalty(changed, source).item(), 0.0
        )
        self.assertEqual(
            fto.target_phase0_base_parity_penalty(source, source).item(), 0.0
        )


@unittest.skipIf(torch is None, "torch is unavailable")
class GateAndControlTests(unittest.TestCase):
    def test_gate_shape_bound_phase0_l2_and_signed_zero(self) -> None:
        phase = torch.full((1, 21), 0.5, dtype=torch.float32)
        phase[:, 0] = 0.0
        block = torch.full((1, 16), 0.5, dtype=torch.float32)
        self.assertAlmostEqual(fto.gate_l2_penalty(phase, block).item(), 0.25)
        self.assertEqual(tuple(fto.flatten_phase_block_code(phase, block).shape), (1, 36))

        signed_zero = torch.zeros(1, 21, dtype=torch.float32)
        signed_zero[:, 0] = -0.0
        with self.assertRaisesRegex(
            fto.FewShotTeacherObjectiveError, "positive zero"
        ):
            fto.gate_l2_penalty(signed_zero, block)

        unbounded = block.clone()
        unbounded[:, 0] = 1.01
        with self.assertRaisesRegex(fto.FewShotTeacherObjectiveError, r"\[-1,1\]"):
            fto.gate_l2_penalty(phase, unbounded)

    def test_reverse_shuffle_are_frozen_bijections_preserving_phase0(self) -> None:
        video = _clean_video()
        for phase in range(21):
            video[:, :, phase] = float(phase)
        reverse = fto.reverse_nonboundary_phases(video)
        shuffled = fto.shuffle_nonboundary_phases(video)
        self.assertTrue(torch.equal(reverse[:, :, 0], video[:, :, 0]))
        self.assertTrue(torch.equal(shuffled[:, :, 0], video[:, :, 0]))
        self.assertEqual(reverse[0, 0, :, 0, 0].tolist(), [0.0] + list(range(20, 0, -1)))
        self.assertEqual(
            shuffled[0, 0, :, 0, 0].tolist(),
            [float(index) for index in fto.SHUFFLE_PHASE_INDICES],
        )
        self.assertEqual(sorted(fto.SHUFFLE_PHASE_INDICES), list(range(21)))

        arbitrary = list(range(21))
        arbitrary[1], arbitrary[2] = arbitrary[2], arbitrary[1]
        with self.assertRaisesRegex(fto.FewShotTeacherObjectiveError, "frozen"):
            fto.permute_nonboundary_phases(video, arbitrary)
        with self.assertRaisesRegex(fto.FewShotTeacherObjectiveError, "phase 0"):
            fto.permute_nonboundary_phases(video, list(range(1, 21)) + [0])

    def test_relative_contrast_margin_has_zero_and_positive_hinges(self) -> None:
        passing = fto.reverse_shuffle_contrast_margin(
            torch.tensor(1.0), torch.tensor(1.1), torch.tensor(1.2)
        )
        self.assertEqual(passing.total.item(), 0.0)
        failing = fto.reverse_shuffle_contrast_margin(
            torch.tensor(1.0), torch.tensor(1.01), torch.tensor(1.2)
        )
        self.assertGreater(failing.reverse_hinge.item(), 0.0)
        self.assertEqual(failing.shuffle_hinge.item(), 0.0)


@unittest.skipIf(torch is None, "torch is unavailable")
class HeldNoiseGoTests(unittest.TestCase):
    @staticmethod
    def _support_codes(cosine: float = 0.8) -> "torch.Tensor":
        codes = torch.zeros(2, 36, dtype=torch.float32)
        codes[0, 0] = 1.0
        codes[1, 0] = cosine
        codes[1, 1] = (1.0 - cosine**2) ** 0.5
        return codes

    @staticmethod
    def _gates(*, saturated: int) -> tuple["torch.Tensor", "torch.Tensor"]:
        phase, block = _zero_gates()
        flattened = torch.zeros(36, dtype=torch.float32)
        flattened[:saturated] = 0.96
        phase[:, 1:] = flattened[:20]
        block[:] = flattened[20:]
        return phase, block

    def test_minimum_pairwise_cosine(self) -> None:
        codes = self._support_codes(0.8)
        self.assertAlmostEqual(
            fto.minimum_pairwise_support_code_cosine(codes), 0.8, places=6
        )
        with self.assertRaisesRegex(fto.FewShotTeacherObjectiveError, "zero"):
            fto.minimum_pairwise_support_code_cosine(torch.zeros(2, 36))

    def test_go_thresholds_and_strict_saturation_boundary(self) -> None:
        phase, block = self._gates(saturated=8)
        stats = fto.build_held_noise_statistics(
            zero_loss=1.0,
            correct_loss=0.85,
            reverse_loss=0.90,
            shuffle_loss=0.90,
            phase_gates=phase,
            block_gates=block,
            support_codes=self._support_codes(0.8),
        )
        decision = fto.evaluate_teacher_go(stats)
        self.assertTrue(decision.go)
        self.assertEqual(decision.failed_checks, ())
        self.assertAlmostEqual(stats.gate_saturation_fraction, 8.0 / 36.0)

        phase, block = self._gates(saturated=9)
        boundary = fto.build_held_noise_statistics(
            zero_loss=1.0,
            correct_loss=0.85,
            reverse_loss=0.90,
            shuffle_loss=0.90,
            phase_gates=phase,
            block_gates=block,
            support_codes=self._support_codes(0.8),
        )
        boundary_decision = fto.evaluate_teacher_go(boundary)
        self.assertFalse(boundary_decision.go)
        self.assertIn(
            "gate_saturation_strictly_below_25pct",
            boundary_decision.failed_checks,
        )

    def test_each_failed_signal_blocks_go_and_zero_baseline_is_required(self) -> None:
        phase, block = self._gates(saturated=0)
        weak = fto.build_held_noise_statistics(
            zero_loss=1.0,
            correct_loss=0.86,
            reverse_loss=0.90,
            shuffle_loss=0.90,
            phase_gates=phase,
            block_gates=block,
            support_codes=self._support_codes(0.59),
        )
        decision = fto.evaluate_teacher_go(weak)
        self.assertFalse(decision.go)
        self.assertIn("zero_improvement_at_least_15pct", decision.failed_checks)
        self.assertIn("support_code_cosine_at_least_0p6", decision.failed_checks)

        with self.assertRaisesRegex(
            fto.FewShotTeacherObjectiveError, "zero_loss must be positive"
        ):
            fto.build_held_noise_statistics(
                zero_loss=0.0,
                correct_loss=0.0,
                reverse_loss=0.0,
                shuffle_loss=0.0,
                phase_gates=phase,
                block_gates=block,
                support_codes=self._support_codes(),
            )


if __name__ == "__main__":
    unittest.main()
