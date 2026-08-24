from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import checkpoint_visual_quality_gate_v3 as gate_v3
import checkpoint_visual_quality_gate_v3_1 as gate
import replay_checkpoint_visual_quality_gate_v3_1 as replay_gate


def _scalars(**overrides):
    values = {
        "hp_kurtosis_median": 10.0,
        "spectral_flatness_median": 0.30,
        "chroma_hp_ratio_to_base": 1.0,
        "motion_compensated_temporal_residual_ratio_to_base": 1.0,
        "frame_hp_rms_retention_to_base_p10": 1.0,
        "frame_laplacian_var_retention_to_base_p10": 1.0,
        "base_salient_low_retention_frame_fraction": 0.0,
        "structure_support_low_retention_frame_fraction": 0.0,
        "candidate_base_windowed_ssim_mean": 1.0,
        "candidate_base_global_ssim_mean": 1.0,
        "candidate_base_edge_correlation_mean": 1.0,
        "base_source_global_ssim_mean": 1.0,
        "candidate_near_duplicate_transition_fraction": 0.0,
        "near_duplicate_fraction_excess_over_base": 0.0,
    }
    values.update(overrides)
    return values


def _features(low=None, high=None):
    return {
        "192x144": {"scalars": _scalars(**(low or {}))},
        "384x288": {"scalars": _scalars(**(high or {}))},
    }


class CheckpointVisualQualityGateV31DecisionTest(unittest.TestCase):
    def test_schema_is_independent_and_v3_is_unchanged(self) -> None:
        self.assertEqual(
            gate.SCHEMA_VERSION,
            "bernini-checkpoint-visual-quality-gate-v3.1",
        )
        self.assertEqual(
            gate.FEATURE_EXTRACTOR_SCHEMA_VERSION,
            "bernini-checkpoint-visual-quality-gate-v3",
        )
        self.assertEqual(
            gate_v3.SCHEMA_VERSION,
            "bernini-checkpoint-visual-quality-gate-v3",
        )
        self.assertEqual(
            gate.TOOL_SHA256,
            hashlib.sha256(Path(gate.__file__).read_bytes()).hexdigest(),
        )
        self.assertEqual(
            gate.FEATURE_EXTRACTOR_TOOL_SHA256,
            hashlib.sha256(Path(gate_v3.__file__).read_bytes()).hexdigest(),
        )

    def test_s359_case00_one_scale_chroma_and_salient_loss_is_not_hard(self) -> None:
        # Metrics are the audited S359/case00 v3 scalars.  The 192 scale has
        # the lone chroma excess; the 384 scale has the lone salient loss.
        decision = gate._decision(
            _features(
                low={
                    "hp_kurtosis_median": 27.17290437275594,
                    "spectral_flatness_median": 0.3230210795971014,
                    "chroma_hp_ratio_to_base": 1.5387707948684692,
                    "motion_compensated_temporal_residual_ratio_to_base": 1.4223779278274358,
                    "frame_hp_rms_retention_to_base_p10": 0.7266230583190918,
                    "frame_laplacian_var_retention_to_base_p10": 0.5462298393249512,
                    "base_salient_low_retention_frame_fraction": 0.0,
                    "structure_support_low_retention_frame_fraction": 0.0,
                    "candidate_base_windowed_ssim_mean": 0.45715240600668355,
                    "candidate_base_global_ssim_mean": 0.5726400014523783,
                    "candidate_base_edge_correlation_mean": 0.16220282919152398,
                    "base_source_global_ssim_mean": 0.38976983564060735,
                },
                high={
                    "hp_kurtosis_median": 43.891430331670755,
                    "spectral_flatness_median": 0.47432440077617977,
                    "chroma_hp_ratio_to_base": 1.3126769065856934,
                    "motion_compensated_temporal_residual_ratio_to_base": 1.2874819458819653,
                    "frame_hp_rms_retention_to_base_p10": 0.7297065854072571,
                    "frame_laplacian_var_retention_to_base_p10": 0.5578593611717224,
                    "base_salient_low_retention_frame_fraction": 0.25925925925925924,
                    "structure_support_low_retention_frame_fraction": 0.32098765432098764,
                    "candidate_base_windowed_ssim_mean": 0.5206039621506209,
                    "candidate_base_global_ssim_mean": 0.5626946359589146,
                    "candidate_base_edge_correlation_mean": 0.1076869931490775,
                    "base_source_global_ssim_mean": 0.3801526305218879,
                },
            )
        )
        self.assertEqual(decision["outcome"], "unresolved")
        self.assertFalse(decision["hard_artifact_failure"])
        self.assertEqual(decision["failure_codes"], [])
        self.assertIn(
            "quality_noise_requires_external_verifier",
            decision["unresolved_codes"],
        )
        self.assertIn(
            "quality_blur_requires_external_verifier",
            decision["unresolved_codes"],
        )

    def test_one_scale_chroma_cannot_hard_fail_noise_or_structure(self) -> None:
        decision = gate._decision(
            _features(
                low={
                    "chroma_hp_ratio_to_base": 1.51,
                    "candidate_base_windowed_ssim_mean": 0.2,
                    "candidate_base_global_ssim_mean": 0.4,
                    "candidate_base_edge_correlation_mean": 0.1,
                },
                high={
                    "candidate_base_windowed_ssim_mean": 0.2,
                    "candidate_base_global_ssim_mean": 0.4,
                    "candidate_base_edge_correlation_mean": 0.1,
                },
            )
        )
        self.assertEqual(decision["outcome"], "unresolved")
        self.assertFalse(
            decision["evidence_families"]["NOISE"]["triggered"]
        )
        self.assertFalse(
            decision["evidence_families"]["ROUTEOFF_STRUCTURE"]["triggered"]
        )

    def test_one_scale_salient_retention_cannot_hard_fail_blur(self) -> None:
        decision = gate._decision(
            _features(
                high={"base_salient_low_retention_frame_fraction": 0.25}
            )
        )
        self.assertEqual(decision["outcome"], "unresolved")
        self.assertFalse(decision["evidence_families"]["BLUR"]["triggered"])

    def test_cross_scale_chroma_keeps_true_noise_hard_fail_closed(self) -> None:
        decision = gate._decision(
            _features(
                low={"chroma_hp_ratio_to_base": 3.0},
                high={"chroma_hp_ratio_to_base": 2.5},
            )
        )
        self.assertEqual(decision["outcome"], "fail")
        self.assertEqual(decision["failure_codes"], ["quality_noise"])
        self.assertEqual(
            decision["evidence_families"]["NOISE"][
                "cross_scale_confirmed_evidence"
            ],
            ["chroma_highpass_excess"],
        )

    def test_two_independent_noise_kinds_can_confirm_at_one_scale(self) -> None:
        decision = gate._decision(
            _features(
                low={
                    "hp_kurtosis_median": 4.0,
                    "spectral_flatness_median": 0.08,
                    "frame_hp_rms_retention_to_base_p10": 1.20,
                    "chroma_hp_ratio_to_base": 1.60,
                }
            )
        )
        self.assertEqual(decision["outcome"], "fail")
        self.assertEqual(
            decision["evidence_families"]["NOISE"]["triggered_scales"],
            ["192x144"],
        )

    def test_cross_scale_blur_and_freeze_remain_hard(self) -> None:
        blur = gate._decision(
            _features(
                low={
                    "frame_hp_rms_retention_to_base_p10": 0.4,
                    "frame_laplacian_var_retention_to_base_p10": 0.2,
                },
                high={
                    "frame_hp_rms_retention_to_base_p10": 0.4,
                    "frame_laplacian_var_retention_to_base_p10": 0.2,
                },
            )
        )
        self.assertEqual(blur["outcome"], "fail")
        self.assertIn("quality_blur", blur["failure_codes"])

        freeze = gate._decision(
            _features(
                low={
                    "candidate_near_duplicate_transition_fraction": 0.95,
                    "near_duplicate_fraction_excess_over_base": 0.70,
                },
                high={
                    "candidate_near_duplicate_transition_fraction": 0.96,
                    "near_duplicate_fraction_excess_over_base": 0.71,
                },
            )
        )
        self.assertEqual(freeze["outcome"], "fail")
        self.assertIn("quality_freeze", freeze["failure_codes"])


class CheckpointVisualQualityGateV31ReplayTest(unittest.TestCase):
    def test_legacy_inventory_remains_seven_pass_thirteen_fail(self) -> None:
        public = REPOSITORY / "artifacts/v16r3_s644_heldout8_site_20260824/public"
        if not public.is_dir():
            self.skipTest("local labelled replay corpus is absent")
        rows = replay_gate.v3_replay._rows(public)
        self.assertEqual(len(rows), 20)
        self.assertEqual(sum(row["expected"] == "PASS" for row in rows), 7)
        self.assertEqual(sum(row["expected"] == "FAIL" for row in rows), 13)

    @unittest.skipUnless(
        os.environ.get("RUN_QUALITY_GATE_V3_1_REPLAY") == "1"
        and shutil.which("ffmpeg")
        and shutil.which("ffprobe"),
        "set RUN_QUALITY_GATE_V3_1_REPLAY=1 for full labelled replay",
    )
    def test_full_legacy_replay_remains_fail_closed_separating(self) -> None:
        public = REPOSITORY / "artifacts/v16r3_s644_heldout8_site_20260824/public"
        report = replay_gate.replay_labelled(public)
        self.assertTrue(report["all_labels_replayed_correctly"])
        self.assertEqual(
            report["confusion_matrix"],
            {
                "expected_PASS_predicted_PASS": 7,
                "expected_PASS_predicted_FAIL": 0,
                "expected_FAIL_predicted_PASS": 0,
                "expected_FAIL_predicted_FAIL": 13,
            },
        )


if __name__ == "__main__":
    unittest.main()
