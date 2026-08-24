from __future__ import annotations

from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import diagnose_saic_partial47_exact81_media_raw_v1 as diagnostic


class Exact47Exact81MediaRawContractTest(unittest.TestCase):
    def test_exact_r4_input_and_runtime_sources_are_pinned(self) -> None:
        self.assertEqual(diagnostic.EXPECTED_CANDIDATE_COUNT, 47)
        self.assertEqual(diagnostic.EXPECTED_REGISTERED_SOURCE_COUNT, 8)
        self.assertEqual(diagnostic.EXPECTED_CANDIDATE_BOUND_SOURCE_COUNT, 7)
        self.assertEqual(
            diagnostic.EXPECTED_INPUT_MANIFEST_SHA256,
            "9fa14916dbfdc36bda0c9ee67180da72b4374f0bdc213b0421c8d48dda9de4cb",
        )
        self.assertEqual(
            diagnostic.EXPECTED_ROOT_SPEC_SHA256,
            "d693d0784530f007888e2825d15db3db808fdf4f1d111b5d080d968c894ff145",
        )
        self.assertEqual(
            diagnostic.EXPECTED_SOURCE_MANIFEST_SHA256,
            "899b5a1dd66fc0bf6d4d0192fb6157f4afe691c50633246dddcaa1db2c2a98a9",
        )
        self.assertEqual(
            diagnostic.PINNED_LOCAL_SOURCES[
                "saic_exact81_media_diagnostics_v1.py"
            ],
            "3658056640b0adc3411c04c029ce99efd5a4d9388be638f659bd8eb472399e0a",
        )
        self.assertEqual(diagnostic.EXPECTED_SELECTED_FRAME_INDICES, list(range(0, 81, 5)))

    def test_all_batch_authority_is_permanently_false(self) -> None:
        self.assertTrue(diagnostic.AUTHORITY)
        self.assertTrue(all(value is False for value in diagnostic.AUTHORITY.values()))
        for field in (
            "scientific_claim_authorized",
            "ranking_authorized",
            "candidate_selection_allowed",
            "training_allowed",
            "training_target_authorized",
            "optimizer_step_allowed",
            "optimizer_or_parameter_update_authorized",
        ):
            self.assertFalse(diagnostic.AUTHORITY[field])

    def test_descriptive_statistics_do_not_rank_or_threshold(self) -> None:
        first = {
            name: float(index + 1)
            for index, name in enumerate(diagnostic._SUMMARY_METRICS)
        }
        second = {name: value + 2.0 for name, value in first.items()}
        summary = diagnostic.descriptive_statistics([first, second])
        self.assertEqual(set(summary), set(diagnostic._SUMMARY_METRICS))
        for name, value in summary.items():
            self.assertEqual(value["count"], 2)
            self.assertEqual(value["mean"], value["median"])
            self.assertLess(value["minimum"], value["maximum"])
            self.assertNotIn("threshold", value)
            self.assertNotIn("rank", value)

    def test_metric_projection_is_raw_closed_camera_technical_temporal(self) -> None:
        comparisons = {
            "camera_trajectory": {
                "global_mean_xy_l2_difference_mean": 1.0,
                "global_mean_xy_l2_difference_p90": 2.0,
                "global_mean_xy_l2_difference_max": 3.0,
                "global_speed_mean_absolute_difference": 4.0,
                "cumulative_global_endpoint_l2_difference": 5.0,
            },
            "scene_cut_ratio_absolute_difference": 6.0,
            "temporal_energy_cv_absolute_difference": 7.0,
            "technical": {
                "sharpness_retention_diagnostic": 8.0,
                "candidate_exposure_diagnostic": 9.0,
                "nonfreeze_retention_diagnostic": 10.0,
                "global_flicker_agreement_diagnostic": 11.0,
                "geometric_mean_technical_diagnostic": 12.0,
            },
        }
        projection = diagnostic._metric_projection({"comparisons": comparisons})
        self.assertEqual(set(projection), set(diagnostic._SUMMARY_METRICS))
        self.assertEqual(len(projection), 12)
        self.assertEqual(projection["technical_geometric_mean_diagnostic"], 12.0)

    def test_exact81_decode_contract_is_rational_25_over_1(self) -> None:
        source = Path(diagnostic.__file__).read_text("utf-8")
        self.assertIn('decode.get("frame_count") != 81', source)
        self.assertIn('decode.get("fps_numerator") != 25', source)
        self.assertIn('decode.get("fps_denominator") != 1', source)
        self.assertNotIn('decode", {}).get("fps")', source)

    def test_launcher_is_allocation_bound_cpu_only_and_exact47(self) -> None:
        launcher = (
            METHOD_ROOT
            / "scripts"
            / "auh_diagnose_saic_partial47_exact81_media_raw_v1.sh"
        ).read_text("utf-8")
        self.assertIn('if [[ "$#" -ne 14 ]]', launcher)
        self.assertIn('${SLURM_JOB_ID:-}', launcher)
        self.assertIn("allocated_cpus >= 32", launcher)
        self.assertIn('export ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""', launcher)
        self.assertIn("--workers 16", launcher)
        self.assertIn("ffmpeg_sha256", launcher)
        self.assertIn("ffprobe_sha256", launcher)
        self.assertIn("-eq 47", launcher)
        self.assertIn("CPU_only=true authority=zero ranking=false selection=false", launcher)
        self.assertNotIn("sbatch ", launcher)
        self.assertNotIn("ROCR_VISIBLE_DEVICES=\"$rank\"", launcher)

    def test_aggregate_contract_has_no_threshold_or_order_by_metric(self) -> None:
        source = Path(diagnostic.__file__).read_text("utf-8")
        self.assertIn(
            '"coverage": "exactly_once_complete_sealed_r4_partial47_exact81_media_raw"',
            source,
        )
        self.assertIn('"thresholds": None', source)
        self.assertIn('"descriptive_statistics_only": True', source)
        self.assertIn('"ranking_or_selection_performed": False', source)
        self.assertIn("runtime_source_and_tool_closure", source)
        self.assertNotIn("sorted(results, key=", source)


if __name__ == "__main__":
    unittest.main()
