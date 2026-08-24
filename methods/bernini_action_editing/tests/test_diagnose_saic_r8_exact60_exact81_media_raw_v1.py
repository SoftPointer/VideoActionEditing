from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import diagnose_saic_r8_exact60_exact81_media_raw_v1 as diagnostic


class R8Exact60Exact81MediaRawContractTest(unittest.TestCase):
    def test_exact_r8_input_and_runtime_sources_are_pinned(self) -> None:
        self.assertEqual(diagnostic.EXPECTED_CANDIDATE_COUNT, 60)
        self.assertEqual(diagnostic.EXPECTED_REGISTERED_SOURCE_COUNT, 8)
        self.assertEqual(diagnostic.EXPECTED_CANDIDATE_BOUND_SOURCE_COUNT, 8)
        self.assertEqual(
            diagnostic.EXPECTED_JOB_ID,
            "135056",
        )
        self.assertEqual(
            diagnostic.EXPECTED_SOURCE_REVISION,
            "ddc8a79199aed1391cf089f51835c2bbfa74ae28",
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
        self.assertEqual(
            diagnostic.EXPECTED_FROZEN_INPUT_SOURCE_SHA256,
            "2839641766b4605311f0c4e7a3ff41ea9a322ad44cf1d12d21fb7f0cca5ab24e",
        )
        self.assertEqual(diagnostic.EXPECTED_SELECTED_FRAME_INDICES, list(range(0, 81, 5)))

    def test_actual_batch_source_hash_is_frozen(self) -> None:
        self.assertEqual(
            hashlib.sha256(Path(diagnostic.__file__).read_bytes()).hexdigest(),
            "af3b953d381fb52a519d12e234cab08135dea66b3f48fd5347324b326eab3cf2",
        )

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

    def test_deep_audit_authority_is_exactly_closed(self) -> None:
        expected = dict(diagnostic.EXPECTED_DEEP_AUDIT_AUTHORITY)
        self.assertEqual(diagnostic._validate_deep_audit_authority(expected), expected)
        missing = dict(expected)
        missing.pop("optimizer")
        extra = {**expected, "hostile_false_authority": False}
        elevated = {**expected, "optimizer": True}
        for hostile in ({}, missing, extra, elevated):
            with self.assertRaises(diagnostic.Exact60MediaRawError):
                diagnostic._validate_deep_audit_authority(hostile)

    def test_nested_r8_exact60_validator_identity_is_restored_after_pollution(self) -> None:
        diagnostic.frozen_input.SCHEMA_VERSION = "hostile-outer"
        diagnostic.frozen_input.core.INPUT_SCHEMA = "hostile-nested"
        diagnostic.frozen_input.EXPECTED_ATTEMPT_COUNT = 47
        diagnostic.frozen_input.core.EXPECTED_WORLD_SIZE = 4
        diagnostic.frozen_input.EXPECTED_PARTITION_SIZES = (60,)
        diagnostic.frozen_input.__file__ = "/tmp/hostile-r7.py"
        diagnostic.frozen_input.core.__file__ = "/tmp/hostile-temporal.py"
        diagnostic.frozen_input.build_manifest = lambda *args, **kwargs: 97
        diagnostic.frozen_input.load_input_manifest = lambda *args, **kwargs: ({}, "")
        diagnostic.frozen_input.aggregate = lambda *args, **kwargs: 98
        diagnostic.frozen_input._worker_common = lambda *args, **kwargs: {}
        diagnostic.frozen_input.partition_indices = lambda *args, **kwargs: ()
        diagnostic.frozen_input.core.partition_indices = lambda *args, **kwargs: ()

        diagnostic._configure_frozen_input()

        expected_path = (
            METHOD_ROOT / "diagnose_saic_r8_exact60_source_bound_dinov2_raw_v1.py"
        ).resolve()
        self.assertEqual(Path(diagnostic.frozen_input.__file__).resolve(), expected_path)
        self.assertEqual(Path(diagnostic.frozen_input.core.__file__).resolve(), expected_path)
        self.assertEqual(diagnostic.frozen_input.EXPECTED_ATTEMPT_COUNT, 60)
        self.assertEqual(diagnostic.frozen_input.core.EXPECTED_ATTEMPT_COUNT, 60)
        self.assertEqual(diagnostic.frozen_input.EXPECTED_WORLD_SIZE, 8)
        self.assertEqual(diagnostic.frozen_input.core.EXPECTED_WORLD_SIZE, 8)
        self.assertEqual(
            diagnostic.frozen_input.EXPECTED_PARTITION_SIZES,
            (8, 8, 8, 8, 7, 7, 7, 7),
        )
        self.assertIs(
            diagnostic.frozen_input.build_manifest,
            diagnostic.frozen_specialization.build_manifest,
        )
        self.assertIs(
            diagnostic.frozen_input.load_input_manifest,
            diagnostic.frozen_specialization.load_input_manifest,
        )
        self.assertIs(
            diagnostic.frozen_input.aggregate,
            diagnostic.frozen_specialization.aggregate,
        )
        self.assertIs(
            diagnostic.frozen_input._worker_common,
            diagnostic.frozen_specialization._worker_common,
        )
        self.assertIs(
            diagnostic.frozen_input.partition_indices,
            diagnostic.frozen_specialization.partition_indices,
        )
        self.assertIs(
            diagnostic.frozen_input.core.partition_indices,
            diagnostic.frozen_specialization.partition_indices,
        )
        self.assertEqual(
            tuple(
                diagnostic.frozen_input.partition_indices(60, rank, 8)
                for rank in range(8)
            ),
            tuple(tuple(range(rank, 60, 8)) for rank in range(8)),
        )

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

    def test_launcher_is_allocation_bound_cpu_only_and_exact60(self) -> None:
        launcher = (
            METHOD_ROOT
            / "scripts"
            / "auh_diagnose_saic_r8_exact60_exact81_media_raw_v1.sh"
        ).read_text("utf-8")
        self.assertIn('if [[ "$#" -ne 15 ]]', launcher)
        self.assertIn('${SLURM_JOB_ID:-}', launcher)
        self.assertIn("allocated_cpus >= 32", launcher)
        self.assertIn('export ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""', launcher)
        self.assertIn("--workers 16", launcher)
        self.assertIn("ffmpeg_sha256", launcher)
        self.assertIn("ffprobe_sha256", launcher)
        self.assertIn("-eq 60", launcher)
        self.assertIn("saic-exact60-terminal-evidence-135056.json", launcher)
        self.assertIn(
            "expected_batch_source_sha=af3b953d381fb52a519d12e234cab08135dea66b3f48fd5347324b326eab3cf2",
            launcher,
        )
        self.assertIn(
            "expected_cyclic_source_sha=2839641766b4605311f0c4e7a3ff41ea9a322ad44cf1d12d21fb7f0cca5ab24e",
            launcher,
        )
        self.assertIn(
            '[[ "$batch_source_sha256" == "$expected_batch_source_sha" ]]',
            launcher,
        )
        self.assertIn(
            '[[ "$exact81_source_sha256" == "$expected_exact81_source_sha" ]]',
            launcher,
        )
        self.assertIn(
            '[[ "$(sha256sum "$cyclic_source" | awk \'{print $1}\')" == "$expected_cyclic_source_sha" ]]',
            launcher,
        )
        self.assertIn('--terminal-evidence "$terminal_evidence"', launcher)
        self.assertIn("CPU_only=true authority=zero ranking=false selection=false", launcher)
        self.assertNotIn("sbatch ", launcher)
        self.assertNotIn("ROCR_VISIBLE_DEVICES=\"$rank\"", launcher)

    def test_aggregate_contract_has_no_threshold_or_order_by_metric(self) -> None:
        source = Path(diagnostic.__file__).read_text("utf-8")
        self.assertIn(
            '"coverage": "exactly_once_complete_terminal_r8_exact60_exact81_media_raw"',
            source,
        )
        self.assertIn('"thresholds": None', source)
        self.assertIn('"descriptive_statistics_only": True', source)
        self.assertIn('"ranking_or_selection_performed": False', source)
        self.assertIn("runtime_source_and_tool_closure", source)
        self.assertNotIn("sorted(results, key=", source)

    def test_terminal_success_and_bank_are_independently_replayed(self) -> None:
        source = Path(diagnostic.__file__).read_text("utf-8")
        for token in (
            'parsed.get("State") != "COMPLETED"',
            'parsed.get("ExitCode") != "0:0"',
            'terminal.get("job_id") != EXPECTED_JOB_ID',
            'master.get("attempt_count") != EXPECTED_CANDIDATE_COUNT',
            'audit.get("completed_prefix_count") != 30',
            'master_row.get("mp4_sha256") != row.get("mp4_sha256")',
        ):
            self.assertIn(token, source)
        self.assertNotIn("t2v-events-topup-r7", source)

    def test_hostile_caller_hash_cannot_rebind_frozen_sources(self) -> None:
        launcher = (
            METHOD_ROOT
            / "scripts"
            / "auh_diagnose_saic_r8_exact60_exact81_media_raw_v1.sh"
        ).read_text("utf-8")
        self.assertIn(
            'fail "r8 exact60 batch source SHA-256 argument differs from hard pin"',
            launcher,
        )
        self.assertIn(
            'fail "frozen exact81 primitive SHA-256 argument differs from hard pin"',
            launcher,
        )
        self.assertIn(
            'fail "frozen r8 exact60 cyclic source SHA-256 differs from hard pin"',
            launcher,
        )

    def test_main_help_is_repeatable_in_one_interpreter(self) -> None:
        for entrypoint in (diagnostic.frozen_specialization.main, diagnostic.main):
            for _ in range(2):
                with self.assertRaises(SystemExit) as raised:
                    entrypoint(["--help"])
                self.assertEqual(raised.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
