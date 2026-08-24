from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


MODULE_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = MODULE_ROOT / "tools"
for entry in (MODULE_ROOT, TOOLS_ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import full644_exploratory_matched_r5g_root_bootstrap_probe_runner_v1 as bootstrap
import full644_exploratory_matched_r5g_static_nomodel_probe_v1 as static
import full644_exploratory_matched_spooled_launcher_auh_r5f as launcher
import materialize_full644_exploratory_matched_r5g_full16_package_v1 as materializer


EXPECTED_TASKS = tuple(
    f"shared8-{index:02d}-{arm}"
    for index in range(8)
    for arm in ("base", "full644")
)
EXPECTED_ROOT = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
    "VideoEditing/VideoEdit_experiments/"
    "bernini_full644_exploratory_matched_eval_auh_r5f_job143812_node293_"
    "full16_847b91a2_c91de7eb_d70eac5c_r1"
)
EXPECTED_CACHE = Path(
    "/tmp/bernini-full644-r5g-job143812-node293-full16-r1-rank-cache"
)
MATERIALIZER_SHA256 = (
    "9eac713809960a319013cf6c8ce00f849a28dff5b631d91a5e33316d90091001"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class R5GFull16PackageContractTests(unittest.TestCase):
    def test_full16_selection_is_exact_and_ordered(self) -> None:
        self.assertEqual(materializer.CAMPAIGN, launcher.FULL16_CAMPAIGN)
        self.assertEqual(materializer.SELECTED_TASK_IDS, EXPECTED_TASKS)
        self.assertEqual(static.CAMPAIGN, launcher.FULL16_CAMPAIGN)
        self.assertEqual(static.ALL_TASKS, EXPECTED_TASKS)
        self.assertEqual(static.SELECTED, EXPECTED_TASKS)
        self.assertEqual(bootstrap.CAMPAIGN, launcher.FULL16_CAMPAIGN)
        self.assertEqual(bootstrap.TASK_IDS, EXPECTED_TASKS)
        bootstrap.validate_campaign_contract(launcher.FULL16_CAMPAIGN, EXPECTED_TASKS)
        with self.assertRaisesRegex(
            bootstrap.R5DRootBootstrapProbeError, "full16 selection differs"
        ):
            bootstrap.validate_campaign_contract(
                launcher.CASE00_CANARY_CAMPAIGN, EXPECTED_TASKS[:2]
            )

    def test_fresh_root_cache_and_final_paths_are_disjoint_from_case00(self) -> None:
        key = ("143812", "auh7-1b-gpu-293")
        self.assertEqual(set(materializer.TARGETS), {key})
        root = materializer.TARGETS[key]
        self.assertEqual(str(root), EXPECTED_ROOT)
        self.assertEqual(materializer.PRODUCTION_RANK_CACHE_ROOT, EXPECTED_CACHE)
        self.assertEqual(static.PRODUCTION_RANK_CACHE_ROOT, EXPECTED_CACHE)
        plan = root / "plan/full644_exploratory_matched_plan_auh_r5d.json"
        value = materializer.launch_input(root, *key, plan)
        self.assertEqual(value["campaign_mode"], launcher.FULL16_CAMPAIGN)
        self.assertEqual(value["rank_cache_root"], str(EXPECTED_CACHE))
        self.assertEqual(
            value["output_report"], str(root / "final/full16_report_auh_r5.json")
        )
        self.assertEqual(
            value["runner_attestation"],
            str(root / "final/full16_runner_attestation_auh_r5.json"),
        )
        self.assertNotIn("case00", str(root))
        self.assertNotIn("case00", value["output_report"])
        self.assertNotIn("r5f-job143812-node293-r1-rank-cache", value["rank_cache_root"])
        self.assertNotEqual(
            os.path.commonpath((value["rank_cache_root"], str(root))), str(root)
        )

    def test_diagnostic_source_closure_and_pins_are_current(self) -> None:
        self.assertEqual(len(materializer.RELEASE_FILES), 17)
        self.assertEqual(materializer.RELEASE_FILES, static.RELEASE_FILES)
        self.assertEqual(len(materializer.DIAGNOSTIC_SOURCE_PINS), 3)
        for relative, expected in materializer.DIAGNOSTIC_SOURCE_PINS.items():
            with self.subTest(relative=relative):
                self.assertEqual(sha(MODULE_ROOT.parent.parent / relative), expected)
        self.assertTrue(
            materializer.ROOT_BOOTSTRAP_PROBE.endswith(
                "full644_exploratory_matched_r5g_root_bootstrap_probe_runner_v1.py"
            )
        )
        self.assertTrue(
            materializer.STATIC_NOMODEL_PROBE.endswith(
                "full644_exploratory_matched_r5g_static_nomodel_probe_v1.py"
            )
        )

    def test_one_shot_materializer_controller_pins_the_fresh_snapshot(self) -> None:
        controller = (
            MODULE_ROOT
            / "scripts/auh_materialize_full644_exploratory_matched_r5g_"
            "full16_job143812_node293_once_v1.sh"
        )
        source = controller.read_text(encoding="utf-8")
        self.assertEqual(
            sha(TOOLS_ROOT / "materialize_full644_exploratory_matched_r5g_full16_package_v1.py"),
            MATERIALIZER_SHA256,
        )
        self.assertIn(MATERIALIZER_SHA256, source)
        self.assertIn("r5g_full16_source_snapshot_21_20260820_r1", source)
        self.assertIn("full16_847b91a2_c91de7eb_d70eac5c_r1", source)
        self.assertNotIn("r5f_source_snapshot_21_20260820_r1", source)
        self.assertNotIn("case00_847b91a2_c91de7eb_d70eac5c_r1", source)

    def test_one_shot_gpu_controller_is_exact8_full16_and_compiles(self) -> None:
        controller = (
            MODULE_ROOT
            / "scripts/auh_launch_full644_exploratory_matched_r5g_"
            "full16_job143812_node293_once_v1.sh"
        )
        source = controller.read_text(encoding="utf-8")
        embedded = source.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
        compile(embedded, str(controller) + ":preflight", "exec")
        self.assertIn("--job-name=f644-r5g-full16-gpu", source)
        self.assertIn("--exclusive --exact --immediate=10", source)
        self.assertIn("--cpus-per-task=64 --mem=64G --gpus-per-node=8", source)
        self.assertIn("--export=NONE --time=03:00:00", source)
        self.assertIn("R5G_FULL16_PREFLIGHT_PASS", source)
        self.assertIn("module.validate_release_tree", source)
        self.assertIn("module.validate_plan", source)
        self.assertIn("module.validate_launch_receipt", source)
        self.assertIn("module.replay_identity_row", source)
        self.assertIn(str(EXPECTED_CACHE), source)
        self.assertIn("full16_847b91a2_c91de7eb_d70eac5c_r1", source)
        self.assertNotIn("case00_847b91a2_c91de7eb_d70eac5c_r1", source)

    def test_static_input_accepts_only_fresh_full16_cache(self) -> None:
        root = materializer.TARGETS[("143812", "auh7-1b-gpu-293")]
        plan = root / "plan/full644_exploratory_matched_plan_auh_r5d.json"
        value = materializer.launch_input(
            root, "143812", "auh7-1b-gpu-293", plan
        )
        environment = {
            "SLURM_JOB_ID": "143812",
            "SLURM_JOB_NODELIST": "auh7-1b-gpu-293",
        }
        with mock.patch.dict(os.environ, environment, clear=False), mock.patch.object(
            static.os.path, "lexists", return_value=False
        ) as lexists:
            static.validate_input(value, root, plan)
            lexists.assert_called_once_with(str(EXPECTED_CACHE))
        with mock.patch.dict(os.environ, environment, clear=False), mock.patch.object(
            static.os.path, "lexists", return_value=True
        ):
            with self.assertRaisesRegex(
                static.R5FStaticProbeError, "launch input semantics differ"
            ):
                static.validate_input(value, root, plan)

    def test_existing_target_stops_before_any_source_read_or_write(self) -> None:
        key = ("143812", "auh7-1b-gpu-293")
        with mock.patch.object(Path, "exists", return_value=True), mock.patch.object(
            materializer.os, "geteuid", return_value=2012
        ), mock.patch.object(
            materializer.os, "getegid", return_value=2000
        ), mock.patch.object(
            materializer, "stable_file"
        ) as stable, mock.patch.object(
            materializer, "mkdir_fresh"
        ) as mkdir:
            with self.assertRaisesRegex(
                materializer.R5FMaterializationError, "fresh r5f root exists"
            ):
                materializer._materialize(
                    types.SimpleNamespace(
                        job_id=key[0],
                        node=key[1],
                        source_root=str(MODULE_ROOT.parents[1]),
                    )
                )
            stable.assert_not_called()
            mkdir.assert_not_called()


if __name__ == "__main__":
    unittest.main()
